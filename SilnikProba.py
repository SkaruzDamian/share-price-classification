import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, LinearRegression
import json
from difflib import get_close_matches
import os
import time
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

class ConfigReader:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self._read_config()

    def _read_config(self):
        with open(self.config_path, 'r', encoding='utf-8') as config_file:
            config = json.load(config_file)
        return config

    def get_parameters(self):
        weeks = self.config.get('weeks', 20) 
        history = self.config.get('history', 100) 
        features = [f for f in self.config['features'] if not f.startswith('#')]
        random_state = self.config.get('random_state', 42)
        if random_state == 0:
            random_state = int(time.time())
        algorithm = self.config['algorithm']
        initial_capital = self.config.get('initial_capital', 50000)
        
        split_date = self.config.get('split_date', '2015-01-05')
        split_date_exit = self.config.get('split_date_exit', '2020-01-02')
        
        return weeks, features, history, random_state, algorithm, initial_capital, split_date, split_date_exit, self.config

class ClassifierFactory:
    def __init__(self, config):
        self.config = config

    def get_classifier(self):
        algorithm = self.config['algorithm']
        params = {k: v for k, v in self.config[algorithm].items() if not k.startswith('#')}
        random_state = params.pop('random_state', self.config.get('random_state'))

        if algorithm == "Nearest Neighbors":
            return KNeighborsClassifier(**params)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

class DataProcessor:
    @staticmethod
    def find_matching_columns(df, features):
        df_columns = df.columns.tolist()
        matching_columns = []
        for feature in features:
            matches = get_close_matches(feature, df_columns, n=1, cutoff=0.6)
            if matches:
                matching_columns.append(matches[0])
            else:
                print(f"Warning: No close match found for feature '{feature}'")
        return matching_columns

    @staticmethod
    def prepare_data(df, weeks_ahead, features, history):
        matching_features = DataProcessor.find_matching_columns(df, features)
        X, y, dates, prices = [], [], [], []
        closing_price_col = get_close_matches('Kurs zamkniecia', df.columns, n=1, cutoff=0.6)[0]

        for i in range(len(df)-history+1):
            if i + history + weeks_ahead >= len(df)+1:
                pass
            else:
                X.append(df[matching_features].iloc[i:i + history].values)
                prices.append(df[closing_price_col].iloc[i + history + weeks_ahead - 1])
                y.append(1 if df[closing_price_col].iloc[i + history + weeks_ahead - 1] >= df[closing_price_col].iloc[i + history - 1] else 0)
                dates.append(df.index[i + history + weeks_ahead - 1])
      
        return np.array(X), np.array(y), dates, np.array(prices)

class ModelTrainer:
    def __init__(self, classifier, random_state, logger, initial_capital): 
        self.classifier = classifier
        self.random_state = random_state
        self.initial_capital = initial_capital
        self.NumberOfStocks = 0
        self.LastBuyPrice = 0
        self.logger = logger
        self.logger.initial_capital_start = self.initial_capital
        self.profit_history = []
        self.date_history = []

    def train_and_predict(self, X, y, prices, dates, weeks, history, df, split_date, split_date_exit):
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)
        
        if len(X) == 0 or len(y) == 0:
            print("Empty input data. Skipping processing.")
            return None, None, None

        predictions = []
        actual_values = []
        
        if 'Data' not in df.columns:
            print("'Data' column not found in dataframe. Skipping processing.")
            return None, None, None
        
        split_indices = df[df['Data'] == split_date].index
        if len(split_indices) == 0:
            print(f"Split date {split_date} not found in dataframe. Skipping processing.")
            return None, None, None
        
        split_index = split_indices[0] + 2
        if split_index < 0 or split_index >= len(X):
            print(f"Split index {split_index} is out of range for X with length {len(X)}. Skipping processing.")
            return None, None, None
        
        split_indices_exit = df[df['Data'] == split_date_exit].index
        if len(split_indices_exit) == 0:
            print(f"Split exit date {split_date_exit} not found in dataframe. Skipping processing.")
            return None, None, None
        
        print(split_indices_exit)
        split_point = split_index
        split_point_exit = split_indices_exit[0] - history
    
        
        X_train_initial = X[:split_point]
        X_test = X[split_point:split_point_exit]
        y_train_initial = y[:split_point]
        y_test = y[split_point:split_point_exit]
        dates_test = dates[split_point:] if dates is not None else None
        prices_test = prices[split_point:] if prices is not None else None
        
        if len(X_train_initial) == 0 or len(X_test) == 0:
            print("Insufficient samples for training or testing. Skipping processing.")
            return None, None, None
            
        X_train = X_train_initial.copy()
        y_train = y_train_initial.copy()
        max_hold_days=weeks-1
        days_holding = 0
        buy_index = -1
        wait = 3
        i=0

        self.classifier.fit(X_train, y_train)
                
        for i in range(weeks-1):
            X_current = X_test[i:i+1].copy()
            warmup_pred = self.classifier.predict(X_current).astype(int)[0]
            print(X_train[-1])
            X_train = np.vstack([X_train, X_current])
            y_train = np.append(y_train, warmup_pred)
            print(X_current)
            self.classifier.fit(X_train, y_train)
    
        try:
            for test_idx in range(weeks-1, len(X_test)):
                data_idx = len(X_train_initial) + history + test_idx
                X_current = X_test[test_idx:test_idx+1].copy()
              
                 
                print(X_train[-1])
                print(X_current)
                current_date = df["Data"].iloc[data_idx]
                current_price = prices_test[test_idx-weeks+1]
                current_price_open = df["Kurs otwarcia"].iloc[len(X_train_initial) + history + test_idx]

                if test_idx < len(X_test):
                    if self.NumberOfStocks > 0:
                        days_holding += 1
                    
                    final_pred = self.classifier.predict(X_test[test_idx:test_idx+1]).astype(int)[0]
                    
                    if self.NumberOfStocks > 0 and days_holding >= max_hold_days:
                        self.logger.log_transaction("sell (max hold)", X_train[-10:], 
                                                None, current_price, current_date, None, None)
                        self.NumberOfStocks, self.initial_capital = self.sell_stocks(current_price, self.NumberOfStocks, self.initial_capital)
                        days_holding = 0
      
                        
                        current_profit_pct = (self.initial_capital / self.logger.initial_capital_start - 1) * 100
                        self.profit_history.append(current_profit_pct)
                        self.date_history.append(current_date)
                        wait = 1
                    
                    if final_pred == 1 and wait == 0:
                        if current_price and self.NumberOfStocks == 0:
                            predictions.append(final_pred)
                            actual_values.append(y_test[test_idx])
                            self.logger.update_predictions(final_pred, y_test[test_idx])
                            
                            
                            self.NumberOfStocks, self.initial_capital = self.buy_stocks(current_price_open, self.initial_capital, self.NumberOfStocks)
                            self.LastBuyPrice = current_price
                            self.logger.log_transaction("buy", X_train[-10:], 
                                                    X_test[test_idx:test_idx+1], current_price_open, current_date, split_date, split_date_exit)
                            
                            days_holding = 0
                            current_profit_pct = ((self.initial_capital + (self.NumberOfStocks * current_price)) / self.logger.initial_capital_start - 1) * 100
                            self.profit_history.append(current_profit_pct)
                            self.date_history.append(current_date)
                    
                    wait = 0
                    
                    X_train = np.vstack([X_train, X_current])
                    y_train = np.append(y_train, final_pred)
                    self.classifier.fit(X_train, y_train)   
                    
            if self.NumberOfStocks > 0 and prices_test is not None and len(prices_test) > 0:
                last_price = current_price
                last_date = current_date
                
                self.logger.log_transaction("sell (final)", X_train[-10:] if len(X_train) > 10 else X_train, 
                                        None, last_price, last_date, None, None)
                self.NumberOfStocks, self.initial_capital = self.sell_stocks(last_price, self.NumberOfStocks, self.initial_capital)
                self.logger.initial_capital_last_sell = self.initial_capital
                
                final_profit_pct = ((self.initial_capital / self.logger.initial_capital_start) - 1) * 100
                self.profit_history.append(final_profit_pct)
                if last_date is not None:
                    self.date_history.append(last_date)
            
            if not predictions:
                print("No predictions were made. Skipping processing.")
                return None, None, None
                
            self.logger.current_capital = self.initial_capital
            self.logger.profit_history = self.profit_history
            self.logger.date_history = self.date_history
            
            test_score = sum(p == a for p, a in zip(predictions, actual_values)) / len(predictions) if predictions else 0
            
            results = {
                "weeks": weeks,
                "history": history,
                "test_accuracy": float(f'{test_score:.4f}'),
                "random_state": self.random_state,
                "predicted_values": predictions,
                "actual_values": actual_values,
                "num_samples": len(predictions)
            }
            
            return self.classifier, results, None
            
        except Exception as e:
            print(f"Error during training and prediction: {str(e)}. Skipping processing.")
            return None, None, None

    def buy_stocks(self, price, initial_capital, NumberOfStocks):
        num_stocks = self.initial_capital//price
        total_cost = num_stocks * price
        initial_capital = initial_capital - total_cost
        NumberOfStocks += num_stocks
        return NumberOfStocks, initial_capital

    def sell_stocks(self, price, NumberOfStocks, initial_capital):
        SellPrice = self.NumberOfStocks * price
        Bill = SellPrice * 0.0039
        PriceAfterBill = SellPrice - Bill
        initial_capital += PriceAfterBill
        NumberOfStocks = 0
        return NumberOfStocks, initial_capital
    
class TradeLogger:
    def __init__(self, output_file, graph_file):
        self.output_file = output_file
        self.graph_file = graph_file
        self.transaction_count = 0
        self.iteration_count = 0
        self.initial_capital_start = None
        self.initial_capital_last_sell = None
        self.transactions = []
        self.start_time = time.time()  
        self.end_time = None
        self.total_predictions = 0
        self.correct_predictions = 0
        self.x_train_initial_size = None
        self.x_test_normal_size = None
        self.total_x_size = None
        self.last_buy_iteration = None
        self.hold_periods = []
        self.total_predictions = 0
        self.correct_predictions = 0
        self.current_capital = None
        self.profit_history = []
        self.date_history = []
        self.start_date = None
        self.end_date = None
        self.start_date_obj = None
        self.end_date_obj = None
        self.df = None  
        self.start_index = None  
        self.end_index = None    

    def set_dataframe(self, df):
        self.df = df

    def log_transaction(self, transaction_type, X_train_last10, X_test_current, current_price, current_date, split_date, split_date_exit):
        if transaction_type == "buy":
            self.last_buy_iteration = self.iteration_count
            self.transaction_count += 1
        elif transaction_type == "sell" and self.last_buy_iteration is not None:
            hold_period = self.iteration_count - self.last_buy_iteration
            self.hold_periods.append(hold_period)
            self.last_buy_iteration = None

        if self.start_date is None:
            self.start_date = split_date
            self.start_date_obj = self._parse_date(split_date)
            if self.df is not None and "Data" in self.df.columns:
                try:
                    matching_indices = self.df[self.df["Data"] == split_date].index
                    if not matching_indices.empty:
                        self.start_index = matching_indices[0]
                except Exception as e:
                    print(f"Error finding start date index: {e}")
        if self.end_date is None:
            self.end_date = split_date_exit
            self.end_date_obj = self._parse_date(split_date_exit)

        if self.df is not None and "Data" in self.df.columns:
            try:
                matching_indices = self.df[self.df["Data"] == split_date_exit].index
                if not matching_indices.empty:
                    self.end_index = matching_indices[0]
            except Exception as e:
                print(f"Error finding end date index: {e}")

        transaction = {
            'type': transaction_type,
            'current_price': current_price,
            'date': current_date,
            'system_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'iterations_since_last_buy': self.iteration_count - self.last_buy_iteration if self.last_buy_iteration is not None else None
        }
        
        if transaction_type == "buy":
            transaction['X_train_last10'] = X_train_last10.tolist() if hasattr(X_train_last10, 'tolist') else X_train_last10
            transaction['X_test_current'] = X_test_current.tolist() if hasattr(X_test_current, 'tolist') else X_test_current
        
        self.transactions.append(transaction)
    
    def _parse_date(self, date_str):
        if not date_str or not isinstance(date_str, str):
            return None
            
        date_formats = ['%Y-%m-%d', '%d.%m.%Y', '%d-%m-%Y', '%m/%d/%Y']
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
                
        print(f"Warning: Could not parse date: {date_str}")
        return None
    
    def set_dataset_sizes(self, x_train_initial_size, x_test_normal_size, total_x_size):
        self.x_train_initial_size = x_train_initial_size
        self.x_test_normal_size = x_test_normal_size
        self.total_x_size = total_x_size

    def update_predictions(self, predicted, actual):
        if predicted==1:
            self.total_predictions += 1
        if predicted==1 and predicted == actual:
            self.correct_predictions += 1
    
    def finalize(self):
        self.end_time = time.time()
    
        if self.df is not None and "Data" in self.df.columns and not self.df.empty:
            last_date = self.df["Data"].iloc[-1]
            if last_date is not None:
                self.end_date = last_date
                self.end_date_obj = self._parse_date(last_date) if isinstance(last_date, str) else last_date
                try:
                    matching_indices = self.df[self.df["Data"] == last_date].index
                    if not matching_indices.empty:
                        self.end_index = matching_indices[0]
                except Exception as e:
                    print(f"Error finding end date index: {e}")
        
        self.create_profit_chart()
    
    def create_profit_chart(self):
        if not self.profit_history:
            print("No profit history data available to create chart. Creating an empty chart instead.")
            fig, ax = plt.figure(figsize=(12, 6)), plt.subplot(111)
            ax.text(0.5, 0.5, 'No trading activity in this period', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title('Profit Percentage Over Time', fontsize=16)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Profit (%)', fontsize=12)
            plt.savefig(self.graph_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return
            
        if not self.date_history:
            print("No date history data available to create chart. Creating an empty chart instead.")
            fig, ax = plt.figure(figsize=(12, 6)), plt.subplot(111)
            ax.text(0.5, 0.5, 'No date information available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title('Profit Percentage Over Time', fontsize=16)
            plt.savefig(self.graph_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return
        
        dates = []
        valid_profits = []
        
        for i, date_str in enumerate(self.date_history):
            date_obj = self._parse_date(date_str)
            if date_obj:
                dates.append(date_obj)
                valid_profits.append(self.profit_history[i])
        
        if not dates:
            print("No valid dates for charting. Creating an empty chart instead.")
            fig, ax = plt.figure(figsize=(12, 6)), plt.subplot(111)
            ax.text(0.5, 0.5, 'No valid date information available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title('Profit Percentage Over Time', fontsize=16)
            plt.savefig(self.graph_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return
        
        fig, ax = plt.figure(figsize=(12, 6)), plt.subplot(111)
        
        try:
            ax.plot(dates, valid_profits, 'b-', linewidth=2, marker='o', markersize=4)
            
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_title('Profit Percentage Over Time', fontsize=16)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Profit (%)', fontsize=12)
            
            ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            
        except Exception as e:
            print(f"Error creating chart: {str(e)}. Creating an error message chart instead.")
            ax.text(0.5, 0.5, f'Error creating chart: {str(e)}', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12, color='red')
        
        plt.tight_layout()
        
        try:
            plt.savefig(self.graph_file, dpi=300, bbox_inches='tight')
            print(f"Profit chart saved to: {self.graph_file}")
        except Exception as e:
            print(f"Error saving chart to file: {str(e)}")
        finally:
            plt.close(fig)
    
    def write_results(self):
        accuracy = (self.correct_predictions / self.total_predictions * 100) if self.total_predictions > 0 else 0
        avg_hold_period = sum(self.hold_periods) / len(self.hold_periods) if self.hold_periods else 0
        execution_time = self.end_time - self.start_time 
        
        final_capital = self.initial_capital_start  
        if self.initial_capital_last_sell is not None:
            final_capital = self.initial_capital_last_sell
        elif self.current_capital is not None:
            final_capital = self.current_capital

        profit = 0
        if final_capital is not None and self.initial_capital_start is not None:
            profit = final_capital - self.initial_capital_start

        if self.df is not None and "Data" in self.df.columns and not self.df.empty:
            self.end_date = self.df["Data"].iloc[-1143]
            self.end_date_obj = self._parse_date(self.end_date) if isinstance(self.end_date, str) else self.end_date
            try:
                matching_indices = self.df[self.df["Data"] == self.end_date].index
                if not matching_indices.empty:
                    self.end_index = matching_indices[-1]
            except Exception as e:
                print(f"Error finding end date index: {e}")

        days_diff = 0
        if self.start_index is not None and self.end_index is not None:
            days_diff = abs(self.end_index - self.start_index)
            print(f"Date indices difference: {days_diff} days (from index {self.start_index} to {self.end_index})")
        elif self.start_date_obj and self.end_date_obj: 
            days_diff = (self.end_date_obj - self.start_date_obj).days
            print(f"Date difference: {days_diff} days (from {self.start_date} to {self.end_date})")
        
        days_active = days_diff if days_diff > 0 else self.iteration_count

        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(f"Liczba wszystkich prognoz: {self.total_predictions}\n")
            f.write(f"Procent poprawnych prognoz: {accuracy:.2f}%\n\n")
            f.write(f"Dlugosc wykonywania programu: {execution_time:.2f} sekund\n")
            f.write(f"Ilosc transakcji: {self.transaction_count}\n")
            
            f.write(f"Dni dzialania silnika: {days_active}\n")
            
            if self.start_date is not None and self.end_date is not None:
                f.write(f"Okres dzialania silnika: od {self.start_date} do {self.end_date}\n")
                if self.start_index is not None and self.end_index is not None:
                    f.write(f"Indeksy w danych: od {self.start_index} do {self.end_index} (roznica: {days_diff})\n")
            
            if self.initial_capital_start is not None:
                f.write(f"Kapital na poczatku: {self.initial_capital_start:.2f}\n")
            else:
                f.write("Kapital na poczatku: Brak danych\n")
                
            if final_capital is not None:
                f.write(f"Kapital na koncu dzialania: {final_capital:.2f}\n")
                f.write(f"Profit: {profit:.2f}\n")
                if self.initial_capital_start > 0:
                    f.write(f"Profit procentowy: {(profit/self.initial_capital_start*100):.2f}%\n")
            else:
                f.write("Kapital na koncu dzialania: Brak danych\n")
                f.write("Profit: Brak danych\n")
            
            if self.total_predictions > 0:
                f.write(f"\nLiczba prognoz gdy final_pred==1: {self.total_predictions}\n")
                f.write(f"Procent poprawnych prognoz gdy final_pred==1: {accuracy:.2f}%\n")
            
            if self.transactions:
                f.write("\nTransakcje w kolejnosci chronologicznej:\n")
                
                current_capital = self.initial_capital_start
                current_stocks = 0
                
                for i, transaction in enumerate(self.transactions, 1):
                    f.write(f"\nTransakcja {i} ({transaction['type'].upper()}):\n")
                    f.write(f"Data transakcji z pliku: {transaction['date']}\n")
                    f.write(f"Obecna cena: {transaction['current_price']}\n")
                    
                    if transaction['type'] == "buy":
                        price = transaction['current_price']
                        stocks_bought = current_capital // price
                        cost = stocks_bought * price
                        current_capital -= cost
                        current_stocks += stocks_bought
                        
                        f.write(f"Zakupiono akcji: {stocks_bought}\n")
                        f.write(f"Kapital po zakupie: {current_capital:.2f}\n")
                        f.write(f"Liczba posiadanych akcji po zakupie: {current_stocks}\n")
                        
                        f.write("\nOstatnie 10 obiektow z X_train:\n")
                        for idx, x_train_row in enumerate(transaction['X_train_last10']):
                            f.write(f"Obiekt {idx + 1}: {x_train_row}\n")
                        
                        f.write("\nObiekt X_test uzyty do predykcji:\n")
                        f.write(f"{transaction['X_test_current']}\n")
                    
                    elif transaction['type'].startswith("sell"):
                        price = transaction['current_price']
                        sell_value = current_stocks * price
                        sell_fee = sell_value * 0.0039 
                        net_proceeds = sell_value - sell_fee
                        
                        current_capital += net_proceeds
                        stocks_sold = current_stocks
                        current_stocks = 0
                        
                        f.write(f"Sprzedano akcji: {stocks_sold}\n")
                        f.write(f"Wartosc sprzedazy (brutto): {sell_value:.2f}\n")
                        f.write(f"Oplata transakcyjna: {sell_fee:.2f}\n")
                        f.write(f"Wartosc sprzedazy (netto): {net_proceeds:.2f}\n")
                        f.write(f"Kapital po sprzedazy: {current_capital:.2f}\n")
                        f.write(f"Liczba posiadanych akcji po sprzedazy: {current_stocks}\n")
            else:
                f.write("\nBrak transakcji w tym okresie\n")

class TradingSummary:
    def __init__(self, results_directory):
        self.results_directory = results_directory
        self.total_profit = 0
        self.total_transactions = 0
        self.file_count = 0
        self.total_predictions = 0
        self.total_accuracy = 0
        self.company_stats = []
    
    def process_results(self):
        for filename in os.listdir(self.results_directory):
            if filename.startswith('Results_') and filename.endswith('.txt'):
                self.file_count += 1
                with open(os.path.join(self.results_directory, filename), 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    company_name = filename.split('_')[1]
                    
                    transactions_match = re.search(r'Ilosc transakcji: (\d+)', content)
                    iterations_match = re.search(r'Dni dzialania silnika: (\d+)', content)
                    initial_capital_match = re.search(r'Kapital na poczatku: (\d+\.?\d*)', content)
                    final_capital_match = re.search(r'Kapital na koncu dzialania: (\d+\.?\d*)', content)
                    predictions_match = re.search(r'Liczba prognoz gdy final_pred==1: (\d+)', content)
                    accuracy_match = re.search(r'Procent poprawnych prognoz gdy final_pred==1: (\d+\.?\d*)%', content)
                    
                    date_range_match = re.search(r'Okres dzialania silnika: od (.*) do (.*)', content)
                    start_date_str = date_range_match.group(1) if date_range_match else "N/A"
                    end_date_str = date_range_match.group(2) if date_range_match else "N/A"
                    
                    indices_match = re.search(r'Indeksy w danych: od (\d+) do (\d+) \(roznica: (\d+)\)', content)
                    start_index = int(indices_match.group(1)) if indices_match else None
                    end_index = int(indices_match.group(2)) if indices_match else None
                    index_diff = int(indices_match.group(3)) if indices_match else 0
                    
                    transactions_count = int(transactions_match.group(1)) if transactions_match else 0
                    predictions_count = int(predictions_match.group(1)) if predictions_match else 0
                    accuracy = float(accuracy_match.group(1)) if accuracy_match else 0
                    
                    if transactions_match:
                        self.total_transactions += transactions_count
                    
                    if predictions_match:
                        self.total_predictions += predictions_count
                    
                    if accuracy_match:
                        self.total_accuracy += accuracy * predictions_count  
                    
                    if initial_capital_match and final_capital_match:
                        initial = float(initial_capital_match.group(1))
                        final = float(final_capital_match.group(1))
                        
                        iterations = index_diff if index_diff > 0 else (int(iterations_match.group(1)) if iterations_match else 0)
                        
                        profit = final - initial
                        avg_profit_per_iteration = profit / iterations if iterations > 0 else 0
                        
                        percent_return = (profit / initial) * 100 if initial > 0 else 0
                       
                        self.company_stats.append({
                            'company': company_name,
                            'iterations': iterations,
                            'profit': profit,
                            'avg_profit_per_iteration': avg_profit_per_iteration,
                            'percent_return': percent_return,
                          
                            'prediction_accuracy': accuracy,
                            'predictions_count': predictions_count,
                            'transactions_count': transactions_count,
                            'start_date': start_date_str,
                            'end_date': end_date_str,
                            'start_index': start_index,
                            'end_index': end_index
                        })
                        
                        self.total_profit += profit
    
    def write_summary(self, output_file):
        avg_accuracy = self.total_accuracy / self.total_predictions if self.total_predictions > 0 else 0
    
        sorted_stats = sorted(self.company_stats, key=lambda x: x['profit'], reverse=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Podsumowanie\n")
            f.write("=====================\n\n")
            f.write(f"Liczba przeanalizowanych plikow: {self.file_count}\n")
            f.write(f"Suma wszystkich prognoz gdy final_pred==1: {self.total_predictions}\n")
            f.write(f"Srednia dokladnosc prognoz gdy final_pred==1: {avg_accuracy:.2f}%\n\n")
            f.write(f"Suma wszystkich transakcji: {self.total_transactions}\n")
            f.write(f"Srednia liczba transakcji na plik: {self.total_transactions/self.file_count:.2f}\n\n")
            f.write(f"Sumaryczny profit: {self.total_profit:.2f}\n")
            f.write(f"Sredni profit na plik: {self.total_profit/self.file_count:.2f}\n\n")
            
            f.write("Statystyki per spolka (posortowane wedlug zarobku):\n")
            
            max_company_width = max(len(stat['company']) for stat in sorted_stats)
            max_company_width = max(max_company_width, len("Nazwa spolki"))
            
            header = (
                f"{'Nazwa spolki':<{max_company_width}} | "
                f"{'Okres od':<12} | "
                f"{'Okres do':<12} | "
                f"{'Dni dzialania silnika':>20} | "
                f"{'Zarobek z spolki':>15} | "
                f"{'Sredni zarobek na dzien':>25} | "
                f"{'Zwrot [%]':>9} | "
          
                f"{'Dokladnosc [%]':>15} | "
                f"{'Ilosc transakcji':>15}"  
            )
            
            f.write(f"{header}\n")
            f.write("=" * len(header) + "\n")
            
            for stat in sorted_stats:
                row = (
                    f"{stat['company']:<{max_company_width}} | "
                    f"{stat['start_date']:<12} | "
                    f"{stat['end_date']:<12} | "
                    f"{stat['iterations']:>20} | "
                    f"{stat['profit']:>15.2f} | "
                    f"{stat['avg_profit_per_iteration']:>25.2f} | "
                    f"{stat['percent_return']:>9.2f} | "
                
                    f"{stat['prediction_accuracy']:>15.2f} | "
                    f"{stat['transactions_count']:>15}" 
                )
                f.write(f"{row}\n")
    
    def create_summary_graph(self, output_file):
        if not self.company_stats:
            print("No company statistics to plot.")
            return
            
        try:
            sorted_stats = sorted(self.company_stats, key=lambda x: x['percent_return'])
            
            companies = [stat['company'] for stat in sorted_stats]
            percent_returns = [stat['percent_return'] for stat in sorted_stats]
        
            plt.figure(figsize=(12, 8))
            
            bars = plt.barh(companies, percent_returns)
            for i, bar in enumerate(bars):
                if percent_returns[i] >= 0:
                    bar.set_color('green')
                else:
                    bar.set_color('red')
            
            plt.title('Profit Comparison by Company')
            plt.xlabel('Profit (%)')
            plt.grid(axis='x', alpha=0.3)
        
            for i, v in enumerate(percent_returns):
                plt.text(v + (1 if v >= 0 else -5), i, f"{v:.1f}%", 
                         va='center', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(output_file)
            plt.close()
            print(f"Summary graph saved to {output_file}")
            
        except Exception as e:
            print(f"Error creating summary graph: {e}")

def main():
    config_reader = ConfigReader('configAl.json')
    weeks, features, history, random_state, algorithm, initial_capital, split_date, split_date_exit, config = config_reader.get_parameters()  
    
    classifier_factory = ClassifierFactory(config)
    classifier = classifier_factory.get_classifier()

    data_directory = 'DataDaily'
    results_directory = 'Results'
    graphs_directory = os.path.join(results_directory, 'Graphs')
    
    os.makedirs(results_directory, exist_ok=True)
    os.makedirs(graphs_directory, exist_ok=True)

    for filename in os.listdir(data_directory):
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            print(f"Processing file: {filename}")
            excel_file = filename
            
            try:
                df = pd.read_excel(os.path.join(data_directory, excel_file))
                
                if 'Data' not in df.columns:
                    print(f"Skipping file {excel_file} - 'Data' column not found")
                    continue
                    
                if split_date not in df['Data'].values:
                    print(f"Skipping file {excel_file} - required split date '{split_date}' not found")
                    continue
                
                base_name = excel_file.split('.')[0]
                output_file = os.path.join(results_directory, f"Results_{base_name}_{history}_{weeks}.txt")
                graph_file = os.path.join(graphs_directory, f"Profit_Graph_{base_name}_{history}_{weeks}.png")
                
                logger = TradeLogger(output_file, graph_file)
            
                logger.set_dataframe(df)
                
                X, y, dates, prices = DataProcessor.prepare_data(df, weeks, features, history=history)
                
                if len(X) == 0 or len(y) == 0:
                    print(f"Skipping file {excel_file} - insufficient data")
                    continue
                
                split_point = int(len(X)*0.8)
                logger.set_dataset_sizes(
                    x_train_initial_size=split_point,
                    x_test_normal_size=len(X)-split_point,
                    total_x_size=len(X)
                )
                
                model_trainer = ModelTrainer(classifier, random_state, logger, initial_capital)
                train_results = model_trainer.train_and_predict(
                    X, y, prices, dates, 
                    weeks=weeks, 
                    history=history, 
                    df=df, 
                    split_date=split_date,
                    split_date_exit=split_date_exit
                )
                
                if train_results is None or len(train_results) != 3:
                    print(f"Skipping processing for {excel_file} - training failed")
                    continue
                
                logger.finalize()
                logger.write_results()
                print(f"Results saved in: {output_file}")
                print(f"Profit graph saved in: {graph_file}")
                print()
             
            except Exception as e:
                print(f"Error processing {excel_file}: {str(e)}")
                continue
    
    print("\nGenerating trading summary...")
    summary = TradingSummary(results_directory)
    summary.process_results()
    summary_file = os.path.join(results_directory, 'trading_summary.txt')
    summary.write_summary(summary_file)
    print(f"Summary saved in: {summary_file}")
main()