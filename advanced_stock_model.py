import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

class RobustStockPredictor:
    def __init__(self, stock_symbol='RELIANCE.NS', period='2y'):
        self.stock_symbol = stock_symbol
        self.period = period
        self.data = None
        self.features = None
        self.target = None
        self.model = None
        self.scaler = RobustScaler()
        self.feature_importance = None
        self.is_trained = False
        
    def fetch_data(self):
        """Fetch stock data from Yahoo Finance with robust error handling"""
        print(f"Fetching data for {self.stock_symbol}...")
        try:
            stock = yf.Ticker(self.stock_symbol)
            self.data = stock.history(period=self.period)
            
            if self.data.empty or len(self.data) < 100:
                raise ValueError(f"Insufficient data for {self.stock_symbol}. Only {len(self.data)} rows found.")
                
            # Data cleaning
            self.data = self.data.dropna()
            self.data = self.data.sort_index()
            
            # Ensure we have enough data after cleaning
            if len(self.data) < 50:
                raise ValueError(f"Not enough data after cleaning for {self.stock_symbol}")
            
            print(f"Data fetched successfully! Shape: {self.data.shape}")
            print(f"Date range: {self.data.index[0]} to {self.data.index[-1]}")
            return self.data
            
        except Exception as e:
            print(f"Error fetching data for {self.stock_symbol}: {e}")
            raise
    
    def calculate_robust_technical_indicators(self):
        """Calculate technical indicators with robust error handling"""
        if self.data is None:
            self.fetch_data()
            
        df = self.data.copy()
        
        print("Calculating technical indicators...")
        
        try:
            # Basic price features
            df['Returns'] = df['Close'].pct_change()
            df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
            df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
            
            # Moving Averages
            for window in [5, 10, 20]:
                df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
                df[f'EMA_{window}'] = df['Close'].ewm(span=window).mean()
                df[f'Price_SMA_Ratio_{window}'] = df['Close'] / df[f'SMA_{window}']
            
            # RSI
            df['RSI'] = self.calculate_safe_rsi(df['Close'])
            
            # MACD
            df['MACD'] = self.calculate_safe_macd(df['Close'])
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # Bollinger Bands
            df = self.calculate_safe_bollinger_bands(df)
            
            # Volume features
            df['Volume_SMA'] = df['Volume'].rolling(10).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            
            # Volatility
            df['Volatility_10'] = df['Returns'].rolling(10).std()
            
            # Momentum
            df['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
            df['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
            
            # Support and Resistance
            df['Resistance_20'] = df['High'].rolling(20).max()
            df['Support_20'] = df['Low'].rolling(20).min()
            df['Distance_to_Resistance'] = (df['Resistance_20'] - df['Close']) / df['Close']
            df['Distance_to_Support'] = (df['Close'] - df['Support_20']) / df['Close']
            
            # Lagged returns
            for lag in [1, 2, 3]:
                df[f'Return_lag_{lag}'] = df['Returns'].shift(lag)
            
            # Remove rows with any NaN values
            initial_shape = df.shape[0]
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna()
            final_shape = df.shape[0]
            
            print(f"Removed {initial_shape - final_shape} rows with NaN/infinite values")
            
            if final_shape < 30:
                raise ValueError("Not enough valid data after cleaning")
                
            self.data = df
            print(f"Final data shape: {self.data.shape}")
            return self.data
            
        except Exception as e:
            print(f"Error calculating indicators for {self.stock_symbol}: {e}")
            raise
    
    def calculate_safe_rsi(self, prices, period=14):
        """Safe RSI calculation"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        except:
            return np.nan
    
    def calculate_safe_macd(self, prices):
        """Safe MACD calculation"""
        try:
            exp1 = prices.ewm(span=12).mean()
            exp2 = prices.ewm(span=26).mean()
            return exp1 - exp2
        except:
            return np.nan
    
    def calculate_safe_bollinger_bands(self, df):
        """Safe Bollinger Bands calculation"""
        try:
            df['BB_Middle'] = df['Close'].rolling(20).mean()
            bb_std = df['Close'].rolling(20).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
            return df
        except:
            df['BB_Middle'] = np.nan
            df['BB_Upper'] = np.nan
            df['BB_Lower'] = np.nan
            df['BB_Position'] = np.nan
            return df
    
    def create_target_variable(self):
        """Create target variable"""
        try:
            self.data['Tomorrow_Close'] = self.data['Close'].shift(-1)
            self.data['Target'] = (self.data['Tomorrow_Close'] > self.data['Close']).astype(int)
            self.data = self.data.dropna()
            
            self.target = self.data['Target']
            print(f"Target distribution: {self.target.value_counts().to_dict()}")
            return self.target
        except Exception as e:
            print(f"Error creating target variable: {e}")
            raise
    
    def prepare_features(self):
        """Prepare features for modeling"""
        try:
            # Select only numeric columns and exclude target columns
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            exclude_cols = ['Tomorrow_Close', 'Target']
            feature_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            self.features = self.data[feature_cols]
            
            # Remove columns with too many NaN values
            self.features = self.features.dropna(axis=1, how='all')
            
            print(f"Using {len(self.features.columns)} features")
            return self.features
        except Exception as e:
            print(f"Error preparing features: {e}")
            raise
    
    def train_models(self):
        """Train machine learning models"""
        try:
            if self.features is None or self.target is None:
                self.calculate_robust_technical_indicators()
                self.create_target_variable()
                self.prepare_features()
            
            X = self.features
            y = self.target
            
            # Time-based split
            split_index = int(0.8 * len(X))
            X_train = X.iloc[:split_index]
            X_test = X.iloc[split_index:]
            y_train = y.iloc[:split_index]
            y_test = y.iloc[split_index:]
            
            print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Initialize models with simpler parameters for stability
            models = {
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
                'XGBoost': XGBClassifier(random_state=42, n_estimators=50)
            }
            
            results = {}
            
            for name, model in models.items():
                print(f"Training {name}...")
                try:
                    if name == 'Logistic Regression':
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        y_pred_proba = model.predict_proba(X_test)[:, 1]
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
                    
                    results[name] = {
                        'model': model,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'predictions': y_pred,
                        'probabilities': y_pred_proba
                    }
                    
                    print(f"{name} - Accuracy: {accuracy:.4f}")
                    
                except Exception as e:
                    print(f"Error training {name}: {e}")
                    continue
            
            if not results:
                raise Exception("No models were successfully trained!")
            
            # Select best model
            best_model_name = max(results, key=lambda x: results[x]['accuracy'])
            self.model = results[best_model_name]['model']
            self.is_trained = True
            
            print(f"\nBest Model: {best_model_name}")
            print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
            
            return results, X_test, y_test
            
        except Exception as e:
            print(f"Error in model training: {e}")
            raise
    
    def predict_next_day(self):
        """Predict next day movement"""
        if not self.is_trained or self.model is None:
            return {
                'prediction': 'UNKNOWN',
                'confidence': 0.0,  # Changed to float for JSON serialization
                'probability_up': 50.0,  # Changed to float for JSON serialization
                'probability_down': 50.0,  # Changed to float for JSON serialization
                'current_price': 0.0,  # Changed to float for JSON serialization
                'signal_strength': 'UNKNOWN',
                'error': 'Model not trained'
            }
        
        try:
            if self.features is None or len(self.features) == 0:
                return {
                    'prediction': 'UNKNOWN',
                    'confidence': 0.0,  # Changed to float for JSON serialization
                    'probability_up': 50.0,  # Changed to float for JSON serialization
                    'probability_down': 50.0,  # Changed to float for JSON serialization
                    'current_price': 0.0,  # Changed to float for JSON serialization
                    'signal_strength': 'UNKNOWN',
                    'error': 'No features available'
                }
            
            latest_features = self.features.iloc[-1:].copy()
            
            if isinstance(self.model, LogisticRegression):
                latest_features_scaled = self.scaler.transform(latest_features)
                prediction = self.model.predict(latest_features_scaled)[0]
                probability = self.model.predict_proba(latest_features_scaled)[0][1]
            else:
                prediction = self.model.predict(latest_features)[0]
                probability = self.model.predict_proba(latest_features)[0][1]
            
            direction = "UP" if prediction == 1 else "DOWN"
            confidence = probability if prediction == 1 else (1 - probability)
            
            # Convert all numeric values to native Python types for JSON serialization
            result = {
                'prediction': direction,
                'confidence': float(round(confidence * 100, 2)),  # Explicit float conversion
                'probability_up': float(round(probability * 100, 2)),  # Explicit float conversion
                'probability_down': float(round((1 - probability) * 100, 2)),  # Explicit float conversion
                'current_price': float(round(self.data['Close'].iloc[-1], 2)),  # Explicit float conversion
                'signal_strength': 'STRONG' if confidence > 0.7 else 'MODERATE' if confidence > 0.6 else 'WEAK',
                'error': None
            }
            
            return result
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return {
                'prediction': 'ERROR',
                'confidence': 0.0,  # Changed to float for JSON serialization
                'probability_up': 50.0,  # Changed to float for JSON serialization
                'probability_down': 50.0,  # Changed to float for JSON serialization
                'current_price': 0.0,  # Changed to float for JSON serialization
                'signal_strength': 'ERROR',
                'error': str(e)
            }

# For testing
if __name__ == "__main__":
    predictor = RobustStockPredictor(stock_symbol='RELIANCE.NS', period='2y')
    
    try:
        predictor.fetch_data()
        predictor.calculate_robust_technical_indicators()
        predictor.create_target_variable()
        predictor.prepare_features()
        results, X_test, y_test = predictor.train_models()
        
        prediction = predictor.predict_next_day()
        print(f"\nPrediction: {prediction}")
        
    except Exception as e:
        print(f"Test failed: {e}")