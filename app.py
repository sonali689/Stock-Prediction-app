from flask import Flask, render_template, request, jsonify
import pickle
import yfinance as yf
from datetime import datetime
import pandas as pd
from advanced_stock_model import RobustStockPredictor
import os
import numpy as np
import json

app = Flask(__name__)

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super(NumpyEncoder, self).default(obj)

app.json_encoder = NumpyEncoder

# Store multiple predictors in a dictionary
predictors = {}

def convert_to_native_types(obj):
    """Recursively convert numpy types to native Python types"""
    if isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    else:
        return obj

def train_or_load_model(stock_symbol='RELIANCE.NS'):
    """Train model or load existing one with robust error handling"""
    model_file = f'model_{stock_symbol.replace(".", "_")}.pkl'
    
    # Check if we already have this predictor in memory
    if stock_symbol in predictors:
        print(f"Using cached model for {stock_symbol}")
        return True
    
    if os.path.exists(model_file):
        # Load existing model
        try:
            with open(model_file, 'rb') as f:
                predictors[stock_symbol] = pickle.load(f)
            print(f"Loaded existing model for {stock_symbol}")
            return True
        except Exception as e:
            print(f"Error loading model for {stock_symbol}: {e}. Retraining...")
            # Remove corrupted file
            try:
                os.remove(model_file)
            except:
                pass
    
    # Train new model
    try:
        print(f"Training new model for {stock_symbol}...")
        predictor = RobustStockPredictor(stock_symbol=stock_symbol, period='2y')
        
        # Fetch and prepare data
        predictor.fetch_data()
        predictor.calculate_robust_technical_indicators()
        predictor.create_target_variable()
        predictor.prepare_features()
        
        # Train models
        results, X_test, y_test = predictor.train_models()
        
        # Save model
        with open(model_file, 'wb') as f:
            pickle.dump(predictor, f)
        
        predictors[stock_symbol] = predictor
        print(f"Trained and saved new model for {stock_symbol}")
        return True
        
    except Exception as e:
        print(f"Error training model for {stock_symbol}: {e}")
        # Create a dummy predictor to avoid repeated failures
        dummy_predictor = RobustStockPredictor(stock_symbol=stock_symbol)
        predictors[stock_symbol] = dummy_predictor
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        stock_symbol = request.json.get('stock_symbol', 'RELIANCE.NS').upper()
        
        # Validate stock symbol
        if not stock_symbol.endswith('.NS'):
            stock_symbol += '.NS'
        
        # Train or load model for the selected stock
        success = train_or_load_model(stock_symbol)
        
        if not success:
            return jsonify({
                'success': False,
                'error': f'Failed to train or load model for {stock_symbol}. The stock might not have sufficient data.'
            })
        
        # Get the predictor for this specific stock
        predictor = predictors[stock_symbol]
        
        # Get prediction
        prediction = predictor.predict_next_day()
        
        # Check if prediction has error
        if prediction.get('error'):
            return jsonify({
                'success': False,
                'error': prediction['error']
            })
        
        # Get current stock info
        try:
            stock = yf.Ticker(stock_symbol)
            info = stock.info
            hist = stock.history(period='1d')
            
            if not hist.empty:
                current_price = float(hist['Close'].iloc[-1])
                prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
                price_change = ((current_price - prev_close) / prev_close) * 100
            else:
                current_price = prediction['current_price']
                price_change = 0
                
        except Exception as e:
            print(f"Error getting stock info: {e}")
            current_price = prediction['current_price']
            price_change = 0
        
        # Ensure all values are native Python types
        response = convert_to_native_types({
            'success': True,
            'stock_symbol': stock_symbol,
            'current_price': current_price,
            'price_change': round(price_change, 2),
            'prediction': prediction['prediction'],
            'confidence': prediction['confidence'],
            'probability_up': prediction['probability_up'],
            'probability_down': prediction['probability_down'],
            'signal_strength': prediction['signal_strength'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': 'Robust ML Model'
        })
        
        return jsonify(response)
        
    except Exception as e:
        response = convert_to_native_types({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        })
        return jsonify(response)

@app.route('/stock_info')
def stock_info():
    try:
        stock_symbol = request.args.get('symbol', 'RELIANCE.NS').upper()
        
        if not stock_symbol.endswith('.NS'):
            stock_symbol += '.NS'
            
        stock = yf.Ticker(stock_symbol)
        info = stock.info
        
        # Get historical data
        hist = stock.history(period='1mo')
        
        if not hist.empty:
            current_price = float(hist['Close'].iloc[-1])
            prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
            day_change = current_price - prev_close
            day_change_pct = (day_change / prev_close) * 100
            
            # Calculate recent high/low
            recent_high = float(hist['High'].max())
            recent_low = float(hist['Low'].min())
        else:
            current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
            prev_close = info.get('previousClose', current_price)
            day_change = 0
            day_change_pct = 0
            recent_high = info.get('fiftyTwoWeekHigh', current_price)
            recent_low = info.get('fiftyTwoWeekLow', current_price)
        
        stock_data = convert_to_native_types({
            'name': str(info.get('longName', stock_symbol)),
            'current_price': round(float(current_price), 2),
            'previous_close': round(float(prev_close), 2),
            'day_change': round(float(day_change), 2),
            'day_change_pct': round(float(day_change_pct), 2),
            'market_cap': format_market_cap(info.get('marketCap', 0)),
            'volume': format_volume(info.get('volume', 0)),
            'sector': str(info.get('sector', 'N/A')),
            'industry': str(info.get('industry', 'N/A')),
            'recent_high': round(float(recent_high), 2),
            'recent_low': round(float(recent_low), 2)
        })
        
        return jsonify(stock_data)
    
    except Exception as e:
        return jsonify({'error': f'Failed to get stock info: {str(e)}'})

@app.route('/model_info')
def model_info():
    """Return information about the model for a specific stock"""
    try:
        stock_symbol = request.args.get('symbol', 'RELIANCE.NS').upper()
        
        if not stock_symbol.endswith('.NS'):
            stock_symbol += '.NS'
            
        # Ensure the model is loaded for this stock
        if stock_symbol not in predictors:
            success = train_or_load_model(stock_symbol)
            if not success:
                return jsonify({'error': f'No model available for {stock_symbol}'})
        
        predictor = predictors[stock_symbol]
        
        model_info = convert_to_native_types({
            'stock_symbol': str(stock_symbol),
            'data_points': int(len(predictor.data) if predictor.data is not None else 0),
            'is_trained': bool(predictor.is_trained),
            'model_type': str(type(predictor.model).__name__ if predictor.model else 'None'),
            'features_count': int(len(predictor.features.columns) if predictor.features is not None else 0)
        })
        
        # Add feature information if available
        if predictor.features is not None:
            model_info['feature_names'] = list(predictor.features.columns[:10])  # First 10 features
        
        return jsonify(model_info)
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/available_stocks')
def available_stocks():
    """Return list of available Indian stocks"""
    indian_stocks = {
        'Large Cap': [
            {'symbol': 'RELIANCE.NS', 'name': 'Reliance Industries'},
            {'symbol': 'TCS.NS', 'name': 'Tata Consultancy Services'},
            {'symbol': 'HDFCBANK.NS', 'name': 'HDFC Bank'},
            {'symbol': 'INFY.NS', 'name': 'Infosys'},
            {'symbol': 'HINDUNILVR.NS', 'name': 'Hindustan Unilever'},
            {'symbol': 'ICICIBANK.NS', 'name': 'ICICI Bank'},
            {'symbol': 'SBIN.NS', 'name': 'State Bank of India'},
            {'symbol': 'BHARTIARTL.NS', 'name': 'Bharti Airtel'}
        ],
        'Mid Cap': [
            {'symbol': 'TATAMOTORS.NS', 'name': 'Tata Motors'},
            {'symbol': 'AXISBANK.NS', 'name': 'Axis Bank'},
            {'symbol': 'LT.NS', 'name': 'Larsen & Toubro'},
            {'symbol': 'KOTAKBANK.NS', 'name': 'Kotak Mahindra Bank'},
            {'symbol': 'ASIANPAINT.NS', 'name': 'Asian Paints'},
            {'symbol': 'MARUTI.NS', 'name': 'Maruti Suzuki'}
        ],
        'Tech Stocks': [
            {'symbol': 'WIPRO.NS', 'name': 'Wipro'},
            {'symbol': 'TECHM.NS', 'name': 'Tech Mahindra'},
            {'symbol': 'HCLTECH.NS', 'name': 'HCL Technologies'}
        ]
    }
    return jsonify(indian_stocks)

def format_market_cap(market_cap):
    """Format market cap in readable format"""
    try:
        market_cap = float(market_cap)
        if market_cap == 0:
            return 'N/A'
        elif market_cap >= 10**12:  # Trillion
            return f'₹{market_cap/10**12:.2f}T'
        elif market_cap >= 10**10:  # Thousand Crore
            return f'₹{market_cap/10**10:.2f}K Cr'
        elif market_cap >= 10**7:  # Crore
            return f'₹{market_cap/10**7:.2f} Cr'
        else:
            return f'₹{market_cap:,.0f}'
    except:
        return 'N/A'

def format_volume(volume):
    """Format volume in readable format"""
    try:
        volume = float(volume)
        if volume == 0:
            return 'N/A'
        elif volume >= 10**9:  # Billion
            return f'{volume/10**9:.2f}B'
        elif volume >= 10**6:  # Million
            return f'{volume/10**6:.2f}M'
        elif volume >= 10**3:  # Thousand
            return f'{volume/10**3:.2f}K'
        else:
            return f'{volume:,.0f}'
    except:
        return 'N/A'

@app.route('/clear_cache')
def clear_cache():
    """Clear all cached models (for development)"""
    global predictors
    predictors.clear()
    
    # Also delete pickle files
    for filename in os.listdir('.'):
        if filename.startswith('model_') and filename.endswith('.pkl'):
            try:
                os.remove(filename)
                print(f"Deleted {filename}")
            except:
                pass
                
    return jsonify({'message': 'Cache cleared successfully'})

@app.route('/list_models')
def list_models():
    """List all currently loaded models"""
    model_list = {}
    for symbol, predictor in predictors.items():
        model_list[symbol] = convert_to_native_types({
            'is_trained': predictor.is_trained,
            'data_points': len(predictor.data) if predictor.data is not None else 0,
            'model_type': type(predictor.model).__name__ if predictor.model else 'None'
        })
    return jsonify(model_list)

if __name__ == '__main__':
    print("Initializing Stock Prediction App...")
    
    # Test with a few reliable stocks first
    test_stocks = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']
    
    for stock in test_stocks:
        try:
            print(f"Pre-training model for {stock}...")
            success = train_or_load_model(stock)
            if success:
                print(f"✓ Successfully loaded/trained model for {stock}")
            else:
                print(f"✗ Failed to load/train model for {stock}")
        except Exception as e:
            print(f"✗ Error with {stock}: {e}")
    
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)