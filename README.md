# 📈 Stock Prediction Web App — Local Setup Guide

A machine-learning powered Flask web application that predicts *next-day stock price direction* using advanced technical indicators and an ensemble of ML models.

---

## 🧭 Table of Contents

* [Features](https://www.google.com/search?q=%23features)
* [Prerequisites](https://www.google.com/search?q=%23prerequisites)
* [Quick Start (Local Development)](https://www.google.com/search?q=%23quick-start-local-development)
* [Step 1: Clone or Download](https://www.google.com/search?q=%23step-1-clone-or-download)
* [Step 2: Set Up Virtual Environment](https://www.google.com/search?q=%23step-2-set-up-virtual-environment)
* [Step 3: Install Dependencies](https://www.google.com/search?q=%23step-3-install-dependencies)
* [Step 4: Verify Installation](https://www.google.com/search?q=%23step-4-verify-installation)
* [Step 5: Run the Application](https://www.google.com/search?q=%23step-5-run-the-application)
* [Step 6: Access the Web App](https://www.google.com/search?q=%23step-6-access-the-web-app)


* [📁 Project Structure](https://www.google.com/search?q=%23project-structure)
* [🎯 Using the Application](https://www.google.com/search?q=%23using-the-application)
* [⚙ Configuration Options](https://www.google.com/search?q=%23configuration-options)
* [🐛 Troubleshooting](https://www.google.com/search?q=%23troubleshooting)
* [⚡ Performance Tips](https://www.google.com/search?q=%23performance-tips)
* [🔧 Development Commands](https://www.google.com/search?q=%23development-commands)
* [📊 Understanding the Output](https://www.google.com/search?q=%23understanding-the-output)
* [🎓 Educational Notes](https://www.google.com/search?q=%23educational-notes)
* [⚠ Important Notes](https://www.google.com/search?q=%23important-notes)
* [🆘 Getting Help](https://www.google.com/search?q=%23getting-help)
* [🚀 Next Steps](https://www.google.com/search?q=%23next-steps)
* [License](https://www.google.com/search?q=%23license)

---

## ✨ Features

* Fetches historical data via *Yahoo Finance (yfinance)*
* Computes *technical indicators*
* Trains and evaluates *multiple ML models* (Random Forest, XGBoost, Logistic Regression)
* *Model selection* based on best performance
* *Model persistence & caching* to speed up subsequent runs
* Clean *Flask* web interface with preloaded Indian stocks and custom symbol support

---

## ✅ Prerequisites

* **Python** 3.8 or higher
* **pip** (Python package manager)
* **Git** (for version control)

---

## 🚀 Quick Start (Local Development)

### Step 1: Clone or Download

**Option A: Using Git**

```bash
git clone <https://github.com/sonali689/Stock-Prediction-app>
cd stock_prediction_project

```

**Option B: Manual Download**

1. Download all project files to a folder named `stock_prediction_project`
2. Ensure you have these files:
* `app.py`
* `advanced_stock_model.py`
* `templates/index.html`
* `static/style.css`
* `requirements.txt`



---

### Step 2: Set Up Virtual Environment (Recommended)

**Windows**

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

```

**macOS/Linux**

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

```

> You should see `(venv)` in your terminal prompt indicating the virtual environment is active.

---

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt

```

**If you don't have a requirements.txt, install packages manually:**

```bash
pip install pandas numpy scikit-learn yfinance matplotlib seaborn flask xgboost ta

```

---

### Step 4: Verify Installation

Create a test file `test_install.py`:

```python
import pandas as pd
import numpy as np
import yfinance as yf
import flask
print("All imports successful! Installation complete.")

```

**Run it:**

```bash
python test_install.py

```

---

### Step 5: Run the Application

```bash
python app.py

```

**You should see output similar to:**

```text
Initializing Stock Prediction App...
Pre-training model for RELIANCE.NS...
Training new model for RELIANCE.NS...
Fetching data for RELIANCE.NS...
...
Starting Flask server...
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000

```

---

### Step 6: Access the Web App

Open your browser and navigate to:
**[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 📁 Project Structure

```text
stock_prediction_project/
├── app.py                  # Main Flask application
├── advanced_stock_model.py # ML pipeline and technical indicators
├── templates/
│   └── index.html          # Web interface
├── static/
│   └── style.css           # CSS styling
├── model_*.pkl             # Auto-generated saved models (cache)
├── requirements.txt        # Python dependencies
└── README.md               # This file

```

---

## 🎯 Using the Application

1. **Select a Stock**
* Choose from pre-loaded Indian stocks (RELIANCE.NS, TCS.NS, INFY.NS, etc.)
* Or enter a custom symbol (format: `SYMBOL.NS` for NSE stocks)


2. **Get Stock Information**
* Click *Get Stock Info* to view current market data
* Shows price, volume, market cap, sector info (where available)


3. **Make Predictions**
* Click *Predict Next Day* for ML-powered direction prediction
* View confidence scores and probabilities
* See model information and features used


4. **Supported Stocks (Examples)**
* **Large Cap:** RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK
* **Mid Cap:** TATAMOTORS, AXISBANK, LT, ASIANPAINT
* **Tech:** WIPRO, TECHM, HCLTECH



---

## ⚙ Configuration Options

### Changing Default Settings

In `app.py`, you can change the default symbol and period:

```python
# Change default stock or period
predictor = RobustStockPredictor(stock_symbol='TCS.NS', period='1y')

```

*Available periods: '1y', '2y', '5y'*

### Adding New Stocks

Edit the `available_stocks` endpoint/data structure in `app.py`:

```python
'indian_stocks': {
    'Your Category': [
        {'symbol': 'NEWSTOCK.NS', 'name': 'New Stock Name'},
    ]
}

```

---

## 🐛 Troubleshooting

### 1) Module Not Found Errors

```bash
# Reactivate virtual environment and reinstall
deactivate
# macOS/Linux
source venv/bin/activate
# Windows
# venv\Scripts\activate
pip install -r requirements.txt

```

### 2) Port Already in Use

```bash
# macOS/Linux: Kill process using port 5000
lsof -ti:5000 | xargs kill -9

# Or change port in app.py
app.run(debug=True, host='0.0.0.0', port=5001)

```

### 3) Yahoo Finance Data Issues

* Check internet connection
* Verify stock symbol format (must end with `.NS` for NSE)
* Some tickers may have limited history

### 4) Model Training Failures

```bash
# Clear cached models in current directory
python -c "import os; [os.remove(f) for f in os.listdir('.') if f.startswith('model_')]"

# Or through the web interface
# Visit http://127.0.0.1:5000/clear_cache

```

---

## ⚡ Performance Tips

**For Faster Training**

* Reduce data period to `1y` in `app.py`
* Decrease the number of estimators in models
* Use fewer technical indicators

**For Better Accuracy**

* Increase data period to `3y` or `5y`
* Add more technical indicators
* Experiment with different model parameters

---

## 🔧 Development Commands

**Start the Application**

```bash
python app.py

```

**Stop the Application**
Press **Ctrl + C** in the terminal where the app is running.

**Restart the Application**

```bash
# Stop with Ctrl+C, then:
python app.py

```

**Clear Model Cache**

```bash
# Delete all saved models
rm model_*.pkl

# Or through the web interface:
# Visit http://127.0.0.1:5000/clear_cache

```

---

## 📊 Understanding the Output

**Prediction Results**

* **Prediction:** UP / DOWN direction for next trading day
* **Confidence:** Model certainty (0–100%)
* **Signal Strength:** STRONG / MODERATE / WEAK (derived from confidence)
* **Probabilities:** % likelihood for UP/DOWN movements

**Model Information**

* **Data Points:** Number of historical records used
* **Features Used:** Count of technical indicators
* **Model Type:** Best performing algorithm (Random Forest / XGBoost / Logistic Regression)
* **Training Status:** Model readiness indicator

---

## 🎓 Educational Notes

This application demonstrates:

* End-to-end ML pipeline: data collection → feature engineering → modeling → deployment
* Technical indicator engineering (via `ta` library)
* Comparing multiple algorithms and selecting the best
* Web development with **Flask**
* Model persistence and caching strategies

---

## ⚠ Important Notes

* First run may take **2–5 minutes** as models train for each stock
* Predictions are **not financial advice**
* Model performance varies by stock and market conditions

---

## 🆘 Getting Help

If you encounter issues:

* Check the **terminal output** for error messages
* Verify **dependencies** are installed correctly
* Ensure a **stable internet connection** for data fetching
* Open browser **DevTools Console** (F12) for frontend errors
* Refer to the **Troubleshooting** section above

---

## 🚀 Next Steps

* Experiment with different stocks
* Modify technical indicators in `advanced_stock_model.py`
* Tweak model parameters for better performance
* Extend the feature set with additional indicators
* Deploy to a cloud platform (Render, Railway, Heroku, etc.)

---

Your stock prediction app should now be running at **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

