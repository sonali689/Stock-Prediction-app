import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
print("All imports successful!")

# Test data fetch
data = yf.download('RELIANCE.NS', period='1mo')
print(f"Data shape: {data.shape}")
print("Installation successful!")