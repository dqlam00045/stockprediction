# feature_engineering.py
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from ta.utils import dropna

def compute_technical_indicators(data):
    data = dropna(data)  # Drop missing values

    # Calculate RSI
    data['RSI'] = RSIIndicator(close=data['Close'], window=14).rsi()

    # Calculate MACD
    macd = MACD(close=data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Diff'] = macd.macd_diff()

    # Calculate Bollinger Bands
    bollinger = BollingerBands(close=data['Close'], window=20, window_dev=2)
    data['Bollinger_High'] = bollinger.bollinger_hband()
    data['Bollinger_Low'] = bollinger.bollinger_lband()

    return data
