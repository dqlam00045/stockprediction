# data_handler.py
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os

def load_process_dataset(symbol, start_date, end_date, split_ratio=0.8, scale_features=True, save_data=True):
    """
    Loads and processes the stock market data for a given symbol and date range.
    """
    filename = f"{symbol}_data.csv"
    if os.path.exists(filename):
        data = pd.read_csv(filename, index_col='Date', parse_dates=True)
    else:
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            raise ValueError("No data fetched. Please check the symbol and date range.")
        if save_data:
            data.to_csv(filename)

    data.dropna(inplace=True)
    features = data.columns

    scalers = {}
    if scale_features:
        for feature in features:
            scaler = MinMaxScaler(feature_range=(0, 1))
            data[feature] = scaler.fit_transform(data[feature].values.reshape(-1, 1))
            scalers[feature] = scaler

    if isinstance(split_ratio, float):
        train_size = int(len(data) * split_ratio)
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
    elif isinstance(split_ratio, str) and split_ratio.lower() == 'date':
        split_date = pd.to_datetime(end_date)
        train_data = data[data.index < split_date]
        test_data = data[data.index >= split_date]
    else:
        train_data, test_data = train_test_split(data, test_size=1-split_ratio, shuffle=True, random_state=42)

    return train_data, test_data, scalers
