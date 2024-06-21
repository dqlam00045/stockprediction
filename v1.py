import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os

def load_process_dataset(symbol, start_date, end_date, split_ratio=0.8, scale_features=True, save_data=True):
    """
    Loads and processes the stock market data for a given symbol and date range.
    Params:
        symbol (str): Stock ticker symbol.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        split_ratio (float/str): Ratio to split train and test data or 'date' to split by date.
        scale_features (bool): Whether to scale features using MinMaxScaler.
        save_data (bool): Whether to save the downloaded data locally.
    Returns:
        train_data (pd.DataFrame): Training data.
        test_data (pd.DataFrame): Testing data.
        scalers (dict): Dictionary of scalers for each feature.
    """
    # Check if data already exists locally
    filename = f"{symbol}_data.csv"
    if os.path.exists(filename):
        data = pd.read_csv(filename, index_col='Date', parse_dates=True)
    else:
        # Download data from Yahoo Finance
        data = yf.download(symbol, start=start_date, end=end_date)
        # Save data locally
        if save_data:
            data.to_csv(filename)

    # Handle NaN values
    data.dropna(inplace=True)

    # Select features
    features = data.columns

    # Scale features if required
    scalers = {}
    if scale_features:
        for feature in features:
            scaler = MinMaxScaler(feature_range=(0, 1))
            data[feature] = scaler.fit_transform(data[feature].values.reshape(-1, 1))
            scalers[feature] = scaler

    # Split data into train and test sets
    if isinstance(split_ratio, float):
        train_size = int(len(data) * split_ratio)
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
    elif isinstance(split_ratio, str) and split_ratio.lower() == 'date':
        train_data = data[data.index < end_date]
        test_data = data[data.index >= end_date]
    else:
        train_data, test_data = train_test_split(data, test_size=split_ratio, shuffle=True, random_state=42)

    return train_data, test_data, scalers

def main():
    start_date = '2015-01-01'
    end_date = '2020-01-01'
    symbol = 'TSLA'
    train_data, test_data, scalers = load_process_dataset(symbol, start_date, end_date, split_ratio=0.8, scale_features=True, save_data=True)

    if not train_data.empty and not test_data.empty:
        print("Data loaded successfully!")
        print("\nTraining data sample:")
        print(train_data.head())
        print("\nTest data sample:")
        print(test_data.head())
    else:
        print("Error: Data not loaded successfully!")

if __name__ == "__main__":
    main()
