import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN, InputLayer

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
        if data.empty:
            raise ValueError("No data fetched. Please check the symbol and date range.")
        
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
        split_date = pd.to_datetime(end_date)
        train_data = data[data.index < split_date]
        test_data = data[data.index >= split_date]
    else:
        train_data, test_data = train_test_split(data, test_size=1-split_ratio, shuffle=True, random_state=42)

    return train_data, test_data, scalers

def create_model(input_shape, layer_config, output_units=1, loss='mean_squared_error', optimizer='adam'):
    """
    Create a deep learning model based on the specified configuration.
    
    Params:
        input_shape (tuple): Shape of the input data (timesteps, features).
        layer_config (list of dict): Configuration for each layer. Each dict should have:
            - 'type': Type of the layer (LSTM, GRU, RNN).
            - 'units': Number of units in the layer.
            - 'return_sequences': Whether to return the last output in the output sequence, or the full sequence.
            - 'dropout': Dropout rate for the layer (0-1).
        output_units (int): Number of output units.
        loss (str): Loss function for the model.
        optimizer (str): Optimizer for the model.
    
    Returns:
        model (tf.keras.Model): Compiled deep learning model.
    """
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    
    for layer in layer_config:
        if layer['type'].lower() == 'lstm':
            model.add(LSTM(units=layer['units'], return_sequences=layer['return_sequences']))
        elif layer['type'].lower() == 'gru':
            model.add(GRU(units=layer['units'], return_sequences=layer['return_sequences']))
        elif layer['type'].lower() == 'rnn':
            model.add(SimpleRNN(units=layer['units'], return_sequences=layer['return_sequences']))
        else:
            raise ValueError(f"Unsupported layer type: {layer['type']}")
        
        if 'dropout' in layer:
            model.add(Dropout(rate=layer['dropout']))
    
    model.add(Dense(units=output_units))
    
    model.compile(optimizer=optimizer, loss=loss)
    
    return model

# Example usage:
start_date = '2015-01-01'
end_date = '2020-01-01'
symbol = 'TSLA'
train_data, test_data, scalers = load_process_dataset(symbol, start_date, end_date, split_ratio=0.8, scale_features=True, save_data=True)

# Prepare the training data for the model
PREDICTION_DAYS = 60

x_train, y_train = [], []
train_data_values = train_data['Close'].values

for x in range(PREDICTION_DAYS, len(train_data_values)):
    x_train.append(train_data_values[x-PREDICTION_DAYS:x])
    y_train.append(train_data_values[x])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Prepare the testing data
test_data_values = test_data['Close'].values
x_test, y_test = [], []

for x in range(PREDICTION_DAYS, len(test_data_values)):
    x_test.append(test_data_values[x-PREDICTION_DAYS:x])
    y_test.append(test_data_values[x])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Configuration for the layers
layer_config = [
    {'type': 'LSTM', 'units': 50, 'return_sequences': True, 'dropout': 0.2},
    {'type': 'LSTM', 'units': 50, 'return_sequences': False, 'dropout': 0.2}
]

# Create and compile the model
input_shape = (x_train.shape[1], 1)
model = create_model(input_shape, layer_config, output_units=1, loss='mean_squared_error', optimizer='adam')

# Summary of the model
model.summary()

# Train the model
history = model.fit(x_train, y_train, epochs=25, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model performance
loss = model.evaluate(x_test, y_test)
print(f'Test Loss: {loss}')

# Predictions
predictions = model.predict(x_test)
predictions = scalers['Close'].inverse_transform(predictions)
y_test_scaled = scalers['Close'].inverse_transform(y_test.reshape(-1, 1))

# Check if the model has been trained properly
plt.figure(figsize=(14, 5)) 
plt.plot(y_test_scaled, color='black', label='Actual Price')
plt.plot(predictions, color='green', label='Predicted Price')
plt.title(f'{symbol} Share Price Prediction')
plt.xlabel('Time')
plt.ylabel('Share Price')
plt.legend()
plt.show()

