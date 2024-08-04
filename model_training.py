# model_training.py
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

def prepare_data(data, sentiment):
    data['Sentiment'] = sentiment
    data = data.dropna()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    x, y = [], []
    for i in range(60, len(scaled_data)):
        x.append(scaled_data[i-60:i])
        y.append(scaled_data[i, 3])  # Close price

    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], x.shape[2]))

    return x, y, scaler

def build_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(x_train, y_train, x_test, y_test, input_shape):
    model = build_model(input_shape)
    model.fit(x_train, y_train, epochs=25, batch_size=32, validation_data=(x_test, y_test))
    return model
