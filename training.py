# train_evaluate.py
import numpy as np
import matplotlib.pyplot as plt
from model_utils import create_model, prepare_multivariate_data
from sarima_model import fit_sarima_model

def train_and_evaluate_model(train_data, test_data, scalers, layer_config, epochs=25, batch_size=32):
    """
    Train and evaluate the model, then integrate predictions from SARIMA model.
    """
    PREDICTION_DAYS = 60

    x_train, y_train = prepare_multivariate_data(train_data.values, PREDICTION_DAYS)
    x_test, y_test = prepare_multivariate_data(test_data.values, PREDICTION_DAYS)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    input_shape = (x_train.shape[1], x_train.shape[2])
    model = create_model(input_shape, layer_config, output_units=1, loss='mean_squared_error', optimizer='adam')

    model.summary()
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))

    loss = model.evaluate(x_test, y_test)
    print(f'Test Loss: {loss}')

    lstm_predictions = model.predict(x_test)
    lstm_predictions = scalers['Close'].inverse_transform(lstm_predictions)
    y_test_scaled = scalers['Close'].inverse_transform(y_test.reshape(-1, 1))

    sarima_predictions = fit_sarima_model(train_data, test_data)
    sarima_predictions = scalers['Close'].inverse_transform(sarima_predictions.values.reshape(-1, 1))
    
    min_length = min(len(lstm_predictions), len(sarima_predictions))
    lstm_predictions = lstm_predictions[:min_length]
    sarima_predictions = sarima_predictions[:min_length]
    y_test_scaled = y_test_scaled[:min_length]

    combined_predictions = (lstm_predictions + sarima_predictions) / 2
    
    plt.figure(figsize=(14, 5))
    plt.plot(y_test_scaled, color='black', label='Actual Price')
    plt.plot(combined_predictions, color='blue', label='Combined Predictions')
    plt.plot(lstm_predictions, color='green', label='LSTM Predictions')
    plt.plot(sarima_predictions, color='red', label='SARIMA Predictions')
    plt.title(f'Share Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Share Price')
    plt.legend()
    plt.show()

    return model
