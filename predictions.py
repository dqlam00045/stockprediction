# predictions.py
import numpy as np
import matplotlib.pyplot as plt

def make_predictions(model, x_test, scaler):
    predictions = model.predict(x_test)

    # Create an array of the same shape as the scaled data with zeros
    dummy_data = np.zeros((predictions.shape[0], scaler.scale_.shape[0]))

    # Replace the column that corresponds to the Close price with the predictions
    dummy_data[:, 3] = predictions[:, 0]  # Close price is the 4th column (index 3)

    # Inverse transform only the Close price column
    predictions = scaler.inverse_transform(dummy_data)[:, 3]
    return predictions

def plot_predictions(data, predictions, predictions_index, symbol):
    plt.figure(figsize=(14, 5))
    plt.plot(data['Close'][predictions_index], color='black', label='Actual Price')
    plt.plot(predictions_index, predictions, color='green', label='Predicted Price')
    plt.title(f'Share Price Prediction for {symbol}')
    plt.xlabel('Time')
    plt.ylabel('Share Price')
    plt.legend()
    plt.show()
