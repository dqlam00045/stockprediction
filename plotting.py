import matplotlib.pyplot as plt

def plot_predictions(y_test_scaled, predictions, symbol):
    """
    Plots the actual vs predicted prices.

    Params:
        y_test_scaled (np.array): Scaled actual prices.
        predictions (np.array): Predicted prices.
        symbol (str): Stock ticker symbol.
    """
    plt.figure(figsize=(14, 5))
    plt.plot(y_test_scaled, color='black', label='Actual Price')
    plt.plot(predictions, color='green', label='Predicted Price')
    plt.title(f'{symbol} Share Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Share Price')
    plt.legend()
    plt.show()
