import numpy as np
from data_loader import load_process_dataset
from model_builder import create_model
from plotting import plot_predictions

def main():
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

    # Plot predictions
    plot_predictions(y_test_scaled, predictions, symbol)

if __name__ == "__main__":
    main()
