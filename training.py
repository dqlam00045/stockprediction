import numpy as np
import matplotlib.pyplot as plt
from model_builder import create_model
from data_preparation import prepare_multivariate_data

def train_and_evaluate_model(train_data, test_data, scalers, layer_config, epochs=25, batch_size=32):
    """
    Train and evaluate the model.
    """
    PREDICTION_DAYS = 60

    # Prepare the training data for the model
    x_train, y_train = prepare_multivariate_data(train_data.values, PREDICTION_DAYS)

    # Prepare the testing data
    x_test, y_test = prepare_multivariate_data(test_data.values, PREDICTION_DAYS)

    # Reshape data to fit the model
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    # Create and compile the model
    input_shape = (x_train.shape[1], x_train.shape[2])
    model = create_model(input_shape, layer_config, output_units=1, loss='mean_squared_error', optimizer='adam')

    # Summary of the model
    model.summary()

    # Train the model
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))

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
    plt.title(f'Share Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Share Price')
    plt.legend()
    plt.show()

    return model
