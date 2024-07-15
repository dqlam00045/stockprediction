import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN, InputLayer

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
