from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN, InputLayer

def create_model(input_shape, layer_config, output_units=1, loss='mean_squared_error', optimizer='adam'):
    """
    Create a deep learning model based on the specified configuration.
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
