from data_loader import load_process_dataset
from training import train_and_evaluate_model

def main():
    start_date = '2015-01-01'
    end_date = '2020-01-01'
    symbol = 'TSLA'
    train_data, test_data, scalers = load_process_dataset(symbol, start_date, end_date, split_ratio=0.8, scale_features=True, save_data=True)

    # Configuration for the layers
    layer_config = [
        {'type': 'LSTM', 'units': 50, 'return_sequences': True, 'dropout': 0.2},
        {'type': 'LSTM', 'units': 50, 'return_sequences': False, 'dropout': 0.2}
    ]

    # Train and evaluate the model
    model = train_and_evaluate_model(train_data, test_data, scalers, layer_config, epochs=25, batch_size=32)

if __name__ == "__main__":
    main()
