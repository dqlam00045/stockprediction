from data_loader import load_process_dataset

def main():
    start_date = '2015-01-01'
    end_date = '2020-01-01'
    symbol = 'TSLA'
    train_data, test_data, scalers = load_process_dataset(symbol, start_date, end_date, split_ratio=0.8, scale_features=True, save_data=True)

    if not train_data.empty and not test_data.empty:
        print("Data loaded successfully!")
        print("\nTraining data sample:")
        print(train_data.head())
        print("\nTest data sample:")
        print(test_data.head())
    else:
        print("Error: Data not loaded successfully!")

if __name__ == "__main__":
    main()
