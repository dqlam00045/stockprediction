from data_loader import load_process_dataset
from plotting import plot_candlestick, plot_boxplot

def main():
    start_date = '2015-01-01'
    end_date = '2020-01-01'
    symbol = 'TSLA'
    train_data, test_data, scalers = load_process_dataset(symbol, start_date, end_date, split_ratio=0.8, scale_features=True, save_data=True)

    # Plot candlestick chart
    plot_candlestick(train_data, symbol)

    # Plot boxplot chart
    plot_boxplot(train_data, ['Open', 'High', 'Low', 'Close', 'Volume'])

if __name__ == "__main__":
    main()
