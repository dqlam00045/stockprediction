import matplotlib.pyplot as plt
import plotly.graph_objects as go

def plot_candlestick(data, ticker):
    """
    Plots the candlestick chart for the given stock market data.
    Params:
        data (pd.DataFrame): the dataframe containing stock market data with 'Open', 'High', 'Low', and 'Close' columns
        ticker (str): the stock ticker symbol
    """
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close'])])
    fig.update_layout(title=f'Candlestick Chart for {ticker}',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      xaxis_rangeslider_visible=False)
    fig.show()

def plot_boxplot(data, feature_columns):
    """
    Plots the boxplot chart for the given stock market data.
    Params:
        data (pd.DataFrame): the dataframe containing stock market data
        feature_columns (list): the list of feature columns to include in the boxplot
    """
    plt.figure(figsize=(10, 6))
    data[feature_columns].boxplot()
    plt.title('Boxplot of Stock Market Data')
    plt.xlabel('Features')
    plt.ylabel('Values')
    plt.show()
