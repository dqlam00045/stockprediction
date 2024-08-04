# main.py
import numpy as np
from data_collection import load_stock_data
from feature_engineering import compute_technical_indicators
from sentiment_analysis import get_reddit_sentiment
from model_training import prepare_data, train_model
from predictions import make_predictions, plot_predictions

symbol = 'AAPL'
start_date = '2020-01-01'
end_date = '2023-01-01'
subreddit = 'stocks'
query = 'AAPL'

data = load_stock_data(symbol, start_date, end_date)
data = compute_technical_indicators(data)

# Get average sentiment score from Reddit
reddit_sentiment = get_reddit_sentiment(subreddit, query, start_date, end_date)

# Prepare data for model
x, y, scaler = prepare_data(data, reddit_sentiment)

# Split data into train and test sets
train_size = int(len(x) * 0.8)
x_train, y_train = x[:train_size], y[:train_size]
x_test, y_test = x[train_size:], y[train_size:]

# Train the model
input_shape = (x_train.shape[1], x_train.shape[2])
model = train_model(x_train, y_train, x_test, y_test, input_shape)

# Make predictions
predictions = make_predictions(model, x_test, scaler)

# Align predictions index with the original data index
predictions_index = data.index[-len(predictions):]

# Plot the predictions
plot_predictions(data, predictions, predictions_index, symbol)
