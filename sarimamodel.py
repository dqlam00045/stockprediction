# sarima_model.py
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def fit_sarima_model(train_data, test_data):
    """
    Fit a SARIMA model and make predictions.
    """
    train_series = train_data['Close']
    test_series = test_data['Close']

    if train_series.index.freq is None:
        train_series = train_series.asfreq('D')
    if test_series.index.freq is None:
        test_series = test_series.asfreq('D')

    model = SARIMAX(train_series, 
                    order=(5, 1, 0),  # (p, d, q) parameters
                    seasonal_order=(1, 1, 1, 12))  # (P, D, Q, s) parameters
    sarima_model = model.fit(disp=False)

    sarima_predictions = sarima_model.get_forecast(steps=len(test_series))
    sarima_predictions = sarima_predictions.predicted_mean
    
    sarima_predictions = pd.Series(sarima_predictions, index=test_series.index)

    return sarima_predictions
