import numpy as np

def prepare_multistep_data(data, prediction_days, future_days):
    """
    Prepare data for multistep prediction.
    """
    x, y = [], []
    for i in range(len(data) - prediction_days - future_days + 1):
        x.append(data[i:i+prediction_days])
        y.append(data[i+prediction_days:i+prediction_days+future_days])
    return np.array(x), np.array(y)

def prepare_multivariate_data(data, prediction_days):
    """
    Prepare data for multivariate prediction.
    """
    x, y = [], []
    for i in range(prediction_days, len(data)):
        x.append(data[i-prediction_days:i])
        y.append(data[i][-1])  # Assuming the last column is the target (closing price)
    return np.array(x), np.array(y)
