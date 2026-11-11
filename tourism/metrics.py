import numpy as np

def mae(actual, forecast):
    """Mean absolute error: on average, how many tourists the forecast was off by."""
    errors = np.asarray(actual) - np.asarray(forecast)
    return np.mean(np.abs(errors))

def rmse(actual, forecast):
    """Root mean squared error: like MAE, but big misses count more."""
    errors = np.asarray(actual) - np.asarray(forecast)
    return np.sqrt(np.mean(errors ** 2))

def mape(actual, forecast):
    """Mean absolute percentage error: on average, how many % of tourists the forecast was off by."""
    actual = np.asarray(actual)
    forecast = np.asarray(forecast)
    return np.mean(np.abs((actual - forecast) / actual)) * 100
