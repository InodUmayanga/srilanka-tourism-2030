"""Forecasting models. Every function has the same shape:

    forecast = model_name(train, horizon)

train   : monthly arrivals (pandas Series), disrupted months are NaN
horizon : how many months ahead to forecast
returns : pandas Series of forecasts, indexed by the future months
"""
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.simplefilter("ignore")


def future_index(train, horizon):
    """The months that come right after the training data."""
    return pd.date_range(train.index[-1] + pd.DateOffset(months=1), periods=horizon, freq="MS")


def seasonal_naive(train, horizon):
    """Each future month = the same month in the last 12 months of training."""
    last_year = train.iloc[-12:].values
    values = [last_year[i % 12] for i in range(horizon)]
    return pd.Series(values, index=future_index(train, horizon))


def _regression_features(index, use_post_covid):
    X = pd.DataFrame(index=index)
    X["t"] = ((index.year - 2010) * 12 + (index.month - 1)).astype(float)
    for m in range(2, 13):
        X[f"month_{m}"] = (index.month == m).astype(float)
    if use_post_covid:
        X["post_covid"] = (index >= "2024-01-01").astype(float)
    return sm.add_constant(X, has_constant="add")


def regression(train, horizon):
    """log(arrivals) = trend + month effects (+ post-COVID level when 2024+ data exists)."""
    clean = train.dropna()
    use_post = clean.index.max() >= pd.Timestamp("2024-01-01")
    model = sm.OLS(np.log(clean), _regression_features(clean.index, use_post)).fit()
    future = future_index(train, horizon)
    return np.exp(model.predict(_regression_features(future, use_post)))


def holt_winters(train, horizon, damped=True):
    """Holt-Winters on logs, with the disrupted years cut out and the two ends joined."""
    clean = pd.concat([train[:"2018-12"], train["2024-01":]]).dropna()
    log_series = pd.Series(np.log(clean.values),
                           index=pd.date_range("2000-01-01", periods=len(clean), freq="MS"))
    model = ExponentialSmoothing(log_series, trend="add", damped_trend=damped,
                                 seasonal="add", seasonal_periods=12).fit()
    return pd.Series(np.exp(model.forecast(horizon).values), index=future_index(train, horizon))


def sarima(train, horizon, order=(0, 1, 1), seasonal_order=(1, 1, 1, 12)):
    """SARIMA on logs with the settings chosen by AIC in Step 9. Handles NaN months itself."""
    log_train = np.log(train)
    log_train.index.freq = "MS"
    model = SARIMAX(log_train, order=order, seasonal_order=seasonal_order).fit(disp=False)
    return pd.Series(np.exp(model.forecast(horizon).values), index=future_index(train, horizon))


_chronos_pipeline = None      # loaded once, then reused (loading takes a few seconds)

def chronos2(train, horizon):
    """Amazon Chronos-2, a pre-trained foundation model. Zero-shot: no training on our data.
    It reads the history (NaN months included) and returns the median forecast."""
    global _chronos_pipeline
    if _chronos_pipeline is None:
        from chronos import Chronos2Pipeline
        _chronos_pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    context = pd.DataFrame({"id": "sri_lanka", "timestamp": train.index, "target": train.values})
    prediction = _chronos_pipeline.predict_df(
        context, prediction_length=horizon, quantile_levels=[0.1, 0.5, 0.9],
        id_column="id", timestamp_column="timestamp", target="target")
    return pd.Series(prediction["0.5"].values, index=future_index(train, horizon))


# name -> function, used by the comparison scripts
MODELS = {
    "Seasonal naive": seasonal_naive,
    "Regression": regression,
    "Holt-Winters damped": holt_winters,
    "SARIMA": sarima,
    "Chronos-2": chronos2,
}