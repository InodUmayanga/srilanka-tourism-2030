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

def chronos2_quantiles(train, horizon, levels=(0.1, 0.5, 0.9)):
    """Chronos-2 forecast range: one column per quantile (0.1 = low, 0.5 = median, 0.9 = high)."""
    global _chronos_pipeline
    if _chronos_pipeline is None:
        from chronos import Chronos2Pipeline
        _chronos_pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    context = pd.DataFrame({"id": "sri_lanka", "timestamp": train.index, "target": train.values})
    prediction = _chronos_pipeline.predict_df(
        context, prediction_length=horizon, quantile_levels=list(levels),
        id_column="id", timestamp_column="timestamp", target="target")
    out = prediction[[str(q) for q in levels]]
    out.index = future_index(train, horizon)
    return out

_timesfm_model = None


def timesfm(train, horizon):
    """Google TimesFM 2.5, a pre-trained foundation model (zero-shot).
    Its docs don't promise NaN support, so we join 2010-2018 to 2024+ as for Holt-Winters."""
    global _timesfm_model
    import torch
    if _timesfm_model is None:
        from transformers import TimesFm2_5ModelForPrediction
        _timesfm_model = TimesFm2_5ModelForPrediction.from_pretrained(
            "google/timesfm-2.5-200m-transformers", device_map="cpu")
    clean = pd.concat([train[:"2018-12"], train["2024-01":]]).dropna()
    history = torch.tensor(clean.values, dtype=torch.float32)
    with torch.no_grad():
        output = _timesfm_model(past_values=[history], return_dict=True)
    values = output.mean_predictions[0, :horizon].numpy()   # it forecasts 128 steps; keep the first `horizon`
    return pd.Series(values, index=future_index(train, horizon))


def ensemble(train, horizon, members=("Holt-Winters damped", "SARIMA", "Chronos-2")):
    """Average of several models' forecasts. Different models make different mistakes,
    so averaging usually cancels part of the error."""
    forecasts = [MODELS[name](train, horizon) for name in members]
    return sum(forecasts) / len(forecasts)


# name -> function, used by the comparison scripts
MODELS = {
    "Seasonal naive": seasonal_naive,
    "Regression": regression,
    "Holt-Winters damped": holt_winters,
    "SARIMA": sarima,
    "Chronos-2": chronos2,
    "TimesFM 2.5": timesfm,
}
MODELS["Ensemble (HW + SARIMA + Chronos-2)"] = ensemble
MODELS["Ensemble (HW + SARIMA + Chronos-2 + TimesFM)"] = lambda train, horizon: ensemble(
    train, horizon, members=("Holt-Winters damped", "SARIMA", "Chronos-2", "TimesFM 2.5"))