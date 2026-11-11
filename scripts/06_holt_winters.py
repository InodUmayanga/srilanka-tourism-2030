# =====================================================================
# Step 8: Holt-Winters exponential smoothing
# Idea: keep a running level, trend and seasonal pattern, updated every month
# =====================================================================
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from tourism.data import load_training_arrivals
from tourism.metrics import mae, rmse, mape
from tourism.paths import FIGURES, RESULTS

warnings.simplefilter("ignore")                 # hide the optimizer's technical warnings

# ---- 1. Load and split ----
train_all = load_training_arrivals()
train = train_all[:"2024-12"]
validation = train_all["2025-01":"2025-12"]

# ---- 2. Close the gap: join 2010-2018 directly to 2024 ----
# Whole years only, so December 2018 is followed by January 2024 and months stay aligned.
clean = pd.concat([train["2010-01":"2018-12"], train["2024-01":"2024-12"]])
# The model needs evenly spaced dates, so give the joined series a simple monthly index.
log_series = pd.Series(np.log(clean.values),
                       index=pd.date_range("2000-01-01", periods=len(clean), freq="MS"))

# ---- 3. Fit two versions and forecast 12 months ----
all_scores = []
forecasts = {}
for name, damped in [("Holt-Winters", False), ("Holt-Winters damped", True)]:
    model = ExponentialSmoothing(
        log_series,
        trend="add",                 # there is a trend
        damped_trend=damped,         # True = the trend fades out over time
        seasonal="add",              # there is a seasonal pattern
        seasonal_periods=12,        # months in one seasonal cycle
    ).fit()

    # forecast 12 months ahead (in logs), convert back, and label with the real 2025 dates
    forecast = pd.Series(np.exp(model.forecast(12).values), index=validation.index)
    forecasts[name] = forecast

    all_scores.append({"model": name, "MAE": mae(validation, forecast),
                       "RMSE": rmse(validation, forecast), "MAPE": mape(validation, forecast)})
    print(f"\n{name}")
    print(f"  MAPE on 2025 = {mape(validation, forecast):.1f}%")
    print(f"  Forecast total for 2025 = {forecast.sum():,.0f}  (actual {validation.sum():,.0f})")
    # how fast each notebook is updated: 0 = never changes, 1 = only the latest month counts
    print(f"  alpha (level) = {model.params['smoothing_level']:.2f}, "
          f"beta (trend) = {model.params['smoothing_trend']:.2f}, "
          f"gamma (season) = {model.params['smoothing_seasonal']:.2f}")

# ---- 4. Add the scores to the results file ----
scores = pd.read_csv(RESULTS / "errors_2025.csv")
scores = scores[~scores["model"].str.startswith("Holt-Winters")]
scores = pd.concat([scores, pd.DataFrame(all_scores)], ignore_index=True)
scores.to_csv(RESULTS / "errors_2025.csv", index=False)
print("\nAll models so far:")
print(scores.round(1))

# ---- 5. Chart ----
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(validation.index, validation.values, marker="o", label="Actual 2025")
for name, forecast in forecasts.items():
    ax.plot(forecast.index, forecast.values, marker="o", linestyle="--", label=name)
ax.set_title("Holt-Winters forecasts vs actual arrivals, 2025")
ax.set_xlabel("Month")
ax.set_ylabel("Monthly arrivals")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "08_holt_winters_2025.png", dpi=150)
plt.close(fig)