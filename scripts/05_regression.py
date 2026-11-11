# =====================================================================
# Step 7: Linear regression forecast
# Idea: log(arrivals) = a + b * time + month effect (+ post-COVID level)
# =====================================================================

# ---- Libraries ----
import numpy as np                    # maths: log, exp
import pandas as pd                   # tables and time series
import matplotlib.pyplot as plt       # charts
import statsmodels.api as sm          # the regression model (OLS)
from tourism.data import load_training_arrivals     # our own data loader (Step 6a)
from tourism.metrics import mae, rmse, mape         # our own error metrics (Step 6b)
from tourism.paths import FIGURES, RESULTS          # folders to save outputs

# ---- 1. Load the data and split it ----
train_all = load_training_arrivals()            # 2010-2025, disrupted months are NaN
train = train_all[:"2024-12"].dropna()          # training: up to Dec 2024, drop the NaN months
validation = train_all["2025-01":"2025-12"]     # validation: the 12 months of 2025


# ---- 2. A function that builds the input table (X) for the model ----
def make_features(index, use_post_covid):
    """Build the X table: time trend, 11 month dummies, optional post-COVID level."""
    X = pd.DataFrame(index=index)               # empty table with one row per month

    # "t" = number of months since January 2010 (0, 1, 2, ...). This is the TREND.
    X["t"] = ((index.year - 2010) * 12 + (index.month - 1)).astype(float)

    # 11 dummy columns, one for each month from February (2) to December (12).
    # A dummy is 1 if the row is that month, otherwise 0.
    # January has no column: it is the base month the others are compared with.
    for m in range(2, 13):
        X[f"month_{m}"] = (index.month == m).astype(float)

    # Optional extra column: 1 from January 2024 onwards, 0 before.
    # It lets the model learn that the level after COVID is different.
    if use_post_covid:
        X["post_covid"] = (index >= "2024-01-01").astype(float)

    # Add a column of 1s called "const". Its coefficient is "a" (the starting level).
    return sm.add_constant(X, has_constant="add")


# ---- 3. Fit two models and forecast 2025 with each ----
all_scores = []       # error scores of each model (list of dictionaries)
forecasts = {}        # the 2025 forecast of each model, for the chart

# Loop twice: first Model A (no post-COVID column), then Model B (with it)
for name, use_post in [("Regression A: trend + months", False),
                       ("Regression B: + post-COVID level", True)]:

    # TRAIN: find the coefficients that best fit log(arrivals) from the features
    model = sm.OLS(np.log(train), make_features(train.index, use_post)).fit()

    # FORECAST: build the features for 2025, predict log(arrivals), convert back to arrivals
    forecast = np.exp(model.predict(make_features(validation.index, use_post)))

    forecasts[name] = forecast                  # keep it for the chart

    # SCORE: compare the forecast with the real 2025 numbers
    all_scores.append({"model": name, "MAE": mae(validation, forecast),
                       "RMSE": rmse(validation, forecast), "MAPE": mape(validation, forecast)})

    # PRINT what the model learned. exp(coefficient) - 1 turns a log effect into a %.
    print(f"\n{name}")
    print(f"  MAPE on 2025 = {mape(validation, forecast):.1f}%")
    # monthly trend coefficient * 12 = yearly growth
    print(f"  Trend growth per year = {(np.exp(model.params['t'] * 12) - 1) * 100:.1f}%")
    # how much busier December is than January
    print(f"  December vs January = {(np.exp(model.params['month_12']) - 1) * 100:+.1f}%")
    # how much quieter May is than January
    print(f"  May vs January = {(np.exp(model.params['month_5']) - 1) * 100:+.1f}%")
    if use_post:
        # how far the post-COVID level is from where the old trend line would be
        print(f"  Post-COVID level vs old trend line = {(np.exp(model.params['post_covid']) - 1) * 100:+.1f}%")
        print(model.summary().tables[0])        # statsmodels' own summary (R-squared etc.)

# ---- 4. Add the scores to the results file from Step 6 ----
scores = pd.read_csv(RESULTS / "errors_2025.csv")                   # has the seasonal naive row
scores = scores[~scores["model"].str.startswith("Regression")]      # remove old regression rows (if re-run)
scores = pd.concat([scores, pd.DataFrame(all_scores)], ignore_index=True)   # add the two new rows
scores.to_csv(RESULTS / "errors_2025.csv", index=False)             # save
print("\nAll models so far:")
print(scores.round(1))

# ---- 5. Chart: actual 2025 vs the two forecasts ----
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(validation.index, validation.values, marker="o", label="Actual 2025")
for name, forecast in forecasts.items():        # one dashed line per model
    ax.plot(forecast.index, forecast.values, marker="o", linestyle="--", label=name)
ax.set_title("Regression forecasts vs actual arrivals, 2025")
ax.set_xlabel("Month")
ax.set_ylabel("Monthly arrivals")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "07_regression_2025.png", dpi=150)
plt.close(fig)