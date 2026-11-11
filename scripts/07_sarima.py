# =====================================================================
# Step 9: SARIMA
# Idea: forecast from the series' own past values and past forecast errors
# =====================================================================
import itertools
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from tourism.data import load_training_arrivals
from tourism.metrics import mae, rmse, mape
from tourism.paths import FIGURES, RESULTS

warnings.simplefilter("ignore")                 # hide the optimizer's technical warnings

# ---- 1. Load and split ----
train_all = load_training_arrivals()
train = train_all[:"2024-12"]                   # keep the NaN months: SARIMA can skip over them
validation = train_all["2025-01":"2025-12"]
log_train = np.log(train)
log_train.index.freq = "MS"                     # tell statsmodels the data is monthly

# ---- 2. Try 16 small models and compare them by AIC (lower = better) ----
# order = (p, d, q): p past values, d differences, q past errors
# seasonal_order = (P, D, Q, 12): the same three, but looking 12 months back
candidates = []
for p, q, P, Q in itertools.product([0, 1], repeat=4):      # every 0/1 combination
    fitted = SARIMAX(log_train, order=(p, 1, q), seasonal_order=(P, 1, Q, 12)).fit(disp=False)
    candidates.append({"order": (p, 1, q), "seasonal_order": (P, 1, Q, 12), "AIC": fitted.aic})
candidates = pd.DataFrame(candidates).sort_values("AIC").reset_index(drop=True)
print("Candidate models, best AIC first:")
print(candidates.head(5).round(1))

# ---- 3. Refit the best one and forecast 2025 ----
best = candidates.iloc[0]                       # first row = lowest AIC
model = SARIMAX(log_train, order=best["order"], seasonal_order=best["seasonal_order"]).fit(disp=False)
forecast = pd.Series(np.exp(model.forecast(12).values), index=validation.index)

name = f"SARIMA {best['order']}{best['seasonal_order']}"
print(f"\n{name}")
print(f"  MAPE on 2025 = {mape(validation, forecast):.1f}%")
print(f"  Forecast total for 2025 = {forecast.sum():,.0f}  (actual {validation.sum():,.0f})")

# ---- 4. Residual check: is there any pattern left that the model missed? ----
residuals = model.resid.iloc[13:].dropna()      # skip the first 13 start-up values and the gap
ljung_box = acorr_ljungbox(residuals, lags=[12])
p_value = ljung_box["lb_pvalue"].iloc[0]
print(f"  Ljung-Box p-value = {p_value:.3f}  ->",
      "OK, residuals look random" if p_value > 0.05 else "pattern left in residuals")

# ---- 5. Add the score to the results file ----
scores = pd.read_csv(RESULTS / "errors_2025.csv")
scores = scores[~scores["model"].str.startswith("SARIMA")]
new_row = {"model": name, "MAE": mae(validation, forecast),
           "RMSE": rmse(validation, forecast), "MAPE": mape(validation, forecast)}
scores = pd.concat([scores, pd.DataFrame([new_row])], ignore_index=True)
scores.to_csv(RESULTS / "errors_2025.csv", index=False)
print("\nAll models so far, best first:")
print(scores.sort_values("MAPE").round(1))

# ---- 6. Chart ----
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(validation.index, validation.values, marker="o", label="Actual 2025")
ax.plot(forecast.index, forecast.values, marker="o", linestyle="--", label=name)
ax.set_title("SARIMA forecast vs actual arrivals, 2025")
ax.set_xlabel("Month")
ax.set_ylabel("Monthly arrivals")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "09_sarima_2025.png", dpi=150)
plt.close(fig)