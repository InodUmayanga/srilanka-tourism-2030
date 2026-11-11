import pandas as pd
import matplotlib.pyplot as plt
from tourism.data import load_training_arrivals
from tourism.metrics import mae, rmse, mape
from tourism.paths import FIGURES, RESULTS

train_all = load_training_arrivals()

# ---- Split: train on everything up to Dec 2024, validate on 2025 ----
train = train_all[:"2024-12"]
validation = train_all["2025-01":"2025-12"]

# ---- Seasonal naive: each month of 2025 = the same month of 2024 ----
forecast = train["2024-01":"2024-12"].copy()
forecast.index = forecast.index + pd.DateOffset(years=1)

# ---- Score it ----
scores = {
    "model": "Seasonal naive",
    "MAE": mae(validation, forecast),
    "RMSE": rmse(validation, forecast),
    "MAPE": mape(validation, forecast),
}
print("\nSeasonal naive, validation year 2025:")
print(f"MAE  = {scores['MAE']:,.0f} tourists")
print(f"RMSE = {scores['RMSE']:,.0f} tourists")
print(f"MAPE = {scores['MAPE']:.1f}%")

# month-by-month comparison
comparison = pd.DataFrame({"actual": validation, "forecast": forecast})
comparison["error_pct"] = (comparison["actual"] - comparison["forecast"]) / comparison["actual"] * 100
print("\nMonth by month:")
print(comparison.round(1))

# save the scores (later models will be added to this file)
pd.DataFrame([scores]).to_csv(RESULTS / "errors_2025.csv", index=False)

# ---- Chart ----
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(validation.index, validation.values, marker="o", label="Actual 2025")
ax.plot(forecast.index, forecast.values, marker="o", linestyle="--", label="Seasonal naive forecast")
ax.set_title("Seasonal naive forecast vs actual arrivals, 2025")
ax.set_xlabel("Month")
ax.set_ylabel("Monthly arrivals")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "06_baseline_2025.png", dpi=150)
plt.close(fig)