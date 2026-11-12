# =====================================================================
# Step 12: Sealed test on January-September 2026
# The final model was chosen by the backtest BEFORE this file was opened.
# =====================================================================

import pandas as pd
import matplotlib.pyplot as plt
from tourism.data import load_training_arrivals
from tourism.metrics import mae, mape
from tourism.models import MODELS
from tourism.paths import DATA_PROCESSED, FIGURES, RESULTS

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
FINAL_MODEL = "Ensemble (HW + SARIMA + Chronos-2 + TimesFM)"    # chosen in Step 11, 2026

# ---- 1. Train on everything up to Dec 2025; open the sealed 2026 file ----
train = load_training_arrivals()                                 # 2010-01 to 2025-12, gap = NaN
actual = pd.read_csv(DATA_PROCESSED / "arrivals_monthly_2026.csv",
                     parse_dates=["date"]).set_index("date")["arrivals"]
horizon = len(actual)                                            # 9 months (Jan-Sep)

# ---- 2. Forecast with every model ----
rows, forecasts = [], {}
for name, model in MODELS.items():
    forecast = model(train, horizon)
    forecasts[name] = forecast
    rows.append({"model": name,
                 "MAPE": mape(actual, forecast),
                 "MAE": mae(actual, forecast),
                 "forecast_total": forecast.sum()})
results = pd.DataFrame(rows).sort_values("MAPE")
results.to_csv(RESULTS / "test_2026.csv", index=False)

# ---- 3. Report ----
actual_total = actual.sum()
same_months_2025 = train["2025-01":"2025-09"].sum()
print(f"Actual Jan-Sep 2026: {actual_total:,.0f}  "
      f"({(actual_total / same_months_2025 - 1) * 100:+.1f}% vs Jan-Sep 2025)")
print("\nAll models on the sealed 2026 test, best first:")
print(results.round(1).to_string(index=False))

final = results.set_index("model").loc[FINAL_MODEL]
print(f"\nFINAL MODEL: MAPE = {final['MAPE']:.1f}%   (backtest average was 7.7%)")
print(f"Forecast total Jan-Sep 2026: {final['forecast_total']:,.0f}")

comparison = pd.DataFrame({"actual": actual, "final_forecast": forecasts[FINAL_MODEL].round()})
comparison["error_pct"] = (comparison["actual"] - comparison["final_forecast"]) / comparison["actual"] * 100
print("\nFinal model, month by month:")
print(comparison.round(1))

# ---- 4. Chart ----
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(actual.index, actual.values, marker="o", linewidth=2, label="Actual 2026")
ax.plot(forecasts[FINAL_MODEL].index, forecasts[FINAL_MODEL].values, marker="o", linestyle="--",
        label="Final model: 4-model ensemble")
ax.plot(forecasts["Seasonal naive"].index, forecasts["Seasonal naive"].values, marker=".", linestyle=":",
        color="grey", label="Seasonal naive (benchmark)")
ax.set_title("Sealed test: forecast vs actual arrivals, January-September 2026")
ax.set_xlabel("Month")
ax.set_ylabel("Monthly arrivals")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "11_test_2026.png", dpi=150)
plt.close(fig)