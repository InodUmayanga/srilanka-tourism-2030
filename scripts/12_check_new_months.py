# =====================================================================
# Check the forecasts against newly published months (September and October 2026)
# =====================================================================
import pandas as pd
from tourism.data import load_training_arrivals, load_all_arrivals
from tourism.models import MODELS
from tourism.paths import DATA_PROCESSED, RESULTS

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

# October 2026 was published after the dataset was built (SLTDA, reported 3 Nov 2026)
NEW_ACTUAL = pd.Series({pd.Timestamp("2026-09-01"): 158557, pd.Timestamp("2026-10-01"): 168625})
MEMBERS = ["Holt-Winters damped", "SARIMA", "Chronos-2", "TimesFM 2.5"]
SINGLE = MEMBERS + ["Seasonal naive"]


def forecasts_from(train, horizon):
    """Each single model's forecast, plus the 4-model ensemble (average of the members)."""
    out = {name: MODELS[name](train, horizon) for name in SINGLE}
    out["Ensemble (final model)"] = sum(out[name] for name in MEMBERS) / len(MEMBERS)
    return pd.DataFrame(out)


# A. Trained to December 2025 (as in Step 12): forecast January-October 2026, keep Sep and Oct
step12 = forecasts_from(load_training_arrivals(), 10).loc[NEW_ACTUAL.index]

# B. Trained to September 2026 (as in Step 13): forecast October 2026 one month ahead
step13 = forecasts_from(load_all_arrivals(), 1)

rows = []
for label, table in [("trained to Dec 2025", step12), ("trained to Sep 2026", step13)]:
    for month, forecasts in table.iterrows():
        actual = NEW_ACTUAL[month]
        for model, value in forecasts.items():
            rows.append({"month": month.strftime("%b %Y"), "trained": label, "model": model,
                         "actual": actual, "forecast": round(value),
                         "error_pct": (actual - value) / actual * 100})
results = pd.DataFrame(rows)
results.to_csv(RESULTS / "check_sep_oct_2026.csv", index=False)

final = results[results["model"] == "Ensemble (final model)"]
print("FINAL MODEL vs actual (positive error = forecast too low):")
print(final[["month", "trained", "actual", "forecast", "error_pct"]].round(1).to_string(index=False))
print("\nAll models:")
print(results.pivot_table(index="model", columns=["month", "trained"], values="error_pct").round(1))

# 2026 so far: the sealed-test months (Jan-Sep) plus October, all forecast from Dec 2025
sealed = pd.read_csv(DATA_PROCESSED / "arrivals_monthly_2026.csv", parse_dates=["date"]).set_index("date")["arrivals"]
actual_2026 = pd.concat([sealed, NEW_ACTUAL[["2026-10-01"]]])
forecast_2026 = forecasts_from(load_training_arrivals(), 10)["Ensemble (final model)"]
mape_2026 = ((actual_2026 - forecast_2026).abs() / actual_2026).mean() * 100
print(f"\nEnsemble MAPE, January-October 2026: {mape_2026:.1f}%  (Jan-Sep was 13.4%)")
print(f"Total: actual {actual_2026.sum():,.0f} vs forecast {forecast_2026.sum():,.0f} "
      f"({(forecast_2026.sum() / actual_2026.sum() - 1) * 100:+.1f}%)")

# saved for the robustness figure (script 14)
pd.DataFrame({"actual": actual_2026, "forecast": forecast_2026.round()}).rename_axis("date").to_csv(RESULTS / "forecast_vs_actual_2026.csv")
