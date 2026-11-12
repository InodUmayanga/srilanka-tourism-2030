# =====================================================================
# Step 13: Forecast to December 2030 with the final ensemble
# =====================================================================
import pandas as pd
import matplotlib.pyplot as plt
from tourism.data import load_all_arrivals
from tourism.models import MODELS, chronos2_quantiles
from tourism.paths import FIGURES, RESULTS

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
MEMBERS = ["Holt-Winters damped", "SARIMA", "Chronos-2", "TimesFM 2.5"]   # the final ensemble

# ---- 1. All data up to September 2026; forecast October 2026 - December 2030 ----
history = load_all_arrivals()
future_months = pd.date_range("2026-10-01", "2030-12-01", freq="MS")
horizon = len(future_months)                                    # 51 months

# ---- 2. Each member's forecast, then the ensemble = their average ----
member_forecasts = {name: MODELS[name](history, horizon) for name in MEMBERS}
ensemble = sum(member_forecasts.values()) / len(MEMBERS)

# ---- 3. 80% range: Chronos-2's spread, applied around the ensemble ----
band = chronos2_quantiles(history, horizon)
low = ensemble * band["0.1"] / band["0.5"]
high = ensemble * band["0.9"] / band["0.5"]


# ---- 4. Annual totals: actual Jan-Sep 2026 + forecast for the remaining months ----
def annual_totals(forecast):
    combined = pd.concat([history["2026-01":"2026-09"], forecast])
    return combined.groupby(combined.index.year).sum()


annual = pd.DataFrame({"low (10%)": annual_totals(low),
                       "forecast": annual_totals(ensemble),
                       "high (90%)": annual_totals(high)})
for name, forecast in member_forecasts.items():                 # each member, to show agreement
    annual[name] = annual_totals(forecast)
annual.to_csv(RESULTS / "forecast_2030.csv")
print("Annual arrivals forecast (millions):")
print((annual / 1e6).round(2))

# ---- 5. The 2030 answer ----
base_2025 = history["2025"].sum()
f2030 = annual.loc[2030, "forecast"]
growth = ((f2030 / base_2025) ** (1 / 5) - 1) * 100
print(f"\n2030 forecast: {f2030 / 1e6:.2f} million  "
      f"(80% range {annual.loc[2030, 'low (10%)'] / 1e6:.2f}-{annual.loc[2030, 'high (90%)'] / 1e6:.2f} million)")
print(f"Implied growth 2025-2030: {growth:.1f}% per year  (4M target needs 11.1%, 5M needs 16.2%)")
for target in (4_000_000, 5_000_000):
    print(f"  {target / 1e6:.0f} million target: "
          f"{'reached' if f2030 >= target else 'NOT reached'} in the base forecast; "
          f"{'inside' if annual.loc[2030, 'high (90%)'] >= target else 'outside'} the 80% range")

# ---- 6. Charts: monthly forecast with range, and annual totals vs targets ----
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))
recent = history["2015-01":].dropna()
ax1.plot(recent.index, recent.values, linewidth=1, label="Actual (disrupted years left out)")
ax1.plot(ensemble.index, ensemble.values, linestyle="--", label="Ensemble forecast")
ax1.fill_between(ensemble.index, low, high, alpha=0.25, label="80% range (normal conditions)")
ax1.set_title("Monthly tourist arrivals: history and forecast to 2030")
ax1.set_ylabel("Monthly arrivals")
ax1.legend()

years = [2025] + list(annual.index)
values = [base_2025] + list(annual["forecast"])
ax2.bar([str(y) for y in years], [v / 1e6 for v in values], label="Actual 2025 / forecast")
errors = [[0] + list((annual["forecast"] - annual["low (10%)"]) / 1e6),
          [0] + list((annual["high (90%)"] - annual["forecast"]) / 1e6)]
ax2.errorbar([str(y) for y in years], [v / 1e6 for v in values], yerr=errors, fmt="none", color="black", capsize=4)
ax2.axhline(4, color="orange", linestyle=":", label="4 million target")
ax2.axhline(5, color="red", linestyle="--", label="5 million target")
ax2.set_title("Annual arrivals vs the 2030 targets")
ax2.set_ylabel("Arrivals (millions)")
ax2.legend()
fig.tight_layout()
fig.savefig(FIGURES / "12_forecast_2030.png", dpi=150)
plt.close(fig)