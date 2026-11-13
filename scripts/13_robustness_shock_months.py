# =====================================================================
# Robustness check: re-forecast 2030 with March-April 2026 treated as disrupted
# The main forecast (Step 13) keeps those months; this script tests how much they matter.
# =====================================================================
import pandas as pd
from tourism.data import load_all_arrivals
from tourism.models import MODELS, chronos2_quantiles
from tourism.paths import RESULTS

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
MEMBERS = ["Holt-Winters damped", "SARIMA", "Chronos-2", "TimesFM 2.5"]
SHOCK = slice("2026-03", "2026-04")                     # the Middle East war months
OCTOBER_ACTUAL = 168625                                 # published after the dataset was built

# ---- 1. Two versions of the history (both end in September 2026) ----
history = load_all_arrivals()

missing = history.copy()                                # version A: the two months are missing (NaN)
missing.loc[SHOCK] = float("nan")

# version B: the two months filled with "what they would have been without the shock":
# the same months of 2025, scaled by how 2026 compared with 2025 in the other months (Jan-Feb, May-Sep)
normal_months = missing["2026-01":"2026-09"].dropna()
ratio = normal_months.sum() / history[normal_months.index - pd.DateOffset(years=1)].sum()
filled = history.copy()
filled.loc[SHOCK] = history["2025-03":"2025-04"].values * ratio
print(f"2026 vs 2025 in the unaffected months: {(ratio - 1) * 100:+.1f}%")
print("Counterfactual March-April 2026 (used by Holt-Winters and TimesFM):")
print(pd.DataFrame({"actual": history[SHOCK], "counterfactual": filled[SHOCK].round()}))

# SARIMA and Chronos-2 can skip missing months; Holt-Winters and TimesFM need a continuous
# series (skipping two months would shift the seasonal cycle), so they get the filled version.
INPUT = {"Holt-Winters damped": filled, "SARIMA": missing, "Chronos-2": missing, "TimesFM 2.5": filled}

# ---- 2. Forecast October 2026 - December 2030 with each member, then average ----
horizon = len(pd.date_range("2026-10-01", "2030-12-01", freq="MS"))
member_forecasts = {name: MODELS[name](INPUT[name], horizon) for name in MEMBERS}
ensemble = sum(member_forecasts.values()) / len(MEMBERS)
band = chronos2_quantiles(missing, horizon)
low = ensemble * band["0.1"] / band["0.5"]
high = ensemble * band["0.9"] / band["0.5"]


def annual_totals(forecast):
    """Real January-September 2026 (shock included: it happened) + forecast months."""
    combined = pd.concat([history["2026-01":"2026-09"], forecast])
    return combined.groupby(combined.index.year).sum()


# ---- 3. Compare with the main forecast (Step 13) ----
main = pd.read_csv(RESULTS / "forecast_2030.csv", index_col=0)
comparison = pd.DataFrame({
    "main forecast (M)": main["forecast"] / 1e6,
    "shock months disrupted (M)": annual_totals(ensemble) / 1e6,
    "low 10% (M)": annual_totals(low) / 1e6,
    "high 90% (M)": annual_totals(high) / 1e6,
})
comparison["difference %"] = (comparison["shock months disrupted (M)"] / comparison["main forecast (M)"] - 1) * 100
comparison.round(3).to_csv(RESULTS / "robustness_shock_months.csv")
print("\nAnnual arrivals: main forecast vs March-April 2026 treated as disrupted")
print(comparison.round(2))

# ---- 4. Which version forecast October 2026 better? (the one month published since) ----
october = ensemble.iloc[0]
print(f"\nOctober 2026: actual {OCTOBER_ACTUAL:,}, this version {october:,.0f} "
      f"(error {(OCTOBER_ACTUAL - october) / OCTOBER_ACTUAL * 100:+.1f}%); main forecast was 162,077 (+3.9%)")

f2030 = comparison.loc[2030, "shock months disrupted (M)"]
growth = ((f2030 * 1e6 / history["2025"].sum()) ** (1 / 5) - 1) * 100
print(f"2030: {f2030:.2f} million (80% range {comparison.loc[2030, 'low 10% (M)']:.2f}-"
      f"{comparison.loc[2030, 'high 90% (M)']:.2f}), implied growth 2025-2030 {growth:.1f}% per year")
print("4 million target:", "inside" if comparison.loc[2030, "high 90% (M)"] >= 4 else "outside", "the 80% range")