# =====================================================================
# Step 14: Scenarios for 2030, including the risk of a new shock
# =====================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tourism.data import load_arrivals
from tourism.paths import FIGURES, RESULTS

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
YEARS = [2026, 2027, 2028, 2029, 2030]

# ---- 1. Starting point: the Step 13 forecast ----
forecast = pd.read_csv(RESULTS / "forecast_2030.csv", index_col=0)
base_2026 = forecast.loc[2026, "forecast"]          # 9 actual months + 3 forecast months
actual_2025 = load_arrivals()["2025"].sum()


def grow(start, rate):
    """Annual totals 2026-2030 if arrivals grow at `rate` per year from the 2026 level."""
    return pd.Series([start * (1 + rate) ** i for i in range(len(YEARS))], index=YEARS)


def rate_needed(target):
    """Constant yearly growth needed to go from the 2026 level to `target` in 2030."""
    return (target / base_2026) ** (1 / 4) - 1


# ---- 2. Four simple scenarios (no shocks) ----
scenarios = pd.DataFrame({
    "1. Momentum (ensemble forecast)": forecast.loc[YEARS, "forecast"],
    "2. Pre-COVID trend (7%/yr)": grow(base_2026, 0.07),
    f"3. Path to 4M ({rate_needed(4e6) * 100:.1f}%/yr)": grow(base_2026, rate_needed(4e6)),
    f"4. Path to 5M ({rate_needed(5e6) * 100:.1f}%/yr)": grow(base_2026, rate_needed(5e6)),
})
print("Annual arrivals under each scenario (millions):")
print((scenarios / 1e6).round(2))

# ---- 3. How often have shocks happened? (1971-2025) ----
yearly = load_arrivals().groupby(lambda d: d.year).sum()
growth = yearly.pct_change().dropna()
shock_drops = growth[growth < -0.10]                 # a "shock year" = arrivals fell more than 10%
p_shock = len(shock_drops) / len(growth)
print(f"\nShock years since 1972: {len(shock_drops)} of {len(growth)} "
      f"({p_shock * 100:.0f}% chance per year)")
print("Their drops (%):", (shock_drops * 100).round(0).astype(int).to_dict())


# ---- 4. Monte Carlo: 10,000 possible futures with random shocks ----
def simulate(path, n=10_000, recovery=0.5, seed=42):
    """Each year 2027-2030: with probability p_shock, a shock knocks the level down by a drop
    drawn from history. The next year recovers `recovery` (50%) of the lost ground."""
    rng = np.random.default_rng(seed)
    results = np.empty(n)
    for i in range(n):
        factor = 1.0                                 # 1.0 = on the scenario path; 0.8 = 20% below it
        for _ in range(4):                           # 2027, 2028, 2029, 2030
            factor += (1 - factor) * recovery        # partly recover from earlier shocks
            if rng.random() < p_shock:
                factor *= 1 + rng.choice(shock_drops.values)
        results[i] = path[2030] * factor
    return results


rows = []
sims = {}
for name in ["1. Momentum (ensemble forecast)", "2. Pre-COVID trend (7%/yr)"]:
    sim = simulate(scenarios[name])
    sims[name] = sim
    rows.append({"scenario": name + " + shocks",
                 "median 2030 (M)": np.median(sim) / 1e6,
                 "10th pct (M)": np.percentile(sim, 10) / 1e6,
                 "90th pct (M)": np.percentile(sim, 90) / 1e6,
                 "P(>= 4M) %": (sim >= 4e6).mean() * 100,
                 "P(>= 5M) %": (sim >= 5e6).mean() * 100,
                 "P(below 2025) %": (sim < actual_2025).mean() * 100})
summary = pd.DataFrame(rows)
summary.to_csv(RESULTS / "scenarios_2030.csv", index=False)
print("\n2030 with shock risk (10,000 simulations each):")
print(summary.round(1).to_string(index=False))

# ---- 5. Charts ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
for name in scenarios.columns:
    ax1.plot([2025] + YEARS, [actual_2025 / 1e6] + list(scenarios[name] / 1e6), marker="o", label=name)
ax1.axhline(4, color="orange", linestyle=":")
ax1.axhline(5, color="red", linestyle="--")
ax1.set_title("Scenarios for annual arrivals (no shocks)")
ax1.set_ylabel("Arrivals (millions)")
ax1.legend(fontsize=8)

for name, sim in sims.items():
    ax2.hist(sim / 1e6, bins=60, alpha=0.5, label=name + " + shocks")
ax2.axvline(4, color="orange", linestyle=":", label="4 million target")
ax2.axvline(5, color="red", linestyle="--", label="5 million target")
ax2.set_title("2030 arrivals in 10,000 simulated futures")
ax2.set_xlabel("Arrivals in 2030 (millions)")
ax2.set_ylabel("Number of simulations")
ax2.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGURES / "13_scenarios_2030.png", dpi=150)
plt.close(fig)