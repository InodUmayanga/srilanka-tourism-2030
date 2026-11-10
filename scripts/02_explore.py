import matplotlib.pyplot as plt
from tourism.data import load_arrivals
from tourism.paths import FIGURES
import pandas as pd

arrivals = load_arrivals()

# ---- Chart 1: normal scale ----
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(arrivals.index, arrivals.values, linewidth=1)
ax.set_title("Monthly tourist arrivals to Sri Lanka, 1971-2025")
ax.set_xlabel("Years")
ax.set_ylabel("Monthly Arrivals")
fig.tight_layout()
fig.savefig(FIGURES / "01_arrivals_linear.png", dpi=150)
plt.close(fig)

# ---- Chart 2: log scale ----
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(arrivals.index, arrivals.values, linewidth=1)
ax.set_yscale("log")
ax.set_title("Monthly tourist arrivals to Sri Lanka, 1971-2025 (log scale)")
ax.set_xlabel("Year")
ax.set_ylabel("Arrivals (log scale)")
fig.tight_layout()
fig.savefig(FIGURES / "02_arrivals_log.png", dpi=150)
plt.close(fig)
print("Charts saved to:", FIGURES)

# ---- Chart 3: seasonal pattern, before vs after COVID ----
def seasonal_profile(series, start, end):
    """Average arrivals per calendar month, as % of the average month in that period."""
    period = series[start:end]
    monthly_avg = period.groupby(period.index.month).mean()
    return monthly_avg / monthly_avg.mean() * 100

before = seasonal_profile(arrivals, "2010", "2018")
after = seasonal_profile(arrivals, "2023", "2025")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(before.index, before.values, marker="o", label="2010-2018 (before shocks)")
ax.plot(after.index, after.values, marker="o", label="2023-2025 (after recovery)")
ax.axhline(100, color="grey", linestyle="--", linewidth=0.8)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
ax.set_title("Seasonal pattern of tourist arrivals")
ax.set_ylabel("% of an average month")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "03_seasonality.png", dpi=150)
plt.close(fig)

print("\nSeasonal profile (% of average month):")
print(pd.DataFrame({"2010-2018": before.round(0), "2023-2025": after.round(0)}))