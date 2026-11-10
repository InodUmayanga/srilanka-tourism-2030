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

# ---- Chart 4: yearly totals, growth, and the 2030 target ----
yearly = arrivals.groupby(arrivals.index.year).sum()
growth = yearly.pct_change() * 100            # % change from the previous year

# growth needed per year from 2025 to reach each target in 2030
need_5m = ((5_000_000 / yearly[2025]) ** (1 / 5) - 1) * 100
need_4m = ((4_000_000 / yearly[2025]) ** (1 / 5) - 1) * 100
print(f"\nGrowth needed per year, 2025-2030: 5M target = {need_5m:.1f}%, 4M target = {need_4m:.1f}%")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax1.bar(yearly.index, yearly.values / 1_000_000)
ax1.set_ylabel("Arrivals (millions)")
ax1.set_title("Yearly tourist arrivals to Sri Lanka and year-on-year growth")

ax2.bar(growth.index, growth.values)
ax2.axhline(0, color="black", linewidth=0.8)
ax2.axhline(need_5m, color="red", linestyle="--", label=f"Needed for 5M by 2030 ({need_5m:.0f}%/yr)")
ax2.axhline(need_4m, color="orange", linestyle=":", label=f"Needed for 4M by 2030 ({need_4m:.0f}%/yr)")
ax2.set_ylim(-80, 100)                 # cuts off the extreme COVID bars so the rest is readable
ax2.set_ylabel("Growth vs previous year (%)")
ax2.set_xlabel("Year")
ax2.legend()
# label bars that are cut off by the y-axis limit
for year, value in growth.items():
    if value > 100:
        ax2.text(year, 95, f"{value:.0f}%", ha="center", va="top",
                 fontsize=8, color="white", rotation=90)

fig.tight_layout()
fig.savefig(FIGURES / "04_yearly_growth.png", dpi=150)
plt.close(fig)

# Has Sri Lanka ever kept up the needed growth for 5 years in a row?
growth_5yr = ((yearly / yearly.shift(5)) ** (1 / 5) - 1) * 100
print("\nAverage growth per year over the previous 5 years (best 10 periods):")
print(growth_5yr.sort_values(ascending=False).head(10).round(1))