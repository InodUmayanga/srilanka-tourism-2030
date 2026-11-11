import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from tourism.data import load_arrivals
from tourism.paths import FIGURES

arrivals = load_arrivals()

# longest clean stretch: January 2010 to March 2019
clean = arrivals["2010-01":"2019-03"]
log_arrivals = np.log(clean)

# split into trend + seasonal + residual (12 = months in one seasonal cycle)
result = STL(log_arrivals, period=12).fit()

fig = result.plot()
fig.set_size_inches(12, 8)
fig.tight_layout()
fig.savefig(FIGURES / "05_decomposition.png", dpi=150)
plt.close(fig)

# 1. Seasonal effect of each month, in % above or below the trend
seasonal_pct = (np.exp(result.seasonal) - 1) * 100
print("Seasonal effect by month (% above/below trend):")
print(seasonal_pct.groupby(seasonal_pct.index.month).mean().round(1))

# 2. Trend growth: % change of the trend over the previous 12 months, shown each December
trend_growth = (np.exp(result.trend.diff(12)) - 1) * 100
print("\nTrend growth per year:")
print(trend_growth[trend_growth.index.month == 12].round(1))

# 3. How big is the random noise, on average?
noise_pct = (np.exp(result.resid.abs()) - 1) * 100
print(f"\nAverage size of the noise: {noise_pct.mean():.1f}%")