# =====================================================================
# Steps 10-11: Rolling backtest
# Idea: repeat the "train on the past, forecast the next year" test for several years
# =====================================================================
import pandas as pd
import matplotlib.pyplot as plt
from tourism.data import load_training_arrivals
from tourism.metrics import mape
from tourism.models import MODELS
from tourism.paths import FIGURES, RESULTS


def backtest(start, years):
    """For each year: train on data from `start` up to the December before, forecast that year."""
    data = load_training_arrivals(start=start)
    rows = []
    for year in years:
        train = data[:f"{year - 1}-12"]               # everything before the test year
        actual = data[f"{year}-01":f"{year}-12"]        # the 12 real months of the test year
        for name, model in MODELS.items():              # run every model the same way
            forecast = model(train, 12)                # forecast 12 months ahead
            rows.append({"model": name, "year": year,
                         "MAPE": mape(actual, forecast),
                         # growth the model predicted vs the previous 12 months
                         "forecast_growth_pct": (forecast.sum() / train.iloc[-12:].sum() - 1) * 100,
                         # growth that really happened
                         "actual_growth_pct": (actual.sum() / train.iloc[-12:].sum() - 1) * 100})
    return pd.DataFrame(rows)


# ---- 1. Main backtest: six normal years ----
YEARS = [2014, 2015, 2016, 2017, 2018, 2025]
results = backtest("2010-01", YEARS)
results.to_csv(RESULTS / "backtest.csv", index=False)

# table: one row per model, one column per year, plus the average
table = results.pivot(index="model", columns="year", values="MAPE")
table["average"] = table.mean(axis=1)
table = table.sort_values("average")
print("MAPE (%) by model and year, best average first:")
print(table.round(1))

growth = results.pivot(index="model", columns="year", values="forecast_growth_pct")
growth.loc["ACTUAL"] = results.groupby("year")["actual_growth_pct"].first()
print("\nGrowth each model forecast for the year (%), compared with ACTUAL:")
print(growth.round(1))

# ---- 2. Sensitivity check: does starting the training data in 2013 change the result? ----
late = backtest("2013-01", [2017, 2018, 2025])
early = results[results["year"].isin([2017, 2018, 2025])]
check = pd.DataFrame({"train from 2010": early.groupby("model")["MAPE"].mean(),
                      "train from 2013": late.groupby("model")["MAPE"].mean()})
print("\nAverage MAPE (%) on 2017, 2018, 2025 by training start:")
print(check.sort_values("train from 2010").round(1))

# ---- 3. Chart ----
fig, ax = plt.subplots(figsize=(10, 5))
for name in table.index:
    ax.plot([str(y) for y in YEARS], table.loc[name, YEARS], marker="o", label=name)
ax.set_title("Backtest: forecast error by model and year")
ax.set_xlabel("Year forecast")
ax.set_ylabel("MAPE (%)")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "10_backtest.png", dpi=150)
plt.close(fig)