# Five Million by 2030? Forecasting Sri Lanka's Tourism Demand

## Modeling decisions

### Training window
Models are trained on monthly arrivals from **January 2010 to December 2025**.
Months from **April 2019 to December 2023** are treated as missing.

**Why:**
- **1971–2009 is excluded.** The industry was much smaller and repeatedly disrupted
  by the civil war, so it does not reflect today's market.
- **April 2019 – December 2023 is excluded** because of the Easter Sunday attacks,
  COVID-19 (airport closed April–December 2020) and the 2022 economic crisis.
  In 2023, arrivals were still recovering (43% of the 2018 level in January,
  83% in December), so it is not normal growth either.
- **2024–2025 is kept** because the seasonal pattern changed after COVID
  (December stronger, July–October weaker; see `figures/03_seasonality.png`).
- **2010–2012 is kept, but tested.** Growth slowed from 46% (2010) to about 10% (2018),
  so the early post-war boom may make models too optimistic. Step 11 compares
  training from 2010 against training from 2013.

### Early evidence on the targets
- Reaching **5 million by 2030** needs **16.2% growth every year** from 2025.
  Sri Lanka has only sustained this when rebounding from a crisis
  (2009–2014, 2020–2025) or in the early days of the industry (1970s).
- Reaching **4 million** needs **11.1% a year**, close to the 2013–2018 average (12.9%).
- STL decomposition (2010-01 to 2019-03) shows the underlying trend growth slowed
  from about 23% a year (2011-2013) to 6-8% a year (2016-2018), well below the
  11.1% needed for 4 million.

### Sealed test (January-September 2026)
The final model (4-model ensemble) was fixed before the 2026 data was opened.
- Backtest (6 normal years): MAPE 7.7%
- Sealed 2026 test: MAPE 13.4% (seasonal naive: 11.3%)
- Arrivals fell 20-30% in March-April 2026 when the Middle East war disrupted
  air transit routes. Excluding those two months, the error was about 10%.
- Lesson: in normal years the ensemble beats simple methods, but sudden shocks
  dominate forecast error. The 2030 outlook therefore includes shock scenarios.