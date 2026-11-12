import numpy as np
import pandas as pd
from tourism.paths import DATA_PROCESSED

def load_arrivals():
    """Load monthly tourist arrivals 1971-2025 as a pandas Series indexed by date."""
    df = pd.read_csv(DATA_PROCESSED / "arrivals_monthly_1971_2025.csv", parse_dates=["date"])
    df = df.set_index("date")
    return df["arrivals"]          # return only the arrivals column (a pandas Series)

def load_annual(filename):
    """Load an annual CSV (e.g. 'capacity_annual.csv') indexed by year."""
    df = pd.read_csv(DATA_PROCESSED / filename)
    df = df.set_index("year")
    return df

# Months disrupted by the Easter attacks, COVID-19 and the economic crisis (see README)
DISRUPTED_START = "2019-04"
DISRUPTED_END = "2023-12"


def load_training_arrivals(start="2010-01", end="2025-12"):
    """Monthly arrivals for model training, with disrupted months set to NaN (missing)."""
    series = load_arrivals()[start:end].astype(float)
    series.loc[DISRUPTED_START:DISRUPTED_END] = np.nan
    return series

def load_all_arrivals():
    """Training series plus the 2026 months. Only for use after the sealed test (Step 13+)."""
    recent = pd.read_csv(DATA_PROCESSED / "arrivals_monthly_2026.csv",
                         parse_dates=["date"]).set_index("date")["arrivals"].astype(float)
    return pd.concat([load_training_arrivals(), recent])