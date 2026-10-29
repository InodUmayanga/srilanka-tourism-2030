import pandas as pd
from tourism.paths import DATA_PROCESSED

def load_arrivals():
    """Load monthly tourist arrivals 1971-2025 as a pandas Series indexed by date."""
    df = pd.read_csv(DATA_PROCESSED / "arrivals_monthly_1971_2025.csv", parse_dates=["date"])
    df = df.set_index("date")
    return df["arrivals"]          # return only the arrivals column (a pandas Series)