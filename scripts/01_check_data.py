from tourism.data import load_arrivals
import pandas as pd

arrivals = load_arrivals()
print("Type:", type(arrivals))
print("Number of months:", len(arrivals))
print("\nFirst 5 rows:")
print(arrivals.head(5))
print("\nLast 5 rows:")
print(arrivals.tail(5))
print("\nMissing values:", arrivals.isna().sum())
print("\nIndex type:", type(arrivals.index))

print("\n--- CHECKS ---")

# 1. Duplicate dates (the same month appearing twice)
n_duplicates = arrivals.index.duplicated().sum()
print("Duplicate dates:", n_duplicates)
assert n_duplicates == 0, "Found duplicate months!"

# 2. Gaps (a month missing from the sequence)
expected_months = pd.date_range(start=arrivals.index.min(), end=arrivals.index.max(), freq="MS")
missing_months = expected_months.difference(arrivals.index)
print("Missing months:", len(missing_months))
assert len(missing_months) == 0, f"Missing months: {list(missing_months)}"

# 3. Negative values (impossible for tourist counts)
n_negative = (arrivals < 0).sum()
print("Negative values:", n_negative)
assert n_negative == 0, "Found negative arrivals!"

# 4. Zero months (should only be the 2020 airport closure)
zero_months = arrivals[arrivals == 0]
print("\nMonths with zero arrivals:")
print(zero_months)

# 5. Yearly totals, to compare against the SLTDA reports
yearly = arrivals.groupby(arrivals.index.year).sum()
print("\nYearly totals (last 8 years):")
print(yearly.tail(8))

print("\nAll checks passed.")