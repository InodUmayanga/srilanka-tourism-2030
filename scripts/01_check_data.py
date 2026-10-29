import pandas as pd
from tourism.data import load_arrivals, load_annual

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


print("\n--- ANNUAL FILES ---")
capacity = load_annual("capacity_annual.csv")
spend = load_annual("stay_spend_annual.csv")

for name, df in [("capacity", capacity), ("stay_spend", spend)]:
    print(f"\n{name}: years {df.index.min()}-{df.index.max()}, {len(df)} rows")
    print("Missing values per column:")
    print(df.isna().sum())

# Cross-check: annual arrivals in stay_spend vs. totals from the monthly file
compare = pd.DataFrame({"annual_file": spend["arrivals"], "monthly_sum": yearly})
compare = compare.dropna()
compare["difference"] = compare["annual_file"] - compare["monthly_sum"]
print("\nYears where the two sources disagree:")
print(compare[compare["difference"] != 0])
assert list(compare[compare["difference"] != 0].index) == [2020], "Unexpected mismatch between sources"
print("\nAll checks passed.")
