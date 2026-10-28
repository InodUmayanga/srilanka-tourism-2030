"""Build the annual datasets from SLTDA reports.

Outputs (data/processed/):
  stay_spend_annual.csv  arrivals, tourist nights, receipts (US$), spend per day, average stay; 1985-2025
  capacity_annual.csv    graded rooms, graded occupancy, total registered rooms; 1985-2025

Sources (data/raw/):
  annual_report_2023.pdf  Table 14 "Tourism Growth Trends 1985 to 2023", pp.81-82  -> 1985-2023
  annual_report_2018.pdf  Table 20 (tourist hotel + supplementary rooms), p.91      -> total rooms 2007-2018
  annual_report_2020.pdf  narrative p.~29 (2019 supplementary rooms; 2020 total)    -> hard-coded below
  year_in_review_2021/2022.pdf  total registered rooms                               -> hard-coded below
  year_in_review_2024.pdf  highlights p.2, Table 06, Table 20                         -> 2024 figures
  year_in_review_2025.pdf  highlights p.3, Table 09, Table 10                         -> 2025 figures
Run from the project root:  python data/extraction/build_annual.py
"""
import csv, re, subprocess
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "raw"
OUT = Path(__file__).resolve().parent.parent / "processed"

def text(pdf, first, last):
    return subprocess.run(["pdftotext", "-layout", "-f", str(first), "-l", str(last), str(RAW / pdf), "-"],
                          capture_output=True, text=True).stdout

def clean_tokens(line):
    toks = [t.replace("*", "") for t in line.split()]
    return [t for t in toks if t and re.fullmatch(r"[\d,]+(\.\d+)?", t)]

def f(t):
    return float(t.replace(",", ""))

# ---- 1. ASR 2023 Table 14 ----------------------------------------------------
FULL = ["arrivals", "excursionists", "tourist_nights_000", "receipts_lkr_mn", "receipts_usd_mn", "receipts_eur_mn",
        "receipts_sdr_mn", "receipt_per_day_usd", "avg_stay_nights", "graded_rooms", "graded_beds",
        "graded_occupancy_pct", "sl_arrivals", "sl_departures", "employment_direct", "employment_indirect"]
t14, flagged = {}, {}
for line in text("annual_report_2023.pdf", 81, 82).splitlines():
    m = re.match(r"^\s*(19[89]\d|20[0-2]\d)\s", line)
    if not m:
        continue
    year = int(m.group(1)); nums = clean_tokens(line)[1:]
    cols = FULL if len(nums) == 16 else [c for c in FULL if c != "receipts_eur_mn"] if len(nums) == 15 else None
    if cols is None:
        raise ValueError(f"Table 14 row {year}: expected 15 or 16 numbers, got {len(nums)}: {line}")
    t14[year] = dict(zip(cols, map(f, nums)))
    flagged[year] = "*" in line
assert sorted(t14) == list(range(1985, 2024)), sorted(t14)

# ---- 2. ASR 2018 Table 20: tourist hotel + supplementary rooms 2007-2018 -----
total_rooms = {}
t20 = text("annual_report_2018.pdf", 91, 92)
hotel = supp = None
for line in t20.splitlines():
    if "(b). No. of Rooms" in line:
        vals = [int(f(v)) for v in clean_tokens(line)]
        if hotel is None: hotel = vals
        else: supp = vals; break
assert hotel and supp and len(hotel) == len(supp) == 12, (hotel, supp)
for i, y in enumerate(range(2007, 2019)):
    total_rooms[y] = (hotel[i] + supp[i], "ASR2018 T20 (tourist hotels + supplementary)")

# ---- 3. Hard-coded values from narrative text / later reports ----------------
total_rooms.update({
    2019: (24831 + 15534, "ASR2023 T14 graded rooms + ASR2020 p.29 supplementary rooms"),
    2020: (42750, "ASR2020 p.29; YIR2021 T06"),
    2021: (47337, "YIR2021 T06"),
    2022: (48120, "YIR2022 T07"),
    2023: (53229, "ASR2023 T.VIII; YIR2024 T07"),
    2024: (53378, "YIR2025 T10 (revised; YIR2024 T07 gave 55,455)"),
    2025: (59578, "YIR2025 T10"),
})
# Graded = boutique hotel + boutique villa + classified + unclassified tourist hotel (ASR definition;
# reproduces ASR2023 T14 2023 value 26,422 from YIR2024 T07: 865+356+16,686+8,515).
graded_2024 = 835 + 345 + 16808 + 8362      # YIR2025 T10, 2024 column (revised)
graded_2025 = 877 + 353 + 18308 + 8510      # YIR2025 T10, 2025 column
assert 865 + 356 + 16686 + 8515 == int(t14[2023]["graded_rooms"])

recent = {
    2024: dict(arrivals=2053465, receipts_usd_mn=3168.601, receipt_per_day_usd=181.15, avg_stay_nights=8.42,
               graded_rooms=graded_2024, graded_occupancy_pct=44.5,
               src="YIR2024 p.2 highlights, T06 income, T20 occupancy (221 responding graded hotels); rooms YIR2025 T10",
               note="provisional"),
    2025: dict(arrivals=2362521, receipts_usd_mn=3219.24, receipt_per_day_usd=None, avg_stay_nights=8.29,
               graded_rooms=graded_2025, graded_occupancy_pct=None,
               src="YIR2025 p.3 highlights, T09 income, T10 rooms",
               note="provisional; spend/day published as 171.74 (Jan-Jul) and 148.26 (Aug-Dec), no annual value; no occupancy published"),
}

# ---- 4. Write files ----------------------------------------------------------
with open(OUT / "stay_spend_annual.csv", "w", newline="") as fh:
    w = csv.writer(fh, lineterminator="\n")
    w.writerow(["year", "arrivals", "tourist_nights_000", "receipts_usd_mn", "receipt_per_day_usd",
                "avg_stay_nights", "source", "notes"])
    for y in range(1985, 2024):
        r = t14[y]
        w.writerow([y, int(r["arrivals"]), r["tourist_nights_000"], r["receipts_usd_mn"], r["receipt_per_day_usd"],
                    r["avg_stay_nights"], "ASR2023 T14", "some values provisional" if flagged[y] else ""])
    for y, r in recent.items():
        w.writerow([y, r["arrivals"], "", r["receipts_usd_mn"], "" if r["receipt_per_day_usd"] is None else r["receipt_per_day_usd"],
                    r["avg_stay_nights"], r["src"], r["note"]])

with open(OUT / "capacity_annual.csv", "w", newline="") as fh:
    w = csv.writer(fh, lineterminator="\n")
    w.writerow(["year", "graded_rooms", "graded_occupancy_pct", "total_registered_rooms", "source", "total_rooms_source"])
    for y in range(1985, 2026):
        if y <= 2023:
            g, occ, src = int(t14[y]["graded_rooms"]), t14[y]["graded_occupancy_pct"], "ASR2023 T14"
        else:
            g, occ, src = recent[y]["graded_rooms"], recent[y]["graded_occupancy_pct"], recent[y]["src"]
        tot, tsrc = total_rooms.get(y, ("", ""))
        w.writerow([y, g, "" if occ is None else occ, tot, src, tsrc])

# ---- 5. Checks ---------------------------------------------------------------
notes = []
monthly = {}
with open(OUT / "arrivals_monthly_1971_2025.csv") as fh:
    for row in csv.DictReader(fh):
        monthly[int(row["year"])] = monthly.get(int(row["year"]), 0) + int(row["arrivals"])
for y in range(1985, 2024):
    if int(t14[y]["arrivals"]) != monthly[y]:
        notes.append(f"{y}: Table 14 arrivals {int(t14[y]['arrivals']):,} vs monthly sum {monthly[y]:,}")
for y in (2024, 2025):
    if recent[y]["arrivals"] != monthly[y]:
        notes.append(f"{y}: annual arrivals {recent[y]['arrivals']:,} vs monthly sum {monthly[y]:,}")
# receipts ~= arrivals x stay x spend/day (how SLTDA computes it)
for y in list(range(1985, 2024)) + [2024]:
    r = t14.get(y) or recent[y]
    if r["receipt_per_day_usd"]:
        implied = r["arrivals"] * r["avg_stay_nights"] * r["receipt_per_day_usd"] / 1e6
        gap = implied / r["receipts_usd_mn"] - 1
        if abs(gap) > 0.05:
            notes.append(f"{y}: receipts {r['receipts_usd_mn']:,.1f} vs arrivals*stay*spend {implied:,.1f} ({gap:+.0%})")
report = ["stay_spend_annual.csv and capacity_annual.csv: 1985-2025 (41 years)",
          "Checks: Table 14 arrivals vs monthly series; receipts vs arrivals*stay*spend (flag if >5% apart)"]
report += ["Notes:"] + ["  - " + n for n in notes] if notes else ["All checks passed."]
(OUT / "validation_report_annual.txt").write_text("\n".join(report) + "\n")
print("\n".join(report))
