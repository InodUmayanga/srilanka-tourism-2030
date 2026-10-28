"""Build a monthly Sri Lanka tourist-arrivals series, 1971-2025, from SLTDA reports.

Sources (all in data/raw/):
  annual_report_2020.pdf  Table 28, pp.101-102  -> 1971-2020
  annual_report_2023.pdf  Table 15, pp.83-84    -> 2021-2023 (and overlap 2010-2020 for checking)
  year_in_review_2025.pdf Table 01, p.5         -> 2024-2025
Requires: pdftotext (poppler). Pure Python otherwise.
Run from the project root:  python data/extraction/extract_arrivals.py
"""
import csv, re, subprocess, sys
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "raw"
OUT = Path(__file__).resolve().parent.parent / "processed"
MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def text(pdf, first, last):
    return subprocess.run(["pdftotext","-layout","-f",str(first),"-l",str(last),str(RAW/pdf),"-"],
                          capture_output=True, text=True).stdout

def num(tok):
    return int(tok.replace(",", ""))

def year_table(pdf, first, last):
    """Rows of form: YEAR m1..m12 TOTAL"""
    rows = {}
    for line in text(pdf, first, last).splitlines():
        toks = line.split()
        if len(toks) == 14 and re.fullmatch(r"(19|20)\d\d", toks[0]) and all(re.fullmatch(r"[\d,]+", t) for t in toks[1:]):
            rows[int(toks[0])] = ([num(t) for t in toks[1:13]], num(toks[13]))
    return rows

problems = []
asr20 = year_table("annual_report_2020.pdf", 101, 102)
asr23 = year_table("annual_report_2023.pdf", 83, 84)

# Year in Review 2025, Table 01: "January 252,761 208,253 238,924 ..." -> 2025, 2024, 2018
yir = {2025: [None]*12, 2024: [None]*12}
yir_tot = {}
full = ["January","February","March","April","May","June","July","August","September","October","November","December"]
for line in text("year_in_review_2025.pdf", 5, 5).splitlines():
    toks = line.split()
    if toks and toks[0] in full and len(toks) >= 4:
        i = full.index(toks[0]); yir[2025][i] = num(toks[1]); yir[2024][i] = num(toks[2])
    if toks and toks[0] == "Total" and len(toks) >= 3:
        yir_tot = {2025: num(toks[1]), 2024: num(toks[2])}

series, totals = {}, {}
for y in range(1971, 2021):
    if y not in asr20: problems.append(f"{y}: missing in ASR2020"); continue
    series[y], totals[y] = asr20[y][0], asr20[y][1]; series[y] = (series[y], "ASR2020 T28")
for y in (2021, 2022, 2023):
    if y not in asr23: problems.append(f"{y}: missing in ASR2023"); continue
    series[y], totals[y] = (asr23[y][0], "ASR2023 T15"), asr23[y][1]
for y in (2024, 2025):
    series[y], totals[y] = (yir[y], "YIR2025 T01"), yir_tot.get(y)

# Check 1: months sum to published annual total
for y, (m, src) in sorted(series.items()):
    if None in m: problems.append(f"{y}: incomplete months"); continue
    if totals[y] is not None and sum(m) != totals[y]:
        problems.append(f"{y}: month sum {sum(m):,} != published total {totals[y]:,} (diff {sum(m)-totals[y]:+,}) [{src}]")
# Check 2: overlap between reports
for y in sorted(set(asr20) & set(asr23)):
    if asr20[y][0] != asr23[y][0]:
        diff = [MONTHS[i] for i in range(12) if asr20[y][0][i] != asr23[y][0][i]]
        problems.append(f"{y}: ASR2020 vs ASR2023 months differ in {diff}")

with open(OUT/"arrivals_monthly_1971_2025.csv", "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n"); w.writerow(["date","year","month","arrivals","source"])
    for y, (m, src) in sorted(series.items()):
        for i, v in enumerate(m):
            w.writerow([f"{y}-{i+1:02d}-01", y, i+1, v, src])

n = sum(len(m) for m, _ in series.values())
report = [f"Years: {min(series)}-{max(series)}  ({len(series)} years, {n} months)",
          f"Overlap years checked (ASR2020 vs ASR2023): {sorted(set(asr20)&set(asr23))}",
          "Validation notes:" if problems else "All checks passed."] + ["  - "+p for p in problems]
(OUT/"validation_report.txt").write_text("\n".join(report)+"\n")
print("\n".join(report))
