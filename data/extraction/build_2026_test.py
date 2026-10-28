"""SEALED TEST SET: monthly tourist arrivals Jan-Sep 2026. Do not open the output before Step 12.

Source: SLTDA "Weekly Report 1st to 30th September 2026", summary table (columns 2018 / 2025 / 2026)
https://www.sltda.gov.lk/storage/common_media/Weekly_Report_1st_to_30th_September_2026.pdf
(The SLTDA site blocks scripted downloads, so values were transcribed from the report.)
Checks: the report's 2018 and 2025 columns must equal our 1971-2025 series exactly; 2026 months must sum to the
report's Jan-Sep total; January cross-checked against the January 2026 weekly report.
October 2026 is excluded because only 1-25 October was published at build time.
"""
import csv
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "processed"
REPORT_2018 = [238924, 235618, 233382, 180429, 129466, 146828, 217829, 200359, 149087, 153123, 195582, 253169]
REPORT_2025 = [252761, 240217, 229298, 174608, 132919, 138241, 200244, 198235, 158971, 165193, 212906, 258928]
ARR_2026 = [277327, 279328, 183979, 135643, 145745, 124551, 196845, 191704, 158557]
REPORT_TOTAL_2026 = 1693679
JAN_2026_FROM_JAN_REPORT = 277327

hist = {}
with open(OUT / "arrivals_monthly_1971_2025.csv") as fh:
    for r in csv.DictReader(fh):
        hist[(int(r["year"]), int(r["month"]))] = int(r["arrivals"])
assert [hist[(2018, m)] for m in range(1, 13)] == REPORT_2018, "2018 column mismatch"
assert [hist[(2025, m)] for m in range(1, 13)] == REPORT_2025, "2025 column mismatch"
assert sum(ARR_2026) == REPORT_TOTAL_2026, "2026 months do not sum to report total"
assert ARR_2026[0] == JAN_2026_FROM_JAN_REPORT

with open(OUT / "arrivals_monthly_2026.csv", "w", newline="") as fh:
    w = csv.writer(fh, lineterminator="\n"); w.writerow(["date", "year", "month", "arrivals", "source"])
    for m, v in enumerate(ARR_2026, start=1):
        w.writerow([f"2026-{m:02d}-01", 2026, m, v, "SLTDA weekly report 1-30 Sep 2026"])
print("Sealed test file written: 9 months (Jan-Sep 2026). All checks passed.")
