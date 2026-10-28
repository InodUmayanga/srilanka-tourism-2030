# Data

All raw sources are official SLTDA (Sri Lanka Tourism Development Authority) reports in `raw/`
(PDFs are git-ignored; `raw/manifest.csv` lists each file with its download URL).
Every processed file can be rebuilt with the scripts in `extraction/` (needs `pdftotext` from poppler):

```
python data/extraction/extract_arrivals.py     # arrivals_monthly_1971_2025.csv
python data/extraction/build_annual.py         # stay_spend_annual.csv, capacity_annual.csv
python data/extraction/build_2026_test.py      # arrivals_monthly_2026.csv (sealed test set)
```

## Files in `processed/`

| File | Rows | What it contains | Source |
|---|---|---|---|
| `arrivals_monthly_1971_2025.csv` | 660 months | monthly foreign tourist arrivals | ASR 2020 Table 28 (1971-2020), ASR 2023 Table 15 (2021-23), Year in Review 2025 Table 01 (2024-25) |
| `arrivals_monthly_2026.csv` | 9 months | **SEALED TEST SET** Jan-Sep 2026. Do not open before Step 12 | SLTDA weekly report 1-30 Sep 2026 |
| `stay_spend_annual.csv` | 1985-2025 | arrivals, tourist nights ('000), receipts (US$ mn), spend per tourist per day (US$), average stay (nights) | ASR 2023 Table 14; Year in Review 2024 & 2025 |
| `capacity_annual.csv` | 1985-2025 | graded rooms, graded occupancy (%), total registered rooms | ASR 2023 Table 14; ASR 2018 Table 20; ASR 2020; Year in Review 2021, 2022, 2024, 2025 |
| `targets.csv` | 9 | official arrival and revenue targets for 2026 and 2030, with links | news reports quoting SLTDA / the ministry |
| `validation_report*.txt` | | results of the automatic checks | |

## Definitions and known issues (read before modeling)

- **Graded rooms** = classified tourist hotels + unclassified tourist hotels + boutique hotels + boutique villas (SLTDA definition).
- **Total registered rooms** is not one consistent series: 2007-2018 = tourist hotels + "supplementary establishments";
  from about 2019 SLTDA registers more types (homestays, bungalows, guest houses...). Treat jumps around 2019-2021 with care.
- **Occupancy method changed.** Up to 2018 graded occupancy was ~70-77%. From 2019 it comes from surveys of responding hotels;
  2024 (44.5%) is the average of 221 responding graded hotels by district. 2025 occupancy was not published.
- **Spend per day** is a survey-based base value applied to all months, not measured each year. 2025 has two values
  (US$171.74 Jan-Jul, US$148.26 Aug-Dec), so the annual column is blank.
- **2024 rooms**: the Year in Review 2025 revised 2024 total rooms to 53,378 (YIR 2024 had said 55,455). The revised value is used.
- **2020 arrivals**: SLTDA's published annual total (507,704) is 393 higher than the sum of its months (507,311). The monthly values are used.
- **2006**: published receipts are 18% below arrivals x stay x spend per day, an inconsistency in SLTDA's own table.
- **Airport closure**: arrivals are 0 from April to December 2020.
- Values marked provisional by SLTDA (mostly 2014 onwards) may be revised in later reports.
