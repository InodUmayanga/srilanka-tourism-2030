"""Run the whole project in order: data checks -> exploration -> models -> backtest -> 2030 forecast."""
import runpy
from pathlib import Path

SCRIPTS = ["01_check_data", "02_explore", "03_decompose", "04_baseline", "05_regression",
           "06_holt_winters", "07_sarima", "08_backtest", "09_test_2026",
           "10_forecast_2030", "11_scenarios"]

for name in SCRIPTS:
    print(f"\n{'=' * 70}\nRunning {name}.py\n{'=' * 70}")
    runpy.run_path(str(Path(__file__).parent / "scripts" / f"{name}.py"), run_name="__main__")
print("\nDone. Figures are in figures/, tables are in results/.")