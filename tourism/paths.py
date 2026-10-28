from pathlib import Path

# This file is at: srilanka-tourism-2030/tourism/paths.py
# __file__ = the path of this file
# .resolve() = turn it into a full absolute path
# .parent = go up one folder

ROOT = Path(__file__).resolve().parent.parent   # go up twice: paths.py -> tourism -> project root

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"