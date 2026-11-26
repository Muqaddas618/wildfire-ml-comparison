"""
config.py - all settings in one place. Edit this file, not the other scripts.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------- paths
# Put the SQLite file from the Forest Service download into the data/ folder.
def _find_db():
    """Prefer the 7th edition; otherwise use whatever .sqlite file is in data/ (e.g. the old FPA_FOD_20170508)."""
    d = BASE_DIR / "data"
    preferred = d / "FPA_FOD_20260615.sqlite"
    if preferred.exists():
        return preferred
    found = sorted(p for p in d.glob("*.sqlite") if "demo" not in p.name.lower())
    return found[0] if found else preferred


DB_PATH = _find_db()
OUTPUT_DIR = BASE_DIR / "outputs"

# ---------------------------------------------------------------- split (by time)
TRAIN_YEARS = (1992, 2019)   # fit models
VAL_YEARS = (2020, 2021)     # check generalization, choose the best model
TEST_YEARS = (2022, 2024)    # final report only; never used for tuning
# Used automatically when the database ends in 2015 or earlier (4th edition, FPA_FOD_20170508):
LEGACY_TRAIN_YEARS = (1992, 2011)
LEGACY_VAL_YEARS = (2012, 2013)
LEGACY_TEST_YEARS = (2014, 2015)

# ---------------------------------------------------------------- sample sizes
TRAIN_SAMPLE = 300_000   # proposal: 200,000 - 500,000 rows. Lower it if your PC is slow.
VAL_SAMPLE = 50_000
TEST_SAMPLE = None       # None = use every fire in the test years
TUNE_SAMPLE = 60_000     # subset of the training sample used for the grid search
CV_SPLITS = 3            # TimeSeriesSplit folds (same for every model)

# ---------------------------------------------------------------- other settings
RANDOM_STATE = 42
N_JOBS = -1              # CPU cores for Random Forest (-1 = all)
MIN_CLASS_SHARE = 0.005  # causes rarer than this (in training years) are merged into "Other causes"
RARE_CLASS_LABEL = "Other causes"

# ---------------------------------------------------------------- report front page
# FILL THESE IN before running build_report.py
PROJECT_TITLE = "Wildfire Cause Classification and Fire Size Regression: A Comparative Analysis of Machine Learning Models"
UNIVERSITY = "COMSATS University Islamabad, Wah Campus"
COURSE = "Machine Learning Fundamentals - Assignment 03 (Comparative Analysis Report)"
GROUP_MEMBERS = [
    ("Zainab Khan", "SP24-BCS-039"),
    ("Emaan Shahid", "SP24-BCS-041"),
    ("Noor e Hira", "SP24-BCS-060"),
    ("Muqaddas Kanwal", "SP24-BCS-117"),
]
NOTEBOOK_LINK = "[ADD GITHUB OR COLAB LINK]"
