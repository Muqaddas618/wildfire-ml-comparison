"""
predict_demo.py - small demo for the project: give ONE fire's discovery-time details and get
(1) cause probabilities and (2) an expected size in acres.

1. Run run_experiments.py first (it saves the best models in outputs/models/).
2. Edit the FIRE dictionary below, then run:  python predict_demo.py
"""
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import config as cfg
from features import FREQ_COLS, CAT_COLS, NUM_COLS, build_features

# ----------------------------------------------------------------- EDIT THIS EXAMPLE
FIRE = dict(
    latitude=34.2, longitude=-118.5,          # where
    discovery_date=date(2023, 7, 15),         # when
    discovery_time="1430",                    # hhmm, or "" if unknown
    state="CA", fips_code="06037",            # county code
    owner="USFS",                             # use values that exist in OWNER_DESCR (unknown values are tolerated)
    agency="FS",                              # NWCG_REPORTING_AGENCY
    source_type="FED",                        # FED / NONFED / INTERAGCY
)
# -----------------------------------------------------------------

models = Path(cfg.OUTPUT_DIR) / "models"
clf = joblib.load(models / "best_classifier.joblib")
reg = joblib.load(models / "best_regressor.joblib")

d = FIRE["discovery_date"]
raw = pd.DataFrame([{
    "FIRE_YEAR": d.year, "DISCOVERY_DOY": d.timetuple().tm_yday, "DISCOVERY_TIME": FIRE["discovery_time"] or np.nan,
    "LATITUDE": FIRE["latitude"], "LONGITUDE": FIRE["longitude"], "STATE": FIRE["state"],
    "FIPS_CODE": FIRE["fips_code"], "OWNER_DESCR": FIRE["owner"], "NWCG_REPORTING_AGENCY": FIRE["agency"],
    "SOURCE_SYSTEM_TYPE": FIRE["source_type"]}])
X = build_features(raw)[NUM_COLS + CAT_COLS + FREQ_COLS]

proba = clf.predict_proba(X)[0]
order = np.argsort(proba)[::-1]
print("Cause probabilities (top 5):")
for i in order[:5]:
    print(f"  {clf.classes_[i]:<45s} {proba[i]:6.1%}")

log_size = float(reg.predict(X)[0])
print(f"\nExpected final size: {max(np.expm1(log_size), 0):.2f} acres  (model predicts ln(1+acres) = {log_size:.2f})")
print("Note: size is a typical value for fires like this; very large fires are usually underpredicted.")
