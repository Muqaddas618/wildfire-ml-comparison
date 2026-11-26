"""
make_demo_db.py - creates a SMALL SYNTHETIC test database (data/demo_fires.sqlite) with the same
columns as the FPA FOD "Fires" table. It is only for checking that the code runs quickly.
The numbers it produces are NOT real results. Never put them in your report.

    python make_demo_db.py
    python run_experiments.py --db data/demo_fires.sqlite --out outputs_demo --demo \
        --train-sample 8000 --val-sample 3000 --tune-sample 4000
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.RandomState(0)
import sys
LEGACY = "legacy" in sys.argv   # python make_demo_db.py legacy  -> old-edition column names, years 1992-2015
nums = [x for x in sys.argv[1:] if x.isdigit()]
N = int(nums[0]) if nums else 40_000   # optional: python make_demo_db.py 200000
states = ["CA", "TX", "FL", "OR", "MT", "GA", "AZ", "NC", "ID", "NM"]
owners = ["PRIVATE", "USFS", "BLM", "STATE", "NPS", "BIA", "MISSING/NOT SPECIFIED"]
agencies = ["ST/C&L", "FS", "BLM", "NPS", "BIA", "FWS"]
causes = ["Arson/incendiarism", "Debris and open burning", "Equipment and vehicle use", "Natural",
          "Recreation and ceremony", "Smoking", "Fireworks", "Missing data/not specified/undetermined"]

year = rng.randint(1992, 2016 if LEGACY else 2025, N)
doy = rng.randint(1, 366, N)
state = rng.choice(states, N)
lat = rng.uniform(26, 48, N)
lon = rng.uniform(-123, -80, N)
# cause depends a little on state and season so that models have something to learn
p = np.tile(np.array([0.15, 0.25, 0.15, 0.15, 0.1, 0.05, 0.05, 0.10]), (N, 1))
p[np.isin(state, ["MT", "ID", "OR"]), 3] += 0.25
p[(doy > 170) & (doy < 250), 3] += 0.15
p[np.isin(state, ["GA", "NC", "FL"]), 1] += 0.2
p /= p.sum(1, keepdims=True)
cause = np.array([rng.choice(causes, p=pi) for pi in p])
size = np.exp(rng.normal(-0.5 + 0.8 * (cause == "Natural") + 0.5 * np.isin(state, ["MT", "ID", "AZ"]), 1.8))
time_ = np.where(rng.rand(N) < 0.3, np.nan, rng.randint(0, 24, N) * 100 + rng.randint(0, 60, N))

df = pd.DataFrame({
    "FOD_ID": np.arange(N), "FIRE_YEAR": year, "DISCOVERY_DATE": "", "DISCOVERY_DOY": doy,
    "DISCOVERY_TIME": time_, "LATITUDE": lat, "LONGITUDE": lon, "STATE": state,
    "FIPS_CODE": [f"{s}{rng.randint(1, 60):03d}" for s in state], "COUNTY": "x",
    "OWNER_DESCR": rng.choice(owners, N), "NWCG_REPORTING_AGENCY": rng.choice(agencies, N),
    "SOURCE_SYSTEM_TYPE": rng.choice(["FED", "NONFED", "INTERAGCY"], N),
    "NWCG_GENERAL_CAUSE": cause, "NWCG_CAUSE_CLASSIFICATION": "x", "FIRE_SIZE": size, "FIRE_SIZE_CLASS": "A"})
if LEGACY:
    df = df.drop(columns=["NWCG_GENERAL_CAUSE", "NWCG_CAUSE_CLASSIFICATION"]).assign(STAT_CAUSE_DESCR=cause)
df.loc[rng.rand(N) < 0.01, "LATITUDE"] = np.nan  # a few missing coordinates to test cleaning

Path("data").mkdir(exist_ok=True)
DEMO = "data/demo_fires_legacy.sqlite" if LEGACY else "data/demo_fires.sqlite"
con = sqlite3.connect(DEMO)
df.to_sql("Fires", con, if_exists="replace", index=False)
con.close()
print("Created", DEMO, "with", len(df), "synthetic rows")
