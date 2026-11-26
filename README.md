# Wildfire ML project - Assignment 03 code (PyCharm)

## 1. Setup (once)
1. PyCharm > Open this folder > create a virtual environment (Python 3.10+).
2. Terminal: `pip install -r requirements.txt`
3. Download the SQLite zip from https://www.fs.usda.gov/rds/archive/products/RDS-2013-0009.7/RDS-2013-0009.7_Data_Format4_SQLITE.zip
   Unzip it and put the .sqlite file in the `data/` folder.
4. Open `config.py`. If your file name differs, change `DB_PATH`. If your PC is slow or has little RAM,
   lower `TRAIN_SAMPLE` (e.g. 150_000).

## 2. Quick check with fake data (2 minutes, optional)
    python make_demo_db.py
    python run_experiments.py --db data/demo_fires.sqlite --out outputs_demo --demo --train-sample 8000 --val-sample 3000 --tune-sample 4000
These numbers are NOT real. Never use them in the report.

## 3. Real run
    python run_experiments.py
Results go to `outputs/` (tables as CSV, 4 figures, saved models, dataset_stats.json).
Expect roughly 20-60 minutes on a normal laptop with the default sample sizes.

## 4. Report and demo
    python build_report.py      # builds outputs/Assignment03_Report.docx from your real results
    python predict_demo.py      # demo for the lab project: one fire in, cause + size out

## Files
- config.py          settings, years, sample sizes, group names
- features.py        feature engineering and preprocessing (shared by all models)
- run_experiments.py data -> models -> metrics -> figures
- build_report.py    results -> report (.docx, export to PDF from Word)
- predict_demo.py    small demo for the lab
- make_demo_db.py    fake data for a quick test

## Old-edition mode (1992-2015)
If the 7th edition cannot be downloaded, put the old `FPA_FOD_20170508.sqlite` in `data/`. `config.py` picks it up automatically,
`run_experiments.py` detects the years (<= 2015), switches the split to train 1992-2011 / validation 2012-2013 / test 2014-2015,
uses `STAT_CAUSE_DESCR` as the cause column, and `build_report.py` adds a paragraph explaining the change from the proposal.
Run `python check_db.py` first to see which edition you have.
