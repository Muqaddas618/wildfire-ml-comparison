"""check_db.py - run this first.  python check_db.py   (or: python check_db.py data/yourfile.sqlite)
Shows which edition of the FPA FOD you have and whether run_experiments.py can use it."""
import sqlite3, sys
import config as cfg

path = sys.argv[1] if len(sys.argv) > 1 else str(cfg.DB_PATH)
print("File:", path)
con = sqlite3.connect(path)
tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
print("Tables:", tables)
for t in tables:
    try:
        cols = [r[1] for r in con.execute(f'PRAGMA table_info("{t}")')]
    except sqlite3.OperationalError:        # SpatiaLite virtual index tables: not needed, skip
        continue
    if "FIRE_YEAR" in cols and "FIRE_SIZE" in cols:
        n = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        lo, hi = con.execute(f'SELECT MIN(FIRE_YEAR), MAX(FIRE_YEAR) FROM "{t}"').fetchone()
        cause = [c for c in ("NWCG_GENERAL_CAUSE", "STAT_CAUSE_DESCR") if c in cols]
        print(f"\nTable '{t}': {n:,} rows, FIRE_YEAR {lo}-{hi}, cause column: {cause}")
        if hi >= 2022 and "NWCG_GENERAL_CAUSE" in cols:
            print("-> 7th edition (1992-2024). Split: train 1992-2019, val 2020-2021, test 2022-2024.")
        elif hi <= 2015 and cause:
            print("-> OLD 4th edition (1992-2015). The code will switch automatically to train 1992-2011, val 2012-2013, test 2014-2015,")
            print("   and the report will include a paragraph explaining the change from the proposal.")
        else:
            print("-> Unexpected edition. Send me this output.")
con.close()