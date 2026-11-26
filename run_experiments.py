"""
run_experiments.py - Assignment 03: comparative analysis of ML models on the FPA FOD wildfire data.

Run (PyCharm: right-click > Run, or in the terminal):
    python run_experiments.py

What it does (same order as the Assignment 02 proposal):
  1. loads the SQLite file, cleans it, builds features
  2. splits by TIME (7th edition: train 1992-2019, val 2020-2021, test 2022-2024; old 1992-2015 file: 1992-2011 / 2012-2013 / 2014-2015)
  3. for each task (cause classification, log fire-size regression):
       baseline + Logistic/Ridge + Decision Tree + Random Forest + Hist Gradient Boosting
       - same preprocessing, same split, same tuning effort (4 candidates x 3 time-ordered folds)
       - tuning never touches the test years
  4. extra checks: random split vs time split, ablations (owner/agency, cause as input)
  5. saves tables, figures, models and a JSON summary into outputs/
Then run build_report.py to produce the report.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, precision_score, r2_score,
                             recall_score)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import config as cfg
from features import (CAT_COLS, FREQ_COLS, NUM_COLS, build_features, make_preprocessor, size_class)

warnings.filterwarnings("ignore")
T0 = time.perf_counter()

# colours: first three slots of a colour-blind-checked palette (blue, orange, aqua)
C_TRAIN, C_VAL, C_TEST = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#222222", "#666666"

NEEDED = ["FOD_ID", "FIRE_YEAR", "DISCOVERY_DOY", "DISCOVERY_TIME", "LATITUDE", "LONGITUDE", "STATE",
          "FIPS_CODE", "COUNTY", "OWNER_DESCR", "NWCG_REPORTING_AGENCY", "SOURCE_SYSTEM_TYPE",
          "NWCG_GENERAL_CAUSE", "STAT_CAUSE_DESCR", "FIRE_SIZE"]
REQUIRED = ["FIRE_YEAR", "DISCOVERY_DOY", "LATITUDE", "LONGITUDE", "FIRE_SIZE"]
CAUSE_COLS = ["NWCG_GENERAL_CAUSE", "STAT_CAUSE_DESCR"]   # 7th edition name, 4th edition (1992-2015) name


def log(msg):
    print(f"[{time.perf_counter() - T0:7.0f}s] {msg}", flush=True)


# ===================================================================== 1. data
def load_raw(db_path):
    import sqlite3
    con = sqlite3.connect(str(db_path))
    tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    tables.sort(key=lambda t: t != "Fires")          # try the main table first
    for t in tables:
        try:
            cols = [r[1] for r in con.execute(f'PRAGMA table_info("{t}")')]
        except sqlite3.OperationalError:             # SpatiaLite index tables (e.g. VirtualSpatialIndex) cannot be opened here; skip them
            continue
        if set(REQUIRED) <= set(cols) and any(c in cols for c in CAUSE_COLS):
            want = [c for c in NEEDED if c in cols]
            log(f"reading table '{t}' ({len(want)} columns)")
            df = pd.read_sql_query(f'SELECT {", ".join(want)} FROM "{t}"', con)
            con.close()
            if "NWCG_GENERAL_CAUSE" not in df.columns:      # old edition: use its cause column under the same name
                df["NWCG_GENERAL_CAUSE"] = df["STAT_CAUSE_DESCR"]
                df.attrs["cause_source"] = "STAT_CAUSE_DESCR"
            else:
                df.attrs["cause_source"] = "NWCG_GENERAL_CAUSE"
            return df
    con.close()
    raise SystemExit(f"No table with columns {REQUIRED} + one of {CAUSE_COLS} found. Tables in file: {tables}")


def prepare(raw):
    """Clean + feature engineering. Returns (data, stats_part)."""
    st = {"n_raw": int(len(raw))}
    df = raw
    if "FOD_ID" in df:
        df = df.drop_duplicates("FOD_ID")
    st["n_duplicates_removed"] = int(st["n_raw"] - len(df))
    before = len(df)
    for c in ["LATITUDE", "LONGITUDE", "FIRE_YEAR", "DISCOVERY_DOY", "FIRE_SIZE"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["LATITUDE", "LONGITUDE", "FIRE_YEAR", "DISCOVERY_DOY", "FIRE_SIZE"])
    df = df[df["FIRE_SIZE"] >= 0]
    st["n_dropped_missing"] = int(before - len(df))
    st["n_clean"] = int(len(df))
    if "FIPS_CODE" not in df and "COUNTY" in df:
        df["FIPS_CODE"] = df["STATE"].astype(str) + "_" + df["COUNTY"].astype(str)
    X = build_features(df)
    data = X.copy()
    data["FIRE_YEAR"] = df["FIRE_YEAR"].astype(int)
    data["DISCOVERY_DOY"] = df["DISCOVERY_DOY"].astype(int)
    data["FIRE_SIZE"] = df["FIRE_SIZE"].astype(float)
    data["size_class"] = size_class(data["FIRE_SIZE"])
    data["CAUSE"] = df["NWCG_GENERAL_CAUSE"].astype(str).str.strip().replace({"None": "Missing", "nan": "Missing"})
    low = data["CAUSE"].str.lower()
    data["cause_labeled"] = ~(low.str.contains("missing") | low.str.contains("undetermined") | low.str.contains("not specified"))
    st["hour_missing_pct"] = float(data["hour"].isna().mean() * 100)
    st["owner_unknown_pct"] = float((data["OWNER_DESCR"] == "Unknown").mean() * 100)
    st["year_min"], st["year_max"] = int(data["FIRE_YEAR"].min()), int(data["FIRE_YEAR"].max())
    return data, st


def by_years(df, years):
    return df[df["FIRE_YEAR"].between(years[0], years[1])]


def strat_sample(df, n, col, seed):
    if n is None or len(df) <= n:
        return df.sample(frac=1.0, random_state=seed)
    return df.groupby(col, group_keys=False).sample(frac=n / len(df), random_state=seed)


def chrono(df):
    return df.sort_values(["FIRE_YEAR", "DISCOVERY_DOY"], kind="stable")


# ===================================================================== 2. models
def model_specs(task, seed, n_jobs):
    """4 models per task, each with exactly 4 grid candidates -> same tuning effort."""
    if task == "classification":
        return {
            "Logistic Regression": dict(
                make=lambda: LogisticRegression(class_weight="balanced", max_iter=300, random_state=seed),
                grid={"model__C": [0.01, 0.1, 1, 10]}, scale=True),
            "Decision Tree": dict(
                make=lambda: DecisionTreeClassifier(class_weight="balanced", min_samples_leaf=10, random_state=seed),
                grid={"model__max_depth": [6, 10, 14, 18]}, scale=False),
            "Random Forest": dict(
                make=lambda: RandomForestClassifier(n_estimators=100, min_samples_leaf=5, class_weight="balanced_subsample",
                                                    n_jobs=n_jobs, random_state=seed),
                grid={"model__max_depth": [8, 12, 16, 20]}, scale=False),
            "Hist Gradient Boosting": dict(
                make=lambda: HistGradientBoostingClassifier(class_weight="balanced", early_stopping=False, random_state=seed),
                grid={"model__learning_rate": [0.03, 0.1], "model__max_iter": [100, 200]}, scale=False),
        }
    return {
        "Ridge Regression": dict(
            make=lambda: Ridge(),
            grid={"model__alpha": [0.1, 1, 10, 100]}, scale=True),
        "Decision Tree": dict(
            make=lambda: DecisionTreeRegressor(min_samples_leaf=10, random_state=seed),
            grid={"model__max_depth": [6, 10, 14, 18]}, scale=False),
        "Random Forest": dict(
            make=lambda: RandomForestRegressor(n_estimators=100, min_samples_leaf=5, n_jobs=n_jobs, random_state=seed),
            grid={"model__max_depth": [8, 12, 16, 20]}, scale=False),
        "Hist Gradient Boosting": dict(
            make=lambda: HistGradientBoostingRegressor(early_stopping=False, random_state=seed),
            grid={"model__learning_rate": [0.03, 0.1], "model__max_iter": [100, 200]}, scale=False),
    }


def make_pipe(spec, colspec):
    return Pipeline([("prep", make_preprocessor(spec["scale"], colspec["num"], colspec["cat"], colspec["freq"])),
                     ("model", spec["make"]())])


def cols_of(colspec):
    return colspec["num"] + colspec["cat"] + colspec["freq"]


# ===================================================================== 3. metrics
def eval_cls(y, p):
    return dict(accuracy=accuracy_score(y, p),
                precision=precision_score(y, p, average="macro", zero_division=0),
                recall=recall_score(y, p, average="macro", zero_division=0),
                f1=f1_score(y, p, average="macro", zero_division=0),
                f1_weighted=f1_score(y, p, average="weighted", zero_division=0))


def to_acres(v):
    return np.clip(np.expm1(v), 0, None)


def eval_reg(y, p):
    mse = mean_squared_error(y, p)
    ya, pa = to_acres(np.asarray(y)), to_acres(np.asarray(p))
    return dict(mae=mean_absolute_error(y, p), mse=mse, rmse=float(np.sqrt(mse)), r2=r2_score(y, p),
                medae_acres=float(np.median(np.abs(ya - pa))), mae_acres=float(np.mean(np.abs(ya - pa))),
                rmse_acres=float(np.sqrt(np.mean((ya - pa) ** 2))))


# ===================================================================== 4. one task
def run_task(task, train, val, test, tune, colspec, specs, seed):
    ev = eval_cls if task == "classification" else eval_reg
    scoring = "f1_macro" if task == "classification" else "neg_mean_squared_error"
    cols = cols_of(colspec)
    rng = np.random.RandomState(seed)
    tr_eval = train.iloc[rng.permutation(len(train))[:50_000]]      # subset used only to score the training fit
    rows, fitted = [], {}

    def record(name, model, params, tune_s):
        t = time.perf_counter(); model.fit(train[cols], train["y"]); train_s = time.perf_counter() - t
        t = time.perf_counter(); p_te = model.predict(test[cols]); pred_s = time.perf_counter() - t
        row = {"model": name, "best_params": json.dumps(params), "tune_s": tune_s,
               "train_s": train_s, "predict_s": pred_s, "n_test": len(test)}
        for tag, d in (("train", tr_eval), ("val", val)):
            row.update({f"{tag}_{k}": v for k, v in ev(d["y"], model.predict(d[cols])).items()})
        row.update({f"test_{k}": v for k, v in ev(test["y"], p_te).items()})
        rows.append(row)
        fitted[name] = dict(model=model, pred_test=p_te, params=params)
        key = "f1" if task == "classification" else "rmse"
        log(f"  {task[:3]} | {name:<24} train {train_s:6.1f}s  pred {pred_s:5.1f}s  test {key} {row['test_' + key]:.4f}")

    base_name = "Baseline (majority class)" if task == "classification" else "Baseline (mean of log size)"
    base = DummyClassifier(strategy="most_frequent") if task == "classification" else DummyRegressor(strategy="mean")
    record(base_name, base, {}, 0.0)

    for name, spec in specs.items():
        pipe = make_pipe(spec, colspec)
        t = time.perf_counter()
        gs = GridSearchCV(pipe, spec["grid"], scoring=scoring, cv=TimeSeriesSplit(n_splits=cfg.CV_SPLITS),
                          n_jobs=1, refit=False)
        gs.fit(tune[cols], tune["y"])
        tune_s = time.perf_counter() - t
        params = {k: (v.item() if hasattr(v, "item") else v) for k, v in gs.best_params_.items()}
        pipe.set_params(**params)
        record(name, pipe, {k.replace("model__", ""): v for k, v in params.items()}, tune_s)

    res = pd.DataFrame(rows)
    cand = res[~res["model"].str.startswith("Baseline")]
    best = (cand.sort_values("val_f1", ascending=False) if task == "classification"
            else cand.sort_values("val_rmse")).iloc[0]["model"]       # best model chosen on VALIDATION years
    return res, fitted, best


def fit_eval(task, spec, params, colspec, train, test):
    """Refit one model with given params (used by ablations and the random-split check)."""
    pipe = make_pipe(spec, colspec)
    pipe.set_params(**{f"model__{k}": v for k, v in params.items()})
    cols = cols_of(colspec)
    pipe.fit(train[cols], train["y"])
    ev = eval_cls if task == "classification" else eval_reg
    return ev(test["y"], pipe.predict(test[cols]))


# ===================================================================== 5. figures
def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#999999")
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)


def fig_confusion(y, p, name, path):
    labels = sorted(set(y) | set(p))
    cm = confusion_matrix(y, p, labels=labels, normalize="true")
    short = [l if len(l) <= 20 else l[:19] + "." for l in labels]
    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=200)
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), short, rotation=70, ha="right", fontsize=6)
    ax.set_yticks(range(len(labels)), short, fontsize=6)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j] >= 0.05:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=5,
                        color="white" if cm[i, j] > 0.5 else INK)
    ax.set_xlabel("Predicted cause", fontsize=7, color=INK)
    ax.set_ylabel("Actual cause", fontsize=7, color=INK)
    ax.set_title(f"{name}: row-normalised confusion matrix (test)", fontsize=7.5, color=INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=6)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_pred_actual(y, p, name, path, seed):
    idx = np.random.RandomState(seed).permutation(len(y))[:40_000]
    y, p = np.asarray(y)[idx], np.asarray(p)[idx]
    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=200)
    hb = ax.hexbin(y, p, gridsize=45, bins="log", cmap="Blues", mincnt=1)
    lim = [min(y.min(), p.min()), max(y.max(), p.max())]
    ax.plot(lim, lim, color=C_TEST, lw=1.5, label="perfect prediction (y = x)")
    ax.set_xlabel("Actual ln(1 + acres)", fontsize=7, color=INK)
    ax.set_ylabel("Predicted ln(1 + acres)", fontsize=7, color=INK)
    ax.set_title(f"{name}: predicted vs actual (test)", fontsize=7.5, color=INK)
    ax.tick_params(colors=MUTED, labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.03); cb.set_label("fires (log count)", fontsize=6); cb.ax.tick_params(labelsize=6)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_generalization(cls_res, reg_res, path):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), dpi=200)
    for ax, res, key, title in ((axes[0], cls_res, "f1", "Classification: macro F1"),
                                (axes[1], reg_res, "r2", "Regression: R² (log scale)")):
        names = [m.replace("Baseline (majority class)", "Baseline").replace("Baseline (mean of log size)", "Baseline")
                  .replace("Hist Gradient Boosting", "Hist\nGB").replace("Logistic Regression", "Logistic\nReg.")
                  .replace("Ridge Regression", "Ridge").replace("Decision Tree", "Decision\nTree")
                  .replace("Random Forest", "Random\nForest") for m in res["model"]]
        x = np.arange(len(names)); w = 0.26
        for k, (tag, col) in enumerate((("train", C_TRAIN), ("val", C_VAL), ("test", C_TEST))):
            ax.bar(x + (k - 1) * w, res[f"{tag}_{key}"], w * 0.92, color=col, label=tag)
        ax.set_xticks(x, names, fontsize=6.5)
        ax.set_title(title, fontsize=8, color=INK, loc="left")
        ax.axhline(0, color="#999999", lw=0.6)
        style_axes(ax)
    axes[0].legend(fontsize=6.5, frameon=False, ncol=3, loc="upper left")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_importance(imp_cls, imp_reg, path):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), dpi=200)
    for ax, imp, title, col in ((axes[0], imp_cls, "Cause: drop in macro F1 when shuffled", C_TRAIN),
                                (axes[1], imp_reg, "Size: rise in MSE when shuffled", C_VAL)):
        imp = imp.sort_values("mean")
        ax.barh(imp.index, imp["mean"], xerr=imp["std"], color=col, height=0.6, ecolor="#999999")
        ax.set_title(title, fontsize=8, color=INK, loc="left")
        style_axes(ax); ax.grid(axis="x", color="#e6e6e6", linewidth=0.6); ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def perm_importance(model, X, y, scoring, seed):
    X = X.iloc[:5000]; y = y.iloc[:5000]
    r = permutation_importance(model, X, y, scoring=scoring, n_repeats=3, random_state=seed, n_jobs=1)
    return pd.DataFrame({"mean": r.importances_mean, "std": r.importances_std}, index=X.columns)


# ===================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(cfg.DB_PATH))
    ap.add_argument("--out", default=str(cfg.OUTPUT_DIR))
    ap.add_argument("--train-sample", type=int, default=cfg.TRAIN_SAMPLE)
    ap.add_argument("--val-sample", type=int, default=cfg.VAL_SAMPLE)
    ap.add_argument("--test-sample", type=int, default=cfg.TEST_SAMPLE)
    ap.add_argument("--tune-sample", type=int, default=cfg.TUNE_SAMPLE)
    ap.add_argument("--demo", action="store_true", help="mark outputs as demo data (synthetic test file)")
    a = ap.parse_args()
    seed = cfg.RANDOM_STATE
    out = Path(a.out); (out / "figures").mkdir(parents=True, exist_ok=True); (out / "models").mkdir(exist_ok=True)

    # ---------------- load + clean
    raw = load_raw(a.db)
    cause_source = raw.attrs.get("cause_source", "NWCG_GENERAL_CAUSE")
    data, stats = prepare(raw)
    del raw
    stats["cause_source"] = cause_source
    stats["legacy"] = bool(stats["year_max"] <= 2015)
    if stats["legacy"]:   # old edition (1992-2015): shift the time-based split to years that exist in the file
        cfg.TRAIN_YEARS, cfg.VAL_YEARS, cfg.TEST_YEARS = cfg.LEGACY_TRAIN_YEARS, cfg.LEGACY_VAL_YEARS, cfg.LEGACY_TEST_YEARS
        log(f"old edition detected (years {stats['year_min']}-{stats['year_max']}): split train {cfg.TRAIN_YEARS}, val {cfg.VAL_YEARS}, test {cfg.TEST_YEARS}")
    log(f"rows: raw {stats['n_raw']:,} -> clean {stats['n_clean']:,}")

    train_all = by_years(data, cfg.TRAIN_YEARS)
    val_all = by_years(data, cfg.VAL_YEARS)
    test_all = by_years(data, cfg.TEST_YEARS)
    colspec = dict(num=NUM_COLS, cat=CAT_COLS, freq=FREQ_COLS)

    # ---------------- classification data
    lab = data[data["cause_labeled"]].copy()
    share = by_years(lab, cfg.TRAIN_YEARS)["CAUSE"].value_counts(normalize=True)
    merged = sorted(share[share < cfg.MIN_CLASS_SHARE].index.tolist())
    lab["y"] = lab["CAUSE"].where(~lab["CAUSE"].isin(merged), cfg.RARE_CLASS_LABEL)
    c_train_pop, c_val_pop, c_test_pop = (by_years(lab, y) for y in (cfg.TRAIN_YEARS, cfg.VAL_YEARS, cfg.TEST_YEARS))
    c_train = chrono(strat_sample(c_train_pop, a.train_sample, "y", seed))
    c_val = strat_sample(c_val_pop, a.val_sample, "y", seed)
    c_test = strat_sample(c_test_pop, a.test_sample, "y", seed)
    c_tune = chrono(strat_sample(c_train, a.tune_sample, "y", seed))

    # ---------------- regression data (all rows, cause not required)
    reg = data.copy()
    reg["y"] = np.log1p(reg["FIRE_SIZE"])
    r_train_pop, r_val_pop, r_test_pop = (by_years(reg, y) for y in (cfg.TRAIN_YEARS, cfg.VAL_YEARS, cfg.TEST_YEARS))
    r_train = chrono(strat_sample(r_train_pop, a.train_sample, "size_class", seed))
    r_val = strat_sample(r_val_pop, a.val_sample, "size_class", seed)
    r_test = strat_sample(r_test_pop, a.test_sample, "size_class", seed)
    r_tune = chrono(strat_sample(r_train, a.tune_sample, "size_class", seed))

    # ---------------- dataset statistics for the report
    classes = sorted(c_train["y"].unique())
    stats["features"] = colspec
    stats["cls"] = dict(
        n_labeled=int(len(lab)), n_unlabeled=int((~data["cause_labeled"]).sum()), merged_classes=merged,
        classes=classes,
        share_train={k: float(v * 100) for k, v in c_train["y"].value_counts(normalize=True).items()},
        share_test={k: float(v * 100) for k, v in c_test["y"].value_counts(normalize=True).items()})
    maj = c_train["y"].value_counts().idxmax()
    stats["cls"]["majority_class"] = maj
    stats["cls"]["majority_share_test"] = float((c_test["y"] == maj).mean() * 100)
    fs = data["FIRE_SIZE"]
    stats["reg"] = dict(
        median=float(fs.median()), mean=float(fs.mean()), max=float(fs.max()),
        pct_class_A=float((data["size_class"] == "A").mean() * 100),
        pct_ge_100=float((fs >= 100).mean() * 100), pct_ge_5000=float((fs >= 5000).mean() * 100),
        log_mean=float(np.log1p(fs).mean()), log_std=float(np.log1p(fs).std()),
        class_share={k: float(v * 100) for k, v in data["size_class"].value_counts(normalize=True).sort_index().items()})
    stats["splits"] = dict(
        train_years=list(cfg.TRAIN_YEARS), val_years=list(cfg.VAL_YEARS), test_years=list(cfg.TEST_YEARS),
        cls=dict(train_pop=len(c_train_pop), val_pop=len(c_val_pop), test_pop=len(c_test_pop),
                 train_n=len(c_train), val_n=len(c_val), test_n=len(c_test), tune_n=len(c_tune)),
        reg=dict(train_pop=len(r_train_pop), val_pop=len(r_val_pop), test_pop=len(r_test_pop),
                 train_n=len(r_train), val_n=len(r_val), test_n=len(r_test), tune_n=len(r_tune)))
    json.dump(stats, open(out / "dataset_stats.json", "w"), indent=2)
    log(f"classes ({len(classes)}): {classes}; merged into '{cfg.RARE_CLASS_LABEL}': {merged}")

    # ---------------- run both tasks
    log("CLASSIFICATION"); cls_specs = model_specs("classification", seed, cfg.N_JOBS)
    cls_res, cls_fit, cls_best = run_task("classification", c_train, c_val, c_test, c_tune, colspec, cls_specs, seed)
    log("REGRESSION"); reg_specs = model_specs("regression", seed, cfg.N_JOBS)
    reg_res, reg_fit, reg_best = run_task("regression", r_train, r_val, r_test, r_tune, colspec, reg_specs, seed)
    cls_res.to_csv(out / "results_classification.csv", index=False)
    reg_res.to_csv(out / "results_regression.csv", index=False)
    log(f"best classifier (validation): {cls_best} | best regressor (validation): {reg_best}")

    # ---------------- per-class and per-size-class detail for the best models
    p_cls = cls_fit[cls_best]["pred_test"]
    rep = pd.DataFrame(classification_report(c_test["y"], p_cls, output_dict=True, zero_division=0)).T
    rep.to_csv(out / "per_class_report.csv")
    p_reg = reg_fit[reg_best]["pred_test"]
    e = r_test.assign(pred=p_reg)
    e["err_log"] = e["pred"] - e["y"]
    e["abs_err_acres"] = np.abs(to_acres(e["pred"]) - e["FIRE_SIZE"])
    size_err = e.groupby("size_class").agg(n=("y", "size"), mean_log_error=("err_log", "mean"),
                                           median_abs_error_acres=("abs_err_acres", "median")).reset_index()
    size_err.to_csv(out / "size_class_errors.csv", index=False)

    # ---------------- extra checks: random split + ablations (best model, best settings)
    log("EXTRA CHECKS")
    extra = {}
    for task, best, fit, specs, pop in (("classification", cls_best, cls_fit, cls_specs, lab.assign(y=lab["y"])),
                                        ("regression", reg_best, reg_fit, reg_specs, reg)):
        spec, params = specs[best], fit[best]["params"]
        tr_n = len(c_train if task == "classification" else r_train)
        te_n = len(c_test if task == "classification" else r_test)
        te_n = min(te_n, int(0.2 * len(pop))); tr_n = min(tr_n, len(pop) - te_n)
        idx = np.random.RandomState(seed).permutation(len(pop))
        rs_train, rs_test = pop.iloc[idx[:tr_n]], pop.iloc[idx[tr_n:tr_n + te_n]]
        m = fit_eval(task, spec, params, colspec, rs_train, rs_test)
        extra[f"{task}_random_split"] = m
        tr, te = (c_train, c_test) if task == "classification" else (r_train, r_test)
        no_oa = dict(num=NUM_COLS, cat=[c for c in CAT_COLS if c not in ("OWNER_DESCR", "NWCG_REPORTING_AGENCY")], freq=FREQ_COLS)
        extra[f"{task}_without_owner_agency"] = fit_eval(task, spec, params, no_oa, tr, te)
        if task == "regression":
            with_cause = dict(num=NUM_COLS, cat=CAT_COLS + ["CAUSE"], freq=FREQ_COLS)
            extra["regression_with_cause"] = fit_eval(task, spec, params, with_cause, tr, te)
        log(f"  {task}: random split done")
    json.dump(extra, open(out / "ablations.json", "w"), indent=2)

    # ---------------- figures
    log("FIGURES")
    fig_confusion(c_test["y"], p_cls, cls_best, out / "figures" / "fig1_confusion_matrix.png")
    fig_pred_actual(r_test["y"], p_reg, reg_best, out / "figures" / "fig2_pred_vs_actual.png", seed)
    fig_generalization(cls_res, reg_res, out / "figures" / "fig3_generalization.png")
    imp_c = perm_importance(cls_fit[cls_best]["model"], c_test[cols_of(colspec)], c_test["y"], "f1_macro", seed)
    imp_r = perm_importance(reg_fit[reg_best]["model"], r_test[cols_of(colspec)], r_test["y"], "neg_mean_squared_error", seed)
    imp_c.to_csv(out / "importance_classification.csv"); imp_r.to_csv(out / "importance_regression.csv")
    fig_importance(imp_c, imp_r, out / "figures" / "fig4_permutation_importance.png")

    # ---------------- save best models + metadata
    joblib.dump(cls_fit[cls_best]["model"], out / "models" / "best_classifier.joblib", compress=3)
    joblib.dump(reg_fit[reg_best]["model"], out / "models" / "best_regressor.joblib", compress=3)
    meta = dict(best_classifier=cls_best, best_regressor=reg_best, demo=bool(a.demo), db=Path(a.db).name,
                python=sys.version.split()[0], sklearn=sklearn.__version__, pandas=pd.__version__,
                platform=platform.platform(), cpu_count=os.cpu_count(), total_runtime_s=time.perf_counter() - T0,
                tune_sample=a.tune_sample, cv_splits=cfg.CV_SPLITS, seed=seed)
    json.dump(meta, open(out / "run_meta.json", "w"), indent=2)
    log(f"DONE. Results are in {out}. Next: python build_report.py")


if __name__ == "__main__":
    main()