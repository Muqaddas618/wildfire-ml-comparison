"""
build_report.py - builds the Assignment 03 report (Word file) from the results made by run_experiments.py.

    python build_report.py                      # reads outputs/, writes outputs/Assignment03_Report.docx
    python build_report.py --out outputs_demo   # only for the synthetic test run

Every number in the report is read from the result files, so the text always matches your results.
Sentences that interpret the numbers are DRAFTS: read each one, check it against Tables/Figures,
and edit it in Word so that it says what YOU conclude (the assignment requires original work).
Yellow text in the document = something you still must fill in (names, reg numbers, link).
Finish: open the .docx in Word, check there are 8 pages or fewer, File > Save As > PDF.
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

import config as cfg

FONT = "Times New Roman"
R2 = "R\u00b2"


# =============================================================== small helpers
def f3(v): return f"{v:.3f}"
def f2(v): return f"{v:.2f}"
def secs(v): return f"{v:.3f}" if v < 0.1 else (f"{v:.2f}" if v < 10 else f"{v:.1f}")
def pct(v): return f"{v:.1f}%"


def params_str(s):
    d = json.loads(s)
    return ", ".join(f"{k}={v}" for k, v in d.items()) if d else "-"


def short(name):
    return (name.replace("Baseline (majority class)", "Baseline").replace("Baseline (mean of log size)", "Baseline")
            .replace("Hist Gradient Boosting", "Hist. Gradient Boosting"))


def row(df, name):
    return df[df["model"] == name].iloc[0]


def cand(df):
    return df[~df["model"].str.startswith("Baseline")]


def base(df):
    return df[df["model"].str.startswith("Baseline")].iloc[0]


def family(name):
    n = name.lower()
    if "logistic" in n or "ridge" in n: return "linear"
    if "decision" in n: return "tree"
    if "forest" in n: return "bagging"
    return "boosting"


# =============================================================== docx helpers
TOKEN = re.compile(r"(\*\*.+?\*\*|\[ADD[^\]]*\])")


def set_font(run, size=12, bold=None, italic=None, color=None):
    run.font.name = FONT
    run._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    run.font.size = Pt(size)
    if bold is not None: run.bold = bold
    if italic is not None: run.italic = italic
    if color: run.font.color.rgb = RGBColor(*color)


def add_runs(p, text, size=12, bold=False, italic=False):
    for part in TOKEN.split(text):
        if not part:
            continue
        if part.startswith("**"):
            r = p.add_run(part[2:-2]); set_font(r, size, True, italic)
        elif part.startswith("[ADD"):
            r = p.add_run(part); set_font(r, size, True, italic); r.font.highlight_color = WD_COLOR_INDEX.YELLOW
        else:
            r = p.add_run(part); set_font(r, size, bold, italic)


def fmt(p, align=WD_ALIGN_PARAGRAPH.JUSTIFY, after=4, before=0, keep_next=False):
    pf = p.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = 1.15
    pf.space_after, pf.space_before = Pt(after), Pt(before)
    pf.keep_with_next = keep_next
    p.alignment = align


class Report:
    def __init__(self, demo):
        self.doc = Document()
        d = self.doc
        s = d.sections[0]
        s.page_height, s.page_width = Inches(11.69), Inches(8.27)          # A4
        s.left_margin = s.right_margin = s.top_margin = s.bottom_margin = Inches(1)
        st = d.styles["Normal"]
        st.font.name = FONT; st.font.size = Pt(12)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        if demo:
            hp = s.header.paragraphs[0]
            add_runs(hp, "DEMO DATA - NOT REAL RESULTS - DO NOT SUBMIT", 10, True)
            for r in hp.runs: r.font.color.rgb = RGBColor(200, 0, 0)
            hp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        # page number, bottom centre
        fp = s.footer.paragraphs[0]; fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = fp.add_run(); set_font(r, 10)
        for t, txt in (("begin", None), (None, "PAGE"), ("end", None)):
            if t:
                e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), t)
            else:
                e = OxmlElement("w:instrText"); e.set(qn("xml:space"), "preserve"); e.text = txt
            r._r.append(e)

    def heading(self, text, new_page=True):
        p = self.doc.add_paragraph()
        add_runs(p, text, 13, True)
        fmt(p, WD_ALIGN_PARAGRAPH.LEFT, after=4, before=0 if new_page else 6, keep_next=True)
        p.paragraph_format.page_break_before = new_page
        return p

    def sub(self, text):
        p = self.doc.add_paragraph(); add_runs(p, text, 12, True)
        fmt(p, WD_ALIGN_PARAGRAPH.LEFT, after=2, before=4, keep_next=True)

    def para(self, text, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=12, after=4, italic=False):
        p = self.doc.add_paragraph(); add_runs(p, text, size, italic=italic); fmt(p, align, after)
        return p

    def bullet(self, text, size=12):
        p = self.doc.add_paragraph(style="List Bullet"); add_runs(p, text, size); fmt(p, after=1)
        p.paragraph_format.left_indent = Inches(0.25)

    def caption(self, text, align=WD_ALIGN_PARAGRAPH.CENTER, keep_next=False):
        p = self.doc.add_paragraph(); add_runs(p, text, 10, italic=False); fmt(p, align, after=4, before=2, keep_next=keep_next)
        return p

    def table(self, header, rows, widths, bold_cells=(), size=10, align_first_left=True):
        t = self.doc.add_table(rows=1, cols=len(header)); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        def fill(cell, text, bold=False, head=False, left=False):
            cell.text = ""
            p = cell.paragraphs[0]; add_runs(p, str(text), size, bold=bold)
            p.paragraph_format.space_after = Pt(0); p.paragraph_format.line_spacing = 1.0
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if left else WD_ALIGN_PARAGRAPH.CENTER
            if head:
                tcPr = cell._tc.get_or_add_tcPr(); sh = OxmlElement("w:shd")
                sh.set(qn("w:val"), "clear"); sh.set(qn("w:fill"), "E7E6E6"); tcPr.append(sh)
        for j, h in enumerate(header):
            fill(t.rows[0].cells[j], h, True, True, left=(j == 0 and align_first_left))
        for i, r in enumerate(rows):
            cells = t.add_row().cells
            for j, v in enumerate(r):
                fill(cells[j], v, bold=((i, j) in bold_cells), left=(j == 0 and align_first_left))
        lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); t._tbl.tblPr.append(lay)
        for j, w in enumerate(widths):
            t.columns[j].width = Inches(w)
        for r in t.rows:
            for j, w in enumerate(widths):
                r.cells[j].width = Inches(w)
        return t

    def image(self, path, width, align=WD_ALIGN_PARAGRAPH.CENTER, keep_next=True):
        p = self.doc.add_paragraph(); p.alignment = align
        p.paragraph_format.space_after = Pt(0); p.paragraph_format.keep_with_next = keep_next
        p.add_run().add_picture(str(path), width=Inches(width))

    def save(self, path):
        self.doc.core_properties.title = cfg.PROJECT_TITLE
        self.doc.save(str(path))


# =============================================================== load results
def load(out):
    out = Path(out)
    R = dict(
        cls=pd.read_csv(out / "results_classification.csv"), reg=pd.read_csv(out / "results_regression.csv"),
        stats=json.load(open(out / "dataset_stats.json")), meta=json.load(open(out / "run_meta.json")),
        abl=json.load(open(out / "ablations.json")), pc=pd.read_csv(out / "per_class_report.csv", index_col=0),
        se=pd.read_csv(out / "size_class_errors.csv"),
        ic=pd.read_csv(out / "importance_classification.csv", index_col=0),
        ir=pd.read_csv(out / "importance_regression.csv", index_col=0), fig=out / "figures")
    return R


LEGACY = False   # set in main() from the run statistics


def ranked(df, key, desc=True):
    d = cand(df).sort_values(key, ascending=not desc)
    return [(short(r["model"]), r[key]) for _, r in d.iterrows()]


def join_ranked(lst, f=f3):
    return "; ".join(f"{n} ({f(v)})" for n, v in lst)


# =============================================================== SECTION 1: title page + introduction
def abstract_text(R):
    c, r, m, S = R["cls"], R["reg"], R["meta"], R["stats"]
    bc, br = row(c, m["best_classifier"]), row(r, m["best_regressor"])
    bcb, brb = base(c), base(r)
    times = pd.concat([cand(c)["train_s"], cand(r)["train_s"]])
    d_f1, r2 = bc["test_f1"] - bcb["test_f1"], br["test_r2"]
    strength = "limited" if (d_f1 < 0.05 and r2 < 0.05) else ("useful" if (d_f1 >= 0.15 or r2 >= 0.2) else "moderate")
    sp = S["splits"]
    return (f"Wildfire managers need an early estimate of what started a fire and how large it will become. Using the US "
            f"FPA FOD database ({S['year_min']}-{S['year_max']}), we compared a baseline and four scikit-learn model families on two tasks: "
            f"classifying the cause of a fire ({len(S['cls']['classes'])} classes) and predicting its size on the log scale. "
            f"All models used the same time-based split (training {sp['train_years'][0]}-{sp['train_years'][1]}, test "
            f"{sp['test_years'][0]}-{sp['test_years'][1]}), the same preprocessing and equal tuning effort. "
            f"{m['best_classifier']} was selected as best classifier (test macro F1 {f3(bc['test_f1'])}; baseline "
            f"{f3(bcb['test_f1'])}) and {m['best_regressor']} as best regressor (test R² {f3(br['test_r2'])}; baseline "
            f"{f3(brb['test_r2'])}). Training time ranged from {secs(times.min())} s to {secs(times.max())} s. "
            f"A random split gave macro F1 {f3(R['abl']['classification_random_split']['f1'])} against "
            f"{f3(bc['test_f1'])} for the time split. Discovery-time information alone gives {strength} predictive power.")


def section1(rep, R):
    p = rep.doc.add_paragraph(); add_runs(p, cfg.PROJECT_TITLE, 15, True); fmt(p, WD_ALIGN_PARAGRAPH.CENTER, after=2)
    p = rep.doc.add_paragraph(); add_runs(p, cfg.COURSE, 12); fmt(p, WD_ALIGN_PARAGRAPH.CENTER, after=0)
    p = rep.doc.add_paragraph(); add_runs(p, cfg.UNIVERSITY, 12); fmt(p, WD_ALIGN_PARAGRAPH.CENTER, after=4)
    rows = [(n, reg) for n, reg in cfg.GROUP_MEMBERS]
    rep.table(["Group member", "Registration number"], rows, [3.3, 2.5], size=11)
    rep.heading("1. Title Page and Introduction", new_page=False)
    rep.para("**Problem statement.** When a wildfire is reported, two questions matter early: what started it, and how large "
             "will it become? The likely cause shows whether prevention should target human activity or natural ignition, and "
             "the expected size helps agencies decide how many resources to send. Using only information known at discovery "
             "(location, date and time, land owner and reporting agency), we study two supervised-learning problems on the US "
             "Fire Program Analysis Fire-Occurrence Database (FPA FOD) [1]: (1) classifying the general cause of a fire "
             "(multi-class classification) and (2) predicting its final size in acres (regression on the log scale).")
    rep.para("**Objective of the comparison.** This report compares a baseline and four model families from scikit-learn [3] "
             "(Logistic/Ridge Regression, Decision Tree, Random Forest and Histogram Gradient Boosting) under identical "
             "conditions: the same time-based split, the same preprocessing and the same tuning effort. The models are judged on "
             "accuracy or error, speed and generalization, and every conclusion is tied to numbers from our results.")
    ab = abstract_text(R)
    n = len(ab.split())
    print(f"Abstract length: {n} words (limit 150)")
    if n > 150:
        print("  WARNING: abstract is over 150 words - shorten it in Word before submitting.")
    rep.para("**Abstract.** " + ab)


# =============================================================== SECTION 2: dataset
def section2(rep, R):
    S, sp = R["stats"], R["stats"]["splits"]
    K = len(S["cls"]["classes"])
    n_feat = sum(len(v) for v in S["features"].values())
    rep.heading("2. Dataset Description")
    edition = ("4th edition (file FPA_FOD_20170508), 1992-2015" if LEGACY else "7th edition (file FPA_FOD_20260615)")
    doi = "10.2737/RDS-2013-0009.4" if LEGACY else "10.2737/RDS-2013-0009.7"
    rep.para(f"**Source.** We use the Fire Program Analysis Fire-Occurrence Database (FPA FOD), {edition}, "
             f"published by the USDA Forest Service Research Data Archive [1] (DOI {doi}; "
             f"other editions are described in [2]). It holds {S['n_raw']:,} geo-referenced wildfire records for "
             f"{S['year_min']}-{S['year_max']}; {S['n_clean']:,} remain after cleaning. Each record has 10 raw input columns "
             f"(location, discovery year/day/time, state, county code, land owner, reporting agency, source type), expanded to "
             f"{n_feat} engineered features. **Targets:** {S.get("cause_source", "NWCG_GENERAL_CAUSE")} (classification, {K} classes after preprocessing; "
             f"{S['cls']['n_labeled']:,} fires have a usable cause) and FIRE_SIZE in acres (regression).")
    # class distribution table (two halves side by side)
    cl = sorted(S["cls"]["classes"], key=lambda k: -S["cls"]["share_train"].get(k, 0))
    half = (len(cl) + 1) // 2
    rows = []
    for i in range(half):
        a = cl[i]; b = cl[i + half] if i + half < len(cl) else None
        rows.append([a, f"{S['cls']['share_train'].get(a, 0):.1f}", f"{S['cls']['share_test'].get(a, 0):.1f}"] +
                    ([b, f"{S['cls']['share_train'].get(b, 0):.1f}", f"{S['cls']['share_test'].get(b, 0):.1f}"] if b else ["", "", ""]))
    rep.caption("**Table 1.** Class distribution of the cause target (% of fires in the training sample and in the test years).",
                keep_next=True)
    rep.table(["Cause", "Train %", "Test %", "Cause", "Train %", "Test %"], rows, [1.75, 0.55, 0.5, 1.75, 0.55, 0.5], size=9)
    g = S["reg"]
    rep.para(f"**Class imbalance and size statistics.** The most common cause is \"{S['cls']['majority_class']}\" "
             f"({S['cls']['majority_share_test']:.1f}% of test fires), so accuracy alone would be misleading and macro-averaged "
             f"scores are used. FIRE_SIZE is very skewed: median {g['median']:.2f} acres, mean {g['mean']:.1f}, maximum "
             f"{g['max']:,.0f}; {g['pct_class_A']:.1f}% of fires are size class A (0.25 acres or less), {g['pct_ge_100']:.1f}% "
             f"reach 100 acres and {g['pct_ge_5000']:.2f}% reach 5,000 acres. The model therefore predicts ln(1 + acres) "
             f"(mean {g['log_mean']:.2f}, SD {g['log_std']:.2f}), converted back to acres for reporting.", after=3)
    merged = S["cls"]["merged_classes"]
    mtxt = (f"causes below {cfg.MIN_CLASS_SHARE * 100:.1f}% of training fires ({'; '.join(merged)}) were merged into "
            f"\"{cfg.RARE_CLASS_LABEL}\"" if merged else
            f"no cause was below the {cfg.MIN_CLASS_SHARE * 100:.1f}% merge threshold, so no classes were merged")
    rep.para("**Preprocessing (identical for all models).**", after=1)
    rep.bullet(f"Cleaning: {S['n_duplicates_removed']:,} duplicate IDs and {S['n_dropped_missing']:,} rows with missing "
               f"coordinates, date or size were removed. Rows without a cause ({S['cls']['n_unlabeled']:,}) are removed for "
               f"classification only; {mtxt}.")
    rep.bullet(f"Missing values: owner, agency, state and county set to \"Unknown\"; missing discovery hour "
               f"({S['hour_missing_pct']:.1f}% of fires) filled with the training median plus a missing-flag.")
    rep.bullet("Features: month, weekday, season, hour; day of year as sine/cosine; county code replaced by the log of its "
               "training frequency. Columns that leak the answer (cause class, cause age, size class, containment dates, IDs) "
               "were excluded; FIRE_YEAR is used only to split the data.")
    rep.bullet("Encoding and scaling: one-hot encoding for state, owner, agency, source type and season (categories under 0.5% "
               "grouped); standard scaling only for the linear models.")
    c, r = sp["cls"], sp["reg"]
    rep.para("**Train / validation / test split (by time).** Models are fitted on earlier years, the best model is chosen on "
             "the next years, and the most recent years are used once for the final test.", after=2)
    rows = [["Train", f"{sp['train_years'][0]}-{sp['train_years'][1]}", f"{c['train_pop']:,}", f"{c['train_n']:,}", f"{r['train_pop']:,}", f"{r['train_n']:,}"],
            ["Validation", f"{sp['val_years'][0]}-{sp['val_years'][1]}", f"{c['val_pop']:,}", f"{c['val_n']:,}", f"{r['val_pop']:,}", f"{r['val_n']:,}"],
            ["Test", f"{sp['test_years'][0]}-{sp['test_years'][1]}", f"{c['test_pop']:,}", f"{c['test_n']:,}", f"{r['test_pop']:,}", f"{r['test_n']:,}"]]
    rep.caption("**Table 2.** Time-based split: fires available and fires used (a stratified sample is used for training).",
                keep_next=True)
    rep.table(["Set", "Fire years", "Cause: available", "Cause: used", "Size: available", "Size: used"], rows,
              [0.9, 0.95, 1.05, 1.0, 1.05, 0.95], size=9)


# =============================================================== SECTION 3: models and setup
def section3(rep, R):
    S, meta, c, r = R["stats"], R["meta"], R["cls"], R["reg"]
    sp = S["splits"]
    rep.heading("3. Models and Experimental Setup")
    def sel(name_c, name_r):
        return params_str(row(c, name_c)["best_params"]), params_str(row(r, name_r)["best_params"])
    lr = sel("Logistic Regression", "Ridge Regression"); dt = sel("Decision Tree", "Decision Tree")
    rf = sel("Random Forest", "Random Forest"); hg = sel("Hist Gradient Boosting", "Hist Gradient Boosting")
    rows = [
        ["Baseline", "Predicts the most common class (cause) or the training mean of ln(1+size).", "none", "-", "-"],
        ["Logistic / Ridge Regression", "Linear models: a weighted sum of the features (Ridge adds an L2 penalty).",
         "C ∈ {0.01, 0.1, 1, 10} (Logistic); alpha ∈ {0.1, 1, 10, 100} (Ridge)", lr[0], lr[1]],
        ["Decision Tree", "One tree of if/else splits on the features; easy to read.",
         "max_depth ∈ {6, 10, 14, 18}; min_samples_leaf = 10", dt[0], dt[1]],
        ["Random Forest", "100 deep trees on bootstrap samples; predictions are averaged (bagging) [4].",
         "max_depth ∈ {8, 12, 16, 20}; min_samples_leaf = 5", rf[0], rf[1]],
        ["Hist. Gradient Boosting", "Trees added one by one, each correcting the errors of the previous ones [5].",
         "learning_rate ∈ {0.03, 0.1} × max_iter ∈ {100, 200}", hg[0], hg[1]],
    ]
    rep.caption("**Table 3.** Models, hyperparameters searched and values selected (cause classifier / size regressor).",
                keep_next=True)
    rep.table(["Model", "One-line description", "Grid searched (4 candidates)", "Selected: cause", "Selected: size"],
              rows, [1.0, 1.6, 1.5, 1.08, 1.09], size=9)
    rep.para(f"**Tuning method.** For each model we ran a grid search of exactly 4 candidates with 3-fold time-ordered "
             f"cross-validation (TimeSeriesSplit) on a {sp['cls']['tune_n']:,}-row subset of the training years, scored by macro F1 "
             f"(cause) or mean squared error (size). Each model therefore got the same tuning effort. The test years were never "
             f"used for tuning. The chosen settings were refitted on the full training sample ({sp['cls']['train_n']:,} fires for "
             f"cause, {sp['reg']['train_n']:,} for size). Balanced class weights were used for all cause models. The best model of "
             f"each task is selected on the validation years, then evaluated once on the test years.", after=4)
    rep.para("**Evaluation metrics.** Cause (classification): accuracy and macro-averaged precision, recall and F1-score "
             "(macro F1 is the main score because classes are unequal). Size (regression): MAE, MSE, RMSE and R² on the "
             "ln(1+acres) scale, plus the median absolute error in acres. Speed: wall-clock training time of the final fit and "
             f"prediction time for the whole test set. Timings were measured on a machine with {meta['cpu_count']} CPU cores "
             f"(Python {meta['python']}, scikit-learn {meta['sklearn']}).", after=4)
    if LEGACY:
        rep.para("**Dataset edition (change from the proposal).** The proposal named the 7th edition (1992-2024), but its "
                 "download was blocked from our network, so we used the 4th edition (FPA_FOD_20170508, 1992-2015). Consequences: "
                 f"the time-based split became training {sp['train_years'][0]}-{sp['train_years'][1]}, validation "
                 f"{sp['val_years'][0]}-{sp['val_years'][1]} and test {sp['test_years'][0]}-{sp['test_years'][1]}; the cause "
                 "target is STAT_CAUSE_DESCR (the cause list used before the 2020 NWCG standard); and the NWCG cause "
                 "classification columns of the 7th edition do not exist. The method is otherwise unchanged.", after=4)
    rep.para("**Changes from the Assignment 02 proposal (and why).** (1) The county code has thousands of categories, so it is "
             "frequency-encoded instead of one-hot encoded. (2) The discovery date is rebuilt from FIRE_YEAR and day of year "
             "(same information, no parsing errors). (3) Grid search runs on a subset of the training years to keep run time "
             "practical. (4) The validation years (" + f"{sp['val_years'][0]}-{sp['val_years'][1]}" + ") were fixed here, as the proposal did not specify them. "
             "(5) Two extra checks were added: a random split and ablations (without owner/agency; size model with cause as input).")


# =============================================================== SECTION 4: results
def bold_positions(df, max_cols, min_cols, col_index):
    """(row, col) positions of the best non-baseline value of each metric column, to print in bold."""
    marks = set()
    nb = [i for i, m in enumerate(df["model"]) if not m.startswith("Baseline")]
    for key, j in col_index.items():
        vals = df.iloc[nb][key]
        best_i = vals.idxmax() if key in max_cols else vals.idxmin()
        marks.add((best_i, j))
    return marks


def two_figures(rep, left, right, cap_l, cap_r, height=2.9):
    t = rep.doc.add_table(rows=1, cols=2); t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for cell, img, cap in ((t.rows[0].cells[0], left, cap_l), (t.rows[0].cells[1], right, cap_r)):
        p = cell.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0); p.add_run().add_picture(str(img), height=Inches(height))
        cp = cell.add_paragraph(); add_runs(cp, cap, 10); cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cp.paragraph_format.space_after = Pt(2); cp.paragraph_format.line_spacing = 1.0


def section4(rep, R):
    c, r, m, S = R["cls"], R["reg"], R["meta"], R["stats"]
    bc, br = m["best_classifier"], m["best_regressor"]
    rep.heading("4. Results")
    rep.para(f"Tables 4 and 5 give the test-set results of every model (best non-baseline value in bold). Test years: "
             f"{S['splits']['test_years'][0]}-{S['splits']['test_years'][1]}; {S['splits']['cls']['test_n']:,} fires for cause "
             f"and {S['splits']['reg']['test_n']:,} for size. Precision, recall and F1 are macro-averaged over the classes. "
             f"Times are in seconds: training = final fit; prediction = whole test set. The best models chosen on the "
             f"validation years are {bc} (cause) and {br} (size).", after=3)
    # Table 4 classification
    rows = [[short(x["model"]), f3(x["test_accuracy"]), f3(x["test_precision"]), f3(x["test_recall"]), f3(x["test_f1"]),
             secs(x["train_s"]), secs(x["predict_s"])] for _, x in c.iterrows()]
    marks = bold_positions(c, {"test_accuracy", "test_precision", "test_recall", "test_f1"}, {"train_s", "predict_s"},
                           {"test_accuracy": 1, "test_precision": 2, "test_recall": 3, "test_f1": 4, "train_s": 5, "predict_s": 6})
    rep.caption("**Table 4.** Cause classification, test set.", keep_next=True)
    rep.table(["Model", "Accuracy", "Precision", "Recall", "F1", "Train (s)", "Predict (s)"], rows,
              [1.75, 0.7, 0.7, 0.65, 0.6, 0.65, 0.75], bold_cells=marks, size=9)
    # Table 5 regression
    rows = [[short(x["model"]), f3(x["test_mae"]), f3(x["test_mse"]), f3(x["test_rmse"]), f3(x["test_r2"]),
             f"{x['test_medae_acres']:.2f}", secs(x["train_s"]), secs(x["predict_s"])] for _, x in r.iterrows()]
    marks = bold_positions(r, {"test_r2"}, {"test_mae", "test_mse", "test_rmse", "test_medae_acres", "train_s", "predict_s"},
                           {"test_mae": 1, "test_mse": 2, "test_rmse": 3, "test_r2": 4, "test_medae_acres": 5, "train_s": 6, "predict_s": 7})
    rep.caption("**Table 5.** Fire size regression, test set (MAE, MSE, RMSE, R² on the ln(1+acres) scale; median absolute "
                "error in acres).", keep_next=True)
    rep.table(["Model", "MAE", "MSE", "RMSE", "R²", "Median AE (acres)", "Train (s)", "Predict (s)"], rows,
              [1.6, 0.55, 0.55, 0.55, 0.55, 0.85, 0.65, 0.7], bold_cells=marks, size=9)
    rep.para("", after=2)
    two_figures(rep, R["fig"] / "fig1_confusion_matrix.png", R["fig"] / "fig2_pred_vs_actual.png",
                f"**Figure 1.** Confusion matrix, {bc} (cause).", f"**Figure 2.** Predicted vs actual size, {br}.")
    # figure text: evidence from the per-class report and size-class errors
    pc = R["pc"].drop(index=[i for i in ("accuracy", "macro avg", "weighted avg") if i in R["pc"].index])
    hi, lo = pc["recall"].idxmax(), pc["recall"].idxmin()
    txt = (f"Figure 1 shows that {bc} recognises \"{hi}\" best (recall {pc.loc[hi, 'recall']:.2f}) and \"{lo}\" worst "
           f"(recall {pc.loc[lo, 'recall']:.2f}); the diagonal is where predicted and actual causes agree. ")
    se = R["se"][R["se"]["n"] >= 30].sort_values("size_class")
    first, last = se.iloc[0], se.iloc[-1]
    txt += (f"In Figure 2 the mean error (predicted minus actual, log scale) is {first['mean_log_error']:+.2f} for size class "
            f"{first['size_class']} and {last['mean_log_error']:+.2f} for class {last['size_class']} ({int(last['n']):,} fires)")
    if first["mean_log_error"] > 0 and last["mean_log_error"] < 0:
        txt += ": small fires are overestimated and the largest fires are underestimated, so predictions are pulled towards the mean."
    else:
        txt += "."
    rep.para(txt, after=3)
    rep.image(R["fig"] / "fig3_generalization.png", 6.2)
    rep.caption("**Figure 3.** Macro F1 (left) and R² (right) of each model on training, validation and test data.")


# =============================================================== SECTION 5: analysis
def gap_text(df, key, label, best_name):
    d = cand(df).copy()
    d["gap"] = d[f"train_{key}"] - d[f"test_{key}"]
    big = d.sort_values("gap", ascending=False).iloc[0]
    bestrow = row(df, best_name)
    s = ""
    if big["gap"] > 0.10:
        s += (f"{short(big['model'])} has the largest train-test gap ({label} {f3(big['train_' + key])} on training data vs "
              f"{f3(big['test_' + key])} on test data, gap {f3(big['gap'])}), which indicates overfitting (high variance). ")
    else:
        s += f"No model has a train-test gap above 0.10 in {label} (largest: {short(big['model'])}, {f3(big['gap'])}). "
    under = d[(d[f"train_{key}"] < bestrow[f"test_{key}"]) & (d["gap"] < 0.05)]
    for _, u in under.iterrows():
        s += (f"{short(u['model'])} has a small gap ({f3(u['gap'])}) but its training {label} ({f3(u['train_' + key])}) is below "
              f"the best model's test {label} ({f3(bestrow['test_' + key])}), a sign of underfitting (high bias). ")
    dv = bestrow[f"test_{key}"] - bestrow[f"val_{key}"]
    s += (f"For {short(best_name)}, {label} goes from {f3(bestrow['val_' + key])} on the validation years to "
          f"{f3(bestrow['test_' + key])} on the test years ({dv:+.3f})")
    s += (", which suggests that performance changes from period to period" +
          ("." if LEGACY else " (for example because of the August 2020 update of the cause standard).") if dv < -0.02 else ".")
    return s


def speed_text(df, task_word, best_name):
    d = cand(df)
    fast, slow = d.sort_values("train_s").iloc[0], d.sort_values("train_s").iloc[-1]
    b = row(df, best_name)
    s = (f"For {task_word}, training takes {secs(fast['train_s'])} s for {short(fast['model'])} and {secs(slow['train_s'])} s "
         f"for {short(slow['model'])}; prediction on {int(b['n_test']):,} test fires takes {secs(d['predict_s'].min())}-"
         f"{secs(d['predict_s'].max())} s. ")
    key = "test_f1" if "test_f1" in df else "test_r2"
    name = "macro F1" if key == "test_f1" else "R²"
    if best_name != fast["model"] and fast["train_s"] > 0:
        s += (f"The selected model ({short(best_name)}) needs {b['train_s'] / fast['train_s']:.1f} times the training time of "
              f"the fastest model and changes test {name} by {b[key] - fast[key]:+.3f} ({f3(fast[key])} to {f3(b[key])}). ")
    return s


def why_text(df, best_name, task, metric_key, metric_name):
    b = row(df, best_name); fam = family(best_name)
    d = cand(df).set_index("model")
    gap = lambda n: d.loc[n, f"train_{metric_key}"] - d.loc[n, f"test_{metric_key}"]
    lin = [n for n in d.index if family(n) == "linear"][0]
    dt = "Decision Tree"; rf = "Random Forest"; hg = "Hist Gradient Boosting"
    pr = params_str(b["best_params"])
    inter = "non-linear effects and interactions between location, season and land owner"
    if fam == "boosting":
        t = (f"**{task}: why {short(best_name)} did best.** Gradient boosting adds small trees one after another, each fitted to "
             f"the errors of the previous ones, which mainly reduces bias; the learning rate and number of iterations ({pr}), "
             f"chosen by time-ordered cross-validation, keep variance in check. It can capture {inter}, which a linear model "
             f"cannot represent without hand-made interaction terms ({short(lin)} test {metric_name} "
             f"{f3(d.loc[lin, 'test_' + metric_key])} vs {f3(b['test_' + metric_key])}). Compared with Random Forest "
             f"({f3(d.loc[rf, 'test_' + metric_key])}), the difference is {b['test_' + metric_key] - d.loc[rf, 'test_' + metric_key]:+.3f}.")
    elif fam == "bagging":
        t = (f"**{task}: why {short(best_name)} did best.** A Random Forest averages many deep trees trained on bootstrap samples "
             f"with random feature subsets. Averaging lowers variance, so it keeps the ability of trees to model {inter} while "
             f"overfitting less than a single tree: its train-test gap is {f3(gap(rf))} against {f3(gap(dt))} for the Decision "
             f"Tree. The linear model cannot represent these interactions without extra terms, which fits its lower test "
             f"{metric_name} ({f3(d.loc[lin, 'test_' + metric_key])} vs {f3(b['test_' + metric_key])}).")
    elif fam == "tree":
        t = (f"**{task}: why {short(best_name)} did best.** A single tree models non-linear rules and interactions "
             f"(for example a state combined with a month), which a linear model cannot ({short(lin)}: "
             f"{f3(d.loc[lin, 'test_' + metric_key])}). Tuning limited its complexity ({pr}), which controls variance. The "
             f"ensembles did not beat it (Random Forest {f3(d.loc[rf, 'test_' + metric_key])}, Hist. Gradient Boosting "
             f"{f3(d.loc[hg, 'test_' + metric_key])}), which suggests that the signal in the discovery-time features is limited "
             f"and that extra capacity mostly fits noise.")
    else:
        others = d.drop(index=best_name)
        t = (f"**{task}: why {short(best_name)} did best.** The linear model was best (test {metric_name} "
             f"{f3(b['test_' + metric_key])}) and has a train-test gap of only {f3(gap(best_name))}, i.e. low variance. The "
             f"more flexible models fit the training years better but their largest train-test gap is "
             f"{f3(max(gap(n) for n in others.index))}, which suggests that their extra capacity did not generalize to the test "
             f"years and that the relations are largely additive after our encoding.")
    return t


def section5(rep, R):
    c, r, m, S, A = R["cls"], R["reg"], R["meta"], R["stats"], R["abl"]
    bc, br = m["best_classifier"], m["best_regressor"]
    cb, rb, cbase, rbase = row(c, bc), row(r, br), base(c), base(r)
    rep.heading("5. Comparative Analysis and Discussion")
    rep.sub("5.1 Accuracy and error")
    top_c = ranked(c, "test_f1")[0][0]
    ratio = cb["test_f1"] / cbase["test_f1"] if cbase["test_f1"] > 0 else None
    ratio_txt = f" ({ratio:.1f} times)" if ratio else ""
    t = (f"**Cause.** Ranked by test macro F1: {join_ranked(ranked(c, 'test_f1'))}; baseline {f3(cbase['test_f1'])}. The selected "
         f"model ({short(bc)}) improves macro F1 by {cb['test_f1'] - cbase['test_f1']:+.3f} over the baseline{ratio_txt}. ")
    if top_c != short(bc):
        t += f"The highest test F1 belongs to {top_c}, but {short(bc)} was chosen on validation data, not on the test set. "
    t += (f"The baseline always predicts \"{S['cls']['majority_class']}\": accuracy {f3(cbase['test_accuracy'])}, macro F1 "
          f"{f3(cbase['test_f1'])}. {short(bc)} reaches accuracy {f3(cb['test_accuracy'])} and macro F1 {f3(cb['test_f1'])}. ")
    if cb["test_accuracy"] < cbase["test_accuracy"]:
        t += ("Its accuracy is lower than the baseline's, which is consistent with balanced class weights trading accuracy on "
              "common causes for recall of rare causes; accuracy alone would favour the baseline although its macro F1 is lower. ")
    if cb["test_accuracy"] - cb["test_f1"] > 0.02:
        t += (f"Accuracy exceeds macro F1 by {cb['test_accuracy'] - cb['test_f1']:.3f}, the effect of class imbalance (Table 1): "
              f"common causes are predicted better than rare ones. ")
    rep.para(t)
    t = (f"**Size.** Ranked by test RMSE on the log scale: {join_ranked(ranked(r, 'test_rmse', desc=False))}; baseline "
         f"{f3(rbase['test_rmse'])}. {short(br)} reaches R² = {f3(rb['test_r2'])}, i.e. it explains {rb['test_r2'] * 100:.1f}% of "
         f"the variance of ln(1+acres) (baseline R² {f3(rbase['test_r2'])}). In acres, the median absolute error is "
         f"{rb['test_medae_acres']:.2f} but the RMSE is {rb['test_rmse_acres']:,.0f}, because a few very large fires dominate "
         f"squared errors.")
    rep.para(t)
    rep.sub("5.2 Speed")
    rep.para(speed_text(c, "cause classification", bc) + speed_text(r, "size regression", br) +
             f"Grid search (4 candidates x {m['cv_splits']} folds per model) is an extra cost paid once: "
             f"{c['tune_s'].sum() + r['tune_s'].sum():,.0f} s in total for both tasks.")
    rep.sub("5.3 Generalization: overfitting and underfitting")
    rep.para(f"Table 6 and Figure 3 compare training, validation and test scores. "
             f"**Cause.** {gap_text(c, 'f1', 'macro F1', bc)} **Size.** {gap_text(r, 'r2', 'R²', br)}", after=3)
    rows = []
    for (_, x), (_, y) in zip(c.iterrows(), r.iterrows()):
        nm = "Baseline" if x["model"].startswith("Baseline") else (
            "Logistic / Ridge" if family(x["model"]) == "linear" else short(x["model"]))
        rows.append([nm, f3(x["train_f1"]), f3(x["val_f1"]), f3(x["test_f1"]), f3(y["train_r2"]), f3(y["val_r2"]), f3(y["test_r2"])])
    rep.caption("**Table 6.** Macro F1 (cause) and R² (size) on training, validation and test data.", keep_next=True)
    rep.table(["Model", "F1 train", "F1 val", "F1 test", "R² train", "R² val", "R² test"], rows,
              [1.7, 0.75, 0.7, 0.7, 0.8, 0.7, 0.7], size=9)
    rs_c, rs_r = A["classification_random_split"], A["regression_random_split"]
    noa_c, noa_r, wc = A["classification_without_owner_agency"], A["regression_without_owner_agency"], A["regression_with_cause"]
    d_c = rs_c["f1"] - cb["test_f1"]
    extra = (f"**Extra checks (best models, same settings).** With a random split, {short(bc)} scores macro F1 "
             f"{f3(rs_c['f1'])} versus {f3(cb['test_f1'])} on the time-based test years ({d_c:+.3f}), and {short(br)} scores R² "
             f"{f3(rs_r['r2'])} versus {f3(rb['test_r2'])} ({rs_r['r2'] - rb['test_r2']:+.3f}); "
             f"{'a random split therefore gives a more optimistic picture of future performance' if d_c > 0 or rs_r['r2'] > rb['test_r2'] else 'a random split did not give a more optimistic picture here'}. "
             f"Removing owner and agency changes macro F1 by {noa_c['f1'] - cb['test_f1']:+.3f} and R² by "
             f"{noa_r['r2'] - rb['test_r2']:+.3f}. Adding the cause as an input to the size model changes R² by "
             f"{wc['r2'] - rb['test_r2']:+.3f} (RMSE {rb['test_rmse']:.3f} to {wc['rmse']:.3f}).")
    rep.para(extra, after=3)
    rep.sub("5.4 Why the best models performed better")
    rep.para(why_text(c, bc, "Cause", "f1", "macro F1"))
    rep.para(why_text(r, br, "Size", "r2", "R²"))
    top3c = ", ".join(R["ic"]["mean"].sort_values(ascending=False).index[:3])
    top3r = ", ".join(R["ir"]["mean"].sort_values(ascending=False).index[:3])
    rep.para(f"Permutation importance (Figure 4) ranks {top3c} highest for the cause model and {top3r} for the size model.", after=2)
    rep.image(R["fig"] / "fig4_permutation_importance.png", 5.4)
    rep.caption("**Figure 4.** Permutation importance of the input features on the test set (best models).")
    rep.sub("5.5 Limitations")
    rep.para(f"(1) The data has no weather, drought or vegetation information, which strongly affects fire behaviour, so "
             f"scores are bounded by what location and calendar can tell. (2) Cause labels may be applied differently by different "
             f"agencies, " + ("the cause list is the older pre-2020 one" if LEGACY else "the cause standard changed in August 2020 (inside our validation and test years)") + f", and "
             f"{S['cls']['n_unlabeled']:,} fires without a cause are excluded from the cause task. (3) Models were fitted on "
             f"a sample of {S['splits']['cls']['train_n']:,} fires, tuned with small 4-candidate grids on a subset, and run once "
             f"(one random seed, no confidence intervals), so small differences between models should not be over-interpreted. "
             f"(4) Timings depend on the hardware ({m['cpu_count']} CPU cores). (5) The data covers the United States only, so the "
             f"results illustrate a method, not a model for other countries.")


# =============================================================== SECTION 6: conclusion
def level(v, small, mid):
    return "only a small" if v < small else ("a moderate" if v < mid else "a clear")


def section6(rep, R):
    c, r, m = R["cls"], R["reg"], R["meta"]
    bc, br = m["best_classifier"], m["best_regressor"]
    cb, rb, cbase, rbase = row(c, bc), row(r, br), base(c), base(r)
    rep.heading("6. Conclusion, References and Code Link")
    rep.para(f"**Best models.** For the **cause** task the best model is **{short(bc)}**: test macro F1 {f3(cb['test_f1'])} "
             f"(baseline {f3(cbase['test_f1'])}), accuracy {f3(cb['test_accuracy'])}, validation macro F1 {f3(cb['val_f1'])}, "
             f"training time {secs(cb['train_s'])} s. It was chosen on the validation years, and it gives "
             f"{level(cb['test_f1'] - cbase['test_f1'], 0.05, 0.15)} improvement over the baseline. For the **size** task the best "
             f"model is **{short(br)}**: test RMSE {f3(rb['test_rmse'])} on the log scale (baseline {f3(rbase['test_rmse'])}), "
             f"R² {f3(rb['test_r2'])}, median absolute error {rb['test_medae_acres']:.2f} acres, training time "
             f"{secs(rb['train_s'])} s; this is {level(rb['test_r2'], 0.05, 0.2)} improvement over predicting the mean. "
             f"Details and the generalization evidence are in Sections 4 and 5.")
    rep.para("**Future improvements.** (1) Add weather, drought and vegetation data for the discovery date and place, the "
             "largest missing information. (2) Train on the full database and use larger or randomised hyperparameter searches. "
             "(3) Handle class imbalance further (resampling, per-class thresholds) and calibrate the predicted probabilities. "
             "(4) Treat very large fires separately (two-stage or quantile models). (5) Repeat the runs with several random seeds "
             "and report confidence intervals.", after=4)
    rep.para("**References**", after=1)
    ref1 = ("[1] K. C. Short, \"Spatial wildfire occurrence data for the United States, 1992-2015 [FPA_FOD_20170508], 4th ed.,\" "
            "Forest Service Research Data Archive, Fort Collins, CO, USA, 2017. doi: 10.2737/RDS-2013-0009.4.") if LEGACY else (
            "[1] K. C. Short, \"Spatial wildfire occurrence data for the United States, 1992-2024 [FPA_FOD_20260615], 7th ed.,\" "
            "Forest Service Research Data Archive, Fort Collins, CO, USA, 2026. doi: 10.2737/RDS-2013-0009.7.")
    refs = [
        ref1,
        "[2] K. C. Short, \"A spatial database of wildfires in the United States, 1992-2011,\" Earth Syst. Sci. Data, vol. 6, "
        "pp. 1-27, 2014. doi: 10.5194/essd-6-1-2014.",
        "[3] F. Pedregosa et al., \"Scikit-learn: Machine learning in Python,\" J. Mach. Learn. Res., vol. 12, pp. 2825-2830, 2011.",
        "[4] L. Breiman, \"Random forests,\" Mach. Learn., vol. 45, no. 1, pp. 5-32, 2001.",
        "[5] J. H. Friedman, \"Greedy function approximation: A gradient boosting machine,\" Ann. Statist., vol. 29, no. 5, "
        "pp. 1189-1232, 2001.",
    ]
    for x in refs:
        rep.para(x, align=WD_ALIGN_PARAGRAPH.LEFT, after=1)
    rep.para(f"**Code (notebook / repository):** {cfg.NOTEBOOK_LINK}. Running run_experiments.py and then build_report.py "
             f"reproduces every table and figure in this report.", after=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(cfg.OUTPUT_DIR))
    a = ap.parse_args()
    R = load(a.out)
    global LEGACY
    LEGACY = bool(R["stats"].get("legacy", False))
    rep = Report(R["meta"].get("demo", False))
    section1(rep, R); section2(rep, R); section3(rep, R); section4(rep, R); section5(rep, R); section6(rep, R)
    path = Path(a.out) / "Assignment03_Report.docx"
    rep.save(path)
    print(f"Saved {path}. Open it in Word, fill the yellow items, edit the draft sentences, check the page count (max 8), "
          f"then File > Save As > PDF.")


if __name__ == "__main__":
    main()
