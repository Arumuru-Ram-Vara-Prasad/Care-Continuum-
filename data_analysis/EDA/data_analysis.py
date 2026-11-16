"""
data_analysis.py
Reads a CSV or Excel (.xlsx) file, prints missing values + preprocessing
metrics for every feature, and saves a styled HTML report.

Usage (run from the 'Care Continuum' folder):
    python data_analysis.py
    python data_analysis.py dataset/diabetic_data.csv
    python data_analysis.py dataset/diabetic_data.xlsx --report my_report.html
    python data_analysis.py dataset/diabetic_data.csv --save metrics.csv --no-open

Notes:
 - The diabetes 130-US hospitals dataset stores missing values as "?", so "?"
   and empty cells are treated as missing by default (change with --na-values).
   The text "None" is kept as a real category.
 - Charts need matplotlib (pip install matplotlib). Without it the report is
   still created, just without charts.
 - Open the HTML in a browser; use Ctrl+P -> "Save as PDF" for a PDF copy.
"""

import argparse
import base64
import html
import io
import os
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

ACCENT = "#2563eb"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_data(path: str, na_values) -> pd.DataFrame:
    """Read .csv or .xlsx. keep_default_na=False so text like "None"
    (= test not performed in this dataset) stays a real category."""
    if path.lower().endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(path, na_values=na_values, keep_default_na=False)
    return pd.read_csv(path, na_values=na_values, keep_default_na=False, low_memory=False)


def find_default_file() -> str:
    for name in ("dataset/diabetic_data.csv", "dataset/diabetic_data.xlsx"):
        if os.path.exists(name):
            return name
    return "dataset/diabetic_data.csv"


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def numeric_metrics(s: pd.Series) -> dict:
    clean = s.dropna()
    if clean.empty:
        return {}
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = int(((clean < lower) | (clean > upper)).sum())
    return {
        "mean": round(clean.mean(), 4),
        "median": round(clean.median(), 4),
        "std": round(clean.std(), 4),
        "min": clean.min(),
        "max": clean.max(),
        "skew": round(clean.skew(), 4),
        "kurtosis": round(clean.kurt(), 4),
        "zeros": int((clean == 0).sum()),
        "outliers_iqr": outliers,
        "outlier_pct": round(outliers / len(clean) * 100, 2),
    }


def categorical_metrics(s: pd.Series) -> dict:
    clean = s.dropna().astype(str)
    if clean.empty:
        return {}
    counts = clean.value_counts()
    top = ", ".join(f"{k} ({v / len(clean) * 100:.1f}%)" for k, v in counts.head(3).items())
    return {
        "mode": counts.index[0],
        "mode_freq": int(counts.iloc[0]),
        "mode_pct": round(counts.iloc[0] / len(clean) * 100, 2),
        "top_values": top,
    }


def suggest(is_numeric, miss_pct, nunique, n_rows, m):
    """Return a list of (category, text) preprocessing suggestions."""
    tips = []

    if miss_pct == 100 or nunique == 0:
        return [("drop", "Drop column (all values missing)")]
    if miss_pct > 50:
        tips.append(("drop", "Consider dropping (>50% missing)"))
    elif miss_pct > 0:
        if is_numeric:
            skewed = abs(m.get("skew", 0)) > 1
            tips.append(("impute", "Impute with median" if skewed else "Impute with mean"))
        else:
            tips.append(("impute", "Impute with mode / 'Unknown'"))

    if nunique <= 1:
        tips.append(("drop", "Constant column - drop"))
    elif nunique == n_rows:
        tips.append(("drop", "Unique per row - looks like an ID, likely drop"))

    if is_numeric and nunique > 1 and nunique != n_rows:
        if nunique <= 10:
            tips.append(("review", "Few distinct values - may be a categorical code"))
        else:
            if m.get("outlier_pct", 0) > 5:
                tips.append(("outliers", "Handle outliers (clip/winsorize)"))
            if abs(m.get("skew", 0)) > 1:
                tips.append(("transform", "Skewed - try log/Box-Cox transform"))
            tips.append(("scale", "Scale (StandardScaler/MinMax)"))

    if not is_numeric and 1 < nunique <= 15:
        tips.append(("encode", "One-hot encode"))
    elif not is_numeric and nunique > 15 and nunique != n_rows:
        tips.append(("encode", "High cardinality - target/frequency encode"))

    return tips


def profile(df: pd.DataFrame):
    n_rows = len(df)
    rows, actions = [], {}

    for col in df.columns:
        s = df[col]
        is_numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        missing = int(s.isna().sum())
        miss_pct = round(missing / n_rows * 100, 2) if n_rows else 0.0
        nunique = int(s.nunique(dropna=True))

        info = {
            "feature": col,
            "dtype": str(s.dtype),
            "type": "numeric" if is_numeric else "categorical",
            "missing": missing,
            "missing_%": miss_pct,
            "unique": nunique,
        }
        extra = numeric_metrics(s) if is_numeric else categorical_metrics(s)
        info.update(extra)

        acts = suggest(is_numeric, miss_pct, nunique, n_rows, extra)
        actions[col] = acts
        info["suggestion"] = "; ".join(t for _, t in acts) if acts else "No action needed"
        rows.append(info)

    return pd.DataFrame(rows).set_index("feature"), actions


# --------------------------------------------------------------------------
# Charts (matplotlib, embedded in the HTML as base64 PNG)
# --------------------------------------------------------------------------
def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _clean_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8)


def usable_numeric(df, report):
    return [c for c in report.index[report["type"] == "numeric"]
            if report.loc[c, "unique"] > 1 and report.loc[c, "unique"] != len(df)]


def chart_missing(report):
    m = report[report["missing_%"] > 0].sort_values("missing_%").tail(30)
    if m.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, max(2.2, 0.34 * len(m) + 1)))
    colors = ["#dc2626" if v > 50 else "#f59e0b" if v >= 5 else ACCENT for v in m["missing_%"]]
    ax.barh([str(i) for i in m.index], m["missing_%"], color=colors)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Missing (%)", fontsize=9)
    for i, v in enumerate(m["missing_%"]):
        if v > 85:
            ax.text(v - 1, i, f"{v:.1f}%", va="center", ha="right", fontsize=8, color="white")
        else:
            ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=8)
    _clean_axes(ax)
    return fig_to_b64(fig)


def chart_distributions(df, report, max_plots=24):
    cols = usable_numeric(df, report)[:max_plots]
    if not cols:
        return None
    ncols = 4
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 2.5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, cols):
        clean = df[c].dropna()
        if report.loc[c, "unique"] <= 15:
            vc = clean.value_counts().sort_index()
            ax.bar([str(i) for i in vc.index], vc.values, color=ACCENT)
        else:
            ax.hist(clean, bins=30, color=ACCENT, edgecolor="white", linewidth=0.3)
        ax.set_title(str(c), fontsize=9)
        _clean_axes(ax)
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_corr(df, report, max_cols=20):
    cols = usable_numeric(df, report)[:max_cols]
    if len(cols) < 2:
        return None
    corr = df[cols].corr()
    size = max(5, 0.55 * len(cols) + 2)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)
    if len(cols) <= 12:
        for i in range(len(cols)):
            for j in range(len(cols)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig_to_b64(fig)


# --------------------------------------------------------------------------
# HTML report
# --------------------------------------------------------------------------
CSS = """
*{box-sizing:border-box}
body{margin:0;background:#f1f5f9;color:#0f172a;font-family:'Segoe UI',system-ui,-apple-system,Arial,sans-serif;line-height:1.5}
.wrap{max-width:1180px;margin:0 auto;padding:32px 24px 64px}
header{background:linear-gradient(135deg,#1e3a8a,#2563eb);color:#fff;border-radius:16px;padding:32px 36px;margin-bottom:24px}
header h1{margin:0 0 6px;font-size:28px}
header p{margin:0;opacity:.85;font-size:14px}
section{background:#fff;border-radius:14px;padding:24px 28px;margin-bottom:20px;box-shadow:0 1px 3px rgba(15,23,42,.08)}
h2{margin:0 0 4px;font-size:20px}
.sub{color:#64748b;font-size:13px;margin:0 0 16px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:14px;margin-bottom:20px}
.card{background:#fff;border-radius:12px;padding:16px 18px;box-shadow:0 1px 3px rgba(15,23,42,.08);border-top:4px solid #2563eb}
.card .v{font-size:24px;font-weight:700}
.card .l{font-size:12px;color:#64748b;text-transform:uppercase;letter-spacing:.04em}
.card.warn{border-top-color:#f59e0b}.card.bad{border-top-color:#dc2626}.card.good{border-top-color:#16a34a}
ul.findings{margin:0;padding-left:20px}ul.findings li{margin:6px 0}
.tw{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px}
th{background:#f8fafc;text-align:left;padding:9px 10px;border-bottom:2px solid #e2e8f0;white-space:nowrap;position:sticky;top:0}
td{padding:8px 10px;border-bottom:1px solid #eef2f7;vertical-align:top}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
tr:hover td{background:#f8fafc}
td.f{font-weight:600;white-space:nowrap}
.bar{display:inline-block;width:90px;height:8px;background:#e2e8f0;border-radius:4px;vertical-align:middle;margin-right:8px;overflow:hidden}
.bar span{display:block;height:100%}
.fill.ok,.fill.low{background:#2563eb}.fill.mid{background:#f59e0b}.fill.high{background:#dc2626}
.pill{display:inline-block;padding:2px 9px;border-radius:999px;font-size:11.5px;margin:1px 3px 1px 0;white-space:nowrap}
.pill.drop{background:#fee2e2;color:#991b1b}.pill.impute{background:#fef3c7;color:#92400e}
.pill.outliers,.pill.transform{background:#ede9fe;color:#5b21b6}.pill.scale{background:#dbeafe;color:#1e40af}
.pill.encode{background:#dcfce7;color:#166534}.pill.review{background:#e2e8f0;color:#334155}
.plan{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}
.plan .box{border:1px solid #e2e8f0;border-radius:10px;padding:14px 16px}
.plan h3{margin:0 0 2px;font-size:15px}.plan p{margin:0 0 8px;color:#64748b;font-size:12.5px}
.chip{display:inline-block;background:#f1f5f9;border-radius:6px;padding:2px 8px;margin:2px 4px 2px 0;font-size:12px}
img.chart{max-width:100%;height:auto;display:block;margin:0 auto}
.note{color:#64748b;font-size:12.5px}
footer{text-align:center;color:#94a3b8;font-size:12px;margin-top:24px}
@media print{body{background:#fff}section,.card{box-shadow:none;border:1px solid #e2e8f0;break-inside:avoid}th{position:static}}
"""

PLAN = [
    ("drop", "Drop", "Mostly empty, constant, or ID-like columns that add little signal."),
    ("impute", "Impute missing values", "Fill gaps with mean/median (numeric) or mode/'Unknown' (categorical)."),
    ("outliers", "Handle outliers", "More than 5% of values fall outside the IQR fences."),
    ("transform", "Transform skewed features", "Absolute skewness above 1 - log or Box-Cox can help."),
    ("scale", "Scale numeric features", "Standardise or min-max scale before distance/gradient-based models."),
    ("encode", "Encode categories", "One-hot for low cardinality; target/frequency encoding for high cardinality."),
    ("review", "Review manually", "Numeric columns with few distinct values - possibly categorical codes."),
]


def esc(x):
    return html.escape(str(x))


def fmt(v):
    if v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)):
        return "-"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        if v == int(v) and abs(v) < 1e15:
            return f"{int(v):,}"
        return f"{v:,.1f}" if abs(v) >= 1000 else f"{v:.4g}"
    return esc(v)


def sev(p):
    return "ok" if p == 0 else "low" if p < 5 else "mid" if p <= 50 else "high"


def miss_cell(p):
    return (f'<div class="bar"><span class="fill {sev(p)}" style="width:{min(p, 100):.1f}%"></span></div>'
            f"{p:.2f}%")


def pills(acts):
    if not acts:
        return '<span class="pill review">No action needed</span>'
    return "".join(f'<span class="pill {c}">{esc(t)}</span>' for c, t in acts)


def names(cols, limit=8):
    cols = [str(c) for c in cols]
    shown = ", ".join(f"<b>{esc(c)}</b>" for c in cols[:limit])
    return shown + (f" and {len(cols) - limit} more" if len(cols) > limit else "")


def img(b64, alt):
    return f'<img class="chart" alt="{esc(alt)}" src="data:image/png;base64,{b64}">'


def build_report(df, report, actions, source, charts):
    n_rows, n_cols = df.shape
    total_missing = int(df.isna().sum().sum())
    total_cells = n_rows * n_cols
    miss_pct_total = total_missing / total_cells * 100 if total_cells else 0
    cols_with_missing = int((report["missing"] > 0).sum())
    dups = int(df.duplicated().sum())
    n_num = int((report["type"] == "numeric").sum())
    n_cat = n_cols - n_num

    # ---- key findings
    f = []
    miss_cols = report[report["missing"] > 0].sort_values("missing_%", ascending=False)
    if miss_cols.empty:
        f.append("No missing values were found.")
    else:
        top = miss_cols.iloc[0]
        f.append(f"{cols_with_missing} of {n_cols} columns contain missing values; the highest is "
                 f"<b>{esc(miss_cols.index[0])}</b> ({top['missing_%']:.1f}% missing).")
    high = report.index[report["missing_%"] > 50]
    if len(high):
        f.append(f"{len(high)} column(s) are more than 50% missing: {names(high)}.")
    const = report.index[report["unique"] <= 1]
    if len(const):
        f.append(f"Constant / empty columns: {names(const)}.")
    ids = report.index[(report["unique"] == n_rows) & (report["unique"] > 1)]
    if len(ids):
        f.append(f"Columns that look like row identifiers: {names(ids)}.")
    if "skew" in report.columns:
        skewed = report.index[(report["skew"].abs() > 1) & (report["unique"] > 10)]
        if len(skewed):
            f.append(f"Highly skewed numeric features: {names(skewed)}.")
        out = report.index[(report["outlier_pct"] > 5) & (report["unique"] > 10)]
        if len(out):
            f.append(f"Features with more than 5% outliers (IQR rule): {names(out)}.")
    f.append(f"{dups:,} duplicate row(s) found." if dups else "No duplicate rows found.")

    # ---- summary cards
    def card(v, l, cls=""):
        return f'<div class="card {cls}"><div class="v">{v}</div><div class="l">{l}</div></div>'
    cards = "".join([
        card(f"{n_rows:,}", "Rows"),
        card(f"{n_cols:,}", "Columns"),
        card(f"{n_num} / {n_cat}", "Numeric / Categorical"),
        card(f"{total_missing:,}", f"Missing cells ({miss_pct_total:.1f}%)",
             "good" if total_missing == 0 else "warn"),
        card(f"{cols_with_missing}", "Columns with missing", "good" if cols_with_missing == 0 else "warn"),
        card(f"{dups:,}", "Duplicate rows", "good" if dups == 0 else "bad"),
    ])

    # ---- missing table
    rows = []
    for c, r in report.sort_values("missing_%", ascending=False).iterrows():
        rows.append(f'<tr><td class="f">{esc(c)}</td><td>{esc(r["type"])}</td>'
                    f'<td class="n">{int(r["missing"]):,}</td><td>{miss_cell(r["missing_%"])}</td></tr>')
    missing_table = ('<div class="tw"><table><tr><th>Feature</th><th>Type</th><th class="n">Missing</th>'
                     "<th>Missing %</th></tr>" + "".join(rows) + "</table></div>")

    # ---- numeric table
    num = report[report["type"] == "numeric"]
    num_html = '<p class="note">No numeric features.</p>'
    if not num.empty and "mean" in num.columns:
        rows = []
        for c, r in num.iterrows():
            rows.append(
                f'<tr><td class="f">{esc(c)}</td><td>{miss_cell(r["missing_%"])}</td>'
                f'<td class="n">{fmt(r["unique"])}</td><td class="n">{fmt(r.get("mean"))}</td>'
                f'<td class="n">{fmt(r.get("median"))}</td><td class="n">{fmt(r.get("std"))}</td>'
                f'<td class="n">{fmt(r.get("min"))}</td><td class="n">{fmt(r.get("max"))}</td>'
                f'<td class="n">{fmt(r.get("skew"))}</td><td class="n">{fmt(r.get("outlier_pct"))}%</td>'
                f"<td>{pills(actions[c])}</td></tr>")
        num_html = ('<div class="tw"><table><tr><th>Feature</th><th>Missing</th><th class="n">Unique</th>'
                    '<th class="n">Mean</th><th class="n">Median</th><th class="n">Std</th>'
                    '<th class="n">Min</th><th class="n">Max</th><th class="n">Skew</th>'
                    '<th class="n">Outliers</th><th>Suggested preprocessing</th></tr>'
                    + "".join(rows) + "</table></div>")

    # ---- categorical table
    cat = report[report["type"] == "categorical"]
    cat_html = '<p class="note">No categorical features.</p>'
    if not cat.empty:
        rows = []
        for c, r in cat.iterrows():
            rows.append(
                f'<tr><td class="f">{esc(c)}</td><td>{miss_cell(r["missing_%"])}</td>'
                f'<td class="n">{fmt(r["unique"])}</td><td>{esc(r.get("top_values", "-") if pd.notna(r.get("top_values", np.nan)) else "-")}</td>'
                f"<td>{pills(actions[c])}</td></tr>")
        cat_html = ('<div class="tw"><table><tr><th>Feature</th><th>Missing</th><th class="n">Unique</th>'
                    "<th>Top values</th><th>Suggested preprocessing</th></tr>" + "".join(rows) + "</table></div>")

    # ---- action plan
    boxes = []
    for key, title, desc in PLAN:
        cols = [c for c, acts in actions.items() if any(k == key for k, _ in acts)]
        if cols:
            chips = "".join(f'<span class="chip">{esc(c)}</span>' for c in cols)
            boxes.append(f'<div class="box"><h3>{title} <span class="note">({len(cols)})</span></h3>'
                         f"<p>{desc}</p>{chips}</div>")
    plan_html = '<div class="plan">' + "".join(boxes) + "</div>" if boxes else \
        '<p class="note">No preprocessing actions suggested.</p>'

    # ---- charts
    def chart_block(title, sub, b64, alt):
        if not b64:
            return ""
        return f"<section><h2>{title}</h2><p class='sub'>{sub}</p>{img(b64, alt)}</section>"

    if HAS_MPL:
        charts_html = (
            chart_block("Missing values by column", "Only columns with missing values are shown. "
                        "Red = over 50%, amber = 5-50%.", charts.get("missing"), "Missing values chart")
            + chart_block("Distributions", "Numeric features (ID-like and constant columns excluded). "
                          "Features with 15 or fewer distinct values are shown as bar charts.",
                          charts.get("dist"), "Distributions")
            + chart_block("Correlation heatmap", "Pearson correlation between numeric features.",
                          charts.get("corr"), "Correlation heatmap"))
    else:
        charts_html = ("<section><h2>Charts</h2><p class='note'>matplotlib is not installed, so charts were "
                       "skipped. Run <code>python -m pip install matplotlib</code> and re-run the script."
                       "</p></section>")

    now = datetime.now().strftime("%d %B %Y, %H:%M")
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Data Analysis Report</title><style>{CSS}</style></head>
<body><div class="wrap">
<header><h1>Data Analysis Report</h1>
<p>Source: {esc(source)} &nbsp;|&nbsp; Generated: {now}</p></header>

<div class="cards">{cards}</div>

<section><h2>Key findings</h2><p class="sub">Generated automatically from the metrics below.</p>
<ul class="findings">{"".join(f"<li>{x}</li>" for x in f)}</ul></section>

<section><h2>Preprocessing action plan</h2>
<p class="sub">Suggestions are rules of thumb - check them against your domain knowledge.</p>{plan_html}</section>

<section><h2>Missing values per column</h2>
<p class="sub">Missing markers treated as NaN: see the note at the end of this report.</p>{missing_table}</section>

{charts_html}

<section><h2>Numeric features</h2><p class="sub">Outliers use the 1.5 x IQR rule. Skew above 1 (or below -1) is considered strong.</p>{num_html}</section>

<section><h2>Categorical features</h2><p class="sub">Top values show the three most frequent categories among non-missing rows.</p>{cat_html}</section>

<footer>Generated by data_analysis.py with pandas {pd.__version__}</footer>
</div></body></html>"""


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Missing values, preprocessing metrics and HTML report")
    parser.add_argument("csv_path", nargs="?", default=None,
                        help="Path to a .csv or .xlsx file (default: dataset/diabetic_data.csv or .xlsx)")
    parser.add_argument("--na-values", nargs="+", default=["?", ""],
                        help="Strings to treat as missing (default: ? and empty cells)")
    parser.add_argument("--save", help="Optional path to save the metrics table as CSV")
    parser.add_argument("--report", default="data_analysis_report.html",
                        help="Where to save the HTML report (default: data_analysis_report.html)")
    parser.add_argument("--no-open", action="store_true", help="Don't open the report in the browser")
    args = parser.parse_args()

    path = args.csv_path or find_default_file()
    print(f"Reading: {path}")
    df = load_data(path, args.na_values)

    print(f"\nRows: {df.shape[0]}  |  Columns: {df.shape[1]}")
    print(f"Duplicate rows: {int(df.duplicated().sum())}")
    print(f"Total missing cells: {int(df.isna().sum().sum())}")

    print("\n=== Missing values per column ===")
    missing = pd.DataFrame({
        "missing": df.isna().sum(),
        "missing_%": (df.isna().mean() * 100).round(2),
    })
    print(missing.to_string())

    report, actions = profile(df)
    pd.set_option("display.max_columns", None, "display.width", 250, "display.max_colwidth", 80)

    print("\n=== Numeric features ===")
    num = report[report["type"] == "numeric"].dropna(axis=1, how="all")
    print(num.to_string() if not num.empty else "None")

    print("\n=== Categorical features ===")
    cat = report[report["type"] == "categorical"].dropna(axis=1, how="all")
    print(cat.to_string() if not cat.empty else "None")

    if args.save:
        report.to_csv(args.save)
        print(f"\nMetrics saved to {args.save}")

    # ---- HTML report
    charts = {}
    if HAS_MPL:
        print("\nBuilding charts...")
        charts = {
            "missing": chart_missing(report),
            "dist": chart_distributions(df, report),
            "corr": chart_corr(df, report),
        }
    html_text = build_report(df, report, actions, path, charts)
    out = Path(args.report)
    out.write_text(html_text, encoding="utf-8")
    print(f"HTML report saved to {out.resolve()}")
    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()