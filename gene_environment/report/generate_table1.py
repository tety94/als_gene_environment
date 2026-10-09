#!/usr/bin/env python3
"""
generate_table1.py

Generates Table 1 for the paper (descriptive statistics for k >= 2 cohorts,
designed for the 3 generations G1/G2/G3) from:
  - a CSV containing clinical/environmental patient data (one row per id)
  - a CSV mapping id -> generation/cohort, produced by build_cohort_mapping.py
    by reading VCF headers (avoids loading gen.parquet, too heavy/unstable)

Statistical tests
  Numeric variables
    k = 2 : Welch t-test (normal) / Mann-Whitney U (non-normal)
    k > 2 : one-way ANOVA (all groups normal AND Levene p > ALPHA)
            otherwise Kruskal-Wallis
            post-hoc (only if overall p < ALPHA): pairwise Welch t-test (after ANOVA)
            or pairwise Mann-Whitney U (after Kruskal-Wallis), Bonferroni-adjusted
  Categorical variables
    2x2 with expected count < 5          : Fisher exact
    r x c with sparse expected counts    : Monte Carlo chi-square (permutation)
    otherwise                            : Chi-square

Output (in OUTPUT_DIR):
  - table1_stats.csv        -> raw statistics table, reusable
  - table1_posthoc.csv      -> pairwise post-hoc comparisons (k > 2)
  - Table1.docx             -> Word table ready for the paper (landscape)
  - figures/*.png           -> boxplots/barplots comparing the cohorts

Usage:
    python -m gene_environment.report.build_cohort_mapping   # generates id -> generation mapping
    python -m gene_environment.report.generate_table1        # generates Table 1

Modify only the CONFIG section below to adapt paths/column names.
"""

from __future__ import annotations

import sys
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must come after matplotlib.use)

from gene_environment.report.word_utils import set_cell_bg

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG — edit here
# ============================================================

CSV_PATH = "/srv/python-projects/gene_environment_v2/data/componenti_ambientali_full.csv"

COHORT_MAPPING_CSV = "output/table1/id_generation_mapping.csv"

OUTPUT_DIR = Path("output/table1")

ID_COL_CSV = "id"
ID_COL_MAPPING = "id"

COHORT_COL = "generation"

# Which cohorts to compare, in the order they appear in the table.
# None = auto-detect all distinct values (>= 2), sorted.
# Example: COHORT_VALUES = ["gen1", "gen2", "gen3"]
COHORT_VALUES = None

# Human-readable labels
COHORT_LABELS = None  # e.g. {"gen1": "G1 (PARALS)", "gen2": "G2", "gen3": "G3"}

CATEGORICAL_VARS = ["sex", "onset_site"]
NUMERIC_VARS = [
    "diagnostic_delay",
    "onset_age",
    "survival",
    "seminativi_1500",
    "vigneti_1500",
    "risaie_1500",
    "seminativi_1000",
    "vigneti_1000",
    "risaie_1000",
]

VAR_LABELS = {
    "sex": "Sex",
    "onset_site": "Onset site",
    "diagnostic_delay": "Diagnostic delay (months)",
    "onset_age": "Age at onset (years)",
    "survival": "Survival (years)",
    "seminativi_1500": "Arable land within 1500 m (%)",
    "vigneti_1500": "Vineyards within 1500 m (%)",
    "risaie_1500": "Rice fields within 1500 m (%)",
    "seminativi_1000": "Arable land within 1000 m (%)",
    "vigneti_1000": "Vineyards within 1000 m (%)",
    "risaie_1000": "Rice fields within 1000 m (%)",
}

ALPHA = 0.05
N_PERM = 10000      # permutations for the Monte Carlo chi-square
RANDOM_SEED = 42

# ============================================================
# DATA LOADING
# ============================================================

def load_data(csv_path: str, cohort_mapping_csv: str) -> pd.DataFrame:
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    if ID_COL_CSV not in df.columns:
        sys.exit(f"ERROR: id column '{ID_COL_CSV}' not found in CSV. Columns: {list(df.columns)}")

    print(f"Loading cohort mapping from CSV: {cohort_mapping_csv}")
    if not Path(cohort_mapping_csv).exists():
        sys.exit(
            f"ERROR: {cohort_mapping_csv} not found.\n"
            f"Generate the mapping first with: python -m gene_environment.report.build_cohort_mapping"
        )
    gen = pd.read_csv(cohort_mapping_csv)

    if ID_COL_MAPPING not in gen.columns:
        sys.exit(f"ERROR: id column '{ID_COL_MAPPING}' not found in mapping CSV. Columns: {list(gen.columns)}")
    if COHORT_COL not in gen.columns:
        sys.exit(f"ERROR: cohort column '{COHORT_COL}' not found in mapping CSV. Columns: {list(gen.columns)}")

    gen = gen[[ID_COL_MAPPING, COHORT_COL]].drop_duplicates()

    merged = df.merge(gen, left_on=ID_COL_CSV, right_on=ID_COL_MAPPING, how="inner")
    n_lost = len(df) - len(merged)
    if n_lost > 0:
        print(f"WARNING: {n_lost} patients in CSV not found in cohort mapping (excluded).")

    return merged


def resolve_cohorts(merged: pd.DataFrame):
    values = sorted(merged[COHORT_COL].dropna().unique().tolist())

    if COHORT_VALUES is not None:
        chosen = list(COHORT_VALUES)
        missing = [v for v in chosen if v not in values]
        if missing:
            sys.exit(f"ERROR: COHORT_VALUES {missing} not present in '{COHORT_COL}'. Found: {values}")
    else:
        chosen = values

    if len(chosen) < 2:
        sys.exit(f"ERROR: need at least 2 cohorts, found {len(chosen)}: {chosen}")

    labels = dict(COHORT_LABELS) if COHORT_LABELS else {}
    for v in chosen:
        labels.setdefault(v, str(v))

    sub = merged[merged[COHORT_COL].isin(chosen)].copy()
    counts = sub[COHORT_COL].value_counts().reindex(chosen).to_dict()
    print(f"Selected cohorts: {chosen} -> N = {counts}")
    return sub, chosen, labels


# ============================================================
# STATISTICS HELPERS
# ============================================================

def is_normal(series: pd.Series, alpha: float = 0.05) -> bool:
    series = series.dropna()
    if len(series) < 8:
        return True
    if series.nunique() < 2:
        return False
    _, p = stats.shapiro(series)
    return p > alpha


def fmt_p(p) -> str:
    if p is None or pd.isna(p):
        return "-"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def pairwise_posthoc(data: dict, groups, labels, parametric: bool, var_label: str) -> list[dict]:
    """Pairwise comparisons with Bonferroni adjustment."""
    pairs = list(combinations(groups, 2))
    n_pairs = len(pairs)
    rows = []
    for a, b in pairs:
        x, y = data[a], data[b]
        if len(x) < 2 or len(y) < 2:
            p_raw, test = np.nan, "n/a"
        elif parametric:
            _, p_raw = stats.ttest_ind(x, y, equal_var=False)
            test = "Welch t-test"
        else:
            _, p_raw = stats.mannwhitneyu(x, y, alternative="two-sided")
            test = "Mann\u2013Whitney U"
        p_adj = min(1.0, p_raw * n_pairs) if not pd.isna(p_raw) else np.nan
        rows.append(
            {
                "variable": var_label,
                "group_a": labels[a],
                "group_b": labels[b],
                "test": test,
                "p_raw": p_raw,
                "p_bonferroni": p_adj,
                "p_bonferroni_fmt": fmt_p(p_adj),
            }
        )
    return rows


def monte_carlo_chi2(x: pd.Series, y: pd.Series, n_perm: int = N_PERM, seed: int = RANDOM_SEED) -> float:
    """Permutation p-value for the chi-square statistic (for sparse r x c tables)."""
    xi, x_levels = pd.factorize(x)
    yi, y_levels = pd.factorize(y)
    nx, ny = len(x_levels), len(y_levels)

    def table(yy):
        return np.bincount(xi * ny + yy, minlength=nx * ny).reshape(nx, ny)

    obs = table(yi)
    expected = np.outer(obs.sum(1), obs.sum(0)) / obs.sum()  # marginals invariant under permutation
    mask = expected > 0
    stat_obs = (((obs - expected) ** 2)[mask] / expected[mask]).sum()

    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        t = table(rng.permutation(yi))
        s = (((t - expected) ** 2)[mask] / expected[mask]).sum()
        if s >= stat_obs - 1e-12:
            hits += 1
    return (hits + 1) / (n_perm + 1)


# ============================================================
# SUMMARIES
# ============================================================

def _empty_row_fields(groups, labels) -> dict:
    return {f"n__{labels[g]}": 0 for g in groups} | {f"stat__{labels[g]}": "" for g in groups}


def summarize_numeric(sub: pd.DataFrame, var: str, cohort_col: str, groups, labels):
    var_label = VAR_LABELS.get(var, var)
    data = {g: sub.loc[sub[cohort_col] == g, var].dropna() for g in groups}
    k = len(groups)

    all_present = all(len(s) > 0 for s in data.values())
    normal = all(is_normal(s) for s in data.values() if len(s) > 0)

    p, stat_name, posthoc_rows = np.nan, "n/a", []
    parametric = False

    if all_present and k == 2:
        a, b = (data[g] for g in groups)
        if normal:
            _, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
            stat_name, parametric = "Welch t-test", True
        else:
            _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            stat_name = "Mann\u2013Whitney U"
    elif all_present and k > 2:
        arrays = [data[g] for g in groups]
        equal_var = False
        if normal:
            try:
                _, p_lev = stats.levene(*arrays, center="median")
                equal_var = p_lev > ALPHA
            except ValueError:
                equal_var = False
        if normal and equal_var:
            _, p = stats.f_oneway(*arrays)
            stat_name, parametric = "One-way ANOVA", True
        else:
            try:
                _, p = stats.kruskal(*arrays)
            except ValueError:  # all values identical
                p = np.nan
            stat_name = "Kruskal\u2013Wallis"
        if not pd.isna(p) and p < ALPHA:
            posthoc_rows = pairwise_posthoc(data, groups, labels, parametric, var_label)

    # descriptive stats: mean ± SD only if the chosen test is parametric
    row = {
        "variable": var_label,
        "type": "numeric",
        "test": stat_name,
        "p_value": p,
        "p_value_fmt": fmt_p(p),
    }
    for g in groups:
        s = data[g]
        row[f"n__{labels[g]}"] = len(s)
        if len(s) == 0:
            row[f"stat__{labels[g]}"] = "-"
        elif parametric or (stat_name in ("n/a",) and normal):
            row[f"stat__{labels[g]}"] = f"{s.mean():.2f} \u00b1 {s.std():.2f}"
        else:
            row[f"stat__{labels[g]}"] = (
                f"{s.median():.2f} [{s.quantile(.25):.2f}-{s.quantile(.75):.2f}]"
            )

    sig_pairs = [
        f"{r['group_a']} vs {r['group_b']} (p={r['p_bonferroni_fmt']})"
        for r in posthoc_rows
        if not pd.isna(r["p_bonferroni"]) and r["p_bonferroni"] < ALPHA
    ]
    row["posthoc"] = "; ".join(sig_pairs)
    return [row], posthoc_rows


def summarize_categorical(sub: pd.DataFrame, var: str, cohort_col: str, groups, labels):
    var_label = VAR_LABELS.get(var, var)
    valid = sub[[var, cohort_col]].dropna()
    ct = pd.crosstab(valid[var], valid[cohort_col]).reindex(columns=groups, fill_value=0)

    p, test_used = np.nan, "n/a"
    if ct.shape[0] >= 2 and (ct.sum(axis=0) > 0).all():
        expected = stats.contingency.expected_freq(ct.values)
        if ct.shape == (2, 2) and (expected < 5).any():
            _, p = stats.fisher_exact(ct.values)
            test_used = "Fisher exact"
        elif (expected < 1).any() or (expected < 5).mean() > 0.2:
            p = monte_carlo_chi2(valid[var], valid[cohort_col])
            test_used = "Chi-square (Monte Carlo)"
        else:
            _, p, _, _ = stats.chi2_contingency(ct.values)
            test_used = "Chi-square"

    tots = ct.sum(axis=0)
    rows, first = [], True
    for level in ct.index:
        row = {
            "variable": var_label if first else f"{var_label} - {level}",
            "type": "categorical",
            "test": test_used if first else "",
            "p_value": p if first else np.nan,
            "p_value_fmt": fmt_p(p) if first else "",
            "posthoc": "",
        }
        # first level row: keep the variable name, but add the level so it is readable
        if first:
            row["variable"] = f"{var_label} - {level}"
        for g in groups:
            n = int(ct.loc[level, g])
            tot = int(tots[g])
            pct = 100 * n / tot if tot else 0
            row[f"n__{labels[g]}"] = tot
            row[f"stat__{labels[g]}"] = f"{n} ({pct:.1f}%)"
        rows.append(row)
        first = False
    return rows


def build_stats_table(sub: pd.DataFrame, cohort_col: str, groups, labels):
    all_rows, posthoc = [], []
    for var in CATEGORICAL_VARS:
        if var not in sub.columns:
            print(f"WARNING: categorical variable '{var}' not found, skipped.")
            continue
        all_rows.extend(summarize_categorical(sub, var, cohort_col, groups, labels))
    for var in NUMERIC_VARS:
        if var not in sub.columns:
            print(f"WARNING: numeric variable '{var}' not found, skipped.")
            continue
        rows, ph = summarize_numeric(sub, var, cohort_col, groups, labels)
        all_rows.extend(rows)
        posthoc.extend(ph)
    return pd.DataFrame(all_rows), pd.DataFrame(posthoc)


# ============================================================
# FIGURES
# ============================================================

def make_figures(sub: pd.DataFrame, cohort_col: str, groups, labels, stats_df: pd.DataFrame,
                 fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")

    order = [labels[g] for g in groups]
    colors = sns.color_palette("deep", len(groups))
    palette = dict(zip(order, colors))

    plot_df = sub.copy()
    plot_df["Cohort"] = plot_df[cohort_col].map(labels)

    def title_for(var):
        base = VAR_LABELS.get(var, var)
        match = stats_df[stats_df["variable"].str.startswith(base)]
        if match.empty:
            return base
        r = match.iloc[0]
        return f"{base}\n{r['test']}, p={r['p_value_fmt']}"

    for var in NUMERIC_VARS:
        if var not in plot_df.columns:
            continue
        fig, ax = plt.subplots(figsize=(1.8 * len(groups) + 2, 4.2))
        sns.boxplot(data=plot_df, x="Cohort", y=var, hue="Cohort", order=order, hue_order=order,
                    palette=palette, legend=False, ax=ax, showfliers=False)
        sns.stripplot(data=plot_df, x="Cohort", y=var, order=order, ax=ax,
                      color="black", alpha=0.3, size=3, jitter=True)
        ax.set_title(title_for(var), fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel(VAR_LABELS.get(var, var))
        fig.tight_layout()
        fig.savefig(fig_dir / f"boxplot_{var}.png", dpi=200)
        plt.close(fig)

    for var in CATEGORICAL_VARS:
        if var not in plot_df.columns:
            continue
        fig, ax = plt.subplots(figsize=(1.8 * len(groups) + 3, 4.2))
        ct = pd.crosstab(plot_df["Cohort"], plot_df[var], normalize="index").reindex(order) * 100
        ct.plot(kind="bar", stacked=True, ax=ax, colormap="tab10", rot=0)
        ax.set_ylabel("%")
        ax.set_xlabel("")
        ax.set_title(title_for(var), fontsize=10)
        ax.legend(title=var, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / f"barplot_{var}.png", dpi=200)
        plt.close(fig)

    print(f"Figures saved in: {fig_dir}")


# ============================================================
# WORD TABLE
# ============================================================

def make_docx_table(stats_df: pd.DataFrame, groups, labels, n_total, output_path: Path) -> None:
    doc = Document()
    k = len(groups)
    show_posthoc = k > 2

    # landscape A4 (3+ cohorts need the width)
    section = doc.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width = Cm(29.7)
    section.page_height = Cm(21.0)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(section, side, Cm(1.5))

    title = doc.add_paragraph()
    run = title.add_run(
        f"Table 1. Clinical and environmental characteristics of the {k} cohorts"
    )
    run.bold = True
    run.font.size = Pt(12)

    col_headers = ["Variable"]
    for g in groups:
        col_headers.append(f"{labels[g]} (n={n_total.get(g, 0)})")
    col_headers += ["Test", "p"]
    if show_posthoc:
        col_headers.append("Post-hoc (Bonferroni)")

    usable = 26.7  # cm
    w_var, w_test, w_p, w_post = 5.5, 3.4, 1.6, (4.5 if show_posthoc else 0.0)
    w_group = (usable - w_var - w_test - w_p - w_post) / k
    widths = [Cm(w_var)] + [Cm(w_group)] * k + [Cm(w_test), Cm(w_p)]
    if show_posthoc:
        widths.append(Cm(w_post))

    table = doc.add_table(rows=1, cols=len(col_headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for i, w in enumerate(widths):
        table.columns[i].width = w

    hdr_cells = table.rows[0].cells
    for i, htext in enumerate(col_headers):
        hdr_cells[i].text = htext
        hdr_cells[i].width = widths[i]
        for p in hdr_cells[i].paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.bold = True
                r.font.size = Pt(10)
        set_cell_bg(hdr_cells[i], "D9D9D9")

    p_col = 1 + k + 1  # index of the "p" column
    for _, row in stats_df.iterrows():
        cells = table.add_row().cells
        values = [row["variable"]]
        values += [row[f"stat__{labels[g]}"] for g in groups]
        values += [row["test"], row["p_value_fmt"]]
        if show_posthoc:
            values.append(row.get("posthoc", ""))
        sig = (not pd.isna(row["p_value"])) and row["p_value"] < ALPHA
        for i, v in enumerate(values):
            cells[i].text = "" if pd.isna(v) else str(v)
            cells[i].width = widths[i]
            for p in cells[i].paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if i == 0 else WD_ALIGN_PARAGRAPH.CENTER
                for r in p.runs:
                    r.font.size = Pt(9 if show_posthoc and i == len(values) - 1 else 10)
                    if i == p_col and sig:
                        r.bold = True

    note_text = (
        "Numeric variables: mean \u00b1 SD if normally distributed, otherwise median [IQR]. "
        + (
            "Overall test: one-way ANOVA (normal distribution and homogeneous variances), "
            "otherwise Kruskal\u2013Wallis; "
            "post-hoc pairwise Welch t-test (after ANOVA) or Mann\u2013Whitney U (after Kruskal\u2013Wallis), "
            "Bonferroni-adjusted, reported only for significant overall tests. "
            if show_posthoc
            else "Welch t-test (normal) or Mann\u2013Whitney U. "
        )
        + "Categorical variables: n (%), Chi-square test (Fisher exact for 2x2 tables, "
        "Monte Carlo Chi-square for sparse tables with expected counts <5)."
    )
    note = doc.add_paragraph()
    note_run = note.add_run(note_text)
    note_run.italic = True
    note_run.font.size = Pt(8)

    doc.save(output_path)
    print(f"Word table saved in: {output_path}")


# ============================================================
# MAIN (callable, reusable from the report runner and the CLI)
# ============================================================

def run_table1(csv_path: str = CSV_PATH, cohort_mapping_csv: str = COHORT_MAPPING_CSV,
               output_dir: Path = OUTPUT_DIR) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"

    merged = load_data(csv_path, cohort_mapping_csv)
    sub, groups, labels = resolve_cohorts(merged)

    n_total = sub[COHORT_COL].value_counts().to_dict()

    stats_df, posthoc_df = build_stats_table(sub, COHORT_COL, groups, labels)

    csv_out = output_dir / "table1_stats.csv"
    stats_df.to_csv(csv_out, index=False)
    print(f"Statistics CSV saved in: {csv_out}")

    if not posthoc_df.empty:
        ph_out = output_dir / "table1_posthoc.csv"
        posthoc_df.to_csv(ph_out, index=False)
        print(f"Post-hoc CSV saved in: {ph_out}")

    make_figures(sub, COHORT_COL, groups, labels, stats_df, fig_dir)

    docx_out = output_dir / "Table1.docx"
    make_docx_table(stats_df, groups, labels, n_total, docx_out)

    print("\nDone. Output in:", output_dir.resolve())


def main() -> None:
    run_table1()


if __name__ == "__main__":
    main()