#!/usr/bin/env python3
"""
generate_table1.py

Generates Table 1 (descriptive statistics for k >= 2 cohorts, e.g. G1/G2/G3).

Data flow (no concatenation of the cohorts is ever performed):
  1) From each heavy genetic CSV only the id column (plus optional GENETIC_COLS)
     is read, in chunks, and saved as a light CSV per cohort in LIGHT_DIR.
     If the light CSV already exists, the heavy genetic file is NOT read again.
  2) Clinical/environmental data are read once from CLINICAL_CSV and filtered
     by the ids of each cohort.
  3) All statistics are computed on those per-cohort DataFrames.

Statistical tests
  Numeric
    k = 2 : Welch t-test (normal) / Mann-Whitney U (non-normal)
    k > 2 : one-way ANOVA (all normal AND Levene p > ALPHA) else Kruskal-Wallis
            post-hoc (only if overall p < ALPHA): pairwise Welch / Mann-Whitney, Bonferroni
  Categorical
    2x2 with expected < 5            : Fisher exact
    r x c with sparse expected       : Monte Carlo chi-square (permutation)
    otherwise                        : Chi-square

Output (OUTPUT_DIR): table1_stats.csv, table1_posthoc.csv, Table1.docx, figures/*.png
"""

from __future__ import annotations

import sys
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gene_environment.report.word_utils import set_cell_bg

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG — edit here
# ============================================================

# Clinical + environmental data (one row per patient, all cohorts)
CLINICAL_CSV = "/srv/python-projects/gene_environment_v2/data/componenti_ambientali_full.csv"
ID_COL = "id"

# Heavy genetic files (dict order = table order)
GENETIC_CSVS = {
    "gen1": "/mnt/cresla_prod/genome_datasets/merged_csv/full_chr_gen1_test1.csv",
    "gen2": "/mnt/cresla_prod/genome_datasets/merged_csv/gen2_variants.csv",
    "gen3": "/mnt/cresla_prod/genome_datasets/merged_csv/gen3_variants.csv",
}

# Extra columns taken from the genetic files besides ID_COL (only if needed in the table).
# If a patient has several rows, the first one is kept.
# If you use them as table variables, also add them to CATEGORICAL_VARS / NUMERIC_VARS.
GENETIC_COLS: list[str] = []

LIGHT_DIR = Path("output/table1/genetic_light")
FORCE_REBUILD_LIGHT = False  # True = regenerate light CSVs even if they exist
CHUNKSIZE = 500_000

OUTPUT_DIR = Path("output/table1")

# None = all cohorts found. Example: ["gen1", "gen2", "gen3"]
COHORT_VALUES = None
COHORT_LABELS = None  # e.g. {"gen1": "G1 (PARALS)", "gen2": "G2", "gen3": "G3"}

CATEGORICAL_VARS = ["sex", "onset_site"]
NUMERIC_VARS = [
    "diagnostic_delay", "onset_age", "survival",
    "seminativi_1500", "vigneti_1500", "risaie_1500",
    "seminativi_1000", "vigneti_1000", "risaie_1000",
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
N_PERM = 10000
RANDOM_SEED = 42
SHAPIRO_MAX_N = 5000  # shapiro is unreliable/slow above this; subsample

ALL_VARS = CATEGORICAL_VARS + NUMERIC_VARS


# ============================================================
# DATA LOADING (no concat of cohorts)
# ============================================================

def _light_path(cohort: str) -> Path:
    return LIGHT_DIR / f"{cohort}_light.csv"


def load_light_genetic(cohort: str, path: str) -> pd.DataFrame:
    """Light CSV per cohort: use the cache if it exists, otherwise build it in chunks."""
    wanted = [ID_COL] + [c for c in GENETIC_COLS if c != ID_COL]
    lp = _light_path(cohort)

    if lp.exists() and not FORCE_REBUILD_LIGHT:
        cached = pd.read_csv(lp, dtype={ID_COL: str})
        if set(wanted) <= set(cached.columns):
            print(f"  '{cohort}': using cached light file {lp} ({len(cached)} patients)")
            return cached
        print(f"  '{cohort}': cache lacks required columns, rebuilding")

    if not Path(path).exists():
        sys.exit(f"ERROR: genetic file for cohort '{cohort}' not found: {path}")
    header = pd.read_csv(path, nrows=0).columns
    missing = [c for c in wanted if c not in header]
    if missing:
        sys.exit(f"ERROR: columns {missing} not found in {path}")

    print(f"  '{cohort}': reading heavy file {path} (columns: {wanted}) ...")
    parts, seen = [], set()
    for chunk in pd.read_csv(path, usecols=wanted, dtype={ID_COL: str}, chunksize=CHUNKSIZE):
        chunk[ID_COL] = chunk[ID_COL].str.strip()
        chunk = chunk.drop_duplicates(subset=ID_COL)
        chunk = chunk[~chunk[ID_COL].isin(seen)]
        seen.update(chunk[ID_COL])
        parts.append(chunk)
    light = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=wanted)

    LIGHT_DIR.mkdir(parents=True, exist_ok=True)
    light.to_csv(lp, index=False)
    print(f"  '{cohort}': light file saved in {lp} ({len(light)} patients)")
    return light


def load_cohorts() -> dict[str, pd.DataFrame]:
    """Return {cohort: DataFrame} with clinical/environmental data, one DataFrame per cohort."""
    if not Path(CLINICAL_CSV).exists():
        sys.exit(f"ERROR: clinical file not found: {CLINICAL_CSV}")
    header = pd.read_csv(CLINICAL_CSV, nrows=0).columns
    if ID_COL not in header:
        sys.exit(f"ERROR: '{ID_COL}' not in {CLINICAL_CSV}")

    # variables coming from the genetic files are not requested from the clinical file
    clin_vars = [v for v in ALL_VARS if v not in GENETIC_COLS]
    missing = [v for v in clin_vars if v not in header]
    if missing:
        print(f"WARNING: variables not in clinical file (will be skipped): {missing}")
    cols = [ID_COL] + [v for v in clin_vars if v in header]

    clin = pd.read_csv(CLINICAL_CSV, usecols=cols, dtype={ID_COL: str})
    clin[ID_COL] = clin[ID_COL].str.strip()
    clin = clin.drop_duplicates(subset=ID_COL)
    print(f"Clinical file: {len(clin)} patients, columns: {cols}")

    data, all_ids = {}, {}
    for cohort, path in GENETIC_CSVS.items():
        light = load_light_genetic(cohort, path)
        ids = set(light[ID_COL])
        sub = clin[clin[ID_COL].isin(ids)]
        if GENETIC_COLS:
            sub = sub.merge(light[[ID_COL] + GENETIC_COLS], on=ID_COL, how="left")
        n_lost = len(ids) - len(sub)
        print(f"Cohort '{cohort}': {len(sub)} patients with clinical data"
              + (f" ({n_lost} genetic ids not found in clinical file)" if n_lost else ""))
        data[cohort] = sub.reset_index(drop=True)
        all_ids[cohort] = ids

    for a, b in combinations(all_ids, 2):
        overlap = len(all_ids[a] & all_ids[b])
        if overlap:
            print(f"WARNING: {overlap} ids present in both '{a}' and '{b}'")
    return data


def resolve_cohorts(data: dict[str, pd.DataFrame]):
    found = list(data)
    if COHORT_VALUES is not None:
        chosen = list(COHORT_VALUES)
        missing = [v for v in chosen if v not in data]
        if missing:
            sys.exit(f"ERROR: COHORT_VALUES {missing} not present. Found: {found}")
    else:
        chosen = found

    if len(chosen) < 2:
        sys.exit(f"ERROR: need at least 2 cohorts, found {len(chosen)}: {chosen}")

    labels = {v: str(v) for v in chosen} | (COHORT_LABELS or {})
    data = {g: data[g] for g in chosen}
    print(f"Selected cohorts: {chosen} -> N = { {g: len(d) for g, d in data.items()} }")
    return data, chosen, labels


# ============================================================
# STATISTICS HELPERS
# ============================================================

def is_normal(series: pd.Series, alpha: float = ALPHA) -> bool:
    series = series.dropna()
    if len(series) < 8:
        return True
    if series.nunique() < 2:
        return False
    if len(series) > SHAPIRO_MAX_N:
        series = series.sample(SHAPIRO_MAX_N, random_state=RANDOM_SEED)
    return stats.shapiro(series)[1] > alpha


def fmt_p(p) -> str:
    if p is None or pd.isna(p):
        return "-"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def pairwise_posthoc(data: dict, groups, labels, parametric: bool, var_label: str) -> list[dict]:
    pairs = list(combinations(groups, 2))
    rows = []
    for a, b in pairs:
        x, y = data[a], data[b]
        if len(x) < 2 or len(y) < 2:
            p_raw, test = np.nan, "n/a"
        elif parametric:
            p_raw, test = stats.ttest_ind(x, y, equal_var=False)[1], "Welch t-test"
        else:
            p_raw, test = stats.mannwhitneyu(x, y, alternative="two-sided")[1], "Mann\u2013Whitney U"
        p_adj = min(1.0, p_raw * len(pairs)) if not pd.isna(p_raw) else np.nan
        rows.append({
            "variable": var_label, "group_a": labels[a], "group_b": labels[b], "test": test,
            "p_raw": p_raw, "p_bonferroni": p_adj, "p_bonferroni_fmt": fmt_p(p_adj),
        })
    return rows


def monte_carlo_chi2(ct: pd.DataFrame, n_perm: int = N_PERM, seed: int = RANDOM_SEED) -> float:
    """Permutation p-value for chi-square, built directly from the contingency table."""
    obs = ct.values
    nx, ny = obs.shape
    # rebuild the two label vectors from the table (no raw data needed)
    xi = np.repeat(np.repeat(np.arange(nx), ny), obs.ravel())
    yi = np.repeat(np.tile(np.arange(ny), nx), obs.ravel())

    expected = np.outer(obs.sum(1), obs.sum(0)) / obs.sum()
    mask = expected > 0
    e = expected[mask]
    stat_obs = (((obs - expected) ** 2)[mask] / e).sum()

    rng = np.random.default_rng(seed)
    idx = xi * ny
    hits = 0
    for _ in range(n_perm):
        t = np.bincount(idx + rng.permutation(yi), minlength=nx * ny).reshape(nx, ny)
        if (((t - expected) ** 2)[mask] / e).sum() >= stat_obs - 1e-12:
            hits += 1
    return (hits + 1) / (n_perm + 1)


# ============================================================
# SUMMARIES
# ============================================================

def summarize_numeric(data: dict, var: str, groups, labels):
    var_label = VAR_LABELS.get(var, var)
    vals = {g: data[g][var].dropna() for g in groups}
    k = len(groups)

    all_present = all(len(s) > 0 for s in vals.values())
    normal = all(is_normal(s) for s in vals.values() if len(s) > 0)

    p, stat_name, posthoc_rows, parametric = np.nan, "n/a", [], False

    if all_present and k == 2:
        a, b = (vals[g] for g in groups)
        if normal:
            p, stat_name, parametric = stats.ttest_ind(a, b, equal_var=False)[1], "Welch t-test", True
        else:
            p, stat_name = stats.mannwhitneyu(a, b, alternative="two-sided")[1], "Mann\u2013Whitney U"
    elif all_present and k > 2:
        arrays = [vals[g] for g in groups]
        equal_var = False
        if normal:
            try:
                equal_var = stats.levene(*arrays, center="median")[1] > ALPHA
            except ValueError:
                pass
        if normal and equal_var:
            p, stat_name, parametric = stats.f_oneway(*arrays)[1], "One-way ANOVA", True
        else:
            try:
                p = stats.kruskal(*arrays)[1]
            except ValueError:  # all values identical
                p = np.nan
            stat_name = "Kruskal\u2013Wallis"
        if not pd.isna(p) and p < ALPHA:
            posthoc_rows = pairwise_posthoc(vals, groups, labels, parametric, var_label)

    row = {
        "var": var, "variable": var_label, "type": "numeric",
        "test": stat_name, "p_value": p, "p_value_fmt": fmt_p(p),
    }
    use_mean = parametric or (stat_name == "n/a" and normal)
    for g in groups:
        s = vals[g]
        row[f"n__{labels[g]}"] = len(s)
        if len(s) == 0:
            row[f"stat__{labels[g]}"] = "-"
        elif use_mean:
            row[f"stat__{labels[g]}"] = f"{s.mean():.2f} \u00b1 {s.std():.2f}"
        else:
            q1, med, q3 = s.quantile([.25, .5, .75])
            row[f"stat__{labels[g]}"] = f"{med:.2f} [{q1:.2f}-{q3:.2f}]"

    row["posthoc"] = "; ".join(
        f"{r['group_a']} vs {r['group_b']} (p={r['p_bonferroni_fmt']})"
        for r in posthoc_rows
        if not pd.isna(r["p_bonferroni"]) and r["p_bonferroni"] < ALPHA
    )
    return [row], posthoc_rows


def contingency(data: dict, var: str, groups) -> pd.DataFrame:
    """Levels x cohorts count table, built per cohort (no concat of raw data)."""
    cols = {g: data[g][var].value_counts() for g in groups}
    return pd.DataFrame(cols).reindex(columns=groups).fillna(0).astype(int)


def summarize_categorical(data: dict, var: str, groups, labels):
    var_label = VAR_LABELS.get(var, var)
    ct = contingency(data, var, groups)

    p, test_used = np.nan, "n/a"
    if ct.shape[0] >= 2 and (ct.sum(axis=0) > 0).all():
        expected = stats.contingency.expected_freq(ct.values)
        if ct.shape == (2, 2) and (expected < 5).any():
            p, test_used = stats.fisher_exact(ct.values)[1], "Fisher exact"
        elif (expected < 1).any() or (expected < 5).mean() > 0.2:
            p, test_used = monte_carlo_chi2(ct), "Chi-square (Monte Carlo)"
        else:
            p, test_used = stats.chi2_contingency(ct.values)[1], "Chi-square"

    tots = ct.sum(axis=0)
    rows = []
    for i, level in enumerate(ct.index):
        first = i == 0
        row = {
            "var": var, "variable": f"{var_label} - {level}", "type": "categorical",
            "test": test_used if first else "",
            "p_value": p if first else np.nan,
            "p_value_fmt": fmt_p(p) if first else "",
            "posthoc": "",
        }
        for g in groups:
            n, tot = int(ct.loc[level, g]), int(tots[g])
            row[f"n__{labels[g]}"] = tot
            row[f"stat__{labels[g]}"] = f"{n} ({100 * n / tot if tot else 0:.1f}%)"
        rows.append(row)
    return rows


def build_stats_table(data: dict, groups, labels):
    all_rows, posthoc = [], []
    for var in CATEGORICAL_VARS:
        if not all(var in data[g].columns for g in groups):
            print(f"WARNING: categorical variable '{var}' missing in some cohort, skipped.")
            continue
        all_rows.extend(summarize_categorical(data, var, groups, labels))
    for var in NUMERIC_VARS:
        if not all(var in data[g].columns for g in groups):
            print(f"WARNING: numeric variable '{var}' missing in some cohort, skipped.")
            continue
        rows, ph = summarize_numeric(data, var, groups, labels)
        all_rows.extend(rows)
        posthoc.extend(ph)

    if not all_rows:
        sys.exit("ERROR: no variable available in all cohorts -> empty table. "
                 "Check column names in CLINICAL_CSV vs NUMERIC_VARS/CATEGORICAL_VARS.")
    return pd.DataFrame(all_rows), pd.DataFrame(posthoc)


# ============================================================
# FIGURES (matplotlib on per-cohort arrays, no merged DataFrame)
# ============================================================

def make_figures(data: dict, groups, labels, stats_df: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    order = [labels[g] for g in groups]
    colors = plt.get_cmap("tab10").colors[: len(groups)]
    rng = np.random.default_rng(RANDOM_SEED)

    def title_for(var):
        base = VAR_LABELS.get(var, var)
        m = stats_df[stats_df["var"] == var] if "var" in stats_df else stats_df.iloc[0:0]
        if m.empty:
            return base
        r = m.iloc[0]
        return f"{base}\n{r['test']}, p={r['p_value_fmt']}"

    for var in NUMERIC_VARS:
        if not all(var in data[g].columns for g in groups):
            continue
        vals = [data[g][var].dropna().values for g in groups]
        fig, ax = plt.subplots(figsize=(1.8 * len(groups) + 2, 4.2))
        bp = ax.boxplot(vals, tick_labels=order, patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.8)
        for i, v in enumerate(vals, start=1):
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=9, c="black", alpha=0.3)
        ax.set_title(title_for(var), fontsize=10)
        ax.set_ylabel(VAR_LABELS.get(var, var))
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / f"boxplot_{var}.png", dpi=200)
        plt.close(fig)

    for var in CATEGORICAL_VARS:
        if not all(var in data[g].columns for g in groups):
            continue
        ct = contingency(data, var, groups)
        pct = (ct / ct.sum(axis=0) * 100).T  # cohorts x levels
        pct.index = order
        fig, ax = plt.subplots(figsize=(1.8 * len(groups) + 3, 4.2))
        pct.plot(kind="bar", stacked=True, ax=ax, colormap="tab10", rot=0)
        ax.set_ylabel("%")
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

    section = doc.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width, section.page_height = Cm(29.7), Cm(21.0)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(section, side, Cm(1.5))

    run = doc.add_paragraph().add_run(
        f"Table 1. Clinical and environmental characteristics of the {k} cohorts"
    )
    run.bold = True
    run.font.size = Pt(12)

    col_headers = ["Variable"] + [f"{labels[g]} (n={n_total.get(g, 0)})" for g in groups] + ["Test", "p"]
    if show_posthoc:
        col_headers.append("Post-hoc (Bonferroni)")

    usable = 26.7
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

    for i, htext in enumerate(col_headers):
        cell = table.rows[0].cells[i]
        cell.text = htext
        cell.width = widths[i]
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.bold = True
                r.font.size = Pt(10)
        set_cell_bg(cell, "D9D9D9")

    p_col = 1 + k + 1
    last = len(col_headers) - 1
    for _, row in stats_df.iterrows():
        cells = table.add_row().cells
        values = [row["variable"]] + [row[f"stat__{labels[g]}"] for g in groups]
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
                    r.font.size = Pt(9 if show_posthoc and i == last else 10)
                    if i == p_col and sig:
                        r.bold = True

    note_text = (
        "Numeric variables: mean \u00b1 SD if normally distributed, otherwise median [IQR]. "
        + (
            "Overall test: one-way ANOVA (normal distribution and homogeneous variances), "
            "otherwise Kruskal\u2013Wallis; post-hoc pairwise Welch t-test (after ANOVA) or "
            "Mann\u2013Whitney U (after Kruskal\u2013Wallis), Bonferroni-adjusted, reported only "
            "for significant overall tests. "
            if show_posthoc
            else "Welch t-test (normal) or Mann\u2013Whitney U. "
        )
        + "Categorical variables: n (%), Chi-square test (Fisher exact for 2x2 tables, "
        "Monte Carlo Chi-square for sparse tables with expected counts <5)."
    )
    note_run = doc.add_paragraph().add_run(note_text)
    note_run.italic = True
    note_run.font.size = Pt(8)

    doc.save(output_path)
    print(f"Word table saved in: {output_path}")


# ============================================================
# MAIN
# ============================================================

def run_table1(output_dir: Path = OUTPUT_DIR) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data, groups, labels = resolve_cohorts(load_cohorts())
    n_total = {g: len(d) for g, d in data.items()}

    stats_df, posthoc_df = build_stats_table(data, groups, labels)

    out_cols = [c for c in stats_df.columns if c != "var"]
    stats_df[out_cols].to_csv(output_dir / "table1_stats.csv", index=False)
    print(f"Statistics CSV saved in: {output_dir / 'table1_stats.csv'}")

    if not posthoc_df.empty:
        posthoc_df.to_csv(output_dir / "table1_posthoc.csv", index=False)
        print(f"Post-hoc CSV saved in: {output_dir / 'table1_posthoc.csv'}")

    make_figures(data, groups, labels, stats_df, output_dir / "figures")
    make_docx_table(stats_df, groups, labels, n_total, output_dir / "Table1.docx")

    print("\nDone. Output in:", output_dir.resolve())


def main() -> None:
    run_table1()


if __name__ == "__main__":
    main()