"""
generation_stats.py

For each generation (1 and 2):
  - for each variant column (0/1), split the cohort into group 0 vs group 1
  - run statistical tests against: sex, onset_site, diagnostic_delay,
    education_years, survival, survival_null (missingness), mutaz_bin
  - save a CSV with all results (raw p-value always included)
  - generate plots for results significant after Bonferroni correction
  - generate a Word report (per generation)

It also produces:
  - a "combined" Word report: Generation 1 section + Generation 2 section,
    each showing its own significant results, followed by a "Combined"
    section that is the UNION of the two generations' significant results
    (simple concatenation, no re-running of stats on pooled data).
  - a dedicated C9 report (mutaz_bin vs every variant), with a Generation 1
    section, a Generation 2 section (ALL variants, regardless of
    significance), and a Combined section that is the union of both.

All output (logs, docx content, column headers) is in English.
"""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
OUT_DIR = Path("/mnt/cresla_prod/stefano_ge/c9_check")
MERGED_CSV = OUT_DIR / "c9_check_merged.csv"
VARIANT_COLS_FILE = OUT_DIR / "variant_columns.json"

STATS_DIR = OUT_DIR / "stats"
PLOTS_DIR = STATS_DIR / "plots"
REPORTS_DIR = STATS_DIR / "reports"
C9_DIR = STATS_DIR / "c9_report"
C9_PLOTS_DIR = C9_DIR / "plots"

ID_COL = "id"
GENERATION_COL = "generation"

SEX_COL = "sex"
ONSET_SITE_COL = "onset_site"
DIAGNOSTIC_DELAY_COL = "diagnostic_delay"
EDUCATION_YEARS_COL = "education_years"
SURVIVAL_COL = "survival"
MUTAZ_RAW_COL = "mutaz"

ALPHA = 0.05  # significance threshold applied to the Bonferroni-corrected p-value

CATEGORICAL_VARS = [SEX_COL, ONSET_SITE_COL, "mutaz_bin", "survival_null"]
CONTINUOUS_VARS = [DIAGNOSTIC_DELAY_COL, EDUCATION_YEARS_COL, SURVIVAL_COL]


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_data():
    df = pd.read_csv(MERGED_CSV)
    with open(VARIANT_COLS_FILE) as f:
        variant_cols = json.load(f)
    variant_cols = [c for c in variant_cols if c in df.columns]

    df["mutaz_bin"] = df[MUTAZ_RAW_COL].astype(str).str.contains("C9ORF72", na=False).astype(int)
    df["survival_null"] = df[SURVIVAL_COL].isna()
    df[EDUCATION_YEARS_COL] = pd.to_numeric(df[EDUCATION_YEARS_COL], errors="coerce").astype("Int64")

    log.info("Data loaded: %d rows, %d variant columns", len(df), len(variant_cols))
    return df, variant_cols


# ----------------------------------------------------------------------
# Statistical tests
# ----------------------------------------------------------------------
def test_categorical(df: pd.DataFrame, group_col: str, var_col: str) -> dict:
    sub = df[[group_col, var_col]].dropna()
    table = pd.crosstab(sub[group_col], sub[var_col])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return {"test": "n/a", "pvalue": None, "n0": None, "n1": None}

    if table.shape == (2, 2):
        _, p = stats.fisher_exact(table.values)
        test_name = "fisher_exact"
    else:
        _, p, _, _ = stats.chi2_contingency(table.values)
        test_name = "chi2"

    n0 = int((sub[group_col] == 0).sum())
    n1 = int((sub[group_col] == 1).sum())
    return {"test": test_name, "pvalue": p, "n0": n0, "n1": n1}


def test_continuous(df: pd.DataFrame, group_col: str, var_col: str) -> dict:
    sub = df[[group_col, var_col]].dropna()
    g0 = sub.loc[sub[group_col] == 0, var_col]
    g1 = sub.loc[sub[group_col] == 1, var_col]
    if len(g0) < 2 or len(g1) < 2:
        return {"test": "n/a", "pvalue": None, "n0": len(g0), "n1": len(g1)}

    _, p = stats.mannwhitneyu(g0, g1, alternative="two-sided")
    return {
        "test": "mannwhitney",
        "pvalue": p,
        "n0": len(g0),
        "n1": len(g1),
        "median0": float(g0.median()),
        "median1": float(g1.median()),
    }


def run_tests_for_generation(df_gen: pd.DataFrame, variant_cols: list) -> pd.DataFrame:
    rows = []
    for variant in variant_cols:
        if variant not in df_gen.columns:
            continue
        group = df_gen[variant]
        if group.dropna().nunique() < 2:
            continue  # monomorphic variant in this generation, skip

        for var_col in CATEGORICAL_VARS:
            res = test_categorical(df_gen.assign(_grp=group), "_grp", var_col)
            rows.append({"variant": variant, "variable": var_col, "type": "categorical", **res})

        for var_col in CONTINUOUS_VARS:
            res = test_continuous(df_gen.assign(_grp=group), "_grp", var_col)
            rows.append({"variant": variant, "variable": var_col, "type": "continuous", **res})

    results = pd.DataFrame(rows)

    # Bonferroni correction: pvalue_bonf = min(1, pvalue * n_tests_run)
    # n_tests_run = number of tests with an actual computed p-value
    # (excludes 'n/a' rows from groups that were too small/missing).
    n_tests = int(results["pvalue"].notna().sum())
    results["pvalue_bonf"] = (results["pvalue"] * n_tests).clip(upper=1.0)
    results.attrs["n_tests"] = n_tests
    log.info("Bonferroni correction applied over %d tests run", n_tests)

    return results


def bonferroni_threshold(results: pd.DataFrame) -> float:
    """Raw p-value threshold equivalent to pvalue_bonf < ALPHA."""
    n_tests = results.attrs.get("n_tests") or int(results["pvalue"].notna().sum())
    if n_tests == 0:
        return 0.0
    return ALPHA / n_tests


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------
def plot_categorical(df_gen: pd.DataFrame, variant: str, var_col: str, generation: int, out_path: Path):
    sub = df_gen[[variant, var_col]].dropna()
    prop = pd.crosstab(sub[variant], sub[var_col], normalize="index")
    fig, ax = plt.subplots(figsize=(6, 4))
    prop.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title(f"gen{generation} | {variant} vs {var_col}")
    ax.set_xlabel(f"{variant} (0/1)")
    ax.set_ylabel("proportion")
    ax.legend(title=var_col, bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_continuous(df_gen: pd.DataFrame, variant: str, var_col: str, generation: int, out_path: Path):
    sub = df_gen[[variant, var_col]].dropna()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.boxplot(data=sub, x=variant, y=var_col, ax=ax)
    sns.stripplot(data=sub, x=variant, y=var_col, ax=ax, color="black", alpha=0.4, size=3)
    ax.set_title(f"gen{generation} | {variant} vs {var_col}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_plots(df_gen: pd.DataFrame, results: pd.DataFrame, generation: int, plots_dir: Path) -> pd.DataFrame:
    plots_dir.mkdir(parents=True, exist_ok=True)
    sig = results[results["pvalue_bonf"].notna() & (results["pvalue_bonf"] < ALPHA)].copy()
    sig["plot_path"] = None

    for idx, row in sig.iterrows():
        variant, var_col, vtype = row["variant"], row["variable"], row["type"]
        fname = f"gen{generation}_{variant}_{var_col}.png".replace("/", "_")
        out_path = plots_dir / fname
        try:
            if vtype == "categorical":
                plot_categorical(df_gen, variant, var_col, generation, out_path)
            else:
                plot_continuous(df_gen, variant, var_col, generation, out_path)
            sig.at[idx, "plot_path"] = str(out_path)
        except Exception as e:
            log.warning("Plot failed for %s/%s: %s", variant, var_col, e)

    log.info("Generation %d: %d plots generated (Bonferroni p < %.2f)", generation, sig["plot_path"].notna().sum(), ALPHA)
    return sig


# ----------------------------------------------------------------------
# Word report helpers
# ----------------------------------------------------------------------
def _set_cell_text(cell, text: str, bold: bool = False):
    cell.text = ""
    run = cell.paragraphs[0].add_run(text)
    run.bold = bold


def _add_significant_results_table(doc, sig: pd.DataFrame):
    """Table WITHOUT a Bonferroni column; the raw p-value is bolded when
    the result is significant after Bonferroni correction (it always is,
    for rows already filtered into `sig`, but we keep the bold logic
    generic in case an unfiltered frame is passed in)."""
    table = doc.add_table(rows=1, cols=6)
    table.style = "Light Grid Accent 1"
    headers = ["Variant", "Variable", "Test", "p-value", "N group 0", "N group 1"]
    for i, h in enumerate(headers):
        _set_cell_text(table.rows[0].cells[i], h, bold=True)

    for _, row in sig.sort_values("pvalue_bonf").iterrows():
        cells = table.add_row().cells
        is_sig = pd.notna(row["pvalue_bonf"]) and row["pvalue_bonf"] < ALPHA
        _set_cell_text(cells[0], str(row["variant"]))
        _set_cell_text(cells[1], str(row["variable"]))
        _set_cell_text(cells[2], str(row["test"]))
        _set_cell_text(cells[3], f"{row['pvalue']:.4g}", bold=is_sig)
        _set_cell_text(cells[4], str(row.get("n0", "")))
        _set_cell_text(cells[5], str(row.get("n1", "")))


def _add_generation_section(doc, results: pd.DataFrame, sig: pd.DataFrame, generation: int, heading_level: int = 2):
    from docx.shared import Inches

    threshold = bonferroni_threshold(results)
    doc.add_heading(f"Generation {generation}", level=heading_level)
    doc.add_paragraph(
        f"Total tests run: {len(results)}. "
        f"Bonferroni-corrected significance threshold: raw p-value < {threshold:.4g} "
        f"(alpha={ALPHA} / {results.attrs.get('n_tests', len(results))} tests). "
        f"Significant results: {len(sig)}. Significant p-values are shown in bold."
    )

    if sig.empty:
        doc.add_paragraph("No significant results in this generation.")
        return

    _add_significant_results_table(doc, sig)

    doc.add_heading("Plots", level=heading_level + 1)
    for _, row in sig.sort_values("pvalue_bonf").iterrows():
        if not row["plot_path"]:
            continue
        doc.add_paragraph(f"{row['variant']} vs {row['variable']} (p = {row['pvalue']:.4g})")
        doc.add_picture(row["plot_path"], width=Inches(5))


def build_word_report(results: pd.DataFrame, sig: pd.DataFrame, generation: int, out_path: Path):
    from docx import Document

    doc = Document()
    doc.add_heading(f"Variant analysis - Generation {generation}", level=1)
    _add_generation_section(doc, results, sig, generation, heading_level=2)

    doc.save(out_path)
    log.info("Report saved: %s", out_path)


def build_combined_report(results1: pd.DataFrame, results2: pd.DataFrame,
                           sig1: pd.DataFrame, sig2: pd.DataFrame, out_path: Path):
    from docx import Document

    doc = Document()
    doc.add_heading("Variant analysis - Generation 1 vs Generation 2", level=1)
    doc.add_paragraph(
        "This report lists Generation 1 and Generation 2 significant results "
        "separately, followed by a Combined section that is the union of "
        "both generations' significant results (simple concatenation, "
        "no statistics re-run on pooled data)."
    )

    _add_generation_section(doc, results1, sig1, 1, heading_level=2)
    _add_generation_section(doc, results2, sig2, 2, heading_level=2)

    # Combined = union of significant results from both generations
    doc.add_heading("Combined (Generation 1 + Generation 2)", level=2)
    sig1_tagged = sig1.copy()
    sig1_tagged["generation"] = 1
    sig2_tagged = sig2.copy()
    sig2_tagged["generation"] = 2
    union = pd.concat([sig1_tagged, sig2_tagged], ignore_index=True)
    doc.add_paragraph(
        f"Union of significant (variant, variable) results across both generations: {len(union)} rows "
        f"({len(sig1)} from Generation 1, {len(sig2)} from Generation 2)."
    )

    from docx.shared import Inches
    table = doc.add_table(rows=1, cols=7)
    table.style = "Light Grid Accent 1"
    headers = ["Generation", "Variant", "Variable", "Test", "p-value", "N group 0", "N group 1"]
    for i, h in enumerate(headers):
        _set_cell_text(table.rows[0].cells[i], h, bold=True)
    for _, row in union.sort_values(["generation", "pvalue_bonf"]).iterrows():
        cells = table.add_row().cells
        _set_cell_text(cells[0], str(row["generation"]))
        _set_cell_text(cells[1], str(row["variant"]))
        _set_cell_text(cells[2], str(row["variable"]))
        _set_cell_text(cells[3], str(row["test"]))
        _set_cell_text(cells[4], f"{row['pvalue']:.4g}", bold=True)
        _set_cell_text(cells[5], str(row.get("n0", "")))
        _set_cell_text(cells[6], str(row.get("n1", "")))

    doc.add_heading("Plots", level=2)
    for _, row in union.sort_values(["generation", "pvalue_bonf"]).iterrows():
        if not row["plot_path"]:
            continue
        doc.add_paragraph(f"Gen{row['generation']} | {row['variant']} vs {row['variable']} (p = {row['pvalue']:.4g})")
        doc.add_picture(row["plot_path"], width=Inches(5))

    doc.save(out_path)
    log.info("Combined report saved: %s", out_path)


# ----------------------------------------------------------------------
# C9 (mutaz_bin) dedicated report - ALL variants, regardless of significance
# ----------------------------------------------------------------------
def _c9_results_for_generation(df: pd.DataFrame, results_by_gen: dict, generation: int) -> pd.DataFrame:
    results = results_by_gen[generation]
    c9 = results[results["variable"] == "mutaz_bin"].copy()
    c9 = c9.sort_values("pvalue")

    df_gen = df[df[GENERATION_COL] == generation]
    C9_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    c9["plot_path"] = None
    for idx, row in c9.iterrows():
        variant = row["variant"]
        fname = f"gen{generation}_{variant}_mutaz_bin.png".replace("/", "_")
        out_img = C9_PLOTS_DIR / fname
        try:
            plot_categorical(df_gen, variant, "mutaz_bin", generation, out_img)
            c9.at[idx, "plot_path"] = str(out_img)
        except Exception as e:
            log.warning("C9 plot failed for %s: %s", variant, e)

    return c9


def _add_c9_generation_section(doc, results: pd.DataFrame, c9: pd.DataFrame, generation: int, heading_level: int = 2):
    from docx.shared import Inches

    threshold = bonferroni_threshold(results)
    doc.add_heading(f"Generation {generation}", level=heading_level)
    doc.add_paragraph(
        f"All variants tested against mutaz_bin (C9ORF72), regardless of significance. "
        f"Total variants: {len(c9)}. "
        f"Bonferroni-corrected significance threshold: raw p-value < {threshold:.4g} "
        f"(alpha={ALPHA} / {results.attrs.get('n_tests', len(results))} tests). "
        f"Significant p-values are shown in bold."
    )

    table = doc.add_table(rows=1, cols=5)
    table.style = "Light Grid Accent 1"
    headers = ["Variant", "Test", "p-value", "N group 0", "N group 1"]
    for i, h in enumerate(headers):
        _set_cell_text(table.rows[0].cells[i], h, bold=True)
    for _, row in c9.iterrows():
        cells = table.add_row().cells
        is_sig = pd.notna(row["pvalue_bonf"]) and row["pvalue_bonf"] < ALPHA
        _set_cell_text(cells[0], str(row["variant"]))
        _set_cell_text(cells[1], str(row["test"]))
        pv_txt = f"{row['pvalue']:.4g}" if pd.notna(row["pvalue"]) else "n/a"
        _set_cell_text(cells[2], pv_txt, bold=is_sig)
        _set_cell_text(cells[3], str(row.get("n0", "")))
        _set_cell_text(cells[4], str(row.get("n1", "")))

    doc.add_heading("Plots", level=heading_level + 1)
    for _, row in c9.iterrows():
        if not row["plot_path"]:
            continue
        doc.add_paragraph(f"{row['variant']} (p = {row['pvalue']:.4g})" if pd.notna(row["pvalue"]) else str(row["variant"]))
        doc.add_picture(row["plot_path"], width=Inches(4.5))


def build_c9_report(df: pd.DataFrame, results_by_gen: dict, out_path: Path):
    from docx import Document
    from docx.shared import Inches

    C9_DIR.mkdir(parents=True, exist_ok=True)

    doc = Document()
    doc.add_heading("C9ORF72 (mutaz_bin) report - all variants", level=1)
    doc.add_paragraph(
        "This report includes ALL tested variants against mutaz_bin, regardless "
        "of statistical significance. Generation 1 and Generation 2 are shown "
        "separately, followed by a Combined section that is the union of both "
        "generations' rows (no statistics re-run on pooled data)."
    )

    c9_by_gen = {}
    for generation in (1, 2):
        if generation not in results_by_gen:
            continue
        c9 = _c9_results_for_generation(df, results_by_gen, generation)
        c9_by_gen[generation] = c9

        csv_path = C9_DIR / f"c9_mutaz_bin_gen{generation}.csv"
        c9.to_csv(csv_path, index=False)
        log.info("C9 CSV saved for generation %d: %s (%d variants)", generation, csv_path, len(c9))

        _add_c9_generation_section(doc, results_by_gen[generation], c9, generation, heading_level=2)

    if 1 in c9_by_gen and 2 in c9_by_gen:
        doc.add_heading("Combined (Generation 1 + Generation 2)", level=2)
        doc.add_paragraph(
            "Union of all variant rows from both generations (concatenation, not a joint re-analysis)."
        )
        c9_1 = c9_by_gen[1].copy()
        c9_1["generation"] = 1
        c9_2 = c9_by_gen[2].copy()
        c9_2["generation"] = 2
        union = pd.concat([c9_1, c9_2], ignore_index=True).sort_values(["generation", "pvalue"])

        combined_csv = C9_DIR / "c9_mutaz_bin_combined.csv"
        union.to_csv(combined_csv, index=False)
        log.info("C9 combined CSV saved: %s", combined_csv)

        table = doc.add_table(rows=1, cols=6)
        table.style = "Light Grid Accent 1"
        headers = ["Generation", "Variant", "Test", "p-value", "N group 0", "N group 1"]
        for i, h in enumerate(headers):
            _set_cell_text(table.rows[0].cells[i], h, bold=True)
        for _, row in union.iterrows():
            gen_results = results_by_gen[row["generation"]]
            threshold = bonferroni_threshold(gen_results)
            is_sig = pd.notna(row["pvalue"]) and row["pvalue"] < threshold
            cells = table.add_row().cells
            _set_cell_text(cells[0], str(row["generation"]))
            _set_cell_text(cells[1], str(row["variant"]))
            _set_cell_text(cells[2], str(row["test"]))
            pv_txt = f"{row['pvalue']:.4g}" if pd.notna(row["pvalue"]) else "n/a"
            _set_cell_text(cells[3], pv_txt, bold=is_sig)
            _set_cell_text(cells[4], str(row.get("n0", "")))
            _set_cell_text(cells[5], str(row.get("n1", "")))

    doc.save(out_path)
    log.info("C9 report saved: %s", out_path)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df, variant_cols = load_data()

    sig_by_gen = {}
    results_by_gen = {}
    for generation in (1, 2):
        df_gen = df[df[GENERATION_COL] == generation].copy()
        log.info("Generation %d: %d samples", generation, len(df_gen))
        if df_gen.empty:
            log.warning("Generation %d is empty, skipping.", generation)
            continue

        results = run_tests_for_generation(df_gen, variant_cols)
        results_by_gen[generation] = results
        results_csv = STATS_DIR / f"gen{generation}_variant_stats.csv"
        results.to_csv(results_csv, index=False)
        log.info("Results CSV saved for generation %d: %s", generation, results_csv)

        sig = generate_plots(df_gen, results, generation, PLOTS_DIR)
        sig_by_gen[generation] = sig

        build_word_report(results, sig, generation, REPORTS_DIR / f"gen{generation}_report.docx")

    if 1 in sig_by_gen and 2 in sig_by_gen:
        build_combined_report(
            results_by_gen[1], results_by_gen[2],
            sig_by_gen[1], sig_by_gen[2],
            REPORTS_DIR / "combined_report.docx",
        )

    build_c9_report(df, results_by_gen, REPORTS_DIR / "c9_report.docx")


if __name__ == "__main__":
    main()