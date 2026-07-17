# -*- coding: utf-8 -*-
"""
Generate Table 2 Word files and figures by calling the stored routine
get_significant_results_table_2 via the project's MySQL connection helpers.

Usage (from project root):
    python3 -m gene_environment.report.generate_table2 [--generation 2] [--alpha 0.05]

Requirements:
    pip install pandas python-docx matplotlib seaborn mysql-connector-python regex scipy statsmodels

Behavior:
- Calls the stored routine `get_significant_results_table_2()` and uses the first
  resultset returned.
- Drops gna.* columns before producing outputs.
- Produces:
    output/table2/Table2_top10.docx
    output/table2/Table2_full_supplementary.docx   (now includes all figures embedded)
    output/table2/figures/variants_per_chromosome.png
    output/table2/figures/genes_vs_variants_scatter.png
    output/table2/figures/empirical_p_g1_histogram.png
    output/table2/figures/significant_by_exposure.png
    output/table2/table2_exposure_enrichment_stats.csv
- Numeric formatting: p-values 3 significant digits, coefficients 2 decimals.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from scipy.stats import binomtest, chisquare
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUT_DIR = Path("output/table2")
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

ASTORE_NAME = "get_significant_results_table_2"

# Columns to include in Word tables (order).
# NOTE: "empirical_p_2" (no "g") is kept exactly as in the original script.
# Every other column follows the "_g1"/"_g2" convention, so this looks like it
# could be a typo for "empirical_p_g2" -- please confirm against the real
# astore output before I rename it, otherwise the column will silently come
# back empty in the report.
TABLE_COLUMNS = [
    "exposure",
    "variant",
    "empirical_p_g1",
    "obs_coef_g1",
    "muted_g1",
    "not_muted_g1",
    "empirical_p_2",
    "obs_coef_g2",
    "muted_g2",
    "not_muted_g2",
]

# Nicer header labels for the Word table (falls back to the raw column name)
COLUMN_LABELS = {
    "exposure": "Exposure",
    "variant": "Variant",
    "empirical_p_g1": "Emp. p (G1)",
    "obs_coef_g1": "Coef. (G1)",
    "muted_g1": "Muted (G1)",
    "not_muted_g1": "Not muted (G1)",
    "empirical_p_2": "Emp. p (G2)",
    "obs_coef_g2": "Coef. (G2)",
    "muted_g2": "Muted (G2)",
    "not_muted_g2": "Not muted (G2)",
}

# Column widths in inches
COL_WIDTHS_IN = {
    "exposure": 1.5,
    "variant": 2.2,
    "empirical_p_g1": 0.85,
    "obs_coef_g1": 0.75,
    "muted_g1": 0.75,
    "not_muted_g1": 0.85,
    "empirical_p_2": 0.85,
    "obs_coef_g2": 0.75,
    "muted_g2": 0.75,
    "not_muted_g2": 0.85,
}

NUMERIC_PREFIXES = ("empirical_p", "obs_coef", "muted", "not_muted")
SIG_ALPHA_DEFAULT = 0.05

# Table color scheme
HEADER_FILL = "44546A"        # dark blue-grey
HEADER_FONT_COLOR = RGBColor(0xFF, 0xFF, 0xFF)
ZEBRA_FILL = "EEF1F6"         # very light blue-grey
SIG_FONT_COLOR = RGBColor(0xC0, 0x00, 0x00)  # highlight p < alpha


# ---------------------------------------------------------------------------
# docx table helpers
# ---------------------------------------------------------------------------

def _set_cell_bg(cell, color_hex: str) -> None:
    """Set background color of a table cell (hex without #)."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), color_hex)
    tcPr.append(shd)


def _set_col_width(cell, width_inches: float) -> None:
    """Set a table cell's width. Word column widths (w:type='dxa') are
    expressed in twips (1/1440 inch) -- NOT EMU (1/914400 inch). The
    original script multiplied by 914400, which produced widths ~635x too
    large and likely made Word ignore/garble the layout.
    """
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcW = OxmlElement("w:tcW")
    tcW.set(qn("w:w"), str(int(width_inches * 1440)))  # twips
    tcW.set(qn("w:type"), "dxa")
    tcPr.append(tcW)


def _set_table_borders(table, color_hex: str = "BFBFBF", size: int = 4) -> None:
    """Apply thin, consistent borders to the whole table."""
    tbl = table._tbl
    tblPr = tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(size))
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), color_hex)
        borders.append(el)
    tblPr.append(borders)


def add_table_to_doc(
    doc: Document,
    df: pd.DataFrame,
    title: Optional[str] = None,
    caption: Optional[str] = None,
    max_rows: Optional[int] = None,
    alpha: float = SIG_ALPHA_DEFAULT,
) -> None:
    """Add a formatted table to a python-docx Document.

    - Dark header band with white bold text and friendly column labels
    - Fixed column widths (in inches, correctly converted to twips)
    - Numeric columns right-aligned, text columns left-aligned
    - Zebra row shading + thin consistent borders
    - empirical_p_* values below `alpha` are bolded/highlighted
    - Header row repeats on each printed page
    """
    if title:
        h = doc.add_heading(title, level=2)
        h.alignment = WD_PARAGRAPH_ALIGNMENT.LEFT

    for c in TABLE_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA

    df_to_use = df[TABLE_COLUMNS]
    if max_rows is not None:
        df_to_use = df_to_use.head(max_rows)

    table = doc.add_table(rows=1, cols=len(TABLE_COLUMNS))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    _set_table_borders(table)

    # Header row
    hdr_cells = table.rows[0].cells
    for i, col in enumerate(TABLE_COLUMNS):
        p = hdr_cells[i].paragraphs[0]
        run = p.add_run(COLUMN_LABELS.get(col, col))
        run.bold = True
        run.font.size = Pt(10)
        run.font.color.rgb = HEADER_FONT_COLOR
        p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        hdr_cells[i].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        _set_cell_bg(hdr_cells[i], HEADER_FILL)
        _set_col_width(hdr_cells[i], COL_WIDTHS_IN.get(col, 1.0))

    # Data rows
    for ridx, (_, row) in enumerate(df_to_use.iterrows()):
        cells = table.add_row().cells
        shade = ZEBRA_FILL if ridx % 2 == 1 else "FFFFFF"
        for i, col in enumerate(TABLE_COLUMNS):
            raw_val = row.get(col, pd.NA)
            text = format_value_for_word(col, raw_val)
            cell = cells[i]
            _set_col_width(cell, COL_WIDTHS_IN.get(col, 1.0))
            para = cell.paragraphs[0]
            para.alignment = (
                WD_PARAGRAPH_ALIGNMENT.RIGHT
                if col.startswith(NUMERIC_PREFIXES)
                else WD_PARAGRAPH_ALIGNMENT.LEFT
            )
            run = para.add_run(text)
            run.font.size = Pt(9)

            if col.startswith("empirical_p") and _is_significant(raw_val, alpha):
                run.bold = True
                run.font.color.rgb = SIG_FONT_COLOR

            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            _set_cell_bg(cell, shade)

    # Repeat header row on each page
    first_tr = table.rows[0]._tr
    trPr = first_tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    trPr.append(tbl_header)

    if caption:
        cap = doc.add_paragraph(caption)
        cap.runs[0].italic = True
        cap.runs[0].font.size = Pt(9)


def _is_significant(val, alpha: float) -> bool:
    try:
        return pd.notna(val) and float(val) < alpha
    except (TypeError, ValueError):
        return False


def add_figure_to_doc(doc: Document, fig_path: Path, caption: str, width_in: float = 6.0) -> None:
    """Embed a figure (if it exists) into the document with a caption below it."""
    if not fig_path.exists():
        return
    doc.add_picture(str(fig_path), width=Inches(width_in))
    last_paragraph = doc.paragraphs[-1]
    last_paragraph.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    cap = doc.add_paragraph(caption)
    cap.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    cap.runs[0].italic = True
    cap.runs[0].font.size = Pt(9)


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------

def call_stored_routine_to_df(astore_name: str, get_connection, cursor_scope) -> pd.DataFrame:
    """
    Call the stored routine `astore_name()` and return a pandas DataFrame built
    from the first resultset. Handles drivers that require iterating nextset().
    """
    rows: List[dict] = []

    with get_connection() as conn:
        with cursor_scope(conn, dictionary=True) as cur:
            try:
                cur.execute(f"CALL {astore_name}()")
            except Exception as e:
                raise RuntimeError(f"Failed to CALL {astore_name}(). DB error: {e}") from e

            try:
                fetched = cur.fetchall()
                if fetched:
                    rows = fetched
            except Exception as e:
                print(f"[warn] initial fetchall() failed, will try nextset(): {e}", file=sys.stderr)
                rows = []

            try:
                while (not rows) and cur.nextset():
                    try:
                        fetched = cur.fetchall()
                        if fetched:
                            rows = fetched
                            break
                    except Exception as e:
                        print(f"[warn] fetchall() on a later resultset failed: {e}", file=sys.stderr)
                        continue
            except Exception as e:
                print(f"[warn] nextset() not supported by this driver: {e}", file=sys.stderr)

    if not rows:
        return pd.DataFrame(columns=TABLE_COLUMNS)

    df = pd.DataFrame(rows)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    return df


def drop_gna_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop gna.* columns if present."""
    to_drop = []
    for col in df.columns:
        if col in ("neuro_plausibility_score", "expressed_neurons"):
            to_drop.append(col)
        if isinstance(col, str) and col.startswith("gna."):
            to_drop.append(col)
    return df.drop(columns=to_drop, errors="ignore") if to_drop else df


def fetch_tested_variant_counts_by_exposure(get_connection, cursor_scope, generation: int) -> Dict[str, int]:
    """
    Return a dict exposure -> n_tested_variants for the given generation:
        SELECT exposure, COUNT(*) AS c
        FROM variant_results
        WHERE generation = <generation>
        GROUP BY exposure

    Returns an empty dict if no rows.
    """
    sql = (
        "SELECT exposure, COUNT(*) AS c "
        "FROM variant_results "
        "WHERE generation = %s "
        "GROUP BY exposure"
    )
    counts: Dict[str, int] = {}
    with get_connection() as conn:
        with cursor_scope(conn, dictionary=True) as cur:
            cur.execute(sql, (generation,))
            for r in cur.fetchall():
                exposure = r.get("exposure")
                cnt = r.get("c", 0)
                if exposure is None:
                    continue
                counts[str(exposure)] = int(cnt)
    return counts


# ---------------------------------------------------------------------------
# Formatting / parsing helpers
# ---------------------------------------------------------------------------

def extract_chromosome(variant: str) -> str:
    """
    Extract chromosome from variant strings like:
      - chr1:12345_A/T
      - 1:12345_A/T
      - chrX:...
      - X:...
    """
    if pd.isna(variant):
        return "NA"
    if not isinstance(variant, str):
        variant = str(variant)
    m = re.match(r"^(?:chr)?([^:]+):", variant, flags=re.IGNORECASE)
    chrom = m.group(1) if m else re.split(r"[:_\-]", variant)[0]
    chrom = chrom.lower().lstrip("chr")
    if chrom.isalpha():
        chrom = chrom.upper()
    return chrom


def format_value_for_word(col: str, val) -> str:
    """empirical_p_*: 3 significant digits; obs_coef_*: 2 decimals; else str()."""
    if pd.isna(val):
        return ""
    try:
        if col.startswith("empirical_p"):
            return "{:.3g}".format(float(val))
        if col.startswith("obs_coef"):
            return "{:.2f}".format(float(val))
    except (TypeError, ValueError):
        return str(val)
    return str(val)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(df: pd.DataFrame) -> None:
    """Chromosome-level descriptive figures (unchanged grouping)."""
    df = df.copy()
    df["chromosome"] = df["variant"].apply(lambda v: extract_chromosome(v) if pd.notna(v) else "NA")

    variants_per_chrom = df.groupby("chromosome")["variant"].nunique().rename("n_variants").reset_index()
    genes_per_chrom = df.groupby("chromosome")["exposure"].nunique().rename("n_genes").reset_index()
    merged = pd.merge(variants_per_chrom, genes_per_chrom, on="chromosome", how="outer").fillna(0)

    def chrom_sort_key(ch):
        try:
            return (0, int(ch))
        except ValueError:
            return (1, ch)

    merged = merged.sort_values(by="chromosome", key=lambda s: s.map(chrom_sort_key))

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=merged, x="chromosome", y="n_variants", color="#4472C4")
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("Number of unique variants")
    ax.set_title("Variants per chromosome")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "variants_per_chromosome.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    ax = sns.scatterplot(data=merged, x="n_genes", y="n_variants", s=100, color="#4472C4")
    for _, r in merged.iterrows():
        ax.text(r["n_genes"], r["n_variants"], str(r["chromosome"]), fontsize=9,
                 horizontalalignment="left", verticalalignment="bottom")
    ax.set_xlabel("Number of unique genes (exposures) per chromosome")
    ax.set_ylabel("Number of unique variants per chromosome")
    ax.set_title("Genes vs variants per chromosome")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "genes_vs_variants_scatter.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    pvals = pd.to_numeric(df.get("empirical_p_g1", pd.Series(dtype=float)), errors="coerce").dropna()
    hist_path = FIG_DIR / "empirical_p_g1_histogram.png"
    if not pvals.empty:
        ax = sns.histplot(pvals, bins=50, kde=False, color="#548235")
        ax.set_xlabel("empirical_p_g1")
        ax.set_ylabel("Count")
        ax.set_title("Histogram of empirical_p_g1")
        plt.tight_layout()
    else:
        plt.text(0.5, 0.5, "No empirical_p_g1 data available", ha="center", va="center")
        plt.axis("off")
    plt.savefig(hist_path, dpi=300)
    plt.close()


def _empty_placeholder_figure(path: Path, message: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    plt.axis("off")
    plt.savefig(path, dpi=300)
    plt.close()


def compute_enrichment_stats_by_exposure(
    df: pd.DataFrame,
    tested_counts: Dict[str, int],
    alpha: float = SIG_ALPHA_DEFAULT,
    out_dir: Path = OUT_DIR,
):
    """
    Enrichment of significant variants *per exposure* (gene), using
    `tested_counts` (exposure -> n_tested_variants) as the null baseline.

    Handles exposures with 0 observed significant variants (and, symmetrically,
    exposures present in `tested_counts` but absent from `df`) without raising:
    they simply enter the table with n_significant_observed = 0.

    Returns (merged_df, summary_dict). If there is no usable data at all,
    returns an empty DataFrame and a summary noting that, and still writes a
    placeholder figure instead of crashing the pipeline.
    """
    fig_path = out_dir / "figures" / "significant_by_exposure.png"
    stats_path = out_dir / "table2_exposure_enrichment_stats.csv"

    if not tested_counts:
        print("[warn] tested_counts is empty -- skipping exposure enrichment stats.", file=sys.stderr)
        _empty_placeholder_figure(fig_path, "No tested-variant counts available")
        empty = pd.DataFrame(columns=["exposure", "n_tested", "n_significant_observed",
                                       "expected", "ratio_obs_exp", "binom_p", "binom_p_adj",
                                       "binom_reject_bh05"])
        empty.to_csv(stats_path, index=False)
        return empty, {"total_tested": 0, "total_significant": 0, "chi2_stat": None,
                        "chi2_p": None, "per_exposure_csv": str(stats_path), "figure": str(fig_path)}

    df = df.copy()
    df["empirical_p_g1_num"] = pd.to_numeric(df.get("empirical_p_g1", pd.Series(dtype=float)), errors="coerce")
    df["is_sig_raw"] = df["empirical_p_g1_num"] < alpha

    obs_counts = (
        df.loc[df["is_sig_raw"]]
        .groupby("exposure")["variant"]
        .nunique()
        .rename("n_significant_observed")
        .reset_index()
    )

    tested_df = pd.DataFrame(
        [{"exposure": k, "n_tested": int(v)} for k, v in tested_counts.items()]
    )

    # Outer merge so exposures with 0 significant variants (present only in
    # tested_counts) AND exposures with hits but no DB tested-count entry
    # both survive, with the missing side filled as 0.
    merged = tested_df.merge(obs_counts, on="exposure", how="outer").fillna(0)
    merged["n_tested"] = merged["n_tested"].astype(int)
    merged["n_significant_observed"] = merged["n_significant_observed"].astype(int)

    total_tested = merged["n_tested"].sum()
    total_sig = merged["n_significant_observed"].sum()

    if total_tested == 0:
        print("[warn] total tested variants is zero -- skipping enrichment stats.", file=sys.stderr)
        _empty_placeholder_figure(fig_path, "No tested variants for this generation")
        merged.to_csv(stats_path, index=False)
        return merged, {"total_tested": 0, "total_significant": int(total_sig), "chi2_stat": None,
                         "chi2_p": None, "per_exposure_csv": str(stats_path), "figure": str(fig_path)}

    merged["expected"] = merged["n_tested"] * (total_sig / total_tested)
    merged["ratio_obs_exp"] = merged.apply(
        lambda r: (r["n_significant_observed"] / r["expected"]) if r["expected"] > 0 else float("nan"),
        axis=1,
    )

    mask_nonzero = merged["expected"] > 0
    chi2_stat, chi2_p = None, None
    if mask_nonzero.sum() >= 2:
        chi2_res = chisquare(
            f_obs=merged.loc[mask_nonzero, "n_significant_observed"].values,
            f_exp=merged.loc[mask_nonzero, "expected"].values,
        )
        chi2_stat, chi2_p = float(chi2_res.statistic), float(chi2_res.pvalue)

    p_global = total_sig / total_tested if total_tested > 0 else 0.0
    binom_pvals = []
    for _, row in merged.iterrows():
        n, k = int(row["n_tested"]), int(row["n_significant_observed"])
        if n <= 0:
            binom_pvals.append(1.0)
            continue
        try:
            binom_pvals.append(binomtest(k, n, p_global).pvalue)
        except (ValueError, Exception):
            binom_pvals.append(1.0)
    merged["binom_p"] = binom_pvals

    try:
        rej, p_adj, _, _ = multipletests(merged["binom_p"].fillna(1.0).values, method="fdr_bh")
        merged["binom_p_adj"] = p_adj
        merged["binom_reject_bh05"] = rej
    except ValueError:
        merged["binom_p_adj"] = merged["binom_p"]
        merged["binom_reject_bh05"] = False

    merged = merged.sort_values(by="n_significant_observed", ascending=False)
    merged.to_csv(stats_path, index=False)

    _plot_significant_by_exposure(merged, fig_path, total_tested)

    summary = {
        "total_tested": int(total_tested),
        "total_significant": int(total_sig),
        "chi2_stat": chi2_stat,
        "chi2_p": chi2_p,
        "per_exposure_csv": str(stats_path),
        "figure": str(fig_path),
    }
    return merged, summary


def _plot_significant_by_exposure(merged: pd.DataFrame, fig_path: Path, total_tested: int) -> None:
    """One chart, one group of bars per exposure (observed vs expected).
    Exposures with 0 significant variants simply render as a zero-height bar
    instead of being dropped or raising an error.
    """
    if merged.empty:
        _empty_placeholder_figure(fig_path, "No exposures to plot")
        return

    n_exposures = len(merged)
    fig_width = max(8, min(0.5 * n_exposures, 30))  # scale width, cap it so huge N doesn't explode the image
    plt.figure(figsize=(fig_width, 6))

    x_labels = merged["exposure"].astype(str)
    xi = range(n_exposures)
    obs_vals = merged["n_significant_observed"]
    exp_vals = merged["expected"]
    bar_w = 0.4

    plt.bar([i - bar_w / 2 for i in xi], obs_vals, width=bar_w, label="Observed significant", color="#4472C4")
    plt.bar([i + bar_w / 2 for i in xi], exp_vals, width=bar_w, label="Expected (proportional to tested)",
             color="#ED7D31", alpha=0.85)
    plt.xticks(list(xi), x_labels, rotation=90 if n_exposures > 15 else 45, ha="right")
    plt.ylabel("Count")
    plt.title("Observed vs expected significant variants per exposure")
    plt.legend()

    for i, (_, row) in enumerate(merged.iterrows()):
        p_adj = row.get("binom_p_adj")
        if pd.notna(p_adj):
            top = max(row["n_significant_observed"], row["expected"])
            plt.text(i, top + max(0.5, 0.01 * total_tested), f"p_adj={p_adj:.1e}",
                      ha="center", va="bottom", fontsize=7, rotation=90)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate Table 2 report (Word + figures).")
    p.add_argument("--generation", type=int, default=2,
                    help="Generation number used to pull tested-variant counts per exposure.")
    p.add_argument("--alpha", type=float, default=SIG_ALPHA_DEFAULT,
                    help="Significance threshold for empirical_p_g1.")
    return p.parse_args()


def main():
    from gene_environment.db.connection import cursor_scope, get_connection

    args = parse_args()

    print(f"Calling stored routine: {ASTORE_NAME}() ...")
    df = call_stored_routine_to_df(ASTORE_NAME, get_connection, cursor_scope)
    df = drop_gna_columns(df)
    for c in TABLE_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA

    make_figures(df)

    tested_counts = fetch_tested_variant_counts_by_exposure(get_connection, cursor_scope, args.generation)
    exposure_stats_df, exposure_summary = compute_enrichment_stats_by_exposure(
        df, tested_counts, alpha=args.alpha, out_dir=OUT_DIR
    )
    print("Exposure enrichment summary:", exposure_summary)

    # --- Table 2 (top 10) ---
    doc_top10 = Document()
    doc_top10.add_heading("Table 2. Significant variants (top 10)", level=1)
    doc_top10.add_paragraph(
        "Table shows the top 10 significant variants from get_significant_results_table_2. "
        "p-values highlighted in red are below the significance threshold "
        f"(alpha = {args.alpha}). gna.* columns omitted."
    )
    add_table_to_doc(doc_top10, df, max_rows=10, alpha=args.alpha)
    top10_path = OUT_DIR / "Table2_top10.docx"
    doc_top10.save(top10_path)

    # --- Supplementary Word (full results + all figures) ---
    doc_full = Document()
    doc_full.add_heading("Supplementary Table: full results", level=1)
    doc_full.add_paragraph(
        "Full results from get_significant_results_table_2. All rows included. "
        "gna.* columns omitted from the table but available in the database."
    )
    add_table_to_doc(doc_full, df, alpha=args.alpha)

    doc_full.add_heading("Figures", level=1)
    add_figure_to_doc(doc_full, FIG_DIR / "variants_per_chromosome.png",
                       "Figure 1. Number of unique variants per chromosome.")
    add_figure_to_doc(doc_full, FIG_DIR / "genes_vs_variants_scatter.png",
                       "Figure 2. Genes vs variants per chromosome.")
    add_figure_to_doc(doc_full, FIG_DIR / "empirical_p_g1_histogram.png",
                       "Figure 3. Distribution of empirical_p_g1.")
    add_figure_to_doc(doc_full, FIG_DIR / "significant_by_exposure.png",
                       "Figure 4. Observed vs expected significant variants per exposure "
                       f"(generation {args.generation}).")

    full_path = OUT_DIR / "Table2_full_supplementary.docx"
    doc_full.save(full_path)

    print("Done.")
    print(f"Top 10 Word: {top10_path}")
    print(f"Full supplementary Word: {full_path}")
    print(f"Figures saved in: {FIG_DIR}")
    print(f"Per-exposure enrichment CSV: {exposure_summary['per_exposure_csv']}")


if __name__ == "__main__":
    main()