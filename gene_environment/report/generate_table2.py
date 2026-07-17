# -*- coding: utf-8 -*-
"""
Generate Table 2 Word files and figures by calling the stored routine
get_significant_results_table_2 via the project's MySQL connection helpers.

Usage (from project root):
    python3 -m gene_environment.report.generate_table2

Requirements:
    pip install pandas python-docx matplotlib seaborn mysql-connector-python regex

Behavior:
- Calls the stored routine `get_significant_results_table_2()` and uses the first
  resultset returned.
- Drops gna.* columns before producing outputs.
- Produces:
    output/table2/Table2_top10.docx
    output/table2/Table2_full_supplementary.docx
    output/table2/figures/variants_per_chromosome.png
    output/table2/figures/genes_vs_variants_scatter.png
    output/table2/figures/empirical_p_g1_histogram.png
- Numeric formatting: p-values 3 significant digits, coefficients 2 decimals.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from docx import Document
from docx.shared import Pt, Inches
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from scipy.stats import chisquare, binomtest
from statsmodels.stats.multitest import multipletests


# Output directories
OUT_DIR = Path("output/table2")
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Columns to include in Word tables (order)
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

ASTORE_NAME = "get_significant_results_table_2"


def _set_cell_bg(cell, color_hex: str):
    """Set background color of a table cell (hex without #)."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), color_hex)
    tcPr.append(shd)

def _set_col_width(cell, width_inches: float):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcW = OxmlElement('w:tcW')
    tcW.set(qn('w:w'), str(int(width_inches * 914400)))  # twips
    tcW.set(qn('w:type'), 'dxa')
    tcPr.append(tcW)

def add_table_to_doc(doc: Document, df: pd.DataFrame, title: str = None, caption: str = None, max_rows: int | None = None):
    """
    Improved table formatting for python-docx:
    - bold headers, repeated header row
    - fixed column widths (inches)
    - numeric columns right-aligned, text left-aligned
    - alternating row shading
    - table style 'Table Grid'
    """
    if title:
        h = doc.add_heading(title, level=2)
        h.alignment = WD_PARAGRAPH_ALIGNMENT.LEFT

    # Ensure columns exist
    for c in TABLE_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA

    df_to_use = df[TABLE_COLUMNS]
    if max_rows is not None:
        df_to_use = df_to_use.head(max_rows)

    # Create table
    table = doc.add_table(rows=1, cols=len(TABLE_COLUMNS))
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_cells = table.rows[0].cells

    # Define column widths (in inches) - tune as needed
    col_widths = {
        "exposure": 1.6,
        "variant": 2.4,
        "empirical_p_g1": 0.9,
        "obs_coef_g1": 0.8,
        "muted_g1": 0.8,
        "not_muted_g1": 0.9,
        "empirical_p_2": 0.9,
        "obs_coef_g2": 0.8,
        "muted_g2": 0.8,
        "not_muted_g2": 0.9,
    }

    # Header row
    for i, col in enumerate(TABLE_COLUMNS):
        p = hdr_cells[i].paragraphs[0]
        run = p.add_run(col)
        run.bold = True
        run.font.size = Pt(10)
        p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        hdr_cells[i].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        # set width if available
        w = col_widths.get(col, 1.0)
        _set_col_width(hdr_cells[i], w)

    # Add data rows with formatting
    shade_colors = ("F2F2F2", "FFFFFF")  # light gray / white
    numeric_prefixes = ("empirical_p", "obs_coef",)  # treat these as numeric
    for ridx, (_, row) in enumerate(df_to_use.iterrows()):
        cells = table.add_row().cells
        shade = shade_colors[(ridx) % 2]
        for i, col in enumerate(TABLE_COLUMNS):
            val = format_value_for_word(col, row.get(col, pd.NA))
            cell = cells[i]
            # set width to match header
            w = col_widths.get(col, 1.0)
            _set_col_width(cell, w)
            # set text and alignment
            para = cell.paragraphs[0]
            if col.startswith(numeric_prefixes):
                para.alignment = WD_PARAGRAPH_ALIGNMENT.RIGHT
            else:
                para.alignment = WD_PARAGRAPH_ALIGNMENT.LEFT
            run = para.add_run(val)
            run.font.size = Pt(9)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            # shading
            _set_cell_bg(cell, shade)

    # Repeat header row on each page
    tbl_pr = table._tbl.get_or_add_tblPr()
    tbl_header = OxmlElement('w:tblHeader')
    tbl_header.set(qn('w:val'), "true")
    # find first row's trPr and set header
    first_tr = table.rows[0]._tr
    trPr = first_tr.get_or_add_trPr()
    tbl_header = OxmlElement('w:tblHeader')
    tbl_header.set(qn('w:val'), "true")
    trPr.append(tbl_header)

    if caption:
        doc.add_paragraph(caption)


def call_stored_routine_to_df(astore_name: str, get_connection, cursor_scope) -> pd.DataFrame:
    """
    Call the stored routine `astore_name()` and return a pandas DataFrame built
    from the first resultset. Handles drivers that require iterating nextset().
    """
    rows: List[dict] = []
    cols = None

    with get_connection() as conn:
        with cursor_scope(conn, dictionary=True) as cur:
            try:
                cur.execute(f"CALL {astore_name}()")
            except Exception as e:
                raise RuntimeError(f"Failed to CALL {astore_name}(). DB error: {e}") from e

            # Fetch first resultset if present
            try:
                fetched = cur.fetchall()
                if fetched:
                    rows = fetched
                    # dictionary=True ensures rows are dict-like
                    # cur.description may be None when dictionary=True; rely on dict keys
                else:
                    # If empty, still try to detect columns from description
                    if cur.description:
                        cols = [d[0] for d in cur.description]
            except Exception:
                # Some drivers raise if no rows; ignore and try nextset handling below
                rows = []

            # Some stored routines return multiple resultsets; ensure we captured the first non-empty one
            # Move through nextset() until we find rows or exhaust sets
            try:
                while (not rows) and cur.nextset():
                    try:
                        fetched = cur.fetchall()
                        if fetched:
                            rows = fetched
                            break
                    except Exception:
                        continue
            except Exception:
                # nextset may not be supported; ignore
                pass

    if not rows:
        # Return empty DataFrame with expected columns to avoid downstream errors
        return pd.DataFrame(columns=TABLE_COLUMNS)

    # rows is a list of dicts (dictionary=True). Build DataFrame.
    df = pd.DataFrame(rows)
    # Normalize column names: strip whitespace
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    return df


def drop_gna_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop gna.* columns if present (either as 'neuro_plausibility_score' or 'expressed_neurons'
    or with 'gna.' prefix)."""
    to_drop = []
    for col in df.columns:
        if col in ("neuro_plausibility_score", "expressed_neurons"):
            to_drop.append(col)
        if isinstance(col, str) and col.startswith("gna."):
            to_drop.append(col)
    if to_drop:
        df = df.drop(columns=to_drop, errors="ignore")
    return df


def extract_chromosome(variant: str) -> str:
    """
    Extract chromosome from variant strings like:
      - chr1:12345_A/T
      - 1:12345_A/T
      - chrX:...
      - X:...
    Returns normalized chromosome string.
    """
    if pd.isna(variant):
        return "NA"
    if not isinstance(variant, str):
        variant = str(variant)
    m = re.match(r'^(?:chr)?([^:]+):', variant, flags=re.IGNORECASE)
    if m:
        chrom = m.group(1)
    else:
        chrom = re.split(r'[:_\-]', variant)[0]
    chrom = chrom.lower().lstrip("chr")
    if chrom.isalpha():
        chrom = chrom.upper()
    return chrom


def format_value_for_word(col: str, val) -> str:
    """Format numeric columns for Word output:
       - empirical_p_*: 3 significant digits
       - obs_coef_*: two decimals
       - others: string as-is
    """
    if pd.isna(val):
        return ""
    try:
        if col.startswith("empirical_p"):
            return "{:.3g}".format(float(val))
        if col.startswith("obs_coef"):
            return "{:.2f}".format(float(val))
    except Exception:
        return str(val)
    return str(val)


def add_table_to_doc(doc: Document, df: pd.DataFrame, title: str = None, caption: str = None, max_rows: int | None = None):
    """Add a table to a python-docx Document with bold headers and formatted values."""
    if title:
        h = doc.add_heading(title, level=2)
        h.alignment = WD_PARAGRAPH_ALIGNMENT.LEFT

    # Ensure columns exist
    for c in TABLE_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA

    df_to_use = df[TABLE_COLUMNS]
    if max_rows is not None:
        df_to_use = df_to_use.head(max_rows)

    table = doc.add_table(rows=1, cols=len(TABLE_COLUMNS))
    hdr_cells = table.rows[0].cells
    for i, col in enumerate(TABLE_COLUMNS):
        run = hdr_cells[i].paragraphs[0].add_run(col)
        run.bold = True
        run.font.size = Pt(10)

    for _, row in df_to_use.iterrows():
        cells = table.add_row().cells
        for i, col in enumerate(TABLE_COLUMNS):
            cells[i].text = format_value_for_word(col, row.get(col, pd.NA))

    if caption:
        doc.add_paragraph(caption)


def make_figures(df: pd.DataFrame):
    """Create and save figures in FIG_DIR."""
    # Parse chromosome
    df["chromosome"] = df["variant"].apply(lambda v: extract_chromosome(v) if pd.notna(v) else "NA")

    # Aggregations
    variants_per_chrom = df.groupby("chromosome")["variant"].nunique().rename("n_variants").reset_index()
    genes_per_chrom = df.groupby("chromosome")["exposure"].nunique().rename("n_genes").reset_index()
    merged = pd.merge(variants_per_chrom, genes_per_chrom, on="chromosome", how="outer").fillna(0)

    # Sort chromosomes: numeric first then others
    def chrom_sort_key(ch):
        try:
            return (0, int(ch))
        except Exception:
            return (1, ch)
    merged = merged.sort_values(by="chromosome", key=lambda s: s.map(chrom_sort_key))

    sns.set(style="whitegrid")

    # Bar plot: variants per chromosome
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=merged, x="chromosome", y="n_variants", color="C0")
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("Number of unique variants")
    ax.set_title("Variants per chromosome")
    plt.xticks(rotation=45)
    plt.tight_layout()
    variants_plot_path = FIG_DIR / "variants_per_chromosome.png"
    plt.savefig(variants_plot_path, dpi=300)
    plt.close()

    # Scatter: genes vs variants per chromosome
    plt.figure(figsize=(8, 6))
    ax = sns.scatterplot(data=merged, x="n_genes", y="n_variants", s=100)
    for _, r in merged.iterrows():
        ax.text(r["n_genes"], r["n_variants"], str(r["chromosome"]), fontsize=9,
                horizontalalignment="left", verticalalignment="bottom")
    ax.set_xlabel("Number of unique genes (exposures) per chromosome")
    ax.set_ylabel("Number of unique variants per chromosome")
    ax.set_title("Genes vs Variants per chromosome")
    plt.tight_layout()
    scatter_path = FIG_DIR / "genes_vs_variants_scatter.png"
    plt.savefig(scatter_path, dpi=300)
    plt.close()

    # Histogram: empirical_p_g1
    plt.figure(figsize=(8, 5))
    try:
        pvals = pd.to_numeric(df["empirical_p_g1"], errors="coerce").dropna()
    except Exception:
        pvals = pd.Series(dtype=float)
    hist_path = FIG_DIR / "empirical_p_g1_histogram.png"
    if not pvals.empty:
        ax = sns.histplot(pvals, bins=50, kde=False, color="C2")
        ax.set_xlabel("empirical_p_g1")
        ax.set_ylabel("Count")
        ax.set_title("Histogram of empirical_p_g1")
        plt.tight_layout()
        plt.savefig(hist_path, dpi=300)
        plt.close()
    else:
        plt.text(0.5, 0.5, "No empirical_p_g1 data available", ha="center", va="center")
        plt.axis("off")
        plt.savefig(hist_path, dpi=300)
        plt.close()


def main():
    # Import connection helpers here to avoid potential import-time circular issues
    from gene_environment.db.connection import get_connection, cursor_scope


    # 1) Call the astore and load results
    print(f"Calling stored routine: {ASTORE_NAME}() ...")
    df = call_stored_routine_to_df(ASTORE_NAME, get_connection, cursor_scope)
    df = drop_gna_columns(df)

    for c in TABLE_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA

    # 4) Create figures
    make_figures(df)

    # 5) Create Table 2 (top 10) Word document
    doc_top10 = Document()
    doc_top10.add_heading("Table 2. Significant variants (top 10)", level=1)
    caption = (
        "Table shows top 10 significant variants from get_significant_results_table_2. "
        "Columns: exposure (gene), variant (chromosome:position_ref/alt), empirical_p_g1, "
        "obs_coef_g1, muted_g1, not_muted_g1, empirical_p_2, obs_coef_g2, muted_g2, not_muted_g2. "
        "gna.* columns omitted."
    )
    doc_top10.add_paragraph(caption)
    add_table_to_doc(doc_top10, df, title=None, caption=None, max_rows=10)
    top10_path = OUT_DIR / "Table2_top10.docx"
    doc_top10.save(top10_path)

    # 6) Create Supplementary Word (full results)
    doc_full = Document()
    doc_full.add_heading("Supplementary Table: full results", level=1)
    doc_full.add_paragraph(
        "Full results from get_significant_results_table_2. All rows included. "
        "gna.* columns omitted from table but available in the database."
    )
    add_table_to_doc(doc_full, df, title=None, caption=None, max_rows=None)
    full_path = OUT_DIR / "Table2_full_supplementary.docx"
    doc_full.save(full_path)

    # fetch tested counts for the exposure/generation of interest
    exposure_name = "risaie_1500"
    generation_number = 2
    tested_counts = fetch_tested_variant_counts(get_connection, cursor_scope, exposure_name, generation_number)

    # compute enrichment using tested counts
    chrom_stats_df, chrom_summary = compute_enrichment_stats_using_tested(df, tested_counts, alpha=0.05, out_dir=OUT_DIR)
    print("Chromosome enrichment summary:", chrom_summary)

    print("Done.")
    print(f"Top 10 Word: {top10_path}")
    print(f"Full supplementary Word: {full_path}")
    print(f"Figures saved in: {FIG_DIR}")

def fetch_tested_variant_counts(get_connection, cursor_scope, exposure: str, generation: int) -> dict:
    """
    Return a dict chromosome -> n_tested_variants using the provided store:
    SELECT chromosome, count(*) as c
    FROM variant_results
    WHERE exposure = <exposure> AND generation = <generation>
    GROUP BY chromosome

    Returns an empty dict if no rows.
    """
    sql = (
        "SELECT chromosome, COUNT(*) AS c "
        "FROM variant_results "
        "WHERE exposure = %s AND generation = %s "
        "GROUP BY chromosome"
    )
    counts = {}
    with get_connection() as conn:
        with cursor_scope(conn, dictionary=True) as cur:
            cur.execute(sql, (exposure, generation))
            rows = cur.fetchall()
            for r in rows:
                # r expected like {'chromosome': '1', 'c': 123}
                chrom = r.get("chromosome")
                cnt = r.get("c", 0)
                # normalize chromosome string same way as extract_chromosome
                if chrom is None:
                    continue
                chrom_norm = str(chrom).lower().lstrip("chr")
                if chrom_norm.isalpha():
                    chrom_norm = chrom_norm.upper()
                counts[chrom_norm] = int(cnt)
    return counts


def compute_enrichment_stats_using_tested(df: pd.DataFrame,
                                         tested_counts: dict,
                                         alpha: float = 0.05,
                                         out_dir: Path = OUT_DIR):
    """
    Compute enrichment using tested_counts (chromosome -> n_tested_variants)
    - df: DataFrame with at least 'variant' and 'empirical_p_g1'
    - tested_counts: dict mapping chromosome -> number of variants tested (from variant_results)
    Returns (merged_df, summary_dict)
    """
    # prepare df
    df = df.copy()
    df["chromosome"] = df["variant"].apply(lambda v: extract_chromosome(v) if pd.notna(v) else "NA")
    df["empirical_p_g1_num"] = pd.to_numeric(df.get("empirical_p_g1", pd.Series(dtype=float)), errors="coerce")
    df["is_sig_raw"] = df["empirical_p_g1_num"] < alpha

    # observed counts (unique variants tested in the resultset per chromosome)
    obs_counts = df.groupby("chromosome")["variant"].nunique().rename("n_significant_observed").reset_index()
    # tested_counts may include chromosomes not present in df; build base table
    tested_df = pd.DataFrame([
        {"chromosome": k, "n_tested": int(v)}
        for k, v in tested_counts.items()
    ])
    if tested_df.empty:
        raise RuntimeError("tested_counts is empty. Ensure the store returned counts for the given exposure/generation.")

    # merge tested and observed (observed may be zero for some chroms)
    merged = tested_df.merge(obs_counts, on="chromosome", how="left").fillna(0)
    merged["n_tested"] = merged["n_tested"].astype(int)
    merged["n_significant_observed"] = merged["n_significant_observed"].astype(int)

    # total tested and total observed significant
    total_tested = merged["n_tested"].sum()
    total_sig = merged["n_significant_observed"].sum()
    if total_tested == 0:
        raise RuntimeError("Total tested variants is zero; cannot compute expectations.")

    # expected under null: proportional to tested counts
    merged["expected"] = merged["n_tested"] * (total_sig / total_tested)
    merged["ratio_obs_exp"] = merged.apply(lambda r: (r["n_significant_observed"] / r["expected"]) if r["expected"] > 0 else float("nan"), axis=1)

    # chi-square goodness-of-fit (exclude zero-expected bins)
    mask_nonzero = merged["expected"] > 0
    chi2_stat, chi2_p = (None, None)
    if mask_nonzero.sum() >= 2:
        chi2_res = chisquare(f_obs=merged.loc[mask_nonzero, "n_significant_observed"].values,
                             f_exp=merged.loc[mask_nonzero, "expected"].values)
        chi2_stat, chi2_p = float(chi2_res.statistic), float(chi2_res.pvalue)

    # per-chromosome binomial tests (two-sided)
    p_global = total_sig / total_tested
    binom_pvals = []
    for _, row in merged.iterrows():
        n = int(row["n_tested"])
        k = int(row["n_significant_observed"])
        if n <= 0:
            binom_pvals.append(1.0)
            continue
        try:
            bt = binomtest(k, n, p_global)
            binom_pvals.append(bt.pvalue)
        except Exception:
            binom_pvals.append(1.0)
    merged["binom_p"] = binom_pvals

    # BH correction
    try:
        rej, p_adj, _, _ = multipletests(merged["binom_p"].fillna(1.0).values, method="fdr_bh")
        merged["binom_p_adj"] = p_adj
        merged["binom_reject_bh05"] = rej
    except Exception:
        merged["binom_p_adj"] = merged["binom_p"]
        merged["binom_reject_bh05"] = False

    # sort chromosomes sensibly (numeric first)
    def chrom_key(x):
        try:
            return (0, int(x))
        except Exception:
            return (1, x)
    merged = merged.sort_values(by="chromosome", key=lambda s: s.map(chrom_key))

    # save CSV and figure
    stats_path = out_dir / "table2_chromosome_enrichment_stats_tested.csv"
    merged.to_csv(stats_path, index=False)

    # figure: observed vs expected
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    x = merged["chromosome"].astype(str)
    xi = range(len(x))
    obs_vals = merged["n_significant_observed"]
    exp_vals = merged["expected"]
    bar_w = 0.4
    plt.bar([i - bar_w/2 for i in xi], obs_vals, width=bar_w, label="Observed significant", color="C0")
    plt.bar([i + bar_w/2 for i in xi], exp_vals, width=bar_w, label="Expected (proportional to tested)", color="C1", alpha=0.8)
    plt.xticks(xi, x, rotation=45)
    plt.ylabel("Count")
    plt.title("Observed vs Expected significant variants per chromosome (tested-based expectation)")
    plt.legend()
    for i, (_, row) in enumerate(merged.iterrows()):
        p_adj = row.get("binom_p_adj", None)
        if p_adj is not None:
            plt.text(i, max(row["n_significant_observed"], row["expected"]) + max(1, 0.01 * total_tested),
                     f"p_adj={p_adj:.1e}", ha="center", va="bottom", fontsize=8, rotation=90)
    fig_path = out_dir / "figures/observed_vs_expected_tested.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    summary = {
        "total_tested": int(total_tested),
        "total_significant": int(total_sig),
        "chi2_stat": chi2_stat,
        "chi2_p": chi2_p,
        "per_chromosome_csv": str(stats_path),
        "figure": str(fig_path),
    }
    return merged, summary


if __name__ == "__main__":
    main()
