# -*- coding: utf-8 -*-
"""
Generate Table 2 Word files and figures from the SQL store get_significant_results_table_2
using the project's MySQL connection helpers (get_connection, cursor_scope).

README / Requirements:
    pip install pandas python-docx matplotlib seaborn mysql-connector-python regex

Notes:
- This script expects a module (e.g., db.py) that exposes `get_connection` and `cursor_scope`
  as described by the user. Adjust the import path if needed.
- The SQL SELECT is included exactly as provided by the user.
- Set up your project's DB configuration (DBConfig / get_config) so that get_connection()
  can obtain connections from the pool.
- The script will:
    * Query the database using the provided SQL.
    * Drop gna.* columns (if present).
    * Produce two Word documents:
        - output/table2/Table2_top10.docx (top 10 rows, formatted for a paper)
        - output/table2/Table2_full_supplementary.docx (all rows, for supplementary materials)
    * Produce three figures in output/table2/figures:
        - variants_per_chromosome.png
        - genes_vs_variants_scatter.png
        - empirical_p_g1_histogram.png
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from gene_environment.db.connection import get_connection, cursor_scope

# -------------------------
# SQL exactly as provided
# -------------------------
SQL_QUERY = """
select distinct 
\tvrs1.exposure as exposure,
    vrs1.variant,
    vrs1.empirical_p as empirical_p_g1,
    round(vrs1.obs_coef, 2) as obs_coef_g1,
    vrs1.mutati as muted_g1,
    vrs1.non_mutati as not_muted_g1,
    vrs2.empirical_p as empirical_p_2,
    round(vrs2.obs_coef, 2) as obs_coef_g2,
    vrs2.mutati as muted_g2,
    vrs2.non_mutati as not_muted_g2,
\tgna.neuro_plausibility_score,
\tgna.expressed_neurons
"""

# Output paths
OUT_DIR = Path("output/table2")
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Columns to keep in Word tables (in this order)
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

# -------------------------
# Helpers
# -------------------------
def fetch_query_to_dataframe(sql: str) -> pd.DataFrame:
    """
    Execute the SQL using the project's get_connection / cursor_scope helpers
    and return a pandas DataFrame. Uses cursor(dictionary=True) so column names
    are preserved.
    """
    rows: List[dict] = []
    with get_connection() as conn:
        with cursor_scope(conn, dictionary=True) as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            # cur.description may be None for some drivers; dictionary=True gives keys
    # If rows is a list of dicts, create DataFrame directly
    if not rows:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=TABLE_COLUMNS)
    df = pd.DataFrame(rows)
    # Normalize column names: strip whitespace and lower-case
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
    Returns a normalized chromosome string (numbers as-is, X/Y/MT uppercase).
    """
    if pd.isna(variant):
        return "NA"
    if not isinstance(variant, str):
        variant = str(variant)
    m = re.match(r'^(?:chr)?([^:]+):', variant, flags=re.IGNORECASE)
    if m:
        chrom = m.group(1)
    else:
        # fallback: split on non-alphanumeric
        chrom = re.split(r'[:_\-]', variant)[0]
    chrom = chrom.lower().lstrip("chr")
    # Normalize alpha chromosomes to uppercase (X, Y, MT)
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
        # fallback to string
        return str(val)
    return str(val)


def add_table_to_doc(doc: Document, df: pd.DataFrame, title: str = None, caption: str = None, max_rows: int | None = None):
    """Add a table to a python-docx Document with bold headers and formatted values."""
    if title:
        h = doc.add_heading(title, level=2)
        h.alignment = WD_PARAGRAPH_ALIGNMENT.LEFT

    # Ensure all required columns exist in df
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


# -------------------------
# Main
# -------------------------
def main():
    # 1) Load data from DB
    print("Querying database...")
    df = fetch_query_to_dataframe(SQL_QUERY)

    # 2) Drop gna.* columns
    df = drop_gna_columns(df)

    # 3) Ensure expected columns exist (avoid KeyError later)
    for c in TABLE_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA

    # 4) Parse chromosome from variant
    df["chromosome"] = df["variant"].apply(lambda v: extract_chromosome(v) if pd.notna(v) else "NA")

    # 5) Prepare aggregated data for figures
    # variants per chromosome (unique variants)
    variants_per_chrom = df.groupby("chromosome")["variant"].nunique().rename("n_variants").reset_index()
    # unique genes (exposures) per chromosome
    genes_per_chrom = df.groupby("chromosome")["exposure"].nunique().rename("n_genes").reset_index()
    merged = pd.merge(variants_per_chrom, genes_per_chrom, on="chromosome", how="outer").fillna(0)

    # Sort chromosomes: numeric first, then alpha (X,Y,MT,...)
    def chrom_sort_key(ch):
        try:
            return (0, int(ch))
        except Exception:
            return (1, ch)
    merged = merged.sort_values(by="chromosome", key=lambda s: s.map(chrom_sort_key))

    sns.set(style="whitegrid")

    # Figure 1: Bar plot - variants per chromosome
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

    # Figure 2: Scatter - genes vs variants per chromosome
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

    # Figure 3: Histogram of empirical_p_g1 (optional if data present)
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
        # Create a placeholder image indicating no data
        plt.text(0.5, 0.5, "No empirical_p_g1 data available", ha="center", va="center")
        plt.axis("off")
        plt.savefig(hist_path, dpi=300)
        plt.close()

    # 6) Create Word documents
    # Table 2: top 10
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

    # Supplementary: full results
    doc_full = Document()
    # Add header text (simple approach)
    doc_full.add_heading("Supplementary Table: full results", level=1)
    doc_full.add_paragraph(
        "Full results from get_significant_results_table_2. All rows included. "
        "gna.* columns omitted from table but available in the database."
    )
    add_table_to_doc(doc_full, df, title=None, caption=None, max_rows=None)
    full_path = OUT_DIR / "Table2_full_supplementary.docx"
    doc_full.save(full_path)

    print("Done.")
    print(f"Top 10 Word: {top10_path}")
    print(f"Full supplementary Word: {full_path}")
    print(f"Figures saved in: {FIG_DIR}")


if __name__ == "__main__":
    main()
