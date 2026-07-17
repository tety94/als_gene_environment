# -*- coding: utf-8 -*-
"""
Generate Table 2 Word files and figures from the SQL store get_significant_results_table_2.

README / Requirements:
pip install pandas sqlalchemy python-docx matplotlib seaborn regex
(If using a specific DB driver, also install it, e.g., psycopg2-binary for Postgres or pyodbc for MSSQL.)

Usage:
- Set DATABASE_URI to your database connection string (SQLAlchemy format).
- Run this script. It will:
  * Query the provided SQL (exactly as given by the user).
  * Drop gna.* columns.
  * Produce two Word documents:
      - output/table2/Table2_top10.docx (top 10 rows, formatted for a paper)
      - output/table2/Table2_full_supplementary.docx (all rows, for supplementary materials)
  * Produce three figures in output/table2/figures:
      - variants_per_chromosome.png
      - genes_vs_variants_scatter.png
      - empirical_p_g1_histogram.png
"""

import os
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from sqlalchemy import create_engine

# -------------------------
# User must set this value
# -------------------------
DATABASE_URI = "postgresql://user:password@host:port/database"  # <-- replace with your DB URI

# SQL exactly as provided by the user (do not modify)
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

# Helper: safe read SQL into DataFrame
def load_data(sql: str, db_uri: str) -> pd.DataFrame:
    engine = create_engine(db_uri)
    with engine.connect() as conn:
        df = pd.read_sql_query(sql, conn)
    return df

# Helper: parse chromosome from variant string
def extract_chromosome(variant: str) -> str:
    if pd.isna(variant):
        return "NA"
    # common formats: chr1:12345_A/T, 1:12345_A/T, chrX:..., X:...
    m = re.match(r'^(?:chr)?([^:]+):', variant, flags=re.IGNORECASE)
    if m:
        chrom = m.group(1)
    else:
        # fallback: try to split on non-alnum
        chrom = re.split(r'[:_\-]', variant)[0]
    # normalize: remove leading 'chr' if any and uppercase X/Y/MT
    chrom = chrom.lower().lstrip("chr")
    chrom = chrom.upper() if chrom.isalpha() else chrom
    return chrom

# Helper: format numeric columns for Word output
def format_row_for_word(row: pd.Series) -> list:
    # Columns to include (in this order)
    cols = [
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
    out = []
    for c in cols:
        val = row.get(c, None)
        if pd.isna(val):
            out.append("")
            continue
        if c.startswith("empirical_p"):
            # 3 significant digits
            try:
                out.append("{:.3g}".format(float(val)))
            except Exception:
                out.append(str(val))
        elif c.startswith("obs_coef"):
            try:
                out.append("{:.2f}".format(float(val)))
            except Exception:
                out.append(str(val))
        else:
            out.append(str(val))
    return out

# Create a Word table with bold headers
def add_table_to_doc(doc: Document, df: pd.DataFrame, title: str = None, caption: str = None, max_rows: int = None):
    if title:
        h = doc.add_heading(title, level=2)
        h.alignment = WD_PARAGRAPH_ALIGNMENT.LEFT
    # Columns to include
    cols = [
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
    display_headers = [
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
    # limit rows if requested
    if max_rows is not None:
        df_to_use = df.head(max_rows)
    else:
        df_to_use = df

    table = doc.add_table(rows=1, cols=len(cols))
    hdr_cells = table.rows[0].cells
    for i, htxt in enumerate(display_headers):
        p = hdr_cells[i].paragraphs[0].add_run(htxt)
        p.bold = True
        p.font.size = Pt(10)

    for _, row in df_to_use.iterrows():
        cells = table.add_row().cells
        formatted = format_row_for_word(row)
        for i, val in enumerate(formatted):
            cells[i].text = val

    if caption:
        doc.add_paragraph(caption)

# Main processing
def main():
    # Load data
    try:
        df = load_data(SQL_QUERY, DATABASE_URI)
    except Exception as e:
        raise RuntimeError(f"Failed to load data from database. Check DATABASE_URI and DB driver. Error: {e}")

    # Drop gna.* columns if present
    for col in ["neuro_plausibility_score", "expressed_neurons"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # If the gna.* columns were present with prefix 'gna.' (unlikely in pandas columns), drop them too
    for col in list(df.columns):
        if col.startswith("gna."):
            df = df.drop(columns=[col])

    # Ensure required columns exist; if not, create them with NaNs to avoid errors
    required_cols = [
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
    for c in required_cols:
        if c not in df.columns:
            df[c] = pd.NA

    # Parse chromosome
    df["chromosome"] = df["variant"].apply(lambda v: extract_chromosome(v) if pd.notna(v) else "NA")

    # Figures data
    # variants per chromosome
    variants_per_chrom = df.groupby("chromosome")["variant"].nunique().rename("n_variants").reset_index()
    # unique genes (exposures) per chromosome
    genes_per_chrom = df.groupby("chromosome")["exposure"].nunique().rename("n_genes").reset_index()
    merged = pd.merge(variants_per_chrom, genes_per_chrom, on="chromosome", how="outer").fillna(0)
    # sort chromosomes in a reasonable way: numeric first then others
    def chrom_sort_key(ch):
        try:
            return (0, int(ch))
        except Exception:
            # put X, Y, MT, others after numbers
            return (1, ch)
    merged = merged.sort_values(by="chromosome", key=lambda s: s.map(chrom_sort_key))

    sns.set(style="whitegrid")

    # 1) Bar plot: variants per chromosome
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

    # 2) Scatter: unique genes vs variants per chromosome, annotate points
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

    # 3) Histogram of empirical_p_g1
    plt.figure(figsize=(8, 5))
    # convert to numeric, drop NaNs
    try:
        pvals = pd.to_numeric(df["empirical_p_g1"], errors="coerce").dropna()
    except Exception:
        pvals = pd.Series(dtype=float)
    if not pvals.empty:
        ax = sns.histplot(pvals, bins=50, kde=False, color="C2")
        ax.set_xlabel("empirical_p_g1")
        ax.set_ylabel("Count")
        ax.set_title("Histogram of empirical_p_g1")
        plt.tight_layout()
        hist_path = FIG_DIR / "empirical_p_g1_histogram.png"
        plt.savefig(hist_path, dpi=300)
        plt.close()
    else:
        # create an empty placeholder figure
        plt.text(0.5, 0.5, "No empirical_p_g1 data available", ha="center", va="center")
        plt.axis("off")
        hist_path = FIG_DIR / "empirical_p_g1_histogram.png"
        plt.savefig(hist_path, dpi=300)
        plt.close()

    # Prepare Word documents
    # Table 2: top 10
    doc_top10 = Document()
    # Title and short methods note
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
    # Add header text
    section = doc_full.sections[0]
    header = section.header
    header_para = header.paragraphs[0] if header.paragraphs else header.add_paragraph()
    header_para.text = "Supplementary Table: full results"
    header_para.runs[0].font.bold = True
    # Short description
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
