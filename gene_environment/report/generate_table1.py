#!/usr/bin/env python3
"""
generate_table1.py

Genera la Tabella 1 del paper (statistiche descrittive per due coorti) a partire da:
  - un CSV con i dati clinico-ambientali dei pazienti (una riga per id)
  - un file parquet con la mappatura id -> generazione/coorte

Output (in OUTPUT_DIR):
  - table1_stats.csv         -> tabella statistiche "grezza", leggibile/riusabile
  - Table1.docx               -> tabella pronta per il paper (Word)
  - figures/*.png             -> boxplot/barplot di confronto tra le due coorti

Uso:
    python generate_table1.py

Modifica solo la sezione CONFIG qui sotto per adattarlo ai tuoi path/nomi colonna.
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG — modifica qui
# ============================================================

CSV_PATH = "/srv/python-projects/gene_environment_v2/data/componenti_ambientali_full.csv"
PARQUET_PATH = "/mnt/cresla_prod/genome_datasets/merged_csv/gen.parquet"

# Sorgente per la mappatura id -> coorte/generazione:
#   "parquet" -> usa PARQUET_PATH (gen.parquet)
#   "csv"     -> usa COHORT_MAPPING_CSV (es. generato da build_cohort_mapping.py
#                 leggendo gli header dei VCF, utile se il parquet è corrotto/pesante)
COHORT_SOURCE = "parquet"
COHORT_MAPPING_CSV = "output/table1/id_generation_mapping.csv"

OUTPUT_DIR = Path("output/table1")

# Nome colonna id nel CSV e nel parquet (se diverso, imposta ID_COL_PARQUET)
ID_COL_CSV = "id"
ID_COL_PARQUET = "id"

# Nome colonna nel parquet che identifica la coorte/generazione
COHORT_COL = "generazione"

# Se il parquet contiene più di 2 valori distinti in COHORT_COL, indica qui
# ESATTAMENTE quali due valori vuoi confrontare in Tabella 1 (altrimenti lo
# script si ferma e ti stampa i valori trovati per farti scegliere).
# Esempio: COHORT_VALUES = ["gen1", "gen2"]
COHORT_VALUES = None  # None = auto-detect (richiede esattamente 2 valori distinti)

# Etichette leggibili per la tabella (adatta ai due valori reali della tua coorte)
COHORT_LABELS = None  # es. {"gen1": "Coorte 1 (PARALS)", "gen2": "Coorte 2 (Svezia)"}

# Variabili categoriche e numeriche da includere in Tabella 1
CATEGORICAL_VARS = ["sex", "onset_site"]
NUMERIC_VARS = [
    "diagnostic_delay",
    "onset_age",
    "survival",
    "seminativi_1500",
    "vigneti_1500",
    "risaie_1500",
]

# Etichette leggibili per le variabili (per la tabella finale)
VAR_LABELS = {
    "sex": "Sesso",
    "onset_site": "Sede d'esordio",
    "diagnostic_delay": "Ritardo diagnostico (mesi)",
    "onset_age": "Età all'esordio (anni)",
    "survival": "Sopravvivenza (anni)",
    "seminativi_1500": "Seminativi entro 1500 m (%)",
    "vigneti_1500": "Vigneti entro 1500 m (%)",
    "risaie_1500": "Risaie entro 1500 m (%)",
}

ALPHA = 0.05

# ============================================================
# FUNZIONI
# ============================================================


def read_parquet_robust(path):
    """
    pd.read_parquet può fallire con:
      OSError: Could not open Parquet input source ... Exceeded size limit
    Succede quando pyarrow deserializza il thrift metadata del file (tipico
    con parquet scritti in tanti row-group/partizioni o con schema molto
    ampio) e i limiti di default (thrift_string_size_limit /
    thrift_container_size_limit) vengono superati.
    Qui si prova prima con limiti alzati via pyarrow, poi con fastparquet
    come ultima spiaggia.
    """
    import pyarrow.parquet as pq

    try:
        return pd.read_parquet(path)
    except OSError as e:
        if "size limit" not in str(e).lower():
            raise
        print("  -> limite thrift di default superato, riprovo con limiti alzati...")

    try:
        table = pq.read_table(
            path,
            thrift_string_size_limit=2_000_000_000,
            thrift_container_size_limit=2_000_000_000,
        )
        return table.to_pandas()
    except TypeError:
        # versioni di pyarrow che non accettano questi kwarg su read_table:
        # provo passandoli al ParquetFile
        pf = pq.ParquetFile(
            path,
            thrift_string_size_limit=2_000_000_000,
            thrift_container_size_limit=2_000_000_000,
        )
        return pf.read().to_pandas()
    except OSError:
        pass

    print("  -> ancora fallito con pyarrow, provo engine fastparquet...")
    try:
        return pd.read_parquet(path, engine="fastparquet")
    except ImportError:
        sys.exit(
            "ERRORE: impossibile leggere il parquet (limite thrift superato) e "
            "'fastparquet' non è installato. Installa con: pip install fastparquet"
        )
    except Exception as e:
        sys.exit(
            f"ERRORE: impossibile leggere il parquet con nessun metodo disponibile.\n"
            f"Ultimo errore: {e}\n"
            f"Il file potrebbe essere corrotto o troncato: verificane l'integrità "
            f"(es. `parquet-tools` o riscrivendolo dalla pipeline sorgente)."
        )


def load_data():
    print(f"Carico CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    if ID_COL_CSV not in df.columns:
        sys.exit(f"ERRORE: colonna id '{ID_COL_CSV}' non trovata nel CSV. Colonne disponibili: {list(df.columns)}")

    if COHORT_SOURCE == "csv":
        print(f"Carico mappatura coorte da CSV: {COHORT_MAPPING_CSV}")
        gen = pd.read_csv(COHORT_MAPPING_CSV)
    else:
        print(f"Carico parquet: {PARQUET_PATH}")
        gen = read_parquet_robust(PARQUET_PATH)

    if ID_COL_PARQUET not in gen.columns:
        sys.exit(f"ERRORE: colonna id '{ID_COL_PARQUET}' non trovata nel parquet. Colonne disponibili: {list(gen.columns)}")
    if COHORT_COL not in gen.columns:
        sys.exit(f"ERRORE: colonna coorte '{COHORT_COL}' non trovata nel parquet. Colonne disponibili: {list(gen.columns)}")

    gen = gen[[ID_COL_PARQUET, COHORT_COL]].drop_duplicates()

    merged = df.merge(
        gen, left_on=ID_COL_CSV, right_on=ID_COL_PARQUET, how="inner"
    )
    n_lost = len(df) - len(merged)
    if n_lost > 0:
        print(f"ATTENZIONE: {n_lost} pazienti del CSV non trovati nel parquet (esclusi dal merge).")

    return merged


def resolve_cohorts(merged):
    values = merged[COHORT_COL].dropna().unique().tolist()

    if COHORT_VALUES is not None:
        chosen = COHORT_VALUES
        missing = [v for v in chosen if v not in values]
        if missing:
            sys.exit(f"ERRORE: i valori COHORT_VALUES {missing} non sono presenti in '{COHORT_COL}'. Valori trovati: {values}")
    else:
        if len(values) != 2:
            sys.exit(
                f"ERRORE: trovati {len(values)} valori distinti in '{COHORT_COL}': {values}.\n"
                f"Imposta COHORT_VALUES = [valore1, valore2] in CONFIG per scegliere le due coorti da confrontare."
            )
        chosen = values

    labels = COHORT_LABELS or {v: str(v) for v in chosen}
    for v in chosen:
        labels.setdefault(v, str(v))

    sub = merged[merged[COHORT_COL].isin(chosen)].copy()
    print(f"Coorti selezionate: {chosen} -> N = {sub[COHORT_COL].value_counts().to_dict()}")
    return sub, chosen, labels


def is_normal(series, alpha=0.05):
    series = series.dropna()
    if len(series) < 8:
        return True  # troppo pochi dati per Shapiro, assume parametrico
    stat, p = stats.shapiro(series)
    return p > alpha


def fmt_p(p):
    if pd.isna(p):
        return "-"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def summarize_numeric(sub, var, cohort_col, groups):
    rows = []
    g1 = sub.loc[sub[cohort_col] == groups[0], var].dropna()
    g2 = sub.loc[sub[cohort_col] == groups[1], var].dropna()

    normal = is_normal(g1) and is_normal(g2)

    if normal:
        desc1 = f"{g1.mean():.2f} ± {g1.std():.2f}"
        desc2 = f"{g2.mean():.2f} ± {g2.std():.2f}"
        stat, p = stats.ttest_ind(g1, g2, equal_var=False, nan_policy="omit")
        test_used = "t-test (Welch)"
    else:
        desc1 = f"{g1.median():.2f} [{g1.quantile(.25):.2f}-{g1.quantile(.75):.2f}]"
        desc2 = f"{g2.median():.2f} [{g2.quantile(.25):.2f}-{g2.quantile(.75):.2f}]"
        if len(g1) > 0 and len(g2) > 0:
            stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
        else:
            p = np.nan
        test_used = "Mann-Whitney U"

    rows.append(
        {
            "variable": VAR_LABELS.get(var, var),
            "type": "numeric",
            "group1_n": len(g1),
            "group1_stat": desc1,
            "group2_n": len(g2),
            "group2_stat": desc2,
            "test": test_used,
            "p_value": p,
            "p_value_fmt": fmt_p(p),
        }
    )
    return rows


def summarize_categorical(sub, var, cohort_col, groups):
    rows = []
    ct = pd.crosstab(sub[var], sub[cohort_col])
    ct = ct[[c for c in groups if c in ct.columns]]

    if ct.shape[0] == 2 and (ct.values < 5).any():
        _, p = stats.fisher_exact(ct.values) if ct.shape == (2, 2) else (None, np.nan)
        test_used = "Fisher exact"
    else:
        chi2, p, _, _ = stats.chi2_contingency(ct)
        test_used = "Chi-square"

    first = True
    for level in ct.index:
        n1 = ct.loc[level, groups[0]] if groups[0] in ct.columns else 0
        n2 = ct.loc[level, groups[1]] if groups[1] in ct.columns else 0
        tot1 = ct[groups[0]].sum() if groups[0] in ct.columns else 0
        tot2 = ct[groups[1]].sum() if groups[1] in ct.columns else 0
        pct1 = 100 * n1 / tot1 if tot1 else 0
        pct2 = 100 * n2 / tot2 if tot2 else 0
        rows.append(
            {
                "variable": f"{VAR_LABELS.get(var, var)} - {level}" if not first else VAR_LABELS.get(var, var),
                "type": "categorical",
                "group1_n": tot1,
                "group1_stat": f"{n1} ({pct1:.1f}%)",
                "group2_n": tot2,
                "group2_stat": f"{n2} ({pct2:.1f}%)",
                "test": test_used if first else "",
                "p_value": p if first else np.nan,
                "p_value_fmt": fmt_p(p) if first else "",
            }
        )
        first = False
    return rows


def build_stats_table(sub, cohort_col, groups):
    all_rows = []
    for var in CATEGORICAL_VARS:
        all_rows.extend(summarize_categorical(sub, var, cohort_col, groups))
    for var in NUMERIC_VARS:
        all_rows.extend(summarize_numeric(sub, var, cohort_col, groups))
    return pd.DataFrame(all_rows)


# ------------------------------------------------------------
# GRAFICI
# ------------------------------------------------------------

def make_figures(sub, cohort_col, groups, labels, fig_dir):
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")
    palette = {groups[0]: "#4C72B0", groups[1]: "#DD8452"}

    plot_df = sub.copy()
    plot_df["Coorte"] = plot_df[cohort_col].map(labels)

    for var in NUMERIC_VARS:
        if var not in plot_df.columns:
            continue
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.boxplot(
            data=plot_df,
            x="Coorte",
            y=var,
            ax=ax,
            palette=[palette[g] for g in groups],
        )
        sns.stripplot(
            data=plot_df, x="Coorte", y=var, ax=ax, color="black", alpha=0.3, size=3, jitter=True
        )
        ax.set_title(VAR_LABELS.get(var, var))
        ax.set_xlabel("")
        ax.set_ylabel(VAR_LABELS.get(var, var))
        fig.tight_layout()
        fig.savefig(fig_dir / f"boxplot_{var}.png", dpi=200)
        plt.close(fig)

    for var in CATEGORICAL_VARS:
        if var not in plot_df.columns:
            continue
        fig, ax = plt.subplots(figsize=(5, 4))
        ct = pd.crosstab(plot_df["Coorte"], plot_df[var], normalize="index") * 100
        ct.plot(kind="bar", stacked=True, ax=ax, colormap="tab10")
        ax.set_ylabel("%")
        ax.set_xlabel("")
        ax.set_title(VAR_LABELS.get(var, var))
        ax.legend(title=var, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / f"barplot_{var}.png", dpi=200)
        plt.close(fig)

    print(f"Grafici salvati in: {fig_dir}")


# ------------------------------------------------------------
# TABELLA WORD
# ------------------------------------------------------------

def set_cell_shading(cell, color_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), color_hex)
    tcPr.append(shd)


def make_docx_table(stats_df, groups, labels, n_total, output_path):
    doc = Document()

    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)

    title = doc.add_paragraph()
    run = title.add_run("Tabella 1. Caratteristiche cliniche e ambientali delle due coorti")
    run.bold = True
    run.font.size = Pt(12)

    n1 = n_total.get(groups[0], 0)
    n2 = n_total.get(groups[1], 0)

    col_headers = [
        "Variabile",
        f"{labels[groups[0]]} (n={n1})",
        f"{labels[groups[1]]} (n={n2})",
        "Test",
        "p",
    ]

    table = doc.add_table(rows=1, cols=len(col_headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    widths = [Cm(6.0), Cm(4.0), Cm(4.0), Cm(3.0), Cm(2.0)]
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
        set_cell_shading(hdr_cells[i], "D9D9D9")

    for _, row in stats_df.iterrows():
        cells = table.add_row().cells
        values = [
            row["variable"],
            row["group1_stat"],
            row["group2_stat"],
            row["test"],
            row["p_value_fmt"],
        ]
        for i, v in enumerate(values):
            cells[i].text = "" if pd.isna(v) else str(v)
            cells[i].width = widths[i]
            for p in cells[i].paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i > 0 else WD_ALIGN_PARAGRAPH.LEFT
                for r in p.runs:
                    r.font.size = Pt(10)
                    # evidenzia p significative
                    if i == 4 and row["p_value"] is not np.nan and not pd.isna(row["p_value"]) and row["p_value"] < ALPHA:
                        r.bold = True

    note = doc.add_paragraph()
    note_run = note.add_run(
        "Variabili numeriche: media ± DS (t-test di Welch) se distribuzione normale, "
        "altrimenti mediana [IQR] (Mann-Whitney U). Variabili categoriche: n (%) "
        "(chi-quadrato o test esatto di Fisher se attese <5)."
    )
    note_run.italic = True
    note_run.font.size = Pt(8)

    doc.save(output_path)
    print(f"Tabella Word salvata in: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = OUTPUT_DIR / "figures"

    merged = load_data()
    sub, groups, labels = resolve_cohorts(merged)

    n_total = sub[COHORT_COL].value_counts().to_dict()

    stats_df = build_stats_table(sub, COHORT_COL, groups)

    csv_out = OUTPUT_DIR / "table1_stats.csv"
    stats_df.to_csv(csv_out, index=False)
    print(f"CSV statistiche salvato in: {csv_out}")

    make_figures(sub, COHORT_COL, groups, labels, fig_dir)

    docx_out = OUTPUT_DIR / "Table1.docx"
    make_docx_table(stats_df, groups, labels, n_total, docx_out)

    print("\nFatto. Output in:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()