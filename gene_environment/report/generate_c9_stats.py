"""
generation_stats.py

Per ciascuna generazione (1 e 2):
  - per ogni colonna-variante (0/1), divide il gruppo in 0 vs 1
  - calcola test statistici per: sex, onset_site, diagnostic_delay,
    education_years, survival, survival_null (missingness), mutaz_bin
  - salva un CSV con tutti i risultati
  - genera grafici per i risultati significativi (p < ALPHA)
  - genera un report Word per generazione

Alla fine produce anche un report Word "combinato" che affianca, per ogni
coppia (variante, variabile) significativa in almeno una generazione, i
grafici delle due generazioni.

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

ID_COL = "id"
GENERATION_COL = "generation"

SEX_COL = "sex"
ONSET_SITE_COL = "onset_site"
DIAGNOSTIC_DELAY_COL = "diagnostic_delay"
EDUCATION_YEARS_COL = "education_years"
SURVIVAL_COL = "survival"
MUTAZ_RAW_COL = "mutaz"

ALPHA = 0.05

CATEGORICAL_VARS = [SEX_COL, ONSET_SITE_COL, "mutaz_bin", "survival_null"]
CONTINUOUS_VARS = [DIAGNOSTIC_DELAY_COL, EDUCATION_YEARS_COL, SURVIVAL_COL]


# ----------------------------------------------------------------------
# Preparazione dati
# ----------------------------------------------------------------------
def load_data():
    df = pd.read_csv(MERGED_CSV)
    with open(VARIANT_COLS_FILE) as f:
        variant_cols = json.load(f)
    variant_cols = [c for c in variant_cols if c in df.columns]

    # variabili derivate
    df["mutaz_bin"] = df[MUTAZ_RAW_COL].astype(str).str.contains("C9ORF72", na=False).astype(int)
    df["survival_null"] = df[SURVIVAL_COL].isna()
    df[EDUCATION_YEARS_COL] = pd.to_numeric(df[EDUCATION_YEARS_COL], errors="coerce").astype("Int64")

    log.info("Dati caricati: %d righe, %d colonne-varianti", len(df), len(variant_cols))
    return df, variant_cols


# ----------------------------------------------------------------------
# Test statistici
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
    return {"test": test_name, "pvalue": p, "n0": n0, "n1": n1, "table": table.to_dict()}


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
            continue  # variante monomorfica in questa generazione, salto

        for var_col in CATEGORICAL_VARS:
            res = test_categorical(df_gen.assign(_grp=group), "_grp", var_col)
            rows.append({"variant": variant, "variable": var_col, "type": "categorical", **res})

        for var_col in CONTINUOUS_VARS:
            res = test_continuous(df_gen.assign(_grp=group), "_grp", var_col)
            rows.append({"variant": variant, "variable": var_col, "type": "continuous", **res})

    results = pd.DataFrame(rows)

    # Correzione Bonferroni: pvalue_bonf = min(1, pvalue * n_test_eseguiti)
    # n_test_eseguiti = numero di test con un pvalue effettivamente calcolato
    # (esclude gli 'n/a' per gruppi troppo piccoli/mancanti).
    n_tests = results["pvalue"].notna().sum()
    results["pvalue_bonf"] = (results["pvalue"] * n_tests).clip(upper=1.0)
    log.info("Correzione Bonferroni applicata su %d test eseguiti", n_tests)

    return results


# ----------------------------------------------------------------------
# Grafici
# ----------------------------------------------------------------------
def plot_categorical(df_gen: pd.DataFrame, variant: str, var_col: str, generation: int, out_path: Path):
    sub = df_gen[[variant, var_col]].dropna()
    prop = pd.crosstab(sub[variant], sub[var_col], normalize="index")
    fig, ax = plt.subplots(figsize=(6, 4))
    prop.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title(f"gen{generation} | {variant} vs {var_col}")
    ax.set_xlabel(f"{variant} (0/1)")
    ax.set_ylabel("proporzione")
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
            log.warning("Plot fallito per %s/%s: %s", variant, var_col, e)

    log.info("Generation %d: %d grafici generati (p < %.2f)", generation, sig["plot_path"].notna().sum(), ALPHA)
    return sig


# ----------------------------------------------------------------------
# Report Word
# ----------------------------------------------------------------------
def build_word_report(results: pd.DataFrame, sig: pd.DataFrame, generation: int, out_path: Path):
    from docx import Document
    from docx.shared import Inches

    doc = Document()
    doc.add_heading(f"Analisi varianti - Generazione {generation}", level=1)
    doc.add_paragraph(
        f"Totale test eseguiti: {len(results)}. "
        f"Risultati significativi dopo correzione Bonferroni (p_bonf < {ALPHA}): "
        f"{(results['pvalue_bonf'] < ALPHA).sum()}."
    )

    doc.add_heading("Tabella risultati significativi (p_bonf < {:.2f})".format(ALPHA), level=2)
    table = doc.add_table(rows=1, cols=7)
    table.style = "Light Grid Accent 1"
    hdr = table.rows[0].cells
    for i, h in enumerate(["Variante", "Variabile", "Test", "p-value", "p-value (Bonferroni)", "N gruppo 0", "N gruppo 1"]):
        hdr[i].text = h

    for _, row in sig.sort_values("pvalue_bonf").iterrows():
        cells = table.add_row().cells
        cells[0].text = str(row["variant"])
        cells[1].text = str(row["variable"])
        cells[2].text = str(row["test"])
        cells[3].text = f"{row['pvalue']:.4g}"
        cells[4].text = f"{row['pvalue_bonf']:.4g}"
        cells[5].text = str(row.get("n0", ""))
        cells[6].text = str(row.get("n1", ""))

    doc.add_heading("Grafici", level=2)
    for _, row in sig.sort_values("pvalue_bonf").iterrows():
        if not row["plot_path"]:
            continue
        doc.add_paragraph(f"{row['variant']} vs {row['variable']} (p = {row['pvalue']:.4g}, p_bonf = {row['pvalue_bonf']:.4g})")
        doc.add_picture(row["plot_path"], width=Inches(5))

    doc.save(out_path)
    log.info("Report salvato: %s", out_path)


def build_combined_report(results1: pd.DataFrame, results2: pd.DataFrame,
                           sig1: pd.DataFrame, sig2: pd.DataFrame, out_path: Path):
    from docx import Document
    from docx.shared import Inches

    doc = Document()
    doc.add_heading("Confronto generazione 1 vs generazione 2", level=1)

    keys1 = set(zip(sig1["variant"], sig1["variable"]))
    keys2 = set(zip(sig2["variant"], sig2["variable"]))
    all_keys = sorted(keys1 | keys2)

    doc.add_paragraph(
        f"Coppie (variante, variabile) significative (Bonferroni) in almeno una generazione: {len(all_keys)}. "
        f"Il p-value mostrato è sempre quello calcolato (su tutti i {len(results1)} test per generazione); "
        f"l'immagine è presente solo se quella generazione ha raggiunto p_bonf < {ALPHA} per quella coppia."
    )

    def lookup_pvalues(results: pd.DataFrame, variant: str, var_col: str):
        row = results[(results["variant"] == variant) & (results["variable"] == var_col)]
        if row.empty or pd.isna(row["pvalue"].iloc[0]):
            return None, None
        return row["pvalue"].iloc[0], row["pvalue_bonf"].iloc[0]

    for variant, var_col in all_keys:
        doc.add_heading(f"{variant} vs {var_col}", level=2)

        p1, p1b = lookup_pvalues(results1, variant, var_col)
        p2, p2b = lookup_pvalues(results2, variant, var_col)

        def fmt(p, pb):
            if p is None:
                return "n/d"
            star = " *" if pb is not None and pb < ALPHA else ""
            return f"p={p:.4g}, p_bonf={pb:.4g}{star}"

        doc.add_paragraph(f"Generazione 1: {fmt(p1, p1b)}    |    Generazione 2: {fmt(p2, p2b)}    (* = p_bonf < {ALPHA})")

        row1 = sig1[(sig1["variant"] == variant) & (sig1["variable"] == var_col)]
        row2 = sig2[(sig2["variant"] == variant) & (sig2["variable"] == var_col)]

        if not row1.empty and row1["plot_path"].iloc[0]:
            doc.add_picture(row1["plot_path"].iloc[0], width=Inches(3.2))
        if not row2.empty and row2["plot_path"].iloc[0]:
            doc.add_picture(row2["plot_path"].iloc[0], width=Inches(3.2))

    doc.save(out_path)
    log.info("Report combinato salvato: %s", out_path)


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
        log.info("Generazione %d: %d campioni", generation, len(df_gen))
        if df_gen.empty:
            log.warning("Generazione %d vuota, salto.", generation)
            continue

        results = run_tests_for_generation(df_gen, variant_cols)
        results_by_gen[generation] = results
        results_csv = STATS_DIR / f"gen{generation}_variant_stats.csv"
        results.drop(columns=["table"], errors="ignore").to_csv(results_csv, index=False)
        log.info("CSV risultati generazione %d salvato: %s", generation, results_csv)

        sig = generate_plots(df_gen, results, generation, PLOTS_DIR)
        sig_by_gen[generation] = sig

        build_word_report(
            results, sig, generation, REPORTS_DIR / f"gen{generation}_report.docx"
        )

    if 1 in sig_by_gen and 2 in sig_by_gen:
        build_combined_report(
            results_by_gen[1], results_by_gen[2],
            sig_by_gen[1], sig_by_gen[2],
            REPORTS_DIR / "combined_report.docx"
        )


if __name__ == "__main__":
    main()