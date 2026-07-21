"""
Step 8 - figure e report.md sui top loci, invariato nella struttura rispetto
all'originale (boxplot per genotipo, scatter fenotipo x esposizione colorato
per genotipo, forest plot di beta_I, executive summary con i caveat
metodologici). Il dosaggio arriva da colonne del DataFrame gia' costruito
(niente rilettura VCF).
"""
from __future__ import annotations

import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from gene_environment.logging_utils import get_logger

from vqtl.config import VqtlConfig
from vqtl.core.data import VqtlDataset
from vqtl.core.filter_candidates import genomic_inflation

log = get_logger(__name__)

N_TOP_FIGURES = 10


def _boxplot_by_genotype(dosage, y, snp_id, exp_col, out_path):
    df = pd.DataFrame({"genotype": pd.array(np.round(dosage), dtype="Int64"), "phenotype": y}).dropna()
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.boxplot(data=df, x="genotype", y="phenotype", hue="genotype", ax=ax, palette="Blues", legend=False)
    sns.stripplot(data=df, x="genotype", y="phenotype", ax=ax, color="black", alpha=0.3, size=3)
    ax.set_title(f"{snp_id}\n(exposure: {exp_col})", fontsize=9)
    ax.set_xlabel("Genotype (ALT dosage)")
    ax.set_ylabel("Phenotype")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _scatter_by_exposure(dosage, y, exposure, snp_id, exp_col, out_path):
    df = pd.DataFrame({"genotype": pd.array(np.round(dosage), dtype="Int64"), "phenotype": y, "exposure": exposure}).dropna()
    fig, ax = plt.subplots(figsize=(4, 4))
    for geno, color in zip([0, 1, 2], ["#c6dbef", "#6baed6", "#08306b"]):
        sub = df[df["genotype"] == geno]
        if len(sub) == 0:
            continue
        ax.scatter(sub["exposure"], sub["phenotype"], s=12, color=color, label=f"genotype={geno}", alpha=0.7)
        if len(sub) > 2:
            z = np.polyfit(sub["exposure"], sub["phenotype"], 1)
            xs = np.linspace(sub["exposure"].min(), sub["exposure"].max(), 20)
            ax.plot(xs, np.polyval(z, xs), color=color, linewidth=1.5)
    ax.set_xlabel(exp_col)
    ax.set_ylabel("Phenotype")
    ax.set_title(f"{snp_id} x {exp_col}", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _forest_plot(top_df, out_path):
    fig, ax = plt.subplots(figsize=(6, max(3, 0.4 * len(top_df))))
    y_pos = np.arange(len(top_df))
    ci = 1.96 * top_df["SE"]
    ax.errorbar(top_df["beta_I"], y_pos, xerr=ci, fmt="o", color="#08306b", ecolor="#6baed6", capsize=3)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    labels = [f"{row.SNP} x {row.exposure}" for row in top_df.itertuples()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("\u03b2_interaction (95% CI)")
    ax.set_title("Top interaction loci - forest plot")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_report(
    dataset: VqtlDataset, vcfg: VqtlConfig, cohort_dir: str, generation: int,
    vqtl_df: pd.DataFrame, candidates: pd.DataFrame, interaction_df: pd.DataFrame,
    rge_df: pd.DataFrame, perm_df: pd.DataFrame, robustness_df: pd.DataFrame,
    target_col: str,
) -> str:
    fig_dir = os.path.join(cohort_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    inv_mapping = {v: k for k, v in dataset.mapping.items()}
    top_interactions = interaction_df.sort_values("pval").head(N_TOP_FIGURES).reset_index(drop=True) if not interaction_df.empty else interaction_df
    y = dataset.df[target_col].to_numpy(dtype=float)

    fig_paths = []
    for i, row in top_interactions.iterrows():
        snp_id, exp_raw = row["SNP"], row["exposure"]
        safe_col = inv_mapping.get(snp_id)
        exp_std_col = dataset.exposure_std_cols.get(exp_raw)
        if safe_col is None or exp_std_col is None:
            continue
        dosage = dataset.df[safe_col].to_numpy(dtype=float)
        exposure_vals = dataset.df[exp_std_col].to_numpy(dtype=float)

        box_path = os.path.join(fig_dir, f"locus{i + 1}_{snp_id}_boxplot.png")
        _boxplot_by_genotype(dosage, y, snp_id, exp_raw, box_path)
        scatter_path = os.path.join(fig_dir, f"locus{i + 1}_{snp_id}_scatter.png")
        _scatter_by_exposure(dosage, y, exposure_vals, snp_id, exp_raw, scatter_path)
        fig_paths.append((snp_id, exp_raw, box_path, scatter_path))

    forest_path = os.path.join(fig_dir, "forest_top_interactions.png")
    if len(top_interactions) > 0:
        _forest_plot(top_interactions, forest_path)
    log.info("Generate %d coppie di figure per-locus + forest plot.", len(fig_paths))

    lam_note = ""
    if not vqtl_df.empty and "Z" in vqtl_df.columns:
        lam_note = f"{genomic_inflation(vqtl_df['Z'].values):.3f}"

    p_col_used = vcfg.filter_p_column if (not vqtl_df.empty and vcfg.filter_p_column in vqtl_df.columns) else "P"

    md = []
    md.append(f"# Report vQTL / G x E -- generazione (coorte): {generation}\n")
    md.append("## Executive summary\n")
    md.append(f"- Campioni analisi-ready: **{len(dataset.df)}**\n")
    md.append(f"- Esposizioni testate: **{', '.join(vcfg.exposures)}**\n")
    md.append(f"- SNP scansionate (Step 3): **{len(vqtl_df)}**\n")
    md.append(
        f"- SNP candidate dopo il filtro Step 4: **{len(candidates)}** (filtrate su `{p_col_used}`"
        f"{' < ' + str(vcfg.filter_p_threshold) if not vcfg.filter_top_n else ', top_n=' + str(vcfg.filter_top_n)})\n"
    )
    md.append(f"- Inflazione genomica (lambda_GC): **{lam_note}**\n")
    if lam_note and float(lam_note) > 1.5:
        md.append(
            "  > **Attenzione:** lambda_GC e' marcatamente > 1. Le p-value asintotiche dello Step 3 sono "
            "anti-conservative per un predittore di dosaggio discreto (0/1/2) e vanno trattate come "
            "screening. Preferisci `P_gc` e le p-value empiriche delle permutazioni (Step 7) per l'inferenza "
            "sui top loci.\n"
        )
    md.append(f"- Test di interazione eseguiti: **{len(interaction_df)}** (coppie SNP candidata x esposizione)\n")
    if not rge_df.empty:
        n_rge = int(rge_df.get("rGE_flag", pd.Series(dtype=bool)).sum())
        n_het = int(rge_df.get("heteroscedasticity_flag", pd.Series(dtype=bool)).sum())
        md.append(f"- Coppie SNP x esposizione flaggate per rGE (p<{vcfg.rge_het_alpha}): **{n_rge}** / {len(rge_df)} (flaggate, non escluse)\n")
        md.append(f"- Coppie flaggate per eteroschedasticita' (Breusch-Pagan p<{vcfg.rge_het_alpha}): **{n_het}** / {len(rge_df)}\n")

    md.append("\n## Top loci vQTL (scan genoma-wide)\n")
    if not vqtl_df.empty:
        md.append(vqtl_df.sort_values("P").head(10).to_markdown(index=False))
    md.append("\n")

    md.append("\n## Top loci di interazione SNP x esposizione\n")
    if not top_interactions.empty:
        md.append(top_interactions.to_markdown(index=False))
    md.append("\n")

    if not perm_df.empty:
        md.append("\n## Robustezza tramite permutazione dei top loci (interazione + varianza per genotipo)\n")
        md.append(perm_df.to_markdown(index=False))
        md.append(
            "\n> Interazione: p-value empirica = (1 + #permutazioni con |beta_I_perm| >= |beta_I_osservato|) / (n_perm + 1), "
            "permutazione Freedman-Lane sui residui del modello ridotto. "
            "Varianza per genotipo (levene_stat_observed/levene_pval): test di Levene (Brown-Forsythe) "
            "permutazionale -- si permutano le ETICHETTE di genotipo (non i residui) sul fenotipo "
            "residualizzato, stessa infrastruttura di permutazione del test di interazione ma statistica "
            "diversa; conferma assumption-light dell'effetto di varianza rilevato dallo scan Step 3 per "
            "questo locus, senza le assunzioni asintotiche della quantile regression. Indipendente "
            "dall'esposizione (stesso valore per righe dello stesso SNP). "
            "Grosse discrepanze tra p asintotica e p empirica (in entrambi i test) indicano che le SE/i "
            "p-value asintotici non sono affidabili per quel locus.\n"
        )

    if not robustness_df.empty:
        md.append("\n## Sensibilita' a trasformazioni del fenotipo / outlier (top loci)\n")
        md.append(robustness_df.to_markdown(index=False))
        md.append(
            "\n> Un locus la cui direzione/significativita' di beta_I e' stabile su `original`, `log_transform`, "
            "`rank_inverse_normal` e `outliers_removed` e' piu' verosimilmente un effetto reale che un artefatto "
            "della distribuzione del fenotipo o di poche osservazioni influenti.\n"
        )

    md.append("\n## Limiti metodologici noti (leggere prima di trarre conclusioni)\n")
    md.append(
        "- **Punto cieco del disegno a due step:** lo Step 3 e' uno *screening* per SNP il cui dosaggio modula "
        "la dispersione del fenotipo; lo Step 5 testa l'interazione solo per le SNP che passano questo filtro. "
        "Una SNP con vera interazione G x E ma **senza effetto marginale sulla varianza** puo' essere persa "
        "interamente da questo disegno. Se hai SNP/geni candidati a priori, testali direttamente con lo Step 5 "
        "indipendentemente dalla loro p-value allo Step 3.\n"
    )
    md.append("- **Inflazione della SE asintotica:** vedi nota su lambda_GC sopra. Usa le p-value da permutazione (Step 7) come parola finale su ogni locus destinato a pubblicazione/follow-up.\n")
    md.append("- **Loci flaggati per rGE:** un'associazione SNP~esposizione significativa non invalida di per se' un risultato G x E, ma complica l'interpretazione causale e va discussa esplicitamente.\n")
    md.append("- **Test multipli:** le tabelle di interazione/rGE/eteroschedasticita' riportano p-value nominali, non corrette per il numero di combinazioni testate (usa Bonferroni o FDR).\n")

    md.append("\n## Figure\n")
    md.append("![Manhattan plot](figures/manhattan_vqtl.png)\n")
    md.append("![QQ plot](figures/qq_vqtl.png)\n")
    if os.path.exists(forest_path):
        md.append("![Forest plot](figures/forest_top_interactions.png)\n")
    for snp_id, exp_raw, box_path, scatter_path in fig_paths:
        md.append(f"\n### {snp_id} x {exp_raw}\n")
        md.append(f"![boxplot](figures/{os.path.basename(box_path)}) ![scatter](figures/{os.path.basename(scatter_path)})\n")

    report_path = os.path.join(cohort_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(md))
    log.info("Scritto %s", report_path)
    return report_path
