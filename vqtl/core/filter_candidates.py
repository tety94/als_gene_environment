"""
Step 4 - visualizzazione + filtro dei candidati vQTL.

Logica statistica invariata rispetto all'originale (correzione genomic
control lambda_GC, colonna P_gc, filtro di default su P_gc non sulla P
asintotica grezza -- vedi l'audit descritto nel README storico). L'unica
aggiunta e' l'FDR (Benjamini-Hochberg) via `gene_environment.utils.
stats_utils.add_fdr`, riusato cosi' com'e' invece di reimplementare
multipletests qui: non c'era nel progetto originale ma e' un'informazione
utile da avere accanto a P_gc, e gene_environment lo fa gia' per il proprio
test di interazione.
"""
from __future__ import annotations

import os

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gene_environment.logging_utils import get_logger
from gene_environment.utils.stats_utils import add_fdr

from vqtl.config import VqtlConfig

log = get_logger(__name__)


def genomic_inflation(z_scores: np.ndarray) -> float:
    z_scores = z_scores[~np.isnan(z_scores)]
    if len(z_scores) == 0:
        return np.nan
    return float(np.median(z_scores ** 2) / 0.4549364)


def _manhattan_plot(df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    df = df.sort_values(["CHR", "POS"]).reset_index(drop=True)
    df["CHR"] = df["CHR"].astype(str)
    chrom_order = sorted(df["CHR"].unique(), key=lambda x: (len(x), x))
    offset = 0
    colors = ["#3B4D8C", "#8C3B5D"]
    ticks, tick_labels = [], []
    for i, chrom in enumerate(chrom_order):
        sub = df[df["CHR"] == chrom]
        x = sub["POS"] + offset
        ax.scatter(x, -np.log10(sub["P"]), s=8, color=colors[i % 2])
        ticks.append(x.mean())
        tick_labels.append(chrom)
        offset += sub["POS"].max() + 1 if len(sub) else 0
    ax.axhline(-np.log10(5e-8), color="red", linestyle="--", linewidth=1, label="genome-wide (5e-8)")
    ax.axhline(-np.log10(1e-5), color="orange", linestyle="--", linewidth=1, label="suggestive (1e-5)")
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("-log10(P)")
    ax.set_title("vQTL Manhattan plot (\u03b2_QI)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Scritto %s", out_path)


def _qq_plot(pvals: np.ndarray, z_scores: np.ndarray, out_path: str) -> float:
    pvals = np.sort(pvals[pvals > 0])
    n = len(pvals)
    expected = -np.log10(np.arange(1, n + 1) / (n + 1))
    observed = -np.log10(pvals)
    lam = genomic_inflation(z_scores)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(expected, observed, s=8, color="#3B4D8C")
    max_v = max(expected.max(), observed.max()) if n else 1
    ax.plot([0, max_v], [0, max_v], color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Expected -log10(P)")
    ax.set_ylabel("Observed -log10(P)")
    ax.set_title(f"QQ plot (\u03bb_GC = {lam:.3f})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Scritto %s (lambda_GC = %.3f)", out_path, lam)
    return lam


def regenerate_plots(vqtl_df: pd.DataFrame, fig_dir: str) -> float:
    """Rigenera SOLO manhattan/qq plot da un vqtl_df gia' completo di P/Z
    (es. gia' in cache a DB da un run precedente), senza toccare P_gc/
    candidati/DB -- usata nel percorso di short-circuit di cli.py quando una
    generazione ha gia' risultati significativi salvati e lo scan viene
    saltato del tutto, ma le figure vanno comunque (ri)generate per il
    report."""
    os.makedirs(fig_dir, exist_ok=True)
    if vqtl_df.empty:
        return float("nan")
    _manhattan_plot(vqtl_df, os.path.join(fig_dir, "manhattan_vqtl.png"))
    return _qq_plot(vqtl_df["P"].values, vqtl_df["Z"].values, os.path.join(fig_dir, "qq_vqtl.png"))


def filter_candidates(vqtl_df: pd.DataFrame, vcfg: VqtlConfig, fig_dir: str, generation: int) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Ritorna (vqtl_df_con_Pgc_e_fdr, candidati, lambda_GC). Persiste anche
    a DB: P_gc/fdr_gc per tutte le varianti (update_gc_correction),
    is_candidate per il sottoinsieme selezionato (mark_candidates), e
    risincronizza vqtl_scan_results_significant (sync_scan_significant) --
    e' quest'ultima tabella a decidere se un run successivo per la stessa
    generazione puo' saltare lo scan (vedi cli.py)."""
    from vqtl.db import repository as repo

    os.makedirs(fig_dir, exist_ok=True)
    if vqtl_df.empty:
        log.warning("vqtl_results vuoto: nessun plot/filtro possibile.")
        return vqtl_df, vqtl_df, float("nan")

    _manhattan_plot(vqtl_df, os.path.join(fig_dir, "manhattan_vqtl.png"))
    lam = _qq_plot(vqtl_df["P"].values, vqtl_df["Z"].values, os.path.join(fig_dir, "qq_vqtl.png"))

    df = vqtl_df.copy()
    if lam and lam > 0 and not np.isnan(lam):
        chi2_corrected = df["Z"] ** 2 / lam
        df["P_gc"] = 1 - stats.chi2.cdf(chi2_corrected, df=1)
        log.info("Genomic inflation lambda_GC = %.3f -- aggiunta colonna P_gc.", lam)
        if lam > 1.5:
            log.warning(
                "lambda_GC = %.3f (>> 1). La P asintotica dello Step 3 e' nota per essere "
                "anti-conservativa con un predittore di dosaggio discreto: trattala come "
                "screening. Usa P_gc per una lista candidati piu' conservativa e conferma "
                "sempre i top loci con la p-value empirica delle permutazioni (Step 7).", lam,
            )
    else:
        df["P_gc"] = df["P"]
        log.warning("Impossibile calcolare lambda_GC; P_gc impostata uguale a P.")

    df = add_fdr(df, p_col="P_gc", fdr_col="fdr_gc")
    repo.update_gc_correction(generation, df[["SNP", "P_gc", "fdr_gc"]].rename(columns={"SNP": "variant", "P_gc": "p_gc"}).to_dict("records"))

    p_col = vcfg.filter_p_column
    if p_col not in df.columns:
        log.warning("filter_p_column='%s' non trovata nei risultati; uso 'P'.", p_col)
        p_col = "P"

    if vcfg.filter_top_n:
        candidates = df.sort_values(p_col).head(vcfg.filter_top_n)
        log.info("Filtro: top_n=%d (per %s)", vcfg.filter_top_n, p_col)
    else:
        candidates = df[df[p_col] < vcfg.filter_p_threshold]
        log.info("Filtro: %s < %s", p_col, vcfg.filter_p_threshold)

    log.info("Candidati selezionati: %d / %d", len(candidates), len(df))

    old_candidates = set(df[df["is_candidate"] == 1]["SNP"]) if "is_candidate" in df.columns else set()
    new_candidates = set(candidates["SNP"])
    dropped = old_candidates - new_candidates
    if dropped:
        log.warning(
            "%d varianti erano candidate in un run precedente e non lo sono piu' (soglia/top_n cambiati): "
            "ripulisco le loro righe da interaction/rge_het/robustness/permutation.", len(dropped),
        )
        repo.clear_downstream_for_variants(generation, list(dropped))

    repo.mark_candidates(generation, candidates["SNP"].tolist())
    repo.sync_scan_significant(generation)
    candidates = candidates.assign(is_candidate=1)
    return df, candidates, lam
