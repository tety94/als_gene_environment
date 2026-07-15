#!/usr/bin/env python3
"""
Genera boxplot e forest plot per la differenza di età d'esordio (onset_age)
delle varianti significative. (ex analyze_variant_onset_age.py)

Le statistiche (mediana, delta, IC bootstrap, p-value, FDR) NON vengono
ricalcolate qui: sono già state calcolate una volta sola in modeling.py e
salvate a DB per ogni variante (vedi analysis/modeling.py e
db/repository.py). Questo script si limita a: (1) leggere quelle
statistiche dal DB, (2) leggere i dati grezzi paziente-per-paziente
(genotipo + onset_age) solo per disegnare i boxplot, (3) produrre i grafici.

NOVITA' rispetto alla versione precedente:
- Parallelizzazione a livello di exposure (ogni componente ambientale gira
  in un processo separato, dato che le run sono indipendenti: query DB
  diverse, output su directory diverse).
- Parametro `single_exposure`:
    * False (default) -> recupera dal DB tutte le exposure distinte
      disponibili in variant_results e le processa tutte, in parallelo.
    * True -> calcola solo per la exposure corrente (cfg.exposure), utile
      per debug rapido su una singola componente ambientale.

NOTA IMPORTANTE su get_config(): e' un singleton letto una volta sola dagli
env var (vedi config.py), quindi NON accetta e non puo' accettare un
parametro `exposure` per generare config diverse per ogni componente
ambientale. cfg.onset_age_out_dir e' percio' un path FISSO, identico per
tutti i worker. Per evitare che processi paralleli su exposure diverse
scrivano nella stessa cartella sovrascrivendosi a vicenda (boxplot,
onset_age_report_table.csv, forest_plot.png), ogni worker crea la propria
sottocartella <onset_age_out_dir>/<exposure>/.
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gene_environment.config import get_config
from gene_environment.db.connection import get_connection
from gene_environment.logging_utils import configure_logging, get_logger
from gene_environment.utils.id_utils import clean_sample_id

log = get_logger(__name__)


# --------------------------------------------------------------------------
# Config helper
# --------------------------------------------------------------------------

def _get_config_for_exposure(exposure: str | None = None):
    """Ritorna la config globale.

    get_config() e' un singleton letto una volta sola dagli env var: non
    esiste (e non serve) una config diversa per exposure. Il parametro
    `exposure` qui e' tenuto solo per compatibilita' di firma con le
    chiamate esistenti, ma non viene passato a get_config(). La
    differenziazione per exposure avviene invece a livello di directory di
    output, dentro _process_exposure.
    """
    return get_config()


# --------------------------------------------------------------------------
# Discovery delle exposure disponibili
# --------------------------------------------------------------------------

def get_available_exposures(cfg) -> list[str]:
    query = """
        SELECT DISTINCT exposure
        FROM variant_results
        WHERE completed = 1 AND onset_p_value IS NOT NULL
    """
    with get_connection() as conn:
        df = pd.read_sql(query, conn)
    return sorted(df["exposure"].dropna().unique().tolist())


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------

def load_onset_age(cfg) -> pd.DataFrame:
    df = pd.read_csv(cfg.env_file, usecols=[cfg.sample_id_col, cfg.target_col] + cfg.covariates)
    return df.set_index(cfg.sample_id_col)


def load_genotype_matrix_for_cohort(cfg, cohort: int) -> pd.DataFrame | None:
    combined_path = os.path.join(cfg.significant_matrix_dir, "combined_significant_variants.csv")
    if not os.path.exists(combined_path):
        log.warning("Manca %s (esegui prima extract_matrix)", combined_path)
        return None

    df_all = pd.read_csv(combined_path, index_col="id")
    df_cohort = df_all[df_all["generation"] == cohort].drop(columns=["generation"])
    if df_cohort.empty:
        log.warning("Nessun paziente per coorte %d in %s", cohort, combined_path)
        return None

    df_cohort.index = [clean_sample_id(idx) for idx in df_cohort.index]
    df_cohort.index.name = "id"
    return df_cohort


def load_db_stats(exposure: str, generation: int) -> pd.DataFrame:
    query = """
        SELECT variant, onset_n_mutati, onset_n_non_mutati, onset_median_mutati,
               onset_median_non_mutati, onset_delta_median, onset_ci_low, onset_ci_high,
               onset_p_value, onset_low_power, empirical_p
        FROM variant_results
        WHERE exposure = %s AND generation = %s AND completed = 1 AND onset_p_value IS NOT NULL
    """
    with get_connection() as conn:
        return pd.read_sql(query, conn, params=(exposure, generation))


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def make_boxplot(mutati, non_mutati, variant, cohort, out_dir):
    fig, ax = plt.subplots(figsize=(4, 5))
    ax.boxplot([non_mutati, mutati], tick_labels=["WT", "Mutato"], showmeans=True)
    ax.set_ylabel("Età d'esordio")
    ax.set_title(f"{variant}\ncoorte {cohort}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{variant}_cohort{cohort}.png"), dpi=150)
    plt.close(fig)


def make_forest_plot(stats_df: pd.DataFrame, out_path: str):
    stats_df = stats_df.sort_values(["variant", "cohort"]).reset_index(drop=True)
    if stats_df.empty:
        log.warning("Nessun dato per il forest plot")
        return

    y_labels = [f"{r.variant}  (coorte {r.cohort})" for r in stats_df.itertuples()]
    y_pos = np.arange(len(stats_df))[::-1]

    xerr_low = stats_df["onset_delta_median"] - stats_df["onset_ci_low"]
    xerr_high = stats_df["onset_ci_high"] - stats_df["onset_delta_median"]
    palette = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
    colors = stats_df["cohort"].map(palette).fillna("gray")

    fig, ax = plt.subplots(figsize=(7, max(4, 0.35 * len(stats_df))))
    ax.errorbar(stats_df["onset_delta_median"], y_pos, xerr=[xerr_low, xerr_high],
                fmt="none", ecolor="gray", capsize=3, zorder=1)
    ax.scatter(stats_df["onset_delta_median"], y_pos, c=colors, zorder=2)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Delta mediana onset_age (mutati - non mutati) [IC 95% bootstrap]")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=palette.get(c, "gray"), label=f"Coorte {c}")
        for c in sorted(stats_df["cohort"].unique())
    ]
    ax.legend(handles=handles, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Forest plot salvato in %s", out_path)


# --------------------------------------------------------------------------
# Core: report per singola exposure (funzione top-level -> picklabile per
# ProcessPoolExecutor)
# --------------------------------------------------------------------------

def _process_exposure(exposure: str) -> str:
    cfg = _get_config_for_exposure(exposure)
    configure_logging(cfg.log_dir)

    # cfg.onset_age_out_dir e' FISSO (stesso per tutti i processi, dato che
    # cfg e' un singleton). Ogni worker scrive quindi nella propria
    # sottocartella per exposure, per evitare che run parallele su
    # componenti ambientali diverse si sovrascrivano a vicenda.
    exp_out_dir = os.path.join(cfg.onset_age_out_dir, exposure)
    os.makedirs(exp_out_dir, exist_ok=True)
    box_dir = os.path.join(exp_out_dir, "boxplots")
    os.makedirs(box_dir, exist_ok=True)

    df_onset = load_onset_age(cfg)
    all_stats = []

    for cohort in cfg.report_cohorts:
        df_geno = load_genotype_matrix_for_cohort(cfg, cohort)
        db_stats = load_db_stats(exposure, cohort)
        if df_geno is None or db_stats.empty:
            continue

        df = df_geno.join(df_onset, how="inner")
        variants = [v for v in df_geno.columns if v in set(db_stats["variant"])]
        log.info("[%s] Coorte %d: %d pazienti, %d varianti con statistiche in DB",
                  exposure, cohort, len(df), len(variants))

        for variant in variants:
            row = db_stats.loc[db_stats["variant"] == variant].iloc[0]
            sub = df[[variant, cfg.target_col]].dropna()
            sub[variant] = pd.to_numeric(sub[variant], errors="coerce")
            mutati = sub.loc[sub[variant] == 1, cfg.target_col]
            non_mutati = sub.loc[sub[variant] == 0, cfg.target_col]

            if len(mutati) == 0 or len(non_mutati) == 0:
                continue

            make_boxplot(mutati, non_mutati, variant, cohort, box_dir)
            all_stats.append({"variant": variant, "cohort": cohort, **row.to_dict()})

    if not all_stats:
        log.info("[%s] Nessuna statistica onset_age disponibile per il report. Esco.", exposure)
        return exposure

    stats_df = pd.DataFrame(all_stats)
    stats_df.to_csv(os.path.join(exp_out_dir, "onset_age_report_table.csv"), index=False)
    make_forest_plot(stats_df, os.path.join(exp_out_dir, "forest_plot.png"))
    log.info("[%s] Report onset_age completato in %s", exposure, exp_out_dir)
    return exposure


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def run_report_onset_age(single_exposure: bool = False, n_workers: int | None = None) -> None:
    """
    Parametri
    ---------
    single_exposure : bool
        Se False (default), recupera dal DB tutte le exposure disponibili
        (DISTINCT exposure da variant_results) e le processa tutte, in
        parallelo.
        Se True, calcola solo per la exposure corrente (cfg.exposure) —
        utile per debug rapido su una singola componente ambientale.
    n_workers : int | None
        Numero di processi paralleli. Default: min(n. exposure, n. cpu).
    """
    base_cfg = get_config()
    configure_logging(base_cfg.log_dir)

    if single_exposure:
        exposures = [base_cfg.exposure]
    else:
        exposures = get_available_exposures(base_cfg)
        log.info("Trovate %d exposure da processare: %s", len(exposures), exposures)
        if not exposures:
            log.warning("Nessuna exposure trovata in variant_results. Esco.")
            return

    if len(exposures) == 1:
        # Nessun vantaggio a spawnare un processo per una sola exposure
        _process_exposure(exposures[0])
        return

    n_workers = n_workers or min(len(exposures), os.cpu_count() or 1)
    log.info("Avvio pool con %d worker per %d exposure", n_workers, len(exposures))

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_process_exposure, exp): exp for exp in exposures}
        for future in as_completed(futures):
            exp = futures[future]
            try:
                future.result()
            except Exception:
                log.exception("Errore processando exposure %s", exp)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report onset_age (boxplot + forest plot)")
    parser.add_argument(
        "--single-exposure",
        action="store_true",
        help="Se presente, calcola solo per la exposure corrente (cfg.exposure) "
             "invece che per tutte quelle disponibili nel DB.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Numero di processi paralleli (default: min(n. exposure, n. cpu)).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_report_onset_age(single_exposure=args.single_exposure, n_workers=args.workers)