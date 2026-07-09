#!/usr/bin/env python3
"""
Genera boxplot e forest plot per la differenza di età d'esordio (onset_age)
delle varianti significative. (ex analyze_variant_onset_age.py)
"""
from __future__ import annotations

import os

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

COHORTS = (1, 2)  # gen3 escluso di proposito, come nell'originale


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


def load_db_stats(cfg, generation: int) -> pd.DataFrame:
    query = """
        SELECT variant, onset_n_mutati, onset_n_non_mutati, onset_median_mutati,
               onset_median_non_mutati, onset_delta_median, onset_ci_low, onset_ci_high,
               onset_p_value, onset_low_power, empirical_p
        FROM variant_results
        WHERE exposure = %s AND generation = %s AND completed = 1 AND onset_p_value IS NOT NULL
    """
    with get_connection() as conn:
        return pd.read_sql(query, conn, params=(cfg.exposure, generation))


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
    colors = stats_df["cohort"].map({1: "tab:blue", 2: "tab:orange"}).fillna("gray")

    fig, ax = plt.subplots(figsize=(7, max(4, 0.35 * len(stats_df))))
    ax.errorbar(stats_df["onset_delta_median"], y_pos, xerr=[xerr_low, xerr_high],
                fmt="none", ecolor="gray", capsize=3, zorder=1)
    ax.scatter(stats_df["onset_delta_median"], y_pos, c=colors, zorder=2)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Delta mediana onset_age (mutati - non mutati) [IC 95% bootstrap]")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue", label="Coorte 1"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:orange", label="Coorte 2"),
    ]
    ax.legend(handles=handles, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Forest plot salvato in %s", out_path)


def run_report_onset_age() -> None:
    cfg = get_config()
    configure_logging(cfg.log_dir)

    os.makedirs(cfg.onset_age_out_dir, exist_ok=True)
    box_dir = os.path.join(cfg.onset_age_out_dir, "boxplots")
    os.makedirs(box_dir, exist_ok=True)

    df_onset = load_onset_age(cfg)
    all_stats = []

    for cohort in COHORTS:
        df_geno = load_genotype_matrix_for_cohort(cfg, cohort)
        db_stats = load_db_stats(cfg, cohort)
        if df_geno is None or db_stats.empty:
            continue

        df = df_geno.join(df_onset, how="inner")
        variants = [v for v in df_geno.columns if v in set(db_stats["variant"])]
        log.info("Coorte %d: %d pazienti, %d varianti con statistiche in DB", cohort, len(df), len(variants))

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
        log.info("Nessuna statistica onset_age disponibile per il report. Esco.")
        return

    stats_df = pd.DataFrame(all_stats)
    stats_df.to_csv(os.path.join(cfg.onset_age_out_dir, "onset_age_report_table.csv"), index=False)
    make_forest_plot(stats_df, os.path.join(cfg.onset_age_out_dir, "forest_plot.png"))
    log.info("Report onset_age completato in %s", cfg.onset_age_out_dir)


if __name__ == "__main__":
    run_report_onset_age()
