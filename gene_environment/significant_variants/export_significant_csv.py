#!/usr/bin/env python3
"""Independent, repeatable export script for the currently significant variants.

Can be rerun at any time (even while the main run is still in progress on
other variants): it reads the CURRENT state of the variant_results table,
recomputes the FDR on the results with "high" permutations
(iterations = N_PERM_HIGH) already completed, selects those below threshold
and writes a CSV to a SEPARATE folder (config: SIGNIFICANT_EXPORT_DIR),
different from the one used by the genotype-extraction pipeline
(SIGNIFICANT_MATRIX_DIR), with:
  - the observed coefficient and the model's empirical p-value/FDR
  - the onset_age difference statistics (medians, delta, bootstrap CI,
    p-value) already saved on the same row by modeling.py
  - the gene name, if already annotated

Each run writes both a timestamped snapshot and an always-up-to-date
"significant_variants_latest.csv" file, so a downstream consumer
(dashboard, notebook, other script) can always point at the same path
without worrying about the timestamp.

Usage: python -m gene_environment.significant_variants.export_significant_csv
"""
from __future__ import annotations

import os
from datetime import datetime

import pandas as pd

from gene_environment.config import get_config
from gene_environment.db.connection import get_connection
from gene_environment.db.repository import get_significant_results, load_raw_significant_results
from gene_environment.logging_utils import configure_logging, get_logger
from gene_environment.utils.stats_utils import add_fdr

log = get_logger(__name__)

RESULT_COLUMNS = [
    "variant", "chromosome", "position", "mutation", "gene", "gene_name",
    "mutati", "non_mutati", "obs_coef", "mean_coef", "sd_coef", "empirical_p", "fdr", "iterations", "balance",
    "onset_n_mutati", "onset_n_non_mutati", "onset_median_mutati", "onset_median_non_mutati",
    "onset_delta_median", "onset_ci_low", "onset_ci_high", "onset_p_value", "onset_effect_size",
    "onset_low_power", "onset_method",
]


def fetch_current_results(exposure: str, generation: int, iterations: int) -> pd.DataFrame:
    query = """
        SELECT vr.variant, vr.chromosome, vr.position, vr.mutation, vr.gene,
               vr.mutati, vr.non_mutati, vr.obs_coef, vr.mean_coef, vr.sd_coef,
               vr.empirical_p, vr.iterations, vr.balance,
               vr.onset_n_mutati, vr.onset_n_non_mutati, vr.onset_median_mutati, vr.onset_median_non_mutati,
               vr.onset_delta_median, vr.onset_ci_low, vr.onset_ci_high, vr.onset_p_value,
               vr.onset_effect_size, vr.onset_low_power, vr.onset_method
        FROM variant_results vr
        WHERE vr.exposure = %s AND vr.generation = %s AND vr.completed = 1
              AND vr.iterations = %s AND vr.onset_low_power = 0
              AND vr.empirical_p <= 0.05
        ORDER BY vr.empirical_p ASC
    """
    with get_connection() as conn:
        df = pd.read_sql(query, conn, params=(exposure, generation, iterations))

    df["gene_name"] = None

    log.info(
        "fetch_current_results: exposure=%s generation=%s iterations=%s -> %d rows",
        exposure, generation, iterations, len(df),
    )
    if df.empty:
        log.warning("fetch_current_results: 0 rows with these filters, check completed/iterations/onset_low_power in DB")
    else:
        log.info("fetch_current_results: variant distinct=%d", df["variant"].nunique())

    return df

def run_export(alpha: float | None = None, from_export:bool | None = None) -> str | None:
    cfg = get_config()
    configure_logging(cfg.log_dir)
    alpha = alpha if alpha is not None else cfg.pvalue_threshold

    if from_export:
        df = load_raw_significant_results()
        print(df)
        if not df.empty:
            # "Wide" schema (cohort 1 and 2 side by side): use g1 as the reference p-value
            df["empirical_p"] = df["empirical_p_g1"]
            df["obs_coef"] = df["obs_coef_g1"]
    else:
        df = fetch_current_results(cfg.exposure, cfg.generation, cfg.n_perm_high)

    if df.empty:
        log.info("No completed results with iterations=%d at the moment. No export produced.", cfg.n_perm_high)
        return None

    df = add_fdr(df, p_col="empirical_p", fdr_col="fdr")
    significant = df.copy()
    log.info("Total results: %d, significant (FDR < %.3f): %d", len(df), alpha, len(significant))

    if significant.empty:
        log.info("No significant variants at the moment. No export produced.")
        return None

    significant = significant.reindex(columns=[c for c in RESULT_COLUMNS if c in significant.columns])
    significant = significant.sort_values("fdr")

    os.makedirs(cfg.significant_export_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = os.path.join(cfg.significant_export_dir, f"significant_variants_{timestamp}.csv")
    latest_path = os.path.join(cfg.significant_export_dir, "significant_variants_latest.csv")

    significant.to_csv(snapshot_path, index=False)
    significant.to_csv(latest_path, index=False)
    log.info("Export written to %s and %s (%d variants)", snapshot_path, latest_path, len(significant))
    return latest_path


if __name__ == "__main__":
    run_export()
