#!/usr/bin/env python3
"""
Script INDIPENDENTE (richiesto: "un CSV con le colonne per le varianti
significative... salvato in maniera separata / con altro script").

Può essere rilanciato in qualsiasi momento (anche mentre il run principale è
ancora in corso su altre varianti): legge lo stato ATTUALE della tabella
variant_results, ricalcola l'FDR sui risultati con permutazioni "alte"
(iterations = N_PERM_HIGH) già completati, seleziona quelli sotto soglia e
scrive un CSV in una cartella SEPARATA (config: SIGNIFICANT_EXPORT_DIR),
diversa da quella usata dalla pipeline di estrazione genotipi
(SIGNIFICANT_MATRIX_DIR), con:
  - il coefficiente osservato e il p-value empirico/FDR del modello
  - le statistiche di differenza onset_age (mediane, delta, IC bootstrap,
    p-value) già salvate nella stessa riga da modeling.py
  - il nome del gene, se già annotato

Ogni esecuzione scrive sia uno snapshot timestampato sia un file
"significant_variants_latest.csv" sempre aggiornato, così un downstream
consumer (dashboard, notebook, altro script) può sempre puntare allo stesso
path senza doversi preoccupare del timestamp.

Usage: python -m gene_environment.significant_variants.export_significant_csv
"""
from __future__ import annotations

import os
from datetime import datetime

import pandas as pd

from gene_environment.config import get_config
from gene_environment.db.connection import get_connection
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
        ORDER BY vr.empirical_p ASC
    """
    with get_connection() as conn:
        df = pd.read_sql(query, conn, params=(exposure, generation, iterations))

    df["gene_name"] = None

    log.info(
        "fetch_current_results: exposure=%s generation=%s iterations=%s -> %d righe",
        exposure, generation, iterations, len(df),
    )
    if df.empty:
        log.warning("fetch_current_results: 0 righe con questi filtri, controlla completed/iterations/onset_low_power in DB")
    else:
        log.info("fetch_current_results: variant distinct=%d", df["variant"].nunique())

    return df

def run_export(alpha: float | None = None) -> str | None:
    cfg = get_config()
    configure_logging(cfg.log_dir)
    alpha = alpha if alpha is not None else cfg.pvalue_threshold

    df = fetch_current_results(cfg.exposure, cfg.generation, cfg.n_perm_high)
    if df.empty:
        log.info("Nessun risultato completato con iterations=%d al momento. Nessun export prodotto.", cfg.n_perm_high)
        return None

    df = add_fdr(df, p_col="empirical_p", fdr_col="fdr")
    significant = df[df["fdr"] < alpha].copy()
    log.info("Risultati totali: %d, significativi (FDR < %.3f): %d", len(df), alpha, len(significant))

    if significant.empty:
        log.info("Nessuna variante significativa al momento. Nessun export prodotto.")
        return None

    significant = significant.reindex(columns=[c for c in RESULT_COLUMNS if c in significant.columns])
    significant = significant.sort_values("fdr")

    os.makedirs(cfg.significant_export_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = os.path.join(cfg.significant_export_dir, f"significant_variants_{timestamp}.csv")
    latest_path = os.path.join(cfg.significant_export_dir, "significant_variants_latest.csv")

    significant.to_csv(snapshot_path, index=False)
    significant.to_csv(latest_path, index=False)
    log.info("Export scritto in %s e %s (%d varianti)", snapshot_path, latest_path, len(significant))
    return latest_path


if __name__ == "__main__":
    run_export()
