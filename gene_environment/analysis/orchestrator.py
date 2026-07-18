"""
Orchestratore del run di analisi per-variante (ex main.py).

NOVITA' (correzione per struttura di popolazione): se cfg.use_pca_covariates
è True (default), carica pca_covariates.csv per la generazione corrente
(cfg.generation) e lo fa il merge nel dataframe principale su IID PRIMA di
salvare temp_df.pkl, cosi' i worker paralleli hanno gia' le colonne PC
disponibili. La lista dei nomi colonna PC (es. ["PC1", ..., "PC5"]) viene
passata a ciascun worker tramite l'initializer di ProcessPoolExecutor (stesso
meccanismo gia' usato per il dataframe stesso), ed e' quello che
modeling.py usa come covariate di correzione (non di interazione)
nell'OLS. Se cfg.use_pca_covariates è False, covariate_cols resta [] e il
comportamento è identico a prima di questa modifica.

Fix rispetto all'originale:
  - Il buffer di risultati veniva scritto a DB con un `for` che chiamava
    `save_variant_result` riga per riga dentro la stessa connessione: ora usa
    `save_variant_results_bulk` (executemany, una sola transazione per batch).
  - Path di `temp_df.pkl` ora configurabile (TEMP_DF_PATH) invece di hardcoded
    relativo alla cwd (rompeva se lo script veniva lanciato da un'altra
    directory).
  - Le statistiche onset_age calcolate in modeling.py vengono salvate nello
    stesso batch, colonna per colonna (vedi db/repository.py).
  - Logging al posto di print(), incluso un riepilogo finale con conteggio
    errori per variante (prima un'eccezione su una variante veniva solo
    stampata e "persa").
"""
from __future__ import annotations

import os
import pickle
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

from gene_environment.analysis import modeling
from gene_environment.config import get_config
from gene_environment.db.repository import (
    insert_new_variants,
    get_variants_to_run,
    save_variant_results_bulk,
    load_variant_results,
)
from gene_environment.logging_utils import configure_logging, get_logger
from gene_environment.utils.id_utils import parse_variant_label
from gene_environment.utils.pca_utils import load_pca_covariates, merge_pca_covariates
from gene_environment.utils.stats_utils import add_fdr, volcano_plot
from gene_environment.vcf_pipeline.build_dataset import load_and_prepare_data

log = get_logger(__name__)

BATCH_SIZE = 50


def init_worker(temp_df_path: str, log_dir: str, covariate_cols: list[str]):
    configure_logging(log_dir)
    with open(temp_df_path, "rb") as f:
        modeling.global_df = pickle.load(f)
    modeling.global_covariate_cols = covariate_cols
    log.info(
        "Worker %d: dataset caricato da %s (covariate di correzione: %s)",
        os.getpid(), temp_df_path, covariate_cols or "nessuna",
    )


def load_pca_covariate_columns(df, cfg):
    """Se cfg.use_pca_covariates e' True, carica pca_covariates.csv per
    cfg.generation e fa il merge in df. Ritorna (df_aggiornato,
    lista_nomi_colonne_pc) -- lista vuota se le PCA sono disattivate."""
    if not cfg.use_pca_covariates:
        log.info("PCA disattivate (cfg.use_pca_covariates=False): nessuna covariata di popolazione nel modello.")
        return df, []

    pca_df = load_pca_covariates(
        cfg.pca_covariates_path_template, cfg.generation, cfg.pca_n_components,
    )
    df, covariate_cols = merge_pca_covariates(df, pca_df)
    log.info("PCA attive come covariate di correzione: %s (generazione=%s)", covariate_cols, cfg.generation)
    return df, covariate_cols


def run_parallel_processing(
    variants: list[str], mapping: dict, Ecols: list[str], covariate_cols: list[str], cfg, description: str = "",
) -> None:
    log.info("Avvio processi paralleli: %s (%d varianti, %d worker)", description, len(variants), cfg.max_workers)

    buffer = []
    completed, skipped, errors = 0, 0, 0

    with ProcessPoolExecutor(
        max_workers=cfg.max_workers,
        initializer=init_worker,
        initargs=(cfg.temp_df_path, cfg.log_dir, covariate_cols),
    ) as ex:
        futures = {ex.submit(modeling.process_single_variant, g, mapping[g], Ecols): g for g in variants}

        for fut in as_completed(futures):
            variant_name = futures[fut]
            try:
                res = fut.result()
                if res is not None:
                    buffer.append(res)
                    completed += 1
                else:
                    skipped += 1

                if len(buffer) >= BATCH_SIZE:
                    save_variant_results_bulk(buffer, cfg.exposure, cfg.generation, cfg.test_label)
                    log.info("Progresso: %d completati, %d saltati, %d errori (su %d totali)",
                              completed, skipped, errors, len(variants))
                    buffer = []

            except Exception:
                errors += 1
                log.exception("Errore imprevisto sulla variante %s", variant_name)

    if buffer:
        save_variant_results_bulk(buffer, cfg.exposure, cfg.generation, cfg.test_label)

    log.info("Run completato: %d completati, %d saltati, %d errori", completed, skipped, errors)


def run_main_pipeline() -> None:
    cfg = get_config()
    configure_logging(cfg.log_dir)

    start_time = datetime.now()
    log.info("Analisi iniziata alle %s", start_time)

    df, variant_cols_safe, mapping, Ecols, variant_cols = load_and_prepare_data(cfg)

    df, covariate_cols = load_pca_covariate_columns(df, cfg)

    with open(cfg.temp_df_path, "wb") as f:
        pickle.dump(df, f)
    log.info("Dataset temporaneo salvato in %s", cfg.temp_df_path)

    variants_to_insert = []
    for v in variant_cols:
        chrom, pos, mutation = parse_variant_label(v)
        variants_to_insert.append({"variant": v, "chromosome": chrom, "position": pos, "mutation": mutation})
    insert_new_variants(variants_to_insert, cfg.exposure, cfg.generation, cfg.test_label)

    variants_to_run = get_variants_to_run(mapping, variant_cols_safe, cfg.exposure, cfg.generation)
    random.shuffle(variants_to_run)  # bilancia il carico fra worker (varianti "pesanti" sparse)

    run_parallel_processing(
        variants_to_run, mapping, Ecols, covariate_cols, cfg, description="run con permutazioni adattive",
    )

    results_df = load_variant_results(cfg.exposure, cfg.n_perm_high)
    if results_df.empty:
        log.warning("Nessun risultato con iterations=%d trovato in DB: volcano plot saltato.", cfg.n_perm_high)
    else:
        results_df = add_fdr(results_df)
        os.makedirs(cfg.log_dir, exist_ok=True)
        volcano_path = os.path.join(cfg.log_dir, "volcano_plot_final.png")
        volcano_plot(results_df, save_path=volcano_path)

    duration = datetime.now() - start_time
    log.info("Analisi terminata. Durata totale: %s", duration)


if __name__ == "__main__":
    run_main_pipeline()
