"""Orchestrates the per-variant analysis run.

Population-structure correction: if cfg.use_pca_covariates is True
(default), loads pca_covariates.csv for the current generation
(cfg.generation) and merges it into the main dataframe on IID BEFORE
saving temp_df.pkl, so the parallel workers already have the PC columns
available. The list of PC column names (e.g. ["PC1", ..., "PC5"]) is
passed to each worker through the ProcessPoolExecutor initializer (the
same mechanism already used for the dataframe itself), and that's what
modeling.py uses as correction (not interaction) covariates in the OLS.
If cfg.use_pca_covariates is False, covariate_cols stays [] and the
behavior is unaffected.

Results are written to the DB in batches via `save_variant_results_bulk`
(executemany, a single transaction per batch), rather than one row at a
time. The `temp_df.pkl` path is configurable (TEMP_DF_PATH) rather than
hardcoded relative to the cwd. The onset_age statistics computed in
modeling.py are saved in the same batch, column by column (see
db/repository.py).
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
        "Worker %d: dataset loaded from %s (correction covariates: %s)",
        os.getpid(), temp_df_path, covariate_cols or "none",
    )



def run_parallel_processing(
    variants: list[str], mapping: dict, Ecols: list[str], covariate_cols: list[str], cfg, description: str = "", full_beta: bool = False,
) -> None:
    log.info("Starting parallel processes: %s (%d variants, %d workers)", description, len(variants), cfg.max_workers)

    buffer = []
    completed, skipped, errors = 0, 0, 0

    with ProcessPoolExecutor(
        max_workers=cfg.max_workers,
        initializer=init_worker,
        initargs=(cfg.temp_df_path, cfg.log_dir, covariate_cols),
    ) as ex:
        futures = {ex.submit(modeling.process_single_variant, g, mapping[g], Ecols, full_beta): g for g in variants}

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
                    log.info("Progress: %d completed, %d skipped, %d errors (out of %d total)",
                              completed, skipped, errors, len(variants))
                    buffer = []

            except Exception:
                errors += 1
                log.exception("Unexpected error on variant %s", variant_name)

    if buffer:
        save_variant_results_bulk(buffer, cfg.exposure, cfg.generation, cfg.test_label)

    log.info("Run complete: %d completed, %d skipped, %d errors", completed, skipped, errors)


def run_main_pipeline() -> None:
    cfg = get_config()
    print("=== DEBUG CONFIG ===")
    print("id(cfg) =", id(cfg))
    print("GENERATION =", cfg.generation)
    print("RAW_FILE =", cfg.raw_file)
    print("ENV_FILE =", cfg.env_file)
    print("SAMPLE_GENERATION_MAP =", repr(cfg.sample_generation_map))
    print("ENV_GENERATION_COL =", repr(cfg.env_generation_col))
    print("OUTPUT_FOLDER =", repr(cfg.output_folder))
    print("cwd =", os.getcwd())
    import gene_environment.config as configmod
    print("resolved map_path =",
          cfg.sample_generation_map or os.path.join(cfg.output_folder, "sample_generation_map.csv"))
    print("map_path exists? =",
          os.path.exists(cfg.sample_generation_map or os.path.join(cfg.output_folder, "sample_generation_map.csv")))
    print("=====================")

    configure_logging(cfg.log_dir)

    start_time = datetime.now()
    log.info("Analysis started at %s", start_time)

    df, variant_cols_safe, mapping, Ecols, variant_cols, covariate_cols = load_and_prepare_data(cfg)

    with open(cfg.temp_df_path, "wb") as f:
        pickle.dump(df, f)
    log.info("Temporary dataset saved to %s", cfg.temp_df_path)

    variants_to_insert = []
    for v in variant_cols:
        chrom, pos, mutation = parse_variant_label(v)
        variants_to_insert.append({"variant": v, "chromosome": chrom, "position": pos, "mutation": mutation})
    insert_new_variants(variants_to_insert, cfg.exposure, cfg.generation, cfg.test_label)

    variants_to_run = get_variants_to_run(mapping, variant_cols_safe, cfg.exposure, cfg.generation)
    random.shuffle(variants_to_run)  # balances load across workers ("heavy" variants spread out)

    run_parallel_processing(
        variants_to_run, mapping, Ecols, covariate_cols, cfg, description="run with adaptive permutations",
        full_beta = False,
    )

    results_df = load_variant_results(cfg.exposure, cfg.n_perm_high)
    if results_df.empty:
        log.warning("No results with iterations=%d found in DB: volcano plot skipped.", cfg.n_perm_high)
    else:
        results_df = add_fdr(results_df)
        os.makedirs(cfg.log_dir, exist_ok=True)
        volcano_path = os.path.join(cfg.log_dir, "volcano_plot_final.png")
        volcano_plot(results_df, save_path=volcano_path)

    duration = datetime.now() - start_time
    log.info("Analysis finished. Total duration: %s", duration)


if __name__ == "__main__":
    run_main_pipeline()
