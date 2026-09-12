"""High-precision final recalculation (default 10000 permutations) on the
variants found significant in BOTH cohorts (gen1 and gen2), saving not
just the interaction beta but the entire coefficient vector (main
effects, PCs) as JSON in `full_model_json`.

Unlike run_replication.py:
  - it doesn't test on a "target" generation different from where the
    variant was significant: it recalculates on the SAME generation (gen1
    and gen2, separately, one row each), just with more iterations and
    more betas.
  - the variant list comes from get_significant_results() (two-cohort
    intersection via the stored procedure), not from
    fetch_current_results() (single generation).
  - test_label is always "final_10k" (fixed, so it isn't confused with the
    main sweep or any replication runs).
"""
from __future__ import annotations

import os
import pickle
import random
from datetime import datetime

from gene_environment.analysis.orchestrator import run_parallel_processing
from gene_environment.config import Config, get_config
from gene_environment.db.repository import get_significant_results, insert_new_variants
from gene_environment.logging_utils import configure_logging, get_logger
from gene_environment.utils.id_utils import parse_variant_label
from gene_environment.vcf_pipeline.build_dataset import load_and_prepare_data

log = get_logger(__name__)

FINAL_TEST_LABEL = "final_10k"


def _recalc_dataset_paths(cfg: Config, generation: int) -> tuple[str, str]:
    """Dedicated paths, same principle as _target_dataset_paths in
    run_replication.py: NEVER share cfg.temp_df_path with the main run or
    with a replication run that might be in progress in parallel."""
    base, ext = os.path.splitext(cfg.temp_df_path)
    df_path = f"{base}_final_gen{generation}{ext}"
    meta_path = f"{base}_final_gen{generation}_meta.pkl"
    return df_path, meta_path


def _load_or_build_dataset(cfg: Config, generation: int, exposure: str, force_rebuild: bool):
    gen_cfg = Config.__new__(Config)
    gen_cfg.__dict__.update(cfg.__dict__)
    object.__setattr__(gen_cfg, "generation", generation)
    object.__setattr__(gen_cfg, "exposure", exposure)
    object.__setattr__(gen_cfg, "test_label", FINAL_TEST_LABEL)

    df_path, meta_path = _recalc_dataset_paths(cfg, generation)
    if os.path.exists(df_path) and os.path.exists(meta_path) and not force_rebuild:
        log.info("Reusing existing dataset for generation=%d: %s", generation, df_path)
        with open(meta_path, "rb") as f:
            variant_cols_safe, mapping, Ecols, variant_cols, covariate_cols = pickle.load(f)
    else:
        log.info("Building the dataset for generation=%d", generation)
        df, variant_cols_safe, mapping, Ecols, variant_cols, covariate_cols = load_and_prepare_data(gen_cfg)
        with open(df_path, "wb") as f:
            pickle.dump(df, f)
        with open(meta_path, "wb") as f:
            pickle.dump((variant_cols_safe, mapping, Ecols, variant_cols, covariate_cols), f)
        log.info("Dataset for generation=%d saved to %s", generation, df_path)

    object.__setattr__(gen_cfg, "temp_df_path", df_path)
    return gen_cfg, variant_cols_safe, mapping, Ecols, variant_cols, covariate_cols


def run_final_recalculation(
    exposure: str | None = None,
    n_perm: int = 10000,
    force_rebuild_dataset: bool = False,
) -> None:
    cfg = get_config()
    configure_logging(cfg.log_dir)
    exposure = exposure if exposure is not None else cfg.exposure

    log.info("Final recalculation: exposure=%s, n_perm=%d, test_label=%s", exposure, n_perm, FINAL_TEST_LABEL)

    sig_df = get_significant_results(exposure=exposure)
    if sig_df.empty:
        log.info("No significant variants for exposure=%s in both cohorts. Exiting.", exposure)
        return

    sig_labels = sorted(sig_df["variant"].unique().tolist())
    log.info("%d variants significant in both cohorts to recalculate", len(sig_labels))

    for generation in (1, 2):
        gen_cfg, variant_cols_safe, mapping, Ecols, variant_cols, covariate_cols = _load_or_build_dataset(
            cfg, generation, exposure, force_rebuild_dataset,
        )
        object.__setattr__(gen_cfg, "n_perm", n_perm)
        # No adaptive early stopping in the final recalculation
        # (full_beta=True in modeling.py bypasses it anyway, but kept
        # consistent here at the config level for clarity)

        orig_to_safe = {v: k for k, v in mapping.items()}
        variants_to_run_safe = []
        missing = []
        for label in sig_labels:
            if label in orig_to_safe:
                variants_to_run_safe.append(orig_to_safe[label])
            else:
                missing.append(label)

        if missing:
            log.warning(
                "%d/%d variants not found in the generation=%d dataset: %s",
                len(missing), len(sig_labels), generation,
                missing[:20] if len(missing) > 20 else missing,
            )
        if not variants_to_run_safe:
            log.warning("No variants found for generation=%d. Skipping.", generation)
            continue

        variants_to_insert = []
        for v_safe in variants_to_run_safe:
            v_orig = mapping[v_safe]
            chrom, pos, mutation = parse_variant_label(v_orig)
            variants_to_insert.append({"variant": v_orig, "chromosome": chrom, "position": pos, "mutation": mutation})
        insert_new_variants(variants_to_insert, exposure, generation, FINAL_TEST_LABEL)

        random.shuffle(variants_to_run_safe)

        start_time = datetime.now()
        run_parallel_processing(
            variants_to_run_safe, mapping, Ecols, covariate_cols, gen_cfg,
            description=f"final recalculation gen{generation} ({len(variants_to_run_safe)} variants, n_perm={n_perm})",
            full_beta=True,
        )
        log.info("Recalculation for generation=%d complete in %s", generation, datetime.now() - start_time)