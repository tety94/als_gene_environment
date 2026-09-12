"""Replication run: takes the variants found significant on an already
completed generation (same `exposure`), and runs `modeling.py` ONLY on
those, but on the target generation -- reusing the pickle/dataset already
built for that generation if present, otherwise building it.

Deliberately does NOT reuse `extract-significant` (extracts raw genotype
from the VCFs for gen1/2/3, for report_onset_age.py's plots) nor
`export-significant-csv` (human-reporting CSV): neither produces an input
that `process_single_variant` can run on. The significant-variant list is
built with the same logic already used by
`export_significant_csv.fetch_current_results` + `add_fdr` (single
generation, not the two-cohort stored procedure -- there's no need here to
compare two already-finished cohorts, just "which variants were
significant in gen X" to then test them on gen Y).

Results are saved with the same `exposure`, `generation=target_generation`,
but a DIFFERENT `test_label` (default: "replication_of_gen{source}") so
they aren't confused in the DB with a full sweep already done/in progress
on the same target generation with the default test_label.
"""
from __future__ import annotations

import os
import pickle
import random
from datetime import datetime

from gene_environment.analysis.orchestrator import run_parallel_processing
from gene_environment.config import Config, get_config
from gene_environment.db.repository import insert_new_variants
from gene_environment.logging_utils import configure_logging, get_logger
from gene_environment.significant_variants.export_significant_csv import fetch_current_results
from gene_environment.utils.id_utils import parse_variant_label
from gene_environment.utils.stats_utils import add_fdr
from gene_environment.vcf_pipeline.build_dataset import load_and_prepare_data

log = get_logger(__name__)


def get_significant_variant_labels(exposure: str, generation: int, iterations: int, alpha: float) -> list[str]:
    """Same logic as `export_significant_csv.run_export`, but returns only
    the list of variant labels (CHROM_POS_MUTATION format), not a CSV."""
    df = fetch_current_results(exposure, generation, iterations)
    if df.empty:
        return []
    df = add_fdr(df, p_col="empirical_p", fdr_col="fdr")
    significant = df.copy()
    return sorted(significant["variant"].unique().tolist())


def _target_dataset_paths(cfg: Config, target_generation: int) -> tuple[str, str]:
    """Paths dedicated to the target generation, SEPARATE from
    cfg.temp_df_path: that's the file `run-model` (full sweep) overwrites
    on every run -- reusing it here would create a race condition if a
    replication run and a full sweep run on the same machine at the same
    time. Returns (df_path, meta_path)."""
    base, ext = os.path.splitext(cfg.temp_df_path)
    df_path = f"{base}_gen{target_generation}{ext}"
    meta_path = f"{base}_gen{target_generation}_meta.pkl"
    return df_path, meta_path


def run_replication_on_significant_variants(
    source_generation: int,
    target_generation: int,
    exposure: str | None = None,
    alpha: float | None = None,
    test_label: str | None = None,
    force_rebuild_dataset: bool = False,
) -> None:
    cfg = get_config()
    configure_logging(cfg.log_dir)
    df, variant_cols_safe, mapping, Ecols, variant_cols, covariate_cols = load_and_prepare_data(cfg)

    exposure = exposure if exposure is not None else cfg.exposure
    alpha = alpha if alpha is not None else cfg.pvalue_threshold
    test_label = test_label or f"replication_of_gen{source_generation}"

    log.info(
        "Replication run: exposure=%s, source_generation=%d -> target_generation=%d, alpha=%.3f, test_label=%s",
        exposure, source_generation, target_generation, alpha, test_label,
    )

    sig_labels = get_significant_variant_labels(exposure, source_generation, cfg.n_perm_high, alpha)
    if not sig_labels:
        log.info(
            "No significant variants found for exposure=%s, generation=%d, iterations=%d, alpha=%.3f. Exiting.",
            exposure, source_generation, cfg.n_perm_high, alpha,
        )
        return
    log.info("%d significant variants in generation=%d to re-test on generation=%d",
              len(sig_labels), source_generation, target_generation)

    # ---- TARGET generation dataset: dedicated path (see
    # _target_dataset_paths), NEVER shares cfg.temp_df_path with the main
    # run. Reuses the pickle if already present, otherwise builds it (same
    # load_and_prepare_data as the main run, with
    # GENERATION=target_generation). ----
    # Config is a frozen=True dataclass: direct assignment
    # (target_cfg.generation = ...) would raise FrozenInstanceError, so we
    # bypass the dataclass's __setattr__ on the LOCAL COPY (doesn't touch
    # the global instance returned by get_config()).
    target_cfg = Config.__new__(Config)
    target_cfg.__dict__.update(cfg.__dict__)
    object.__setattr__(target_cfg, "generation", target_generation)
    object.__setattr__(target_cfg, "exposure", exposure)
    object.__setattr__(target_cfg, "test_label", test_label)

    df_path, meta_path = _target_dataset_paths(cfg, target_generation)
    if os.path.exists(df_path) and os.path.exists(meta_path) and not force_rebuild_dataset:
        log.info("Reusing existing dataset for generation=%d: %s", target_generation, df_path)
        with open(meta_path, "rb") as f:
            variant_cols_safe, mapping, Ecols, variant_cols = pickle.load(f)
    else:
        log.info("Building the dataset for generation=%d (not found or force_rebuild_dataset=True)", target_generation)
        df, variant_cols_safe, mapping, Ecols, variant_cols, covariate_cols = load_and_prepare_data(target_cfg)
        with open(df_path, "wb") as f:
            pickle.dump(df, f)
        with open(meta_path, "wb") as f:
            pickle.dump((variant_cols_safe, mapping, Ecols, variant_cols), f)
        log.info("Dataset for generation=%d saved to %s (reusable in later runs)", target_generation, df_path)

    # init_worker (see orchestrator.py) loads the df from the path in
    # target_cfg.temp_df_path -> point it to this generation's dedicated
    # file, NOT cfg.temp_df_path (that stays free for a main run possibly
    # in progress in parallel).
    object.__setattr__(target_cfg, "temp_df_path", df_path)

    # ---- map from original label -> "safe" column name in the target
    # dataset. A variant significant in source_generation might not be
    # genotyped in the target generation's VCFs: flag it, don't silently
    # skip it. ----
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
            "%d/%d significant variants not found in the generation=%d dataset (not genotyped in those VCFs): %s",
            len(missing), len(sig_labels), target_generation,
            missing[:20] if len(missing) > 20 else missing,
        )

    if not variants_to_run_safe:
        log.warning("None of the significant variants are present in the target generation. Exiting.")
        return

    variants_to_insert = []
    for v_safe in variants_to_run_safe:
        v_orig = mapping[v_safe]
        chrom, pos, mutation = parse_variant_label(v_orig)
        variants_to_insert.append({"variant": v_orig, "chromosome": chrom, "position": pos, "mutation": mutation})
    insert_new_variants(variants_to_insert, exposure, target_generation, test_label)

    random.shuffle(variants_to_run_safe)

    start_time = datetime.now()
    run_parallel_processing(
        variants_to_run_safe, mapping, Ecols, covariate_cols, target_cfg,
        description=f"replication gen{source_generation}->gen{target_generation} ({len(variants_to_run_safe)} variants)",
    )
    log.info("Replication run complete in %s", datetime.now() - start_time)