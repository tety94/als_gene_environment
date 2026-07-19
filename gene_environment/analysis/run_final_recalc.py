"""
Ricalcolo finale ad alta precisione (default 10000 permutazioni) sulle
varianti risultate significative in ENTRAMBE le coorti (gen1 e gen2),
salvando non solo il beta di interazione ma l'intero vettore di
coefficienti (main effects, PC) come JSON in `full_model_json`.

A differenza di run_replication.py:
  - non testa su una generazione "target" diversa da dove la variante era
    significativa: ricalcola sulla STESSA generazione (gen1 e gen2,
    separatamente, una riga a testa), solo con più iterazioni e più beta.
  - la lista di varianti parte da get_significant_results() (intersezione
    a due coorti via stored procedure), non da fetch_current_results()
    (singola generazione).
  - test_label sempre "final_10k" (fisso, per non essere confuso con lo
    sweep principale né con eventuali replication run).
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
    """Path dedicati, stesso principio di _target_dataset_paths in
    run_replication.py: MAI cfg.temp_df_path condiviso col run principale
    o con un eventuale replication run in corso in parallelo."""
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
        log.info("Riuso dataset già presente per generation=%d: %s", generation, df_path)
        with open(meta_path, "rb") as f:
            variant_cols_safe, mapping, Ecols, variant_cols, covariate_cols = pickle.load(f)
    else:
        log.info("Costruisco il dataset per generation=%d", generation)
        df, variant_cols_safe, mapping, Ecols, variant_cols, covariate_cols = load_and_prepare_data(gen_cfg)
        with open(df_path, "wb") as f:
            pickle.dump(df, f)
        with open(meta_path, "wb") as f:
            pickle.dump((variant_cols_safe, mapping, Ecols, variant_cols, covariate_cols), f)
        log.info("Dataset generation=%d salvato in %s", generation, df_path)

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

    log.info("Ricalcolo finale: exposure=%s, n_perm=%d, test_label=%s", exposure, n_perm, FINAL_TEST_LABEL)

    sig_df = get_significant_results(exposure=exposure)
    if sig_df.empty:
        log.info("Nessuna variante significativa per exposure=%s in entrambe le coorti. Esco.", exposure)
        return

    sig_labels = sorted(sig_df["variant"].unique().tolist())
    log.info("%d varianti significative in entrambe le coorti da ricalcolare", len(sig_labels))

    for generation in (1, 2):
        gen_cfg, variant_cols_safe, mapping, Ecols, variant_cols, covariate_cols = _load_or_build_dataset(
            cfg, generation, exposure, force_rebuild_dataset,
        )
        object.__setattr__(gen_cfg, "n_perm", n_perm)
        # niente early stopping adattivo nel ricalcolo finale (full_beta=True
        # in modeling.py lo bypassa comunque, ma per chiarezza lo teniamo
        # coerente qui a livello di config)

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
                "%d/%d varianti non trovate nel dataset di generation=%d: %s",
                len(missing), len(sig_labels), generation,
                missing[:20] if len(missing) > 20 else missing,
            )
        if not variants_to_run_safe:
            log.warning("Nessuna variante trovata per generation=%d. Salto.", generation)
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
            description=f"ricalcolo finale gen{generation} ({len(variants_to_run_safe)} varianti, n_perm={n_perm})",
            full_beta=True,
        )
        log.info("Ricalcolo generation=%d completato in %s", generation, datetime.now() - start_time)