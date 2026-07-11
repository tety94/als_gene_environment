"""
Replication run: prende le varianti risultate significative su una
generazione già completata (stessa `exposure`), e lancia `modeling.py` SOLO
su quelle, ma sulla generazione target — usando il pickle/dataset già
costruito per quella generazione se presente, altrimenti costruendolo.

Deliberatamente NON riusa `extract-significant` (estrae genotipo grezzo dai
VCF per gen1/2/3, per i plot di report_onset_age.py) né
`export-significant-csv` (CSV di reporting umano): nessuno dei due produce
un input eseguibile da `process_single_variant`. La lista di varianti
significative viene presa con la stessa logica già usata da
`export_significant_csv.fetch_current_results` + `add_fdr` (singola
generazione, non la stored procedure a due coorti — qui non serve
confrontare due coorti già finite, serve solo "quali varianti erano
significative in gen X" per poi testarle su gen Y).

Risultati salvati con lo stesso `exposure`, `generation=target_generation`,
ma `test_label` DIVERSO (default: "replication_of_gen{source}") per non
confonderli in DB con un eventuale sweep completo già fatto/in corso sulla
stessa generazione target con lo stesso test_label di default.
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
    """Stessa logica di `export_significant_csv.run_export`, ma ritorna solo
    la lista di label variante (formato CHROM_POS_MUTATION), non un CSV."""
    df = fetch_current_results(exposure, generation, iterations)
    if df.empty:
        return []
    df = add_fdr(df, p_col="empirical_p", fdr_col="fdr")
    significant = df[df["fdr"] < alpha]
    return sorted(significant["variant"].unique().tolist())


def _target_dataset_paths(cfg: Config, target_generation: int) -> tuple[str, str]:
    """Path dedicati alla generazione target, SEPARATI da cfg.temp_df_path:
    quello è il file che `run-model` (sweep completo) sovrascrive ad ogni
    lancio — riusarlo qui creerebbe una race condition se un replication run
    e uno sweep completo girano nello stesso momento sulla stessa macchina.
    Ritorna (df_path, meta_path)."""
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
            "Nessuna variante significativa trovata per exposure=%s, generation=%d, iterations=%d, alpha=%.3f. Esco.",
            exposure, source_generation, cfg.n_perm_high, alpha,
        )
        return
    log.info("%d varianti significative in generation=%d da ritestare su generation=%d",
              len(sig_labels), source_generation, target_generation)

    # ---- dataset della generazione TARGET: path dedicato (vedi
    # _target_dataset_paths), MAI cfg.temp_df_path condiviso col run
    # principale. Riusa il pickle se già presente, altrimenti lo costruisce
    # (stesso load_and_prepare_data del run principale, con
    # GENERATION=target_generation). ----
    # Config è un dataclass frozen=True: l'assegnazione diretta
    # (target_cfg.generation = ...) solleverebbe FrozenInstanceError, quindi
    # bypassiamo l'__setattr__ del dataclass sulla COPIA locale (non tocca
    # l'istanza globale restituita da get_config()).
    target_cfg = Config.__new__(Config)
    target_cfg.__dict__.update(cfg.__dict__)
    object.__setattr__(target_cfg, "generation", target_generation)
    object.__setattr__(target_cfg, "exposure", exposure)
    object.__setattr__(target_cfg, "test_label", test_label)

    df_path, meta_path = _target_dataset_paths(cfg, target_generation)
    if os.path.exists(df_path) and os.path.exists(meta_path) and not force_rebuild_dataset:
        log.info("Riuso dataset già presente per generation=%d: %s", target_generation, df_path)
        with open(meta_path, "rb") as f:
            variant_cols_safe, mapping, Ecols, variant_cols = pickle.load(f)
    else:
        log.info("Costruisco il dataset per generation=%d (non trovato o force_rebuild_dataset=True)", target_generation)
        df, variant_cols_safe, mapping, Ecols, variant_cols = load_and_prepare_data(target_cfg)
        with open(df_path, "wb") as f:
            pickle.dump(df, f)
        with open(meta_path, "wb") as f:
            pickle.dump((variant_cols_safe, mapping, Ecols, variant_cols), f)
        log.info("Dataset generation=%d salvato in %s (riusabile in run successivi)", target_generation, df_path)

    # init_worker (vedi orchestrator.py) carica il df dal path in
    # target_cfg.temp_df_path -> lo puntiamo al file dedicato di questa
    # generazione, NON a cfg.temp_df_path (quello resta libero per un run
    # principale eventualmente in corso in parallelo).
    object.__setattr__(target_cfg, "temp_df_path", df_path)

    # ---- mappa label originale -> nome colonna "safe" nel dataset target.
    # Una variante significativa in source_generation potrebbe non essere
    # genotipata nei VCF della generazione target: la segnaliamo, non la
    # saltiamo silenziosamente. ----
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
            "%d/%d varianti significative non trovate nel dataset di generation=%d (non genotipate in quei VCF): %s",
            len(missing), len(sig_labels), target_generation,
            missing[:20] if len(missing) > 20 else missing,
        )

    if not variants_to_run_safe:
        log.warning("Nessuna delle varianti significative è presente nella generazione target. Esco.")
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
        variants_to_run_safe, mapping, Ecols, target_cfg,
        description=f"replication gen{source_generation}->gen{target_generation} ({len(variants_to_run_safe)} varianti)",
    )
    log.info("Replication run completato in %s", datetime.now() - start_time)