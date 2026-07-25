"""
Builds the "vQTL-ready" dataset: genotype dosage + phenotype + covariates +
PCA, merged into a single DataFrame.

gene_environment already produces (via the `filter-vcf` + `build-matrix`
commands of its CLI) a single genome-wide genotype matrix, already
MAF/LD-pruned (RAW_FILE, parquet), and already knows how to merge it with
the environmental file by sample id, clean the ids, and select the correct
generation/cohort
(`gene_environment.vcf_pipeline.build_dataset.load_and_prepare_data`). This
module:
  1. reuses that function as-is (no reimplementation of the join/id-cleaning/
     generation-filtering logic);
  2. adds the extra exposure columns required by vqtl (gene_environment
     handles a single exposure per run -- see VqtlConfig.exposures -- while
     vqtl can test several in the same run);
  3. merges in the real PCA computed by the QC pipeline
     (quality_control/00_run_plink_qc.sh -> extract_pca_covariates.py); QC
     and PCA are already an output of the official QC pipeline, so there is
     no need to recompute them here.

The resulting genotype dosage does NOT contain raw -1/NaN values from the
VCF: it has already gone through the same missing-data handling strategy
used by the whole gene_environment pipeline (MISSING_GENOTYPE_STRATEGY, see
vcf_pipeline/vcf_to_parquet.py); here we only add a defensive check (NaN or
out-of-range dosage treated as missing) in case this behavior changes in
the future.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from gene_environment.config import Config
from gene_environment.logging_utils import get_logger
from gene_environment.utils.id_utils import parse_variant_label
from gene_environment.vcf_pipeline.build_dataset import load_and_prepare_data

from vqtl.config import VqtlConfig

log = get_logger(__name__)


@dataclass
class VqtlDataset:
    df: pd.DataFrame
    # "safe" genotype columns (variant_0, variant_1, ...), dosage values
    # 0/1/2 (missing -> NaN, already handled upstream)
    variant_cols: list[str]
    # variant_safe -> real label "CHROM_POS_REF_ALT"
    mapping: dict[str, str]
    # requested exposure (raw) -> standardized column used in the models
    exposure_std_cols: dict[str, str]
    # adjustment covariates: cfg.covariates (dummy-encoded if categorical) + PC1..PCn
    covariate_cols: list[str]
    # REQUESTED covariates (vcfg.covariates, before dummy-encoding): only
    # used to detect whether the cache must be invalidated when
    # VQTL_COVARIATES changes between runs (see get_or_build_dataset)
    requested_covariates: list[str] = field(default_factory=list)


def _generation_config(ge_cfg: Config, generation: int | None) -> Config:
    """Returns a copy of ge_cfg with .generation overridden, if requested
    (e.g. via `--generation` in cli.py), without mutating the global
    singleton."""
    if generation is None or generation == ge_cfg.generation:
        return ge_cfg
    return replace(ge_cfg, generation=generation)


def _dummy_encode(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Dummy-encoding (drop_first) of non-numeric covariates (e.g. 'sex',
    'onset_site' if stored as strings/categories in the environmental
    file). Covariates that are already numeric are left untouched.
    Necessary because the statsmodels models used by vqtl
    (residualization, interaction, rGE, permutations) build the design
    matrix directly from numpy arrays (no formula/patsy parsing as in
    gene_environment.analysis.modeling), so covariates must already be
    numeric on input."""
    out_cols = []
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            log.warning("Covariate '%s' not found in the dataset, skipped.", c)
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out_cols.append(c)
            continue
        dummies = pd.get_dummies(df[c], prefix=c, drop_first=True, dtype=float)
        df = pd.concat([df, dummies], axis=1)
        out_cols.extend(dummies.columns.tolist())
        log.info("Non-numeric covariate '%s': dummy-encoded into %s", c, dummies.columns.tolist())
    return df, out_cols


def load_vqtl_dataset(
    ge_cfg: Config, vcfg: VqtlConfig, generation: int | None = None
) -> VqtlDataset:
    ge_cfg = _generation_config(ge_cfg, generation)

    log.info(
        "Loading vQTL dataset (generation=%s) from RAW_FILE=%s + ENV_FILE=%s",
        ge_cfg.generation, ge_cfg.raw_file, ge_cfg.env_file,
    )
    df, variant_cols, mapping, _ecols_default, _variant_cols_real, _covariate_cols_default = load_and_prepare_data(ge_cfg)
    log.info("Dataset loaded: %d samples, %d variants", len(df), len(variant_cols))

    # ---- Exposures: gene_environment only standardizes cfg.exposure by
    # default; here we add (if not already present) a standardized column
    # for EVERY exposure requested by vqtl. ----
    exposure_std_cols: dict[str, str] = {}
    for exp in vcfg.exposures:
        if exp not in df.columns:
            raise ValueError(
                f"Exposure '{exp}' (VQTL_EXPOSURES) not found in the environmental file "
                f"{ge_cfg.env_file}. Available columns: {list(df.columns)}"
            )
        df[exp] = pd.to_numeric(df[exp], errors="coerce")
        std_col = f"{exp}_std" if ge_cfg.standardize else exp
        if std_col not in df.columns:
            if ge_cfg.standardize:
                df[std_col] = StandardScaler().fit_transform(df[[exp]])
            else:
                std_col = exp
        exposure_std_cols[exp] = std_col
    log.info("vqtl exposures: %s", exposure_std_cols)

    # ---- Base covariates (sex, onset_site, diagnostic_delay, ...) ----
    # NOTE: this uses vcfg.covariates (vqtl-specific override, defaulting to
    # ge_cfg.covariates if VQTL_COVARIATES is not set -- see vqtl/config.py),
    # not ge_cfg.covariates directly: this makes it possible to exclude a
    # covariate from vqtl only, without touching COVARIATES in the .env
    # shared with the rest of the pipeline.
    df, covariate_cols = _dummy_encode(df, list(vcfg.covariates))

    # ---- Real PCA (quality_control) ----
    # These must NOT be reloaded/merged here: gene_environment.vcf_pipeline.
    # build_dataset.load_and_prepare_data already merges them into the
    # dataframe (inside _build_narrow_covariates, when cfg.use_pca_covariates
    # is true) before returning it -- df already has the PC1..PC<n> columns
    # at this point. A second merge on the same column name would produce
    # PC1_x/PC1_y instead of PC1 (pandas renames overlapping non-key
    # columns), breaking all downstream code that expects "PC1" etc. We
    # therefore only recognize the columns that are already present.
    if ge_cfg.use_pca_covariates:
        pc_cols = [f"PC{i}" for i in range(1, ge_cfg.pca_n_components + 1)]
        missing_pc = [c for c in pc_cols if c not in df.columns]
        if missing_pc:
            raise ValueError(
                f"USE_PCA_COVARIATES=true but {missing_pc} are not in the dataset already "
                f"built by load_and_prepare_data (expected: already merged there). "
                f"Available columns: {list(df.columns)}"
            )
        n_missing_pca = int(df[pc_cols[0]].isna().sum())
        if n_missing_pca:
            log.warning(
                "%d/%d samples without PCA (missingness already surfaced in load_and_prepare_data's merge).",
                n_missing_pca, len(df),
            )
        covariate_cols = covariate_cols + pc_cols
        log.info("PCA active as adjustment covariates: %s", pc_cols)
    else:
        log.info("PCA disabled (USE_PCA_COVARIATES=false).")

    return VqtlDataset(
        df=df,
        variant_cols=variant_cols,
        mapping=mapping,
        exposure_std_cols=exposure_std_cols,
        covariate_cols=covariate_cols,
        requested_covariates=list(vcfg.covariates),
    )


def variant_chrom_pos(variant_label: str) -> tuple[str | None, int | None]:
    """CHROM/POS from the real label ('CHROM_POS_REF_ALT'), reusing
    gene_environment's shared parser instead of a local regex."""
    chrom, pos, _mutation = parse_variant_label(variant_label)
    return chrom, pos


def get_or_build_dataset(
    ge_cfg: Config, vcfg: VqtlConfig, generation: int | None = None, force: bool = False,
) -> VqtlDataset:
    """Same as `load_vqtl_dataset`, but with an on-disk (pickle) cache inside
    the cohort folder -- the same principle as the TEMP_DF_PATH used by
    gene_environment.analysis.orchestrator, so the genetics+environment+PCA
    join (expensive, reads the entire RAW_FILE parquet) is not redone on
    every single vqtl CLI subcommand (scan/filter/interaction/...). Pass
    force=True to rebuild from scratch (e.g. after rerunning
    filter-vcf/build-matrix upstream)."""
    ge_cfg_eff = _generation_config(ge_cfg, generation)
    cache_path = os.path.join(vcfg.cohort_dir(ge_cfg_eff.generation), "vqtl_dataset.pkl")

    if not force and os.path.exists(cache_path):
        log.info("Loading vQTL dataset from cache: %s", cache_path)
        with open(cache_path, "rb") as f:
            dataset = pickle.load(f)
        # getattr for compatibility with caches written by an earlier
        # version of this module, before requested_covariates was added
        cached_covariates = getattr(dataset, "requested_covariates", None)
        exposures_changed = set(dataset.exposure_std_cols.keys()) != set(vcfg.exposures)
        covariates_changed = cached_covariates is not None and set(cached_covariates) != set(vcfg.covariates)
        if exposures_changed or covariates_changed:
            log.warning(
                "vqtl config changed relative to the cache (exposures: cache=%s requested=%s; "
                "covariates: cache=%s requested=%s): rebuilding.",
                list(dataset.exposure_std_cols.keys()), vcfg.exposures, cached_covariates, vcfg.covariates,
            )
        else:
            return dataset

    dataset = load_vqtl_dataset(ge_cfg, vcfg, generation=generation)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(dataset, f)
    log.info("vQTL dataset saved to cache: %s", cache_path)
    return dataset


def dosage_matrix(dataset: VqtlDataset, safe_cols: list[str]) -> np.ndarray:
    """Extracts the dosage sub-block (n_samples x len(safe_cols)) as a
    numpy array, with any missing/invalid value -> NaN (safety net; the
    dosage produced by gene_environment is already 0/1/2)."""
    sub = dataset.df[safe_cols].apply(pd.to_numeric, errors="coerce")
    arr = sub.to_numpy(dtype=float, copy=True)
    arr[(arr < 0) | (arr > 2)] = np.nan
    return arr

def select_variants_from_significant_results(
    dataset: VqtlDataset, exposure: str | None = None,
) -> list[str]:
    """Restricts variant_cols to the variants already known to be
    significant (gene_environment.db.repository.get_significant_results),
    instead of a genomic range. No change to the dataset in memory/cache."""
    from gene_environment.db.repository import get_significant_results

    sig_df = get_significant_results(exposure=exposure)
    if sig_df.empty:
        log.warning("get_significant_results(exposure=%s): no rows found, empty subset.", exposure)
        return []

    log.debug("Example label from get_significant_results: %s", sig_df["variant"].head(5).tolist())
    log.debug("Example label from dataset.mapping: %s", list(dataset.mapping.values())[:5])

    inv_mapping = {v: k for k, v in dataset.mapping.items()}
    wanted = sig_df["variant"].unique().tolist()
    safe_cols = [inv_mapping[v] for v in wanted if v in inv_mapping]

    missing = [v for v in wanted if v not in inv_mapping]
    if missing:
        log.warning(
            "%d/%d significant variants not found in the current dataset (label did not match): %s%s",
            len(missing), len(wanted), missing[:5], "..." if len(missing) > 5 else "",
        )
    log.info("Subset from get_significant_results: %d/%d variants mapped onto dataset.variant_cols.",
              len(safe_cols), len(wanted))
    return safe_cols
