"""Chromosome-by-chromosome G x E analysis with GENetLib.

Integrates with the rest of the `gene_environment` pipeline (same
config.py, same logging, same id-normalization and PCA-loading functions
from build_dataset.py / pca_utils.py) instead of reinventing them.

--------------------------------------------------------------------------
DIFFERENCES FROM THE REST OF THE PIPELINE (modeling.py / orchestrator.py)
--------------------------------------------------------------------------
The rest of the pipeline tests ONE variant at a time with OLS + matching +
permutations, using the formula:

    onset_age ~ variant * (exposure) + sex + PC1 + ... + PCk

(variant x exposure interaction, sex and PCA as additive covariates: see
modeling.build_formula). This module conceptually does the SAME thing
(PCA and "sex" additive, exposure interacts with the genetics) but with a
different model: instead of N separate OLS regressions (one per variant),
it uses GENetLib (a neural network with MCP + L2 penalty, package
XMU-Kuangnan-Fang-Team/GENetLib) to estimate ALL the SNPs of a chromosome
together, with automatic selection of important variants.

To get the same "PCA/sex additive, never G x PCA interaction" constraint
here, this proceeds in two steps (consistent with the "correct then test"
approach already used elsewhere in the project for population structure):

  STEP A (in this module, `compute_pca_corrected_residuals`):
      onset_age = [PCA_1..k, sex] . delta + eps      (OLS, per generation)
      onset_age' = onset_age - [PCA, sex] . delta_hat

  STEP B (GENetLib, per chromosome):
      onset_age' = G . beta + (G x E) . theta,   E = exposure ONLY

GENetLib (the scalar_ge function) has no "additive only, non-interactive"
parameter for a covariate: any column passed as E is automatically
crossed with EVERY SNP. That's why "sex" and the PCs CANNOT be passed as E
to GENetLib (it would violate the "no G x PCA" constraint): they are
instead removed from the phenotype in the preliminary regression step,
just like the PCs.

--------------------------------------------------------------------------
ACTUAL FORMAT OF RAW_FILE (gen.parquet), as built by
vcf_pipeline/vcf_to_parquet.py:
  - WIDE format: index = sample id (already passed through
    clean_sample_id during the VCF->parquet conversion, cleaned again here
    for safety), columns = one per variant.
  - variant column name: "{CHROM}_{POS}_{REF}_{ALT}" (build_variant_label
    uses the same scheme for the rest of the pipeline). The CHROM prefix
    can be "1".."22" or "chr1".."chr22" depending on how the source VCF
    named its contigs (see extract_matrix.resolve_chrom_name): detected
    automatically here.
  - values: int8 0/1 (presence of at least one mutant allele: binarization
    already happens in vcf_to_parquet.merge_chromosome,
    "arr[arr > 0] = 1" -- NOT a 0/1/2 dosage).
  - there is NO "generation" column in the parquet: each sample's
    generation is obtained ONLY from sample_generation_map.csv (produced
    by vcf_to_parquet.save_sample_generation_map), exactly like
    build_dataset._build_narrow_covariates does. The same resolution logic
    is replicated here in `resolve_generation_map`.
--------------------------------------------------------------------------
"""
from __future__ import annotations

import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

try:
    from docx import Document
    _HAS_DOCX = True
except ImportError:
    _HAS_DOCX = False

from GENetLib.scalar_ge import scalar_ge

from gene_environment.config import Config, get_config, _env, _env_int, _env_float, _env_bool, _env_list
from gene_environment.logging_utils import configure_logging, get_logger
from gene_environment.utils.id_utils import clean_sample_id
from gene_environment.utils.pca_utils import load_pca_covariates, PCA_ID_COLUMN
from gene_environment.vcf_pipeline.vcf_to_parquet import CHROMOSOMES

log = get_logger(__name__)

# Sex encoding identical to build_dataset.py (SEX_ENCODING), kept separate
# here (not imported) because in build_dataset.py it's local to the
# function and not exposed as a module constant.
SEX_ENCODING = {"M": 1, "F": 0}


# ============================================================================
# 1. GENetLib-SPECIFIC CONFIGURATION (GXE_* variables)
# ============================================================================
# Reuses the private helpers from gene_environment.config (_env, _env_int,
# ...) to stay in the same style/convention as the rest of the project: all
# values come from environment variables / .env, with sensible defaults.

def _env_float_optional(name: str) -> Optional[float]:
    val = os.environ.get(name)
    return float(val) if val not in (None, "") else None


def _env_int_list(name: str, default: str) -> list:
    return [int(v) for v in _env_list(name, default)]


@dataclass(frozen=True)
class GXEConfig:
    # Chromosomes to process (default: all those produced by build-matrix)
    chromosomes: list = field(default_factory=lambda: _env_list("GXE_CHROMOSOMES", ",".join(CHROMOSOMES)) or CHROMOSOMES)

    # Generations to include in the run. If empty, uses ONLY cfg.generation
    # (same behavior as the rest of the pipeline, which processes one
    # generation per run). If set (e.g. "1,2"), residualizes them
    # per-generation PCA and combines them into a single GENetLib model per
    # chromosome (joint analysis of both cohorts).
    generations: list = field(default_factory=lambda: _env_int_list("GXE_GENERATIONS", ""))

    # scalar_ge hyperparameters (see the run_genetlib_scalar_ge docstring
    # for the mapping to the "lam"/"alpha"/"max_iter" concepts)
    num_hidden_layers: int = field(default_factory=lambda: _env_int("GXE_NUM_HIDDEN_LAYERS", 2))
    nodes_hidden_layer: list = field(default_factory=lambda: _env_int_list("GXE_NODES_HIDDEN_LAYER", "64,16"))
    num_epochs: int = field(default_factory=lambda: _env_int("GXE_NUM_EPOCHS", 100))
    learning_rate1: float = field(default_factory=lambda: _env_float("GXE_LEARNING_RATE1", 0.02))
    learning_rate2: float = field(default_factory=lambda: _env_float("GXE_LEARNING_RATE2", 0.01))
    lambda1: Optional[float] = field(default_factory=lambda: _env_float_optional("GXE_LAMBDA1"))
    lambda2: float = field(default_factory=lambda: _env_float("GXE_LAMBDA2", 0.05))
    Lambda: float = field(default_factory=lambda: _env_float("GXE_LAMBDA_L2", 0.05))
    split_type: int = field(default_factory=lambda: _env_int("GXE_SPLIT_TYPE", 0))
    ratio: list = field(default_factory=lambda: _env_int_list("GXE_RATIO", "7,3"))

    # Significance threshold (fraction of the max absolute |weight|,
    # GENetLib's native convention -- NOT a p-value)
    significance_threshold: float = field(default_factory=lambda: _env_float("GXE_SIGNIFICANCE_THRESHOLD", 0.3))

    min_samples_required: int = field(default_factory=lambda: _env_int("GXE_MIN_SAMPLES", 30))

    output_dir: str = field(default_factory=lambda: _env("GXE_OUTPUT_DIR", "./output/gxe_genetlib"))
    save_plots: bool = field(default_factory=lambda: _env_bool("GXE_SAVE_PLOTS", True))
    save_word_summary: bool = field(default_factory=lambda: _env_bool("GXE_SAVE_WORD_SUMMARY", True))
    torch_threads_per_worker: int = field(default_factory=lambda: _env_int("GXE_TORCH_THREADS", 1))


# ============================================================================
# 2. STEP A - PRELIMINARY REGRESSION onset_age ~ PCA + ADDITIVE covariates
# ============================================================================

def resolve_generation_map(cfg: Config) -> Optional[pd.DataFrame]:
    """Replicates the priority logic used in
    vcf_pipeline.build_dataset._build_narrow_covariates:
        1) sample_generation_map.csv (cfg.sample_generation_map, or
           <OUTPUT_FOLDER>/sample_generation_map.csv if not set)
        2) cfg.env_generation_col column in the environmental file (legacy)
        3) no filter possible -> None (all rows, unknown generation: NOT
           usable for Step A, which needs the generation to pick the right
           PCA file)

    Returns a DataFrame [id, generation], or None if there's no way to
    determine the samples' generation.
    """
    map_path = cfg.sample_generation_map or os.path.join(cfg.output_folder, "sample_generation_map.csv")
    if os.path.exists(map_path):
        gen_map = pd.read_csv(map_path, dtype={"id": str})
        gen_map["id"] = gen_map["id"].astype(str)
        log.info("id->generation map loaded from %s (%d samples)", map_path, len(gen_map))
        return gen_map[["id", "generation"]]

    log.warning(
        "No id->generation map found at %s. If the environmental file has a "
        "generation column, set it in ENV_GENERATION_COL; otherwise Step A "
        "(per-generation PCA regression) cannot run correctly.", map_path,
    )
    return None


def _encode_covariates(df_env: pd.DataFrame, covariates: list) -> tuple[pd.DataFrame, list]:
    """Encodes the additive covariates (e.g. 'sex') exactly like
    build_dataset._build_narrow_covariates, returning the dataframe with
    the encoded columns and the list of column names that are actually
    numeric and usable in the regression."""
    df = df_env.copy()
    resolved = []
    for cov in covariates:
        if cov not in df.columns:
            log.warning("Covariate '%s' not found in the environmental file: ignored.", cov)
            continue
        if cov == "sex":
            unmapped = set(df["sex"].dropna().unique()) - set(SEX_ENCODING.keys())
            if unmapped:
                raise ValueError(f"'sex': unrecognized values {unmapped}, update SEX_ENCODING")
            df["sex"] = df["sex"].map(SEX_ENCODING).astype(float)
        else:
            df[cov] = pd.to_numeric(df[cov], errors="coerce")
        resolved.append(cov)
    return df, resolved


def load_environment_and_phenotype(cfg: Config, logger) -> pd.DataFrame:
    """Loads ENV_FILE and validates the required columns (SAMPLE_ID_COL, TARGET_COL)."""
    if not os.path.exists(cfg.env_file):
        raise FileNotFoundError(f"ENV_FILE not found: {cfg.env_file}")

    df_env = pd.read_csv(cfg.env_file, sep=cfg.sep, decimal=cfg.decimal)

    required = [cfg.sample_id_col, cfg.target_col, cfg.exposure]
    missing = [c for c in required if c not in df_env.columns]
    if missing:
        raise ValueError(f"Missing columns in {cfg.env_file}: {missing}. Available: {list(df_env.columns)}")

    df_env[cfg.sample_id_col] = df_env[cfg.sample_id_col].astype(str)
    df_env[cfg.target_col] = pd.to_numeric(df_env[cfg.target_col], errors="coerce")
    df_env[cfg.exposure] = pd.to_numeric(df_env[cfg.exposure], errors="coerce")

    n_before = len(df_env)
    df_env = df_env.dropna(subset=[cfg.target_col, cfg.exposure]).drop_duplicates(cfg.sample_id_col)
    if len(df_env) < n_before:
        logger.warning("[ENV] Removed %d patients with missing %s/%s or duplicate id", n_before - len(df_env), cfg.target_col, cfg.exposure)

    logger.info("[ENV] Loaded %d patients from %s", len(df_env), cfg.env_file)
    return df_env


def build_exposure_column(cfg: Config, df: pd.DataFrame) -> str:
    """Standardizes EXPOSURE like build_dataset.py does (StandardScaler if
    cfg.standardize=True) and returns the column name to use as E."""
    if cfg.standardize:
        col = f"{cfg.exposure}_std"
        df[col] = StandardScaler().fit_transform(df[[cfg.exposure]])
        return col
    return cfg.exposure


def compute_pca_corrected_residuals(
    cfg: Config, gxe_cfg: GXEConfig, df_env: pd.DataFrame, gen_map: Optional[pd.DataFrame], logger,
) -> pd.DataFrame:
    """STEP A: for each requested generation,
        onset_age  = [PCA_1..k, additive covariates] . delta + eps
        onset_age' = onset_age - [PCA, covariates] . delta_hat

    The PCs (and additive covariates like 'sex') are used ONLY here. They
    are never passed to GENetLib.
    """
    generations = gxe_cfg.generations or [cfg.generation]

    if gen_map is None and len(generations) > 0:
        raise RuntimeError(
            "Cannot run Step A: no id->generation map available "
            "(see resolve_generation_map) and the PCA data is generation-specific."
        )

    all_residuals = []
    for generation in generations:
        ids_this_gen = set(gen_map.loc[gen_map["generation"] == generation, "id"])
        sub = df_env[df_env[cfg.sample_id_col].isin(ids_this_gen)].copy()
        if sub.empty:
            logger.warning("[STEP A] generation=%s: no patients found, skipping", generation)
            continue

        sub, covariate_cols = _encode_covariates(sub, cfg.covariates)

        pca_df = None
        pc_cols = []
        if cfg.use_pca_covariates:
            pca_df = load_pca_covariates(cfg.pca_covariates_path_template, generation, cfg.pca_n_components)
            pc_cols = [c for c in pca_df.columns if c != PCA_ID_COLUMN]
            n_before = len(sub)
            sub = sub.merge(pca_df, left_on=cfg.sample_id_col, right_on=PCA_ID_COLUMN, how="inner")
            if len(sub) < n_before:
                logger.warning(
                    "[STEP A] generation=%s: %d patients lost in the merge with the PCA data (id not in common)",
                    generation, n_before - len(sub),
                )

        design_cols = pc_cols + covariate_cols
        sub = sub.dropna(subset=[cfg.target_col] + design_cols)
        if sub.empty or not design_cols:
            logger.warning(
                "[STEP A] generation=%s: empty dataset or no correction covariate "
                "available (PCA=%s, covariates=%s), skipping", generation, cfg.use_pca_covariates, cfg.covariates,
            )
            continue

        X = sub[design_cols].to_numpy(dtype=float)
        y = sub[cfg.target_col].to_numpy(dtype=float)

        reg = LinearRegression(fit_intercept=True)
        reg.fit(X, y)
        residuals = y - reg.predict(X)
        r2 = reg.score(X, y)
        logger.info(
            "[STEP A] generation=%s: regression %s ~ %s on %d patients, R2=%.4f",
            generation, cfg.target_col, design_cols, len(sub), r2,
        )

        out = sub[[cfg.sample_id_col]].copy()
        out["generation"] = generation
        out[cfg.target_col] = y
        out[f"{cfg.target_col}_resid"] = residuals
        all_residuals.append(out)

    if not all_residuals:
        raise RuntimeError("STEP A: no residuals computed for any requested generation.")

    result = pd.concat(all_residuals, ignore_index=True)
    logger.info("[STEP A] Total patients with corrected phenotype (Y'): %d", len(result))
    return result


# ============================================================================
# 3. GENOTYPE READING PER CHROMOSOME (gen.parquet, WIDE format)
# ============================================================================

def get_variant_schema(raw_file: str) -> list:
    """Reads ONLY the parquet schema (column names), without loading the
    data: efficient even with ~1.3M columns (reads only the footer)."""
    return pq.ParquetFile(raw_file).schema_arrow.names


def _chrom_prefix_candidates(chrom: str) -> set:
    """Source VCF contigs may be named '1' or 'chr1' (see
    extract_matrix.resolve_chrom_name): both conventions are accepted here
    without needing to know which one in advance."""
    return {str(chrom), f"chr{chrom}", f"Chr{chrom}", f"CHR{chrom}"}


def select_columns_for_chromosome(all_columns: list, chrom: str, id_col: str = "id") -> list:
    """Selects, among all parquet columns, those belonging to the
    requested chromosome's variants (column name format:
    '{CHROM}_{POS}_{REF}_{ALT}', see id_utils.build_variant_label)."""
    candidates = _chrom_prefix_candidates(chrom)
    variant_cols = [
        c for c in all_columns
        if c != id_col and c.split("_", 1)[0] in candidates
    ]
    return variant_cols


def load_genotype_matrix_for_chromosome(
    cfg: Config, chromosome: str, all_columns: list, patient_ids: set, logger,
) -> pd.DataFrame:
    """Reads from gen.parquet, via DuckDB, ONLY the id column + the SNP
    columns for the requested chromosome (column pruning: DuckDB/parquet
    read from disk only the selected columns, not the entire ~1.3M-column
    file)."""
    variant_cols = select_columns_for_chromosome(all_columns, chromosome, cfg.sample_id_col)
    if not variant_cols:
        logger.warning("[chr%s] No SNP column found for this chromosome in the Parquet file", chromosome)
        return pd.DataFrame()

    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=4")
    try:
        cols_sql = ", ".join([f'"{cfg.sample_id_col}"'] + [f'"{c}"' for c in variant_cols])
        query = f"SELECT {cols_sql} FROM read_parquet('{cfg.raw_file}')"
        g_wide = con.execute(query).fetchdf()
    finally:
        con.close()

    g_wide[cfg.sample_id_col] = g_wide[cfg.sample_id_col].astype(str).map(clean_sample_id)
    g_wide = g_wide.drop_duplicates(cfg.sample_id_col).set_index(cfg.sample_id_col)

    # Filter immediately to just the patients we need (reduces the cost of
    # subsequent operations, especially for heavily populated chromosomes)
    g_wide = g_wide[g_wide.index.isin(patient_ids)]

    logger.info(
        "[chr%s] Genotype loaded: %d patients x %d SNPs (columns selectively read from Parquet)",
        chromosome, len(g_wide), len(variant_cols),
    )
    return g_wide


# ============================================================================
# 4. DATASET (G, E, Y') CONSTRUCTION FOR THE MODEL
# ============================================================================

def build_chromosome_dataset(
    cfg: Config, gxe_cfg: GXEConfig, chromosome: str,
    g_wide: pd.DataFrame, residual_df: pd.DataFrame, exposure_col: str, logger,
):
    if g_wide.empty:
        return None

    resid_col = f"{cfg.target_col}_resid"
    narrow = residual_df.set_index(cfg.sample_id_col)[[resid_col, exposure_col]]

    merged = g_wide.join(narrow, how="inner").dropna()
    if len(merged) < gxe_cfg.min_samples_required:
        logger.warning(
            "[chr%s] Only %d patients available after the merge (minimum required: %d), skipping",
            chromosome, len(merged), gxe_cfg.min_samples_required,
        )
        return None

    snp_cols = list(g_wide.columns)
    G_df = merged[snp_cols].astype(float)
    E_df = merged[[exposure_col]].astype(float)
    y_resid = merged[resid_col].astype(float).to_numpy()

    variances = G_df.var(axis=0)
    zero_var = variances[variances == 0].index.tolist()
    if zero_var:
        logger.info("[chr%s] Removed %d monomorphic SNPs in the current sample", chromosome, len(zero_var))
        G_df = G_df.drop(columns=zero_var)

    if G_df.shape[1] == 0:
        logger.warning("[chr%s] No polymorphic SNPs left, skipping", chromosome)
        return None

    logger.info(
        "[chr%s] Final dataset: %d patients, %d SNPs, E=%s",
        chromosome, len(merged), G_df.shape[1], list(E_df.columns),
    )
    return G_df, E_df, y_resid, list(G_df.columns), list(merged.index)


# ============================================================================
# 5. GENetLib (scalar_ge) TRAINING AND COEFFICIENT EXTRACTION
# ============================================================================

def run_genetlib_scalar_ge(gxe_cfg: GXEConfig, chromosome: str, G_df: pd.DataFrame, E_df: pd.DataFrame, y_resid: np.ndarray, logger):
    """Y' = G*beta + (G x E)*theta ,  E = exposure ONLY (never PCA/sex).

    NOTE: GENetLib is a neural network (MCP + L2), not a classic linear
    regression: "coefficients" = the network's sparse layer weights after
    training (net.sparse1 = main G effect, net.sparse2 = G x E
    interaction). "lam"/"alpha"/"max_iter" map to
    lambda2/Lambda/num_epochs.
    """
    torch.manual_seed(0)
    G = G_df.to_numpy(dtype=float)
    E = E_df.to_numpy(dtype=float)
    y = y_resid.reshape(-1, 1)

    result = scalar_ge(
        y=y, G=G, E=E, ytype="Continuous",
        num_hidden_layers=gxe_cfg.num_hidden_layers,
        nodes_hidden_layer=gxe_cfg.nodes_hidden_layer,
        num_epochs=gxe_cfg.num_epochs,
        learning_rate1=gxe_cfg.learning_rate1,
        learning_rate2=gxe_cfg.learning_rate2,
        lambda1=gxe_cfg.lambda1,
        lambda2=gxe_cfg.lambda2,
        Lambda=gxe_cfg.Lambda,
        threshold=gxe_cfg.significance_threshold,
        split_type=gxe_cfg.split_type,
        ratio=gxe_cfg.ratio,
        important_feature=True,
        plot=False,
    )

    train_res, ifs_g_idx, ifs_ge_idx = result
    train_loss, eval_loss, train_r2, eval_r2, net = train_res
    if torch.is_tensor(train_loss):
        train_loss = train_loss.detach().cpu().numpy()
    if torch.is_tensor(eval_loss):
        eval_loss = eval_loss.detach().cpu().numpy()
    if torch.is_tensor(train_r2):
        train_r2 = float(train_r2.detach().cpu().numpy())
    if torch.is_tensor(eval_r2):
        eval_r2 = float(eval_r2.detach().cpu().numpy())

    logger.info(
        "[chr%s] Training complete: MSE_train=%.4f MSE_valid=%.4f R2_train=%.4f R2_valid=%.4f",
        chromosome, float(np.asarray(train_loss).reshape(-1)[0]), float(np.asarray(eval_loss).reshape(-1)[0]), train_r2, eval_r2,
    )

    n_snps = G.shape[1]
    n_env = E.shape[1]
    coef_main = net.sparse1.weight.data.detach().cpu().numpy().reshape(-1)
    coef_inter_matrix = net.sparse2.weight.data.detach().cpu().numpy().reshape(-1).reshape(n_env, n_snps)

    metrics = {
        "train_loss": float(np.asarray(train_loss).reshape(-1)[0]),
        "eval_loss": float(np.asarray(eval_loss).reshape(-1)[0]),
        "train_r2": float(train_r2),
        "eval_r2": float(eval_r2),
        "n_samples": int(G.shape[0]),
        "n_snps": int(n_snps),
        "n_env": int(n_env),
    }
    return {
        "coef_main": coef_main,
        "coef_inter_matrix": coef_inter_matrix,
        "important_snp_idx": set(ifs_g_idx),
        "important_interaction_idx": set(ifs_ge_idx),
        "metrics": metrics,
    }


def build_results_table(snp_names, env_names, coef_main, coef_inter_matrix, important_snp_idx, important_interaction_idx) -> pd.DataFrame:
    n_snps = len(snp_names)
    max_abs_inter = np.max(np.abs(coef_inter_matrix), axis=0)
    argmax_env = np.argmax(np.abs(coef_inter_matrix), axis=0)
    sig_inter_snp_idx = {idx % n_snps for idx in important_interaction_idx}

    rows = []
    for i, snp in enumerate(snp_names):
        sig_main = i in important_snp_idx
        sig_inter = i in sig_inter_snp_idx
        rows.append({
            "snp_id": snp,
            "coef_main": coef_main[i],
            "abs_coef_main": abs(coef_main[i]),
            "significant_main": sig_main,
            "max_abs_coef_interaction": max_abs_inter[i],
            "top_interacting_env": env_names[argmax_env[i]] if env_names else None,
            "significant_interaction": sig_inter,
            "significant_overall": sig_main or sig_inter,
        })
    return pd.DataFrame(rows).sort_values("abs_coef_main", ascending=False).reset_index(drop=True)


# ============================================================================
# 6. PER-CHROMOSOME OUTPUT SAVING
# ============================================================================

def save_chromosome_outputs(gxe_cfg: GXEConfig, chromosome: str, results_df, coef_main, coef_inter_matrix, snp_names, env_names, metrics, logger) -> Path:
    chrom_dir = Path(gxe_cfg.output_dir) / f"chr{chromosome}"
    chrom_dir.mkdir(parents=True, exist_ok=True)

    np.save(chrom_dir / "coef_main.npy", coef_main)
    np.save(chrom_dir / "coef_interaction_matrix.npy", coef_inter_matrix)
    np.save(chrom_dir / "snp_names.npy", np.array(snp_names))
    np.save(chrom_dir / "env_names.npy", np.array(env_names))
    results_df.to_csv(chrom_dir / "snp_results.csv", index=False)
    with open(chrom_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if gxe_cfg.save_plots:
        try:
            top = results_df.reindex(results_df["abs_coef_main"].sort_values(ascending=False).index).head(25)
            fig, ax = plt.subplots(figsize=(9, 6))
            colors = ["#d62728" if s else "#1f77b4" for s in top["significant_overall"]]
            ax.barh(top["snp_id"].astype(str), top["coef_main"], color=colors)
            ax.set_xlabel("Main G effect coefficient (sparse1 weight)")
            ax.set_title(f"Chr{chromosome}: top 25 SNPs by |coefficient| (red = significant)")
            ax.invert_yaxis()
            fig.tight_layout()
            fig.savefig(chrom_dir / "top_snp_coefficients.png", dpi=150)
            plt.close(fig)
        except Exception:
            logger.warning("[chr%s] Could not generate the plot: %s", chromosome, traceback.format_exc())

    logger.info("[chr%s] Results saved to %s", chromosome, chrom_dir)
    return chrom_dir


# ============================================================================
# 7. SINGLE-CHROMOSOME PIPELINE (parallel worker)
# ============================================================================

def process_chromosome(args) -> dict:
    (chromosome, all_columns, residual_records, exposure_col, gxe_cfg_dict, cfg_dict) = args

    cfg = Config(**cfg_dict) if not isinstance(cfg_dict, Config) else cfg_dict
    gxe_cfg = GXEConfig(**gxe_cfg_dict) if not isinstance(gxe_cfg_dict, GXEConfig) else gxe_cfg_dict

    configure_logging(cfg.log_dir)
    logger = get_logger(f"{__name__}.chr{chromosome}")
    torch.set_num_threads(gxe_cfg.torch_threads_per_worker)

    residual_df = pd.DataFrame(residual_records)
    t0 = time.time()
    status = {"chromosome": chromosome, "status": "unknown", "error": None}

    try:
        logger.info("===== STARTING processing of chromosome %s =====", chromosome)
        patient_ids = set(residual_df[cfg.sample_id_col])

        g_wide = load_genotype_matrix_for_chromosome(cfg, chromosome, all_columns, patient_ids, logger)
        dataset = build_chromosome_dataset(cfg, gxe_cfg, chromosome, g_wide, residual_df, exposure_col, logger)
        if dataset is None:
            status["status"] = "skipped_insufficient_data"
            return status

        G_df, E_df, y_resid, snp_names, patient_ids_used = dataset
        model_out = run_genetlib_scalar_ge(gxe_cfg, chromosome, G_df, E_df, y_resid, logger)

        results_df = build_results_table(
            snp_names, list(E_df.columns), model_out["coef_main"], model_out["coef_inter_matrix"],
            model_out["important_snp_idx"], model_out["important_interaction_idx"],
        )
        n_sig = int(results_df["significant_overall"].sum())
        logger.info("[chr%s] Significant SNPs: %d / %d", chromosome, n_sig, len(results_df))

        chrom_dir = save_chromosome_outputs(
            gxe_cfg, chromosome, results_df, model_out["coef_main"], model_out["coef_inter_matrix"],
            snp_names, list(E_df.columns), model_out["metrics"], logger,
        )

        status.update({
            "status": "success", "n_snps": len(snp_names), "n_samples": len(patient_ids_used),
            "n_significant": n_sig, "output_dir": str(chrom_dir), "metrics": model_out["metrics"],
            "elapsed_sec": round(time.time() - t0, 1),
        })
        logger.info("===== FINISHED chromosome %s in %.1fs =====", chromosome, status["elapsed_sec"])
        return status

    except Exception as e:
        logger.error("[chr%s] ERROR: %s\n%s", chromosome, e, traceback.format_exc())
        status.update({"status": "error", "error": str(e), "elapsed_sec": round(time.time() - t0, 1)})
        return status


# ============================================================================
# 8. SUMMARY WORD REPORT (optional)
# ============================================================================

def write_word_summary(gxe_cfg: GXEConfig, run_results: list, logger) -> Optional[Path]:
    if not gxe_cfg.save_word_summary:
        return None
    if not _HAS_DOCX:
        logger.warning("python-docx not installed: Word report not generated")
        return None

    doc = Document()
    doc.add_heading("G x E Pipeline (GENetLib) - Summary Report", level=1)
    doc.add_paragraph(f"Generated on {datetime.now():%Y-%m-%d %H:%M:%S}")
    doc.add_paragraph(
        "PCA and additive covariates (sex) removed from the phenotype in a preliminary "
        "regression step (onset_age ~ PCA + sex); GENetLib estimated on the residuals, "
        "with E = exposure only (no G x PCA/sex interaction)."
    )
    table = doc.add_table(rows=1, cols=7)
    table.style = "Light Grid Accent 1"
    for i, h in enumerate(["Chr", "Status", "N patients", "N SNPs", "N sig.", "R2 valid", "Time (s)"]):
        table.rows[0].cells[i].text = h
    for r in run_results:
        row = table.add_row().cells
        row[0].text = str(r.get("chromosome", ""))
        row[1].text = str(r.get("status", ""))
        row[2].text = str(r.get("n_samples", "-"))
        row[3].text = str(r.get("n_snps", "-"))
        row[4].text = str(r.get("n_significant", "-"))
        m = r.get("metrics") or {}
        row[5].text = f"{m.get('eval_r2', float('nan')):.3f}" if m else "-"
        row[6].text = str(r.get("elapsed_sec", "-"))

    out_path = Path(gxe_cfg.output_dir) / "summary_report.docx"
    doc.save(out_path)
    logger.info("Word report saved to %s", out_path)
    return out_path


# ============================================================================
# 9. MAIN ORCHESTRATION
# ============================================================================

def run_gxe_genetlib_pipeline(cfg: Optional[Config] = None, gxe_cfg: Optional[GXEConfig] = None) -> None:
    cfg = cfg or get_config()
    gxe_cfg = gxe_cfg or GXEConfig()

    configure_logging(cfg.log_dir)
    logger = get_logger(__name__)
    logger.info("========== STARTING G x E PIPELINE (GENetLib) ==========")
    logger.info("RAW_FILE=%s ENV_FILE=%s EXPOSURE=%s TARGET=%s", cfg.raw_file, cfg.env_file, cfg.exposure, cfg.target_col)
    logger.info("Chromosomes: %s | Generations: %s", gxe_cfg.chromosomes, gxe_cfg.generations or [cfg.generation])

    for path, label in [(cfg.raw_file, "RAW_FILE"), (cfg.env_file, "ENV_FILE")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    all_columns = get_variant_schema(cfg.raw_file)
    if cfg.sample_id_col not in all_columns:
        raise ValueError(
            f"Sample id column '{cfg.sample_id_col}' not found in the schema of {cfg.raw_file}. "
            f"Columns (first 20 of {len(all_columns)}): {all_columns[:20]}"
        )
    logger.info("Parquet schema read: %d total columns (%d expected variants)", len(all_columns), len(all_columns) - 1)

    df_env = load_environment_and_phenotype(cfg, logger)
    exposure_col = build_exposure_column(cfg, df_env)
    gen_map = resolve_generation_map(cfg)

    residual_df = compute_pca_corrected_residuals(cfg, gxe_cfg, df_env, gen_map, logger)
    # exposure_col was computed on df_env (before the merge with the PCA
    # data in Step A): carry it into residual_df for the rest of the pipeline.
    residual_df = residual_df.merge(df_env[[cfg.sample_id_col, exposure_col]], on=cfg.sample_id_col, how="inner")

    residual_records = residual_df.to_dict("records")

    from dataclasses import asdict
    cfg_dict = asdict(cfg) if not isinstance(cfg, dict) else cfg
    # The nested DBConfig isn't needed by workers and may hold credentials:
    # don't propagate it to child processes.
    cfg_dict.pop("db", None)
    cfg_for_workers = Config(**{k: v for k, v in cfg_dict.items()})
    gxe_cfg_dict = asdict(gxe_cfg)

    tasks = [
        (chrom, all_columns, residual_records, exposure_col, gxe_cfg_dict, asdict(cfg_for_workers))
        for chrom in gxe_cfg.chromosomes
    ]

    logger.info("Starting parallel processing of %d chromosomes with %d workers", len(tasks), cfg.max_workers)
    run_results = []
    with ProcessPoolExecutor(max_workers=cfg.max_workers) as executor:
        futures = {executor.submit(process_chromosome, t): t[0] for t in tasks}
        iterator = as_completed(futures)
        if _HAS_TQDM:
            iterator = tqdm(iterator, total=len(futures), desc="Chromosomes processed")
        for future in iterator:
            chrom = futures[future]
            try:
                res = future.result()
            except Exception as e:
                logger.error("[chr%s] Unhandled exception in worker: %s", chrom, e)
                res = {"chromosome": chrom, "status": "error", "error": str(e)}
            run_results.append(res)

    n_ok = sum(1 for r in run_results if r["status"] == "success")
    n_err = sum(1 for r in run_results if r["status"] == "error")
    n_skip = sum(1 for r in run_results if r["status"] == "skipped_insufficient_data")
    logger.info("===== PIPELINE COMPLETE: %d ok, %d errors, %d skipped =====", n_ok, n_err, n_skip)

    os.makedirs(gxe_cfg.output_dir, exist_ok=True)
    pd.DataFrame(run_results).to_csv(Path(gxe_cfg.output_dir) / "run_summary.csv", index=False)
    write_word_summary(gxe_cfg, run_results, logger)


if __name__ == "__main__":
    run_gxe_genetlib_pipeline()
