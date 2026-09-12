"""Prepares the final dataset (genetics + environment + PCA merge) used by modeling.

The genetic dataframe (df_gen) has ~1.3M columns. Every merge/rename that
touches it rebuilds pandas' entire BlockManager, so the cost depends not
just on the number of rows but on how many times the wide object gets
"reshuffled". To minimize that:

  1. All the "narrow" parts (env, gen_map, PCA) are merged together FIRST
     -- they're small, so it's cheap to do as many times as needed.
  2. df_gen is renamed (to safe names) once, right after loading.
  3. A SINGLE final merge joins the narrow covariate block with df_gen.

The PCA-loading logic (load_pca_covariates) stays in pca_utils.py but is
called from here instead of from the orchestrator, so the merge with the
PCA data happens on the narrow dataframe, not the wide one.
merge_pca_covariates (in pca_utils.py) is no longer used.
"""
from __future__ import annotations

import os

import pandas as pd
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler

from gene_environment.config import Config, get_config
from gene_environment.logging_utils import get_logger
from gene_environment.utils.id_utils import clean_sample_id
from gene_environment.utils.pca_utils import load_pca_covariates, PCA_ID_COLUMN

log = get_logger(__name__)

NON_GEN_COLS = ["FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE", "id"]


def _load_genetic_data(cfg: Config) -> tuple[pd.DataFrame, list[str], dict, list[str]]:
    fmt = cfg.raw_file_format
    if fmt == "auto":
        fmt = "parquet" if cfg.raw_file.endswith(".parquet") else "csv"

    log.info("Loading genetic file from %s (format=%s)", cfg.raw_file, fmt)
    if fmt == "parquet":
        df_gen = pq.ParquetFile(
            cfg.raw_file,
            thrift_string_size_limit=2_000_000_000,
            thrift_container_size_limit=2_000_000_000,
        ).read(use_pandas_metadata=True).to_pandas()
        df_gen.index = df_gen.index.astype(str).map(clean_sample_id)
        df_gen.index.name = "id"
        df_gen = df_gen.reset_index()
    else:
        df_gen = pd.read_csv(cfg.raw_file, sep=cfg.sep, decimal=cfg.decimal, low_memory=False)
        if "IID" in df_gen.columns:
            df_gen = df_gen.rename(columns={"IID": "id"})
        if "id" in df_gen.columns:
            df_gen["id"] = df_gen["id"].astype(str).map(clean_sample_id)

    variant_cols = [c for c in df_gen.columns if c not in NON_GEN_COLS]
    log.info("Variant columns detected: %d", len(variant_cols))

    # Rename to "safe" names (variant_i) immediately: a single touch of the
    # wide frame for this operation, instead of doing it after other merges.
    safe = {g: f"variant_{i}" for i, g in enumerate(variant_cols)}
    df_gen = df_gen.rename(columns=safe)
    variant_cols_safe = list(safe.values())
    mapping = {v: k for k, v in safe.items()}

    return df_gen, variant_cols_safe, mapping, variant_cols


def _build_narrow_covariates(cfg: Config, gen_ids: pd.Series) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Builds the 'narrow' block (env + generation map + PCA), all cheap
    operations since the frames involved are small. gen_ids is passed
    purely for diagnostic purposes (logging how many ids match), it is
    never merged directly with df_gen here."""

    log.info("Loading environmental file from %s", cfg.env_file)
    df_env = pd.read_csv(cfg.env_file, sep=cfg.sep, decimal=cfg.decimal)
    df_env["id"] = df_env["id"].astype(str)
    SEX_ENCODING = {"M": 1, "F": 0}
    if "sex" in df_env.columns:
        unmapped = set(df_env["sex"].dropna().unique()) - set(SEX_ENCODING.keys())
        if unmapped:
            raise ValueError(f"'sex': unrecognized values {unmapped}, update SEX_ENCODING in {__name__}")
        df_env["sex"] = df_env["sex"].map(SEX_ENCODING).astype(float)
        log.info("sex encoded with %s", SEX_ENCODING)
    if "onset_site" in df_env.columns:
        df_env["onset_site"] = df_env["onset_site"].astype("category")

    df = df_env

    # ---- id -> generation map ----
    map_path = cfg.sample_generation_map or os.path.join(cfg.output_folder, "sample_generation_map.csv")
    if os.path.exists(map_path):
        gen_map = pd.read_csv(map_path, dtype={"id": str})
        print("DEBUG requested generation:", cfg.generation)
        print("DEBUG generations present in gen_map:", gen_map["generation"].unique())
        print("DEBUG rows before generation filter:", len(df))
        print("DEBUG sample filtered ids:", df["id"].head(5).tolist())
        n_before = len(df)
        df = df.merge(gen_map, on="id", how="left")
        n_missing_map = df["generation"].isna().sum()
        if n_missing_map:
            log.warning(
                "%d patients are not present in the id->generation map (%s): "
                "they will be excluded from the run (unknown generation).", n_missing_map, map_path,
            )
        df = df[df["generation"] == cfg.generation].drop(columns=["generation"])
        log.info(
            "Filter for generation=%s (id->generation map from build-matrix): %d -> %d rows",
            cfg.generation, n_before, len(df),
        )
    elif cfg.env_generation_col and cfg.env_generation_col in df_env.columns:
        n_before = len(df)
        df = df[df[cfg.env_generation_col].astype(str) == str(cfg.generation)]
        log.info(
            "Filter for generation=%s (column '%s' in the environmental file): %d -> %d rows",
            cfg.generation, cfg.env_generation_col, n_before, len(df),
        )
    else:
        log.warning(
            "No id->generation map found (%s) and no ENV_GENERATION_COL configured: "
            "using ALL rows with no generation filter.", map_path,
        )

    df = df.drop_duplicates("id")

    # ---- Exposure standardization (on the narrow frame) ----
    df[cfg.target_col] = pd.to_numeric(df[cfg.target_col], errors="coerce")
    log.info("Standardizing exposure '%s' (standardize=%s)", cfg.exposure, cfg.standardize)
    Ecols = []
    df[cfg.exposure] = pd.to_numeric(df[cfg.exposure], errors="coerce")
    if cfg.standardize:
        df[cfg.exposure + "_std"] = StandardScaler().fit_transform(df[[cfg.exposure]])
        Ecols.append(cfg.exposure + "_std")
    else:
        Ecols.append(cfg.exposure)

    # ---- PCA (narrow frame, cheap merge) ----
    covariate_cols: list[str] = []
    if cfg.use_pca_covariates:
        pca_df = load_pca_covariates(cfg.pca_covariates_path_template, cfg.generation, cfg.pca_n_components)

        if "id" in df.columns and PCA_ID_COLUMN not in df.columns:
            n_match = df["id"].isin(pca_df[PCA_ID_COLUMN]).sum()
            log.info("PCA: %d/%d ids in the covariate block matched in IID.", n_match, len(df))
            df = df.merge(pca_df, left_on="id", right_on=PCA_ID_COLUMN, how="left")
        else:
            df = df.merge(pca_df, on=PCA_ID_COLUMN, how="left")

        covariate_cols = [c for c in pca_df.columns if c != PCA_ID_COLUMN]
        n_missing = int(df[covariate_cols[0]].isna().sum())
        if n_missing:
            pct = 100 * n_missing / len(df)
            log.warning(
                "PCA: %d/%d samples (%.1f%%) had no match after the merge (narrow block).",
                n_missing, len(df), pct,
            )
        else:
            log.info("PCA: merge complete, all %d samples have PCs.", len(df))
    else:
        log.info("PCA disabled (cfg.use_pca_covariates=False): no population-structure covariate.")

    if "sex" in df.columns:
        n_missing_sex = int(df["sex"].isna().sum())
        if n_missing_sex:
            pct = 100 * n_missing_sex / len(df)
            log.warning("sex: %d/%d samples (%.1f%%) have no value, will be excluded by dropna().", n_missing_sex,
                        len(df), pct)
        else:
            log.info("sex: no missing values across %d samples.", len(df))
        covariate_cols = covariate_cols + ["sex"]
    else:
        log.warning("Column 'sex' not found in the environmental file: covariate not added.")

    log.info("Correction covariates used in the model: %s", covariate_cols or "none")
    print(df.columns)
    return df, Ecols, covariate_cols


def load_and_prepare_data(cfg: Config | None = None):
    cfg = cfg or get_config()

    df_gen, variant_cols_safe, mapping, variant_cols = _load_genetic_data(cfg)

    covariates, Ecols, covariate_cols = _build_narrow_covariates(cfg, df_gen["id"])

    log.info("Final merge (single touch of the wide genetic dataframe) on 'id'")
    print("DEBUG covariates id (after generation filter):", covariates["id"].head(10).tolist())
    print("DEBUG df_gen id (genetics):", df_gen["id"].head(10).tolist())
    print("DEBUG overlap:", len(set(covariates["id"]) & set(df_gen["id"])))
    df = pd.merge(covariates, df_gen, on="id", how="inner")

    n_cov, n_gen, n_merged = len(covariates), len(df_gen), len(df)
    log.info("Rows covariates=%d, genetics=%d, after final merge (inner)=%d", n_cov, n_gen, n_merged)
    if n_merged == 0:
        log.warning(
            "The final merge produced 0 rows: no id in common between covariates and genetics. "
            "Check the id format (genN_ prefixes, XXX_XXX duplication, etc.)."
        )
    elif n_merged < 0.5 * min(n_cov, n_gen):
        log.warning(
            "The final merge 'lost' more than 50%% of the expected rows (%d out of min(%d,%d)): "
            "check id consistency across the files.", n_merged, n_cov, n_gen
        )

    log.info("Unique ids post-merge: %d (total rows: %d)", df["id"].nunique(), len(df))

    return df, variant_cols_safe, mapping, Ecols, variant_cols, covariate_cols