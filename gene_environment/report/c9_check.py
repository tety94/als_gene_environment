"""
c9_check.py

Pipeline:
1. Calls the stored procedure `get_annotated_results()` -> gets
   (exposure, gene_name, variant, gna.*)
2. For EACH generation reads its own genetic parquet, keeping only the
   columns matching `variant` or `"char_" + variant` from (1):
       gen 1  -> GEN1_FILE
       gen 23 -> GEN23_FILE  (gen2 + gen3 pooled, as in keller_sensitivity.py)
   The generation is decided by the SOURCE FILE, not by the VCFs.
3. Overlap rule (same as keller_sensitivity.py): a sample present in the
   higher-generation file belongs to that generation and is removed from
   the lower one.
4. Concatenates the cohorts (variant columns missing in one file -> NaN)
   and saves the "restricted" CSV.
5. Joins with ENV_FILE (validated m:1) and with CODICE_GEN_FILE.
6. Saves the coverage-by-generation report, the list of variant columns,
   the variant -> exposure map, the variant -> gene map (reused by
   generate_c9_stats.py) and the final merged CSV.

NB: the `generation` column now takes the values 1 and 23 (23 = gen2+gen3),
the same labels used by keller_sensitivity.py. Change GEN23_LABEL if
generate_c9_stats.py expects 2.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
import pyarrow.parquet as pq

from gene_environment.db.repository import get_annotated_results
from gene_environment.utils.id_utils import clean_sample_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
GEN1_FILE = "/mnt/cresla_prod/genome_datasets/merged_csv/gen.parquet"
GEN23_FILE = "/mnt/cresla_prod/genome_datasets/gen3/gen2_gen3_selected_variants.parquet"
GEN1_LABEL = 1
GEN23_LABEL = 23  # gen2 + gen3 pooled

# Ordered from the HIGHEST generation to the lowest: higher ones win on overlaps.
SOURCES: List[Tuple[int, str]] = [(GEN23_LABEL, GEN23_FILE), (GEN1_LABEL, GEN1_FILE)]

ENV_FILE = "/srv/python-projects/gene_environment_v2/data/componenti_ambientali_full.csv"
OUT_DIR = Path("/mnt/cresla_prod/stefano_ge/c9_check")

RESTRICTED_CSV = OUT_DIR / "gen_restricted_variants.csv"
MERGED_CSV = OUT_DIR / "c9_check_merged.csv"
CODICE_GEN_FILE = OUT_DIR / "codice_gen.csv"
COVERAGE_CSV = OUT_DIR / "variant_coverage_by_generation.csv"
VARIANT_COLS_FILE = OUT_DIR / "variant_columns.json"
VARIANT_EXPOSURE_MAP_FILE = OUT_DIR / "variant_exposure_map.json"
VARIANT_GENE_MAP_FILE = OUT_DIR / "variant_gene_map.json"

# In the parquet the sample id is the DataFrame index; in the env CSV it's
# the first column.
ID_COL_RAW = "id"

THRIFT_LIMITS = dict(
    thrift_string_size_limit=2_000_000_000,
    thrift_container_size_limit=2_000_000_000,
)


def first_column_name(path: str) -> str:
    return pd.read_csv(path, nrows=0).columns[0]


def get_target_columns(annotated_df: pd.DataFrame) -> Set[str]:
    """Column names to look for in the parquet: `variant` and `char_<variant>`."""
    variants = annotated_df["variant"].dropna().unique().tolist()
    return set(variants) | {f"char_{v}" for v in variants}


def filter_parquet_columns(raw_file: str, target_variants: Set[str]) -> List[str]:
    """Reads only the parquet schema and returns the matching variant columns."""
    schema_cols = pq.ParquetFile(raw_file, **THRIFT_LIMITS).schema.names
    matched = [c for c in schema_cols if c in target_variants]
    log.info("%s: variant columns found %d / %d targets", Path(raw_file).name, len(matched), len(target_variants))
    if not matched:
        log.warning("%s: no variant column matched (char_ prefix?).", raw_file)
    return matched


def load_generation(path: str, target_variants: Set[str], label: int) -> Tuple[pd.DataFrame, List[str]]:
    """Reads one genetic file restricted to the target columns (+ index as id)."""
    if not Path(path).exists():
        raise FileNotFoundError(f"genetic file for generation {label} not found: {path}")
    cols = filter_parquet_columns(path, target_variants)
    pf = pq.ParquetFile(path, **THRIFT_LIMITS)
    # the index (sample id) is rebuilt by use_pandas_metadata=True
    df = pf.read(columns=cols, use_pandas_metadata=True).to_pandas()
    df.index = df.index.astype(str).map(clean_sample_id)
    df.index.name = ID_COL_RAW
    df = df.reset_index()

    dup = df[ID_COL_RAW].duplicated(keep=False)
    if dup.any():
        log.warning("gen %s: %d rows with duplicated ids (e.g. %s)",
                    label, int(dup.sum()), df.loc[dup, ID_COL_RAW].unique()[:5].tolist())
    df["generation"] = label
    log.info("gen %s: %d samples, %d variant columns", label, len(df), len(cols))
    return df, cols


def build_cohorts(target_variants: Set[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Loads every generation (highest first) and removes from each the
    samples that already belong to a higher generation."""
    frames, all_cols, seen_ids = [], [], set()
    for label, path in SOURCES:
        df, cols = load_generation(path, target_variants, label)
        ids = df[ID_COL_RAW].astype(str)
        hit = ids.isin(seen_ids)
        if hit.any():
            log.warning("gen %s: removed %d samples that belong to a higher generation", label, int(hit.sum()))
        seen_ids |= set(ids)  # all ids of this file, before removal (as keller_sensitivity.py)
        frames.append(df[~hit])
        all_cols += [c for c in cols if c not in all_cols]

    raw_df = pd.concat(frames, ignore_index=True)  # missing columns -> NaN
    selected_cols = [c for c in all_cols if c in raw_df.columns]
    raw_df = raw_df[[ID_COL_RAW, "generation"] + selected_cols]
    raw_df["generation"] = raw_df["generation"].astype(int)

    if raw_df[ID_COL_RAW].duplicated().any():
        log.warning("Duplicated ids remain after concat (within-file duplicates): %d",
                    int(raw_df[ID_COL_RAW].duplicated().sum()))
    log.info("Cohorts: %s", raw_df["generation"].value_counts().sort_index().to_dict())
    return raw_df, selected_cols


def log_generation_coverage(df: pd.DataFrame, variant_cols: List[str]) -> None:
    """% non-null genotypes per variant column and generation. Highlights
    variants populated in only one cohort (absent from the other file, or
    stored as v vs char_v)."""
    cov = df.groupby("generation")[variant_cols].agg(lambda s: s.notna().mean()).T
    cov.columns = [f"gen{c}_notnull" for c in cov.columns]
    cov.to_csv(COVERAGE_CSV)
    one_only = cov[(cov.min(axis=1) == 0) & (cov.max(axis=1) > 0)]
    if not one_only.empty:
        log.warning("%d variant columns populated in only one generation (see %s):\n%s",
                    len(one_only), COVERAGE_CSV, one_only.head(20).to_string())


def _bare(col: str) -> str:
    return col[len("char_"):] if col.startswith("char_") else col


def build_variant_exposure_map(annotated_df: pd.DataFrame, matched_cols: List[str]) -> Dict[str, List[str]]:
    """column -> sorted list of RAW exposures (matches bare and char_ names)."""
    mapping = {}
    for col in matched_cols:
        matches = annotated_df[annotated_df["variant"] == _bare(col)]
        exposures = sorted(matches["exposure"].dropna().unique().tolist())
        mapping[col] = exposures
        if not exposures:
            log.warning("No exposure found for column %s", col)
    return mapping


def build_variant_gene_map(annotated_df: pd.DataFrame, matched_cols: List[str]) -> Dict[str, Dict[str, str]]:
    """column -> {gene_name, gene_symbol}; first value kept if several."""
    gene_cols = [c for c in ("gene_name", "gene_symbol") if c in annotated_df.columns]
    if not gene_cols:
        log.warning("Neither 'gene_name' nor 'gene_symbol' in get_annotated_results() output.")
    mapping = {}
    for col in matched_cols:
        matches = annotated_df[annotated_df["variant"] == _bare(col)]
        info = {}
        for gc in gene_cols:
            vals = matches[gc].dropna().unique().tolist()
            info[gc] = vals[0] if vals else None
            if len(vals) > 1:
                log.warning("Variant %s has multiple distinct %s values %s, using the first.", col, gc, vals)
        mapping[col] = info
    return mapping


def json_dump(path: Path, items) -> None:
    with open(path, "w") as f:
        json.dump(items, f, indent=2)


def run_c9_check() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    id_col_env = first_column_name(ENV_FILE)
    log.info("ID column in ENV_FILE: %s", id_col_env)

    # 1. Stored procedure
    log.info("Calling get_annotated_results()...")
    annotated_df = get_annotated_results()
    if annotated_df.empty:
        raise RuntimeError("get_annotated_results() returned 0 rows, check the DB.")
    log.info("Annotations received: %d rows, %d columns", *annotated_df.shape)

    # 2-4. One genetic file per generation, overlap removal, concat
    target_variants = get_target_columns(annotated_df)
    raw_df, selected_cols = build_cohorts(target_variants)
    log_generation_coverage(raw_df, selected_cols)
    raw_df.to_csv(RESTRICTED_CSV, index=False)
    log.info("Restricted CSV saved to %s (%d rows, %d columns)", RESTRICTED_CSV, *raw_df.shape)

    # 5a. Join with the environmental file
    env_df = pd.read_csv(ENV_FILE)
    env_df[id_col_env] = env_df[id_col_env].astype(str).map(clean_sample_id)

    dup_mask = env_df[id_col_env].duplicated(keep=False)
    if dup_mask.any():
        env_df[dup_mask].sort_values(id_col_env).to_csv(OUT_DIR / "env_duplicated_ids.csv", index=False)
        n_dup_ids = env_df.loc[dup_mask, id_col_env].nunique()
        # 1) fully identical rows are harmless: collapse them
        env_df = env_df.drop_duplicates()
        # 2) same id with DIFFERENT values: real conflict, keep the first and warn
        conflict = env_df[id_col_env].duplicated(keep=False)
        if conflict.any():
            log.warning(
                "ENV_FILE: %d ids duplicated (see env_duplicated_ids.csv); %d ids still have "
                "CONFLICTING rows after removing exact copies, e.g. %s. Keeping the first row.",
                n_dup_ids, env_df.loc[conflict, id_col_env].nunique(),
                env_df.loc[conflict, id_col_env].unique()[:5].tolist(),
            )
            env_df = env_df.drop_duplicates(subset=id_col_env, keep="first")
        else:
            log.info("ENV_FILE: %d duplicated ids were exact copies, collapsed.", n_dup_ids)

    merged = raw_df.merge(
        env_df, left_on=ID_COL_RAW, right_on=id_col_env,
        how="inner", validate="m:1",  # fails if ENV has duplicated ids
    )
    log.info("Join with ENV: %d rows, %d columns (dropped %d samples without env data)",
             *merged.shape, len(raw_df) - len(merged))

    # 5b. Join with codice_gen.csv
    codice_gen_df = pd.read_csv(CODICE_GEN_FILE)
    for required_col in ("corretto", "parals_codals", "mutaz"):
        if required_col not in codice_gen_df.columns:
            raise KeyError(f"Column '{required_col}' missing in {CODICE_GEN_FILE}. "
                           f"Available: {list(codice_gen_df.columns)}")
    codice_gen_df["corretto"] = codice_gen_df["corretto"].astype(str).map(clean_sample_id)
    n_dup = int(codice_gen_df["corretto"].duplicated().sum())
    if n_dup:
        log.warning("codice_gen has %d duplicated 'corretto' ids: keeping the first.", n_dup)
        codice_gen_df = codice_gen_df.drop_duplicates(subset="corretto")

    merged[ID_COL_RAW] = merged[ID_COL_RAW].astype(str)
    merged = merged.merge(
        codice_gen_df[["corretto", "parals_codals", "mutaz"]],
        left_on=ID_COL_RAW, right_on="corretto", how="left", validate="m:1",
    ).drop(columns=["corretto"])
    log.info("Added parals_codals / mutaz: %d rows, %d columns", *merged.shape)
    log.info("Final cohorts: %s", merged["generation"].value_counts().sort_index().to_dict())

    # 6. Metadata for generate_c9_stats.py
    json_dump(VARIANT_COLS_FILE, selected_cols)
    json_dump(VARIANT_EXPOSURE_MAP_FILE, build_variant_exposure_map(annotated_df, selected_cols))
    json_dump(VARIANT_GENE_MAP_FILE, build_variant_gene_map(annotated_df, selected_cols))
    log.info("Metadata saved to %s", OUT_DIR)

    # 7. Final save
    merged.to_csv(MERGED_CSV, index=False)
    log.info("Final file saved to %s", MERGED_CSV)
    return MERGED_CSV


def main() -> None:
    run_c9_check()


if __name__ == "__main__":
    main()