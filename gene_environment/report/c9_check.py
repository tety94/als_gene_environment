"""
c9_check.py

Pipeline:
1. Calls the stored procedure `get_annotated_results()` -> gets
   (exposure, gene_name, variant, gna.*)
2. Filters the RAW_FILE parquet columns, keeping only those matching
   `variant` or `"char_" + variant` present in (1), plus the sample
   identifier column(s).
3. Saves a "restricted" CSV (genotypes of the annotated variants only).
4. Joins with ENV_FILE (environmental components) on the patient/sample key.
5. Saves the final result to OUT_DIR.

"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

# Import of the function already existing in the codebase.
# Adjust the module path if get_annotated_results lives elsewhere
# (e.g. `from db_utils import get_annotated_results`)
from gene_environment.db.repository import get_annotated_results
from gene_environment.utils.id_utils import clean_sample_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
RAW_FILE = "/mnt/cresla_prod/genome_datasets/merged_csv/gen.parquet"
ENV_FILE = "/srv/python-projects/gene_environment_v2/data/componenti_ambientali_full.csv"
OUT_DIR = Path("/mnt/cresla_prod/stefano_ge/c9_check")

RESTRICTED_CSV = OUT_DIR / "gen_restricted_variants.csv"
MERGED_CSV = OUT_DIR / "c9_check_merged.csv"
CODICE_GEN_FILE = OUT_DIR / "codice_gen.csv"

# In the parquet, the sample id is NOT a column: it's the DataFrame index
# (as in _load_genetic_data). In the environmental CSV, the id is instead
# a real column (the first one).
ID_COL_RAW = "id"  # name we give the index after reset_index()


def first_column_name(path: str) -> str:
    return pd.read_csv(path, nrows=0).columns[0]


ID_COLS_RAW = [ID_COL_RAW]

# If the id column names differ between RAW and ENV, map them here:
# {"name_in_env": "name_in_raw"}
JOIN_RENAME_ENV_TO_RAW: dict[str, str] = {}


def get_target_columns(annotated_df: pd.DataFrame) -> set:
    """Builds the set of column names to look for in the parquet: both
    `variant` and `char_<variant>`."""
    variants = annotated_df["variant"].dropna().unique().tolist()
    targets = set(variants) | {f"char_{v}" for v in variants}
    return targets


def filter_parquet_columns(raw_file: str, target_variants: set) -> list:
    """Reads only the parquet schema (without loading it all into memory)
    and returns the list of variant columns matching target_variants.
    The sample id is NOT among these columns: it's the index, handled
    separately at read time (use_pandas_metadata=True)."""
    schema_cols = pq.ParquetFile(
        raw_file,
        thrift_string_size_limit=2_000_000_000,
        thrift_container_size_limit=2_000_000_000,
    ).schema.names

    matched = [c for c in schema_cols if c in target_variants]

    log.info("Variant columns found in the parquet: %d / %d targets", len(matched), len(target_variants))
    if not matched:
        log.warning("No variant column matched. Check the naming format (char_ prefix?).")

    return matched


def run_c9_check() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    id_cols_env = [first_column_name(ENV_FILE)]

    log.info("ID column detected in the parquet (RAW): %s", ID_COLS_RAW[0])
    log.info("ID column detected in ENV_FILE: %s", id_cols_env[0])

    # 1. Stored procedure
    log.info("Calling get_annotated_results()...")
    annotated_df = get_annotated_results()
    if annotated_df.empty:
        raise RuntimeError("get_annotated_results() returned 0 rows, check the DB.")
    log.info("Annotations received: %d rows, %d columns", *annotated_df.shape)

    # 2. Target columns in the parquet
    target_variants = get_target_columns(annotated_df)
    selected_cols = filter_parquet_columns(RAW_FILE, target_variants)

    # 3. Restricted parquet read and CSV save
    # NB: the index (sample id) is reconstructed automatically by
    # use_pandas_metadata=True even though it's not among the requested
    # `columns`, because pyarrow tracks it separately from the data.
    log.info("Reading the parquet limited to %d columns (+ index)...", len(selected_cols))
    pf = pq.ParquetFile(
        RAW_FILE,
        thrift_string_size_limit=2_000_000_000,
        thrift_container_size_limit=2_000_000_000,
    )
    raw_df = pf.read(columns=selected_cols, use_pandas_metadata=True).to_pandas()
    raw_df.index = raw_df.index.astype(str).map(clean_sample_id)
    raw_df.index.name = ID_COL_RAW
    raw_df = raw_df.reset_index()

    raw_df.to_csv(RESTRICTED_CSV, index=False)
    log.info("Restricted CSV saved to %s (%d rows, %d columns)", RESTRICTED_CSV, *raw_df.shape)

    # 4. Join with the environmental file
    log.info("Loading ENV_FILE...")
    env_df = pd.read_csv(ENV_FILE)
    if JOIN_RENAME_ENV_TO_RAW:
        env_df = env_df.rename(columns=JOIN_RENAME_ENV_TO_RAW)

    missing_env_ids = [c for c in id_cols_env if c not in env_df.columns]
    if missing_env_ids:
        raise KeyError(
            f"ID column(s) {missing_env_ids} do not exist in ENV_FILE. "
            f"Available columns: {list(env_df.columns)}"
        )

    merged = raw_df.merge(
        env_df,
        left_on=ID_COLS_RAW,
        right_on=id_cols_env,
        how="inner",  # change to "left" to keep all samples from the parquet
    )
    log.info("Join completed: %d rows, %d columns", *merged.shape)

    # 5. Join with codice_gen.csv to attach the parals_codals column
    log.info("Loading CODICE_GEN_FILE (%s)...", CODICE_GEN_FILE)
    codice_gen_df = pd.read_csv(CODICE_GEN_FILE)
    if "corretto" not in codice_gen_df.columns:
        raise KeyError(
            f"Column 'corretto' does not exist in {CODICE_GEN_FILE}. "
            f"Available columns: {list(codice_gen_df.columns)}"
        )
    if "parals_codals" not in codice_gen_df.columns:
        raise KeyError(
            f"Column 'parals_codals' does not exist in {CODICE_GEN_FILE}. "
            f"Available columns: {list(codice_gen_df.columns)}"
        )

    codice_gen_df["corretto"] = codice_gen_df["corretto"].astype(str)
    merged[ID_COL_RAW] = merged[ID_COL_RAW].astype(str)

    merged = merged.merge(
        codice_gen_df[["corretto", "parals_codals", "mutaz"]],
        left_on=ID_COL_RAW,
        right_on="corretto",
        how="left",  # keeps all rows of `merged` even without a match
    )
    merged = merged.drop(columns=["corretto"])
    log.info("Added parals_codals: %d rows, %d columns", *merged.shape)

    # 6. Final save
    merged.to_csv(MERGED_CSV, index=False)
    log.info("Final file saved to %s", MERGED_CSV)
    return MERGED_CSV


def main() -> None:
    run_c9_check()


if __name__ == "__main__":
    main()
