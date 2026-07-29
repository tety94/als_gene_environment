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
5. Joins with CODICE_GEN_FILE to attach the parals_codals / mutaz columns.
6. Assigns a `generation` column: 1 if the id is in the gen1 VCF, 2 if
   it's in the gen2 VCF. Rows whose id appears in NEITHER VCF are
   dropped.
7. Saves the list of variant columns and the variant -> exposure(s) map
   (both reused by generate_c9_stats.py to build the per-exposure
   reports), and the final merged CSV, to OUT_DIR.

"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
import pyarrow.parquet as pq

# Import of the function already existing in the codebase.
# Adjust the module path if get_annotated_results lives elsewhere
# (e.g. `from db_utils import get_annotated_results`)
from gene_environment.db.repository import get_annotated_results
from gene_environment.report.vcf_utils import get_sample_ids
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
VCF_FILE_GEN1 = "/mnt/cresla_prod/genome_datasets/gen1/gen1_onlycases_vcf_chr22.vcf.gz"
VCF_FILE_GEN2 = "/mnt/cresla_prod/genome_datasets/gen2/gen2_vcf_chr22.vcf.gz"
VARIANT_COLS_FILE = OUT_DIR / "variant_columns.json"  # list of variant columns, reused by generate_c9_stats.py
VARIANT_EXPOSURE_MAP_FILE = OUT_DIR / "variant_exposure_map.json"  # column -> list of exposures (raw, untranslated)

# In the parquet the sample id is NOT a column: it's the DataFrame index
# (as in _load_genetic_data). In the environmental CSV, the id is instead
# a real column (the first one).
ID_COL_RAW = "id"  # name we give the index after reset_index()

ID_COLS_RAW = [ID_COL_RAW]

# If the id column names differ between RAW and ENV, map them here:
# {"name_in_env": "name_in_raw"}
JOIN_RENAME_ENV_TO_RAW: Dict[str, str] = {}


def first_column_name(path: str) -> str:
    return pd.read_csv(path, nrows=0).columns[0]


def get_target_columns(annotated_df: pd.DataFrame) -> Set[str]:
    """Builds the set of column names to look for in the parquet: both
    `variant` and `char_<variant>`."""
    variants = annotated_df["variant"].dropna().unique().tolist()
    targets = set(variants) | {f"char_{v}" for v in variants}
    return targets


def filter_parquet_columns(raw_file: str, target_variants: Set[str]) -> List[str]:
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


def get_vcf_sample_ids(vcf_path: str) -> Set[str]:
    """Reads ONLY the VCF header (bcftools query -l, no writing, no
    modification of the file). Applies clean_sample_id to normalize
    doubled ids (e.g. ACH10008_ACH10008 -> ACH10008)."""
    if not os.path.exists(vcf_path):
        raise FileNotFoundError(f"VCF not found: {vcf_path}")

    raw_ids = get_sample_ids(vcf_path)
    cleaned = {clean_sample_id(rid) for rid in raw_ids}
    log.info("Sample ids found in the VCF: %d (raw example: %s)", len(raw_ids), raw_ids[:1])
    return cleaned


def build_variant_exposure_map(annotated_df: pd.DataFrame, matched_cols: List[str]) -> Dict[str, List[str]]:
    """For each variant column actually present in the parquet, finds
    which exposure(s) it's associated with according to
    get_annotated_results() (matching both the bare variant name and the
    char_-prefixed name). Exposure values are kept RAW (untranslated):
    this map is intermediate data consumed by generate_c9_stats.py to
    match against the merged CSV's own (raw) environmental-component
    column names, not something displayed directly."""
    mapping = {}
    for col in matched_cols:
        is_char = col.startswith("char_")
        bare = col[len("char_"):] if is_char else col
        matches = annotated_df[annotated_df["variant"] == bare]
        exposures = sorted(matches["exposure"].dropna().unique().tolist())
        mapping[col] = exposures
        if not exposures:
            log.warning("No exposure found for column %s", col)
    return mapping


def json_dump(path: Path, items) -> None:
    with open(path, "w") as f:
        json.dump(items, f, indent=2)


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

    # 5. Join with codice_gen.csv to attach the parals_codals / mutaz columns
    log.info("Loading CODICE_GEN_FILE (%s)...", CODICE_GEN_FILE)
    codice_gen_df = pd.read_csv(CODICE_GEN_FILE)
    for required_col in ("corretto", "parals_codals", "mutaz"):
        if required_col not in codice_gen_df.columns:
            raise KeyError(
                f"Column '{required_col}' does not exist in {CODICE_GEN_FILE}. "
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

    # 6. 'generation' column: 1 if the id is in the gen1 VCF, 2 if it's in
    # the gen2 VCF. Rows whose id appears in NEITHER VCF are dropped.
    log.info("Reading sample ids from the gen1 VCF (%s)...", VCF_FILE_GEN1)
    vcf1_ids = get_vcf_sample_ids(VCF_FILE_GEN1)
    log.info("Reading sample ids from the gen2 VCF (%s)...", VCF_FILE_GEN2)
    vcf2_ids = get_vcf_sample_ids(VCF_FILE_GEN2)

    both = vcf1_ids & vcf2_ids
    if both:
        log.warning("%d id(s) appear in BOTH VCFs (gen1 and gen2): %s", len(both), sorted(both)[:5])

    def assign_generation(x):
        in1 = x in vcf1_ids
        in2 = x in vcf2_ids
        if in1 and not in2:
            return 1
        if in2 and not in1:
            return 2
        if in1 and in2:
            return 2  # ambiguous: present in both, priority to gen2 (see warning above)
        return None  # not found in either VCF

    merged[ID_COL_RAW] = merged[ID_COL_RAW].astype(str)
    merged["generation"] = merged[ID_COL_RAW].apply(assign_generation)

    n_before = len(merged)
    n_dropped = merged["generation"].isna().sum()
    merged = merged.dropna(subset=["generation"]).copy()
    merged["generation"] = merged["generation"].astype(int)

    log.info(
        "Rows dropped because the id isn't in either VCF: %d / %d",
        n_dropped, n_before,
    )
    log.info(
        "Generation assigned: gen1=%d, gen2=%d",
        (merged["generation"] == 1).sum(),
        (merged["generation"] == 2).sum(),
    )

    # Save the list of variant columns and the column -> exposure(s) map,
    # reused by generate_c9_stats.py to build the per-environmental-component reports.
    json_dump(VARIANT_COLS_FILE, selected_cols)
    log.info("Variant-column list saved to %s (%d columns)", VARIANT_COLS_FILE, len(selected_cols))

    variant_exposure_map = build_variant_exposure_map(annotated_df, selected_cols)
    json_dump(VARIANT_EXPOSURE_MAP_FILE, variant_exposure_map)
    log.info("Variant -> exposure map saved to %s", VARIANT_EXPOSURE_MAP_FILE)

    # 7. Final save
    merged.to_csv(MERGED_CSV, index=False)
    log.info("Final file saved to %s", MERGED_CSV)
    return MERGED_CSV


def main() -> None:
    run_c9_check()


if __name__ == "__main__":
    main()
