"""Loads the principal components (PCA) computed by the QC pipeline
(00_run_plink_qc.sh -> extract_pca_covariates.py), used as population-structure
correction covariates in the OLS model.

PCA is computed SEPARATELY per generation (gen1 = discovery, gen2 =
replication): there is no "combined" PCA to use here. Each generation has
its own pca_covariates.csv, produced by extract_pca_covariates.py from the
PCA computed on that cohort alone. That's why the file path depends on
cfg.generation (see load_pca_covariates).

NOTE: cfg.generation is an int (1, 2, 3 -- see config.py), not the string
"gen1"/"gen2": the "gen" prefix is already hardcoded in the default
template (PCA_COVARIATES_PATH_TEMPLATE), not in the substituted value.
"""
from __future__ import annotations

import pandas as pd

from gene_environment.logging_utils import get_logger

log = get_logger(__name__)

# Column used as the merge key between the main dataframe and the PCA data.
# Must match the sample ID column name in the main dataframe produced by
# build_dataset.py. IDs are in the "NAME_NAME" format (duplicated
# FamilyID_IndividualID) both in the main dataframe and in
# pca_covariates.csv (extracted WITHOUT --strip-doubled-id, by explicit
# choice), so the merge works without any transformation.
PCA_ID_COLUMN = "IID"


def load_pca_covariates(path_template: str, generation: int, n_components: int) -> pd.DataFrame:
    """Load pca_covariates.csv for the given generation and return a
    dataframe with columns [PCA_ID_COLUMN, PC1 .. PC<n_components>].

    path_template may contain the {generation} placeholder, substituted with
    cfg.generation as-is (an int: 1, 2, 3 -- the "gen" prefix belongs in the
    template itself), e.g. with the default template and cfg.generation=1:
      "/mnt/cresla_prod/genome_datasets/qc_output_gen{generation}/pca_covariates.csv"
    becomes ".../qc_output_gen1/pca_covariates.csv".
    """
    path = path_template.format(generation=generation)
    df = pd.read_csv(path)
    print(df.head(2))
    if PCA_ID_COLUMN not in df.columns:
        raise ValueError(
            f"Column '{PCA_ID_COLUMN}' not found in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    pc_cols = [f"PC{i}" for i in range(1, n_components + 1)]
    missing = [c for c in pc_cols if c not in df.columns]
    if missing:
        available = sorted(
            (c for c in df.columns if c.startswith("PC")),
            key=lambda c: int(c[2:]) if c[2:].isdigit() else 0,
        )
        raise ValueError(
            f"Requested {n_components} PCs but {missing} are missing in {path}. "
            f"Available PCs: {available}"
        )

    log.info("PCA: loaded %d PCs for %d samples from %s (generation=%s)",
              n_components, len(df), path, generation)
    return df[[PCA_ID_COLUMN] + pc_cols]


def merge_pca_covariates(df: pd.DataFrame, pca_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Merge df with the PCA data on the PCA_ID_COLUMN column.

    PERFORMANCE NOTE: df has ~1.3M columns (one per variant), so a classic
    pd.merge is prohibitively slow (it would rebuild the entire
    BlockManager). Instead: realign pca_df to df's IID order with reindex()
    (O(n_pc x n_rows), not O(n_total_columns)) and assign the PC columns
    directly via numpy.
    """
    pc_cols = [c for c in pca_df.columns if c != PCA_ID_COLUMN]

    if PCA_ID_COLUMN not in df.columns:
        raise ValueError(
            f"Column '{PCA_ID_COLUMN}' not found in the main dataframe: "
            f"cannot merge with the PCA data. Available columns: {list(df.columns)}"
        )

    # Equivalent to validate="many_to_one": check for duplicated IIDs in pca_df.
    dup = pca_df[PCA_ID_COLUMN].duplicated()
    if dup.any():
        raise ValueError(
            f"pca_covariates.csv contains {dup.sum()} duplicated IIDs: "
            "cannot guarantee many_to_one."
        )

    pca_indexed = pca_df.set_index(PCA_ID_COLUMN)
    pca_aligned = pca_indexed.reindex(df[PCA_ID_COLUMN])

    for c in pc_cols:
        df[c] = pca_aligned[c].to_numpy()

    n_missing = int(df[pc_cols[0]].isna().sum())
    if n_missing > 0:
        pct = 100 * n_missing / len(df)
        log.warning(
            "PCA: %d/%d samples (%.1f%%) had no match after the merge "
            "(ID not found in the PCA file). They will be excluded from the "
            "model for every variant that involves them (dropna on the PC "
            "columns in process_single_variant). If the percentage is high, "
            "check the ID format in both files.",
            n_missing, len(df), pct,
        )
    else:
        log.info("PCA: merge complete, all %d samples have PCs.", len(df))

    return df, pc_cols
