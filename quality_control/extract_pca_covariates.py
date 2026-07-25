#!/usr/bin/env python3
"""
extract_pca_covariates.py
==========================
Extracts the first N principal components from pca.eigenvec and saves
them to a CSV ready to be merged with the exposure/phenotype data, for
use as covariates in the gene-environment interaction model.

IMPORTANT: pca.eigenvec must have been computed for THIS COHORT ONLY
(00_run_plink_qc.sh run with only this cohort's VCF directory/directories
as input, per the study's discovery/replication design -- see the root
README). A PCA fit on a pooled multi-cohort dataset would leak
information between cohorts and must never be used to produce this file.
This script does not check that for you -- it only extracts whatever PCs
are in the eigenvec file you point it at, so pass in the right one.

USAGE:
  python3 extract_pca_covariates.py \
      --eigenvec /mnt/genome_datasets/qc_output_cohortA/pca.eigenvec \
      --n-pcs 10 \
      --out /mnt/genome_datasets/qc_output_cohortA/pca_covariates.csv

The output CSV has one row per sample, columns: IID, PC1 ... PC10
(rename to e.g. "sample_id" with --id-column-name if your exposure
dataframe uses a different name for the merge key).
"""

import argparse
from pathlib import Path

from plink_io import load_eigenvec, strip_doubled_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extracts the first N PCs from pca.eigenvec and saves them as a covariate CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--eigenvec", required=True, type=Path, help="path to pca.eigenvec")
    parser.add_argument(
        "--n-pcs", type=int, default=10,
        help="number of principal components to extract (default 10, the max computed by the pipeline)",
    )
    parser.add_argument("--out", required=True, type=Path, help="output CSV path")
    parser.add_argument(
        "--id-column-name", default="IID",
        help="name to give the sample-ID column in the output CSV (default: IID)",
    )
    parser.add_argument(
        "--strip-doubled-id", action="store_true",
        help=(
            "if the IID is in 'NAME_NAME' form (the two halves separated by an "
            "underscore are identical -- typical when FamilyID=IndividualID in "
            "the source VCF), reduce it to 'NAME'. Does not touch IDs where the "
            "two halves differ (there the underscore is probably part of the "
            "real name). Use the SAME setting here as in "
            "interpret_plink_output.py for this cohort, since both scripts need "
            "to agree on what an IID looks like."
        ),
    )
    args = parser.parse_args()

    df = load_eigenvec(args.eigenvec)

    if args.strip_doubled_id:
        df["IID"] = strip_doubled_ids(df["IID"], label="IID")

    pc_cols = [f"PC{i}" for i in range(1, args.n_pcs + 1)]
    missing_pcs = [c for c in pc_cols if c not in df.columns]
    if missing_pcs:
        available = sorted(
            (c for c in df.columns if c.startswith("PC")),
            key=lambda c: int(c[2:]),
        )
        raise ValueError(
            f"Requested {args.n_pcs} PCs but missing: {missing_pcs}. "
            f"PCs available in {args.eigenvec}: {available}"
        )

    out_df = df[["IID"] + pc_cols].copy()
    if args.id_column_name != "IID":
        out_df = out_df.rename(columns={"IID": args.id_column_name})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    print(f"Samples: {len(out_df):,}")
    print(f"Components extracted: PC1 ... PC{args.n_pcs}")
    print(f"Saved to: {args.out}")
    print("\nPreview:")
    print(out_df.head().to_string(index=False))


if __name__ == "__main__":
    main()
