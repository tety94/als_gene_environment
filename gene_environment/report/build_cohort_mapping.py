#!/usr/bin/env python3
"""
build_cohort_mapping.py

If gen.parquet keeps causing problems (thrift size limit / corrupted
file), this alternative rebuilds the id -> generation mapping by reading
the header of the filtered VCFs directly (sample ids are the columns
after #CHROM POS ID REF ALT QUAL FILTER INFO FORMAT).

Only ONE file per generation is needed to get the full list of sample
ids for that cohort (the sample columns are the same across all
chromosomes). Uses bcftools if available (reads only the header, so it's
instant even on multi-GB files), otherwise falls back to zcat + parsing.

Output: output/table1/id_generation_mapping.csv  (columns: id, generation)

Then in generate_table1.py set:
    COHORT_SOURCE = "csv"
    COHORT_MAPPING_CSV = "output/table1/id_generation_mapping.csv"
"""

from __future__ import annotations

import gzip
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

# ============================================================
# CONFIG — edit here
# ============================================================

# For each generation: directory with the filtered VCFs, and WHICH
# representative file to use to read the header (one chromosome is
# enough; use the "_filtered.vcf.gz" file, NOT "_selected_filtered",
# which should contain the full sample set for that generation).
GENERATION_VCF_DIRS = {
    "gen1": "/mnt/cresla_prod/genome_datasets/gen1/vcf_filtered",
    "gen2": "/mnt/cresla_prod/genome_datasets/gen2/vcf_filtered",
}

# Pattern of the representative file used to extract sample ids
# (looked up inside GENERATION_VCF_DIRS[generation]).
REPRESENTATIVE_CHR_PATTERN = "*_vcf_chr1_filtered.vcf.gz"  # avoid '*_selected_*'

OUTPUT_CSV = Path("output/table1/id_generation_mapping.csv")

# ============================================================


def find_representative_file(vcf_dir) -> Path:
    vcf_dir = Path(vcf_dir)
    if not vcf_dir.exists():
        sys.exit(f"ERROR: directory not found: {vcf_dir}")

    candidates = sorted(
        p for p in vcf_dir.glob(REPRESENTATIVE_CHR_PATTERN)
        if "_selected_" not in p.name
    )
    if not candidates:
        # fallback: any *_filtered.vcf.gz that isn't "selected", first match
        candidates = sorted(
            p for p in vcf_dir.glob("*_filtered.vcf.gz")
            if "_selected_" not in p.name and not p.name.endswith(".raw.parquet")
        )
    if not candidates:
        sys.exit(f"ERROR: no filtered VCF found in {vcf_dir}")

    return candidates[0]


def extract_sample_ids_bcftools(vcf_path: Path) -> list[str]:
    result = subprocess.run(
        ["bcftools", "query", "-l", str(vcf_path)],
        capture_output=True, text=True, check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def extract_sample_ids_manual(vcf_path: Path) -> list[str]:
    """Fallback without bcftools: reads line by line until it finds #CHROM."""
    with gzip.open(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#CHROM"):
                cols = line.rstrip("\n").split("\t")
                # fixed VCF columns: CHROM POS ID REF ALT QUAL FILTER INFO FORMAT, then samples
                fixed = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]
                return cols[len(fixed):]
            if not line.startswith("##") and not line.startswith("#CHROM"):
                # header ended without finding #CHROM: malformed file
                break
    sys.exit(f"ERROR: #CHROM line not found in the header of {vcf_path}")


def extract_sample_ids(vcf_path: Path) -> list[str]:
    if shutil.which("bcftools"):
        try:
            return extract_sample_ids_bcftools(vcf_path)
        except subprocess.CalledProcessError as e:
            print(f"  bcftools failed ({e}), trying manual parsing...")
    return extract_sample_ids_manual(vcf_path)


def run_build_cohort_mapping(output_csv: Path = OUTPUT_CSV) -> Path:
    rows = []
    for generation, vcf_dir in GENERATION_VCF_DIRS.items():
        rep_file = find_representative_file(vcf_dir)
        print(f"[{generation}] using representative file: {rep_file}")
        sample_ids = extract_sample_ids(rep_file)
        print(f"[{generation}] found {len(sample_ids)} sample ids")
        for sid in sample_ids:
            rows.append({"id": sid, "generation": generation})

    if not rows:
        sys.exit("ERROR: no sample ids extracted from any generation.")

    mapping = pd.DataFrame(rows)

    # check for ids duplicated across different generations (shouldn't happen)
    mapping["id"] = mapping["id"].str.split("_").str[0]
    dup = mapping[mapping.duplicated("id", keep=False)]
    if not dup.empty:
        print(f"WARNING: {dup['id'].nunique()} id(s) appear in more than one generation:")
        print(dup.sort_values("id").to_string(index=False))

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(output_csv, index=False)
    print(f"\nMapping saved to: {output_csv.resolve()}")
    print(mapping["generation"].value_counts())
    return output_csv


def main() -> None:
    run_build_cohort_mapping()


if __name__ == "__main__":
    main()
