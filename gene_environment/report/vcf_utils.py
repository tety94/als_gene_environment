"""
vcf_utils.py

Shared VCF-header sample-id extraction, used by both
`build_cohort_mapping.py` (looks up one representative file per
generation directory) and `c9_check.py` (looks up sample ids directly
from a given VCF, to assign generation 1/2 by set membership). Reads
only the VCF header (via `bcftools query -l`, or a manual header parse
if bcftools isn't available) -- instant even on multi-GB files, no need
to load genotype data.
"""

from __future__ import annotations

import gzip
import shutil
import subprocess
from pathlib import Path
from typing import List


def bcftools_sample_ids(vcf_path) -> List[str]:
    """Sample ids via `bcftools query -l`. Raises RuntimeError if
    bcftools isn't found or the call fails."""
    result = subprocess.run(
        ["bcftools", "query", "-l", str(vcf_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"bcftools query -l failed on {vcf_path} (exit {result.returncode}).\n"
            f"stderr: {result.stderr.strip()}"
        )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def manual_sample_ids(vcf_path) -> List[str]:
    """Fallback without bcftools: reads the VCF header line by line until
    #CHROM, then returns the columns after the 9 fixed VCF columns."""
    fixed = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]
    with gzip.open(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#CHROM"):
                cols = line.rstrip("\n").split("\t")
                return cols[len(fixed):]
            if not line.startswith("##"):
                break
    raise RuntimeError(f"#CHROM line not found in the header of {vcf_path}")


def get_sample_ids(vcf_path) -> List[str]:
    """bcftools if available, else the manual gzip-header fallback."""
    if shutil.which("bcftools"):
        try:
            return bcftools_sample_ids(vcf_path)
        except RuntimeError:
            pass
    return manual_sample_ids(vcf_path)
