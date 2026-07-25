#!/usr/bin/env python3
"""
plink_io.py
===========
Shared helpers for reading plink2 output tables and normalizing sample
IDs. Every Python script in this pipeline (qc_report.py,
interpret_plink_output.py, qc_supplementary_plots.py,
extract_pca_covariates.py, build_supplementary_report.py) imports from
here instead of re-implementing its own copy.

Why this exists: plink2's column-naming conventions have drifted
slightly across versions (e.g. leading '#' on the first header column,
"ID1"/"ID2" vs "IID1"/"IID2" in .kin0 files). Centralizing the parsing
means a version bump only needs to be handled in one place instead of
five, and the doubled-ID stripping logic (see strip_doubled_ids) behaves
identically everywhere it is used instead of silently diverging between
scripts.

Nothing in this module talks to the filesystem beyond reading the paths
it is given, and nothing here strips or renames sample IDs unless the
caller explicitly asks for it -- see strip_doubled_ids().
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_plink_table(path: Path) -> pd.DataFrame:
    """
    Read a whitespace-separated plink2 output table (.eigenvec, .kin0,
    .sexcheck, .het, .afreq, .vmiss, .smiss, ...) and strip the leading
    '#' plink2 puts on the first header column (e.g. '#FID' -> 'FID').
    """
    df = pd.read_csv(path, sep=r"\s+")
    df.columns = [c.lstrip("#") for c in df.columns]
    return df


def load_eigenvec(path: Path) -> pd.DataFrame:
    """Load pca.eigenvec (#FID IID PC1 PC2 ... PCk)."""
    df = read_plink_table(path)
    if "IID" not in df.columns:
        raise ValueError(
            f"Column IID not found in {path}. Columns found: {list(df.columns)}"
        )
    return df


def load_eigenval(path: Path | None) -> list[float] | None:
    """Load pca.eigenval (one eigenvalue per line). Returns None if path is None or missing."""
    if path is None or not Path(path).exists():
        return None
    with open(path) as f:
        return [float(line.strip()) for line in f if line.strip()]


def load_kinship(path: Path) -> pd.DataFrame:
    """
    Load king.kin0. Column names have varied slightly across plink2
    versions (ID1/ID2 vs IID1/IID2); this normalizes to IID1/IID2 and
    validates that the columns the rest of the pipeline relies on are
    present.
    """
    df = read_plink_table(path)
    df = df.rename(columns={"ID1": "IID1", "ID2": "IID2"})
    required = {"IID1", "IID2", "NSNP", "KINSHIP"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Expected columns missing from {path}: {missing}. "
            f"Columns found: {list(df.columns)}"
        )
    return df


def strip_doubled_id(iid: str) -> str:
    """
    Reduce an IID of the form 'NAME_NAME' (both halves identical -- typical
    when the source VCF had FamilyID == IndividualID, so plink2's
    --double-id produced a doubled string) down to 'NAME'. IDs where the
    two halves differ are left untouched, since the underscore there is
    presumably part of the real sample name rather than a doubling
    artifact.
    """
    if "_" in iid:
        first, _, rest = iid.partition("_")
        if rest == first:
            return first
    return iid


def strip_doubled_ids(series: pd.Series, label: str = "IID") -> pd.Series:
    """
    Vectorized wrapper around strip_doubled_id() that also prints a short
    summary of how many values were changed vs. left untouched. Always go
    through this function (rather than calling strip_doubled_id directly
    on a whole column) so every script reports the same diagnostic and
    doubled-ID stripping never happens silently anywhere in the pipeline.
    """
    stripped = series.apply(strip_doubled_id)
    n_changed = int((stripped != series).sum())
    n_total = len(series)
    print(f"  strip_doubled_id: {n_changed}/{n_total} {label} values in NAME_NAME form reduced to NAME")
    if n_changed < n_total:
        print(
            f"  NOTE: {n_total - n_changed} {label} values left unchanged (halves did not "
            f"match, or no underscore present) -- check these manually if that's unexpected "
            f"for your ID convention."
        )
    return stripped
