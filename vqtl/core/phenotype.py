"""
Phenotype preparation for the vQTL scan.

This is genuinely specific to the vQTL/QUAIL method (z-score, log,
rank-based inverse-normal transform) and has no equivalent in
gene_environment (which does not transform onset_age, using it as-is in
matching+OLS). It operates directly on the DataFrame already merged by
`core.data.load_vqtl_dataset`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from gene_environment.logging_utils import get_logger

log = get_logger(__name__)


def rank_inverse_normal(x: np.ndarray) -> np.ndarray:
    ranks = rankdata(x, method="average")
    n = len(x)
    p = (ranks - 0.5) / n
    return norm.ppf(p)


def prepare_phenotype(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Adds <target_col>_z, _log, _rint. Returns a COPY of df."""
    df = df.copy()
    y = pd.to_numeric(df[target_col], errors="coerce").values.astype(float)

    df[f"{target_col}_z"] = (y - np.nanmean(y)) / np.nanstd(y)

    shift = min(0.0, np.nanmin(y)) - 1.0
    with np.errstate(invalid="ignore"):
        df[f"{target_col}_log"] = np.log(y - shift)

    df[f"{target_col}_rint"] = rank_inverse_normal(y)

    log.info(
        "Phenotype '%s' prepared: n=%d, mean=%.3f, sd=%.3f (columns added: _z, _log, _rint)",
        target_col, len(df), np.nanmean(y), np.nanstd(y),
    )
    return df
