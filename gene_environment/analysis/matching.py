"""Nearest-neighbor matching between mutant and non-mutant patients on covariates."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from gene_environment.logging_utils import get_logger

log = get_logger(__name__)


def _prepare_matching_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if not cols:
        raise ValueError("No columns provided for matching")

    features = []
    for c in cols:
        if c not in df.columns:
            log.warning("Matching column '%s' not found in dataframe: skipped", c)
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            features.append(df[c].fillna(df[c].mean()))
        else:
            dummies = pd.get_dummies(df[c].astype(str), prefix=c, drop_first=True)
            features.append(dummies)

    if not features:
        raise ValueError("No valid matching feature found in dataframe")

    X = pd.concat(features, axis=1)
    X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index)
    return X_scaled


def match_control_units(
    df: pd.DataFrame, variant_col: str, k: int = 2, covariates_for_matching: list[str] | None = None
) -> pd.DataFrame | None:
    group1 = df[df[variant_col] == 1].reset_index(drop=True)
    group0 = df[df[variant_col] == 0].reset_index(drop=True)

    if group1.shape[0] == 0 or group0.shape[0] == 0:
        return None

    if group1.shape[0] <= group0.shape[0]:
        base, other = group1, group0
        base_label, other_label = 1, 0
    else:
        base, other = group0, group1
        base_label, other_label = 0, 1

    df_matching = pd.concat([base, other], ignore_index=True)
    X = _prepare_matching_matrix(df_matching, covariates_for_matching or [])

    mask_base = df_matching[variant_col] == base_label
    X_base = X[mask_base]
    X_other = X[~mask_base]

    k_used = min(k, X_other.shape[0])
    if k_used == 0:
        return None

    nn = NearestNeighbors(n_neighbors=k_used).fit(X_other.values)
    distances, _ = nn.kneighbors(X_base.values)

    # Tie handling: if multiple "other" points are at the SAME distance as
    # the k-th neighbor (e.g. a cluster of identical values in the matching
    # covariate, such as a numerically consistent "unexposed" category),
    # include ALL of them, not just the first k found. This is standard
    # practice in the matching literature (e.g. R's MatchIt). Without this,
    # ties would be arbitrarily truncated to k representatives, biasing the
    # matched sample whenever the covariate had many repeated values.
    kth_dist = distances[:, -1]
    D_full = cdist(X_base.values, X_other.values)
    selected_other_pos = np.unique(np.where(D_full <= kth_dist[:, None] + 1e-9)[1])
    other_idx = X_other.index[selected_other_pos]

    matched_other = df_matching.loc[other_idx]
    matched_base = df_matching.loc[mask_base]

    return pd.concat([matched_base, matched_other], ignore_index=True)


def precompute_scaled_covariates(df: pd.DataFrame, covariates_for_matching: list[str]) -> np.ndarray:
    """Fit the StandardScaler ONCE on the matching covariates.

    The covariates (Ecols) never change between permutations -- only the
    treated/control label changes (_match_variant). Calling this function
    once per variant and passing the result to `match_control_units_indices`
    avoids re-fitting the scaler on every single permutation.

    Note: `_prepare_matching_matrix` scales on `df_matching` (base+other,
    i.e. the whole dataframe passed in), so it's statistically equivalent
    to scaling once on the whole `df_model` as done here: same set of rows,
    same mean/std, regardless of the order base/other are concatenated in.
    """
    return _prepare_matching_matrix(df, covariates_for_matching).values


def match_control_units_indices(
    labels: np.ndarray, X_scaled: np.ndarray, k: int = 2
) -> tuple[np.ndarray, np.ndarray] | None:
    """Fast equivalent of `match_control_units`, used in the permutation
    loop: instead of re-fitting `NearestNeighbors` (building a tree on
    every call) it uses `cdist` + `argpartition` on an ALREADY scaled
    covariate matrix (see `precompute_scaled_covariates`).

    Handles ties like `match_control_units` (see the comment there): if
    multiple "other" points are at the same distance as the k-th neighbor,
    all of them are included, not just the first k found by argpartition.

    Returns (matched_base_idx, matched_other_idx): arrays of integer
    POSITIONS in `X_scaled`/`labels` (not pandas indices), or None if a
    group is empty or no neighbors are available.
    """
    group1 = np.where(labels == 1)[0]
    group0 = np.where(labels == 0)[0]

    if group1.shape[0] == 0 or group0.shape[0] == 0:
        return None

    if group1.shape[0] <= group0.shape[0]:
        base, other = group1, group0
    else:
        base, other = group0, group1

    k_used = min(k, other.shape[0])
    if k_used == 0:
        return None

    D = cdist(X_scaled[base], X_scaled[other])
    idx_part = np.argpartition(D, k_used - 1, axis=1)[:, :k_used]
    kth_dist = np.take_along_axis(D, idx_part, axis=1).max(axis=1)
    selected_other = np.unique(np.where(D <= kth_dist[:, None] + 1e-9)[1])

    return base, selected_other


def check_balance(matched_df: pd.DataFrame | None, variant_col: str, covariates_for_matching: list[str]) -> dict:
    if matched_df is None:
        log.debug("check_balance: matched_df is None, no balance computed")
        return {}

    treated = matched_df[matched_df[variant_col] == 1]
    control = matched_df[matched_df[variant_col] == 0]
    smd_results: dict[str, float] = {}

    for c in covariates_for_matching:
        if c not in matched_df.columns:
            continue
        if pd.api.types.is_numeric_dtype(matched_df[c]):
            c_treated = treated[c].fillna(treated[c].mean())
            c_control = control[c].fillna(control[c].mean())
            mean_t, mean_c = c_treated.mean(), c_control.mean()
            n_t, n_c = c_treated.count(), c_control.count()
            std_t, std_c = c_treated.std(ddof=1), c_control.std(ddof=1)
            pooled_std = np.sqrt(((n_t - 1) * std_t ** 2 + (n_c - 1) * std_c ** 2) / max(n_t + n_c - 2, 1))
            smd_results[c] = 0.0 if pooled_std == 0 or np.isnan(pooled_std) else abs(mean_t - mean_c) / pooled_std
        else:
            dummies = pd.get_dummies(matched_df[c].astype(str), prefix=c, drop_first=True)
            for d_col in dummies.columns:
                d_treated = dummies.loc[treated.index, d_col]
                d_control = dummies.loc[control.index, d_col]
                p_t, p_c = d_treated.mean(), d_control.mean()
                p_pooled = (p_t * len(d_treated) + p_c * len(d_control)) / (len(d_treated) + len(d_control))
                pooled_std = np.sqrt(p_pooled * (1 - p_pooled))
                smd_results[d_col] = 0.0 if pooled_std == 0 else abs(p_t - p_c) / pooled_std

    return smd_results