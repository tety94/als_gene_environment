"""OLS "fast path" for the permutation loop.

`smf.ols(formula=...).fit()` (statsmodels + patsy) re-parses the text
formula and rebuilds the entire Model object (including standard errors,
T-stats, R^2, etc., which the permutation loop doesn't need: it only needs
the interaction coefficient) on every single call. In the permutation loop
this happens N_PERM (up to N_PERM_HIGH) times per variant, making it the
single largest cost in the whole pipeline.

With the formula actually used by the pipeline
(`target ~ variant * (Ecols) [+ covariates]`), the design matrix has a
FIXED structure known in advance:

    [intercept, variant, E_1..E_p, variant*E_1..variant*E_p, C_1..C_q]

where C_1..C_q are additional covariates that do NOT interact with
variant (e.g. the PCs for population-structure correction, see
pca_utils.py) -- patsy always appends them AFTER the interaction block,
because in `build_formula` they're added with "+ covariate" outside the
multiplication parentheses, not inside. Building the design matrix by hand
with numpy instead of calling patsy/statsmodels on every permutation gives
a large speedup on the individual call.

ASSUMPTION (checked against build_dataset.py): Ecols is always numeric
(pd.to_numeric plus optional standardization), NEVER categorical. The PCs
are always numeric by construction (plink2 --pca output). If in the future
Ecols or the additional covariates were to include non-numeric columns
(e.g. `cfg.covariates` with a categorical column like "sex" stored as a
string), this module would no longer be equivalent to smf.ols (patsy would
do automatic one-hot encoding) -- that's why `assert_numeric_covariates`
validates the dtypes and raises an explicit error instead of silently
giving wrong results.
"""
from __future__ import annotations

import numpy as np


def interaction_column_index(n_ecols: int) -> int:
    """Index of the FIRST variant:E_i interaction column in the design
    matrix [intercept, variant, E_1..E_p, variant*E_1..variant*E_p, C_1..C_q].

    In the pipeline Ecols always has a single element (see
    build_dataset.py: Ecols = [exposure (+ "_std")]), so there is only one
    interaction term and this index matches exactly what
    `_find_interaction_term` would find as the FIRST match in
    mod.params.index (same order as patsy: intercept, main effects,
    interactions, then any additional covariates C_1..C_q -- which don't
    shift this index, since they come after the interaction block, not
    before)."""
    return 2 + n_ecols  # 0=intercept, 1=variant, [2 .. 2+p-1]=E_i, then interactions


def build_design_and_solve(
    variant_values: np.ndarray,
    E: np.ndarray,
    y: np.ndarray,
    C: np.ndarray | None = None,
) -> np.ndarray | None:
    """Build the design matrix [1, variant, E, variant*E, C] and solve the
    OLS coefficients via lstsq (numerically equivalent to statsmodels,
    which also uses a pseudo-inverse under the hood).

    variant_values: array (n,) -- RAW variant values (dosage, not
        binarized: binarization is only used for matching, not for the
        regression).
    E: array (n, p) -- covariates/exposures THAT INTERACT with variant,
        ALWAYS numeric.
    y: array (n,) -- target (onset_age).
    C: array (n, q) or None -- additional non-interacting covariates (e.g.
        the population-structure PCs), ALWAYS numeric. None, or an array
        with q=0 columns, is equivalent to having no additional
        covariates: the design matrix falls back to [1, variant, E,
        variant*E], matching the behavior when PCA is disabled
        (cfg.use_pca_covariates=False) or covariate_cols is empty.

    Returns None if the system is too small/degenerate to solve (n <
    number of columns), otherwise the coefficient array in the same order
    as [intercept, variant, E_1..E_p, variant:E_1..variant:E_p, C_1..C_q].
    """
    n = variant_values.shape[0]
    p = E.shape[1]
    q = 0 if C is None else C.shape[1]
    n_cols = 2 + 2 * p + q
    if n < n_cols:
        return None

    v = variant_values.reshape(-1, 1)
    X = np.empty((n, n_cols), dtype=np.float64)
    X[:, 0] = 1.0
    X[:, 1] = variant_values
    X[:, 2:2 + p] = E
    X[:, 2 + p:2 + 2 * p] = v * E
    if q > 0:
        X[:, 2 + 2 * p:] = C

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def assert_numeric_covariates(E_df) -> None:
    """Explicit guard: the fast path assumes numeric covariates (exposures
    E and any additional covariates C, e.g. the PCs). If a categorical
    column is ever passed, an explicit error is better than a silently
    wrong coefficient (patsy would do dummy-encoding, this path doesn't).

    The caller (modeling.py) passes both Ecols and covariate_cols together
    (Ecols + covariate_cols), so validation covers both groups of columns
    that end up in the fast path's design matrix."""
    import pandas as pd

    non_numeric = [c for c in E_df.columns if not pd.api.types.is_numeric_dtype(E_df[c])]
    if non_numeric:
        raise TypeError(
            f"fast_ols requires numeric covariates, found non-numeric: {non_numeric}. "
            "The fast path doesn't do dummy-encoding like patsy: the original smf.ols path would be needed."
        )

def design_column_names(variant_col: str, Ecols: list[str], covariate_cols: list[str]) -> list[str]:
    """Names in the same order as build_design_and_solve:
    [intercept, variant, E_1..E_p, variant:E_1..variant:E_p, C_1..C_q]"""
    names = ["Intercept", variant_col]
    names += list(Ecols)
    names += [f"{variant_col}:{e}" for e in Ecols]
    names += list(covariate_cols)
    return names
