"""Core of the per-variant test: matching + regression + permutation test
+ onset_age difference statistics, all saved together.

Population-structure correction: the OLS model includes, by default, the
first N principal components (PCA) as correction (NOT interaction)
covariates, loaded for the current generation (cfg.generation) from
gene_environment/utils/pca_utils.py. The PCA columns are populated once
per worker by `orchestrator.init_worker` into `global_covariate_cols`
(same pattern as `global_df`), so they don't need to be repassed on every
submit to the ProcessPoolExecutor.

To disable them: cfg.use_pca_covariates = False (env
USE_PCA_COVARIATES=false) -> global_covariate_cols stays [] and the model
has no additional covariates, neither in the smf.ols path nor in the fast
permutation path.

Permutation seeding: `rng` is seeded via a stable MD5-based hash
(`_stable_seed`) rather than Python's built-in `hash()` on a string.
`hash()` on a string is NOT stable across different process runs (hash
randomization has been on by default since 2012, PEP 456) unless
PYTHONHASHSEED=0 is set explicitly -- using it would mean the same data
with the same RANDOM_STATE could still produce slightly different
permutations (and empirical p-values) on every pipeline run, which is a
reproducibility problem for a statistical analysis.

Other notes:
  - tqdm inside ProcessPoolExecutor workers produces garbled output (dozens
    of processes writing progress bars to the same terminal), so periodic
    log lines (every N permutations) through the centralized logger are
    used instead.
  - "Adaptive early stopping" for the LIGHT permutations: every
    `adaptive_perm_check_every` permutations, checks whether the number of
    permutations with |beta_perm| >= |beta_obs| is already clearly too high
    (futility check), in which case it stops before wasting the remaining
    permutations on a variant that won't end up significant anyway. This
    matters because matching+OLS per permutation is the most expensive
    operation in the pipeline and is repeated N_PERM (up to N_PERM_HIGH)
    times per variant.
  - The onset_age difference statistics (mutant vs non-mutant, on the same
    exact dataset used for the model) are computed here, immediately, and
    returned together with the rest -> saved to the DB on the same row, no
    separate script to rerun afterwards.

NOTE: cfg.covariates (e.g. "sex") exists in config.py but is NOT yet used
in build_formula below (passed as an empty list). It hasn't been merged
automatically with the PCA covariates here because if "sex" were stored as
a non-numeric string it would break `assert_numeric_covariates` in the
fast path for EVERY variant (the fast path doesn't do dummy-encoding like
patsy) -- the actual dtype needs to be verified before adding it.
"""
from __future__ import annotations

import hashlib

import numpy as np
import statsmodels.formula.api as smf

from gene_environment.analysis.fast_ols import (
    assert_numeric_covariates,
    build_design_and_solve,
    interaction_column_index,
)
from gene_environment.analysis.matching import (
    check_balance,
    match_control_units,
    match_control_units_indices,
    precompute_scaled_covariates,
)
from gene_environment.analysis.onset_age_stats import compute_onset_age_result
from gene_environment.config import get_config
from gene_environment.db.repository import safe_val
from gene_environment.logging_utils import get_logger

log = get_logger(__name__)

# Populated by the worker initializer (see orchestrator.py) -- avoids
# passing/pickling the entire dataframe (and covariate list) on every
# submit.
global_df = None
global_covariate_cols: list[str] = []


def _stable_seed(base_seed: int, variant_col: str) -> int:
    """Deterministic seed, reproducible across different runs, unlike
    hash() on a string (see the module docstring)."""
    digest = hashlib.md5(variant_col.encode("utf-8")).hexdigest()
    offset = int(digest[:8], 16) % 2_000_000
    return base_seed + offset


def build_formula(onset_col: str, variant_col: str, exposures: list[str], covariates: list[str], df_subset) -> str:
    exposures_str = " + ".join(exposures)
    formula = f"{onset_col} ~ {variant_col} * ({exposures_str})"
    cov_in_df = [c for c in covariates if c in df_subset.columns]
    if cov_in_df:
        formula += " + " + " + ".join(cov_in_df)
    return formula


def _find_interaction_term(mod_params_index, variant_col: str) -> str | None:
    for name in mod_params_index:
        if ":" in name and variant_col in name:
            return name
    return None


from gene_environment.analysis.fast_ols import (
    assert_numeric_covariates,
    build_design_and_solve,
    design_column_names,
    interaction_column_index,
)

def _run_permutation_batch(
    df_model, variant_col, X_scaled, Ecols, covariate_cols, cfg, rng, n_perm, log_prefix,
    full_beta: bool = False,
):
    variant_values = df_model[variant_col].values
    y_values = df_model[cfg.target_col].values
    E_values = df_model[Ecols].values
    C_values = df_model[covariate_cols].values if covariate_cols else None
    n_ecols = E_values.shape[1]
    q = 0 if C_values is None else C_values.shape[1]
    n_cols = 2 + 2 * n_ecols + q
    inter_idx = interaction_column_index(n_ecols)

    betas = np.full((n_perm, n_cols) if full_beta else (n_perm,), np.nan)

    for i in range(n_perm):
        perm_variant = rng.permutation(variant_values)
        perm_labels = (perm_variant > 0).astype(int)

        matched = match_control_units_indices(perm_labels, X_scaled, k=cfg.match_k)
        if matched is None:
            continue
        base_idx, other_idx = matched
        idx = np.concatenate([base_idx, other_idx])
        if idx.shape[0] < cfg.min_sample_size:
            continue

        C_idx = C_values[idx] if C_values is not None else None
        beta = build_design_and_solve(perm_variant[idx], E_values[idx], y_values[idx], C_idx)
        if beta is None:
            continue

        if full_beta:
            betas[i, :] = beta
        else:
            betas[i] = beta[inter_idx]

        if (i + 1) % 500 == 0:
            log.debug("%s: %d/%d permutations completed", log_prefix, i + 1, n_perm)

    if full_beta:
        valid = ~np.isnan(betas[:, 0])
        return betas[valid]
    return betas[~np.isnan(betas)]

def process_single_variant(variant_col: str, variant_original: str, Ecols: list[str],  full_beta: bool = False) -> dict | None:
    cfg = get_config()
    df = global_df
    covariate_cols = global_covariate_cols  # e.g. the PCs, populated by init_worker; [] if disabled

    df = df[df[variant_col] != "."].copy()
    df[variant_col] = df[variant_col].astype(int)
    df["_match_variant"] = (df[variant_col] > 0).astype(int)

    n_treated = int((df["_match_variant"] == 1).sum())
    n_control = int((df["_match_variant"] == 0).sum())

    def _empty(obs_coef=None, max_smd=None, iterations=cfg.n_perm, onset=None):
        return {
            "variant": variant_original,
            "n_treated": n_treated,
            "n_control": n_control,
            "obs_coef": obs_coef,
            "perm_mean": None,
            "perm_std": None,
            "p_emp": 1,
            "max_smd": max_smd,
            "iterations": iterations,
            "empirical_p_significant": False,
            "onset": onset,
        }

    if n_treated < cfg.min_treated or n_control < cfg.min_treated:
        return _empty()

    # covariate_cols (the PCs) enters here into the column selection: if a
    # sample has no PCA (merge failed for that IID, see pca_utils.py) it is
    # dropped by dropna() exactly like any other missing covariate -- same
    # treatment, no special handling.
    cols = [cfg.target_col, variant_col, "_match_variant"] + Ecols + covariate_cols
    df_model = df[cols].dropna()
    if df_model.shape[0] < cfg.min_sample_size:
        return _empty()

    # ---- POOLED onset_age statistics (mutant vs non-mutant, exposure
    # ignored): computed here, on the same dataset used for the model, so
    # they're consistent with the rest of the result and get saved to the
    # DB on the same row/transaction.
    # NB: the version STRATIFIED by exposure (mutant/non-mutant x
    # exposed/unexposed) is NOT computed here -- it would cost work on
    # millions of variants, the vast majority of which aren't significant.
    # It's computed downstream, only for significant variants, by the
    # dedicated script (see significant_variants/). ----
    mutati_age = df_model.loc[df_model["_match_variant"] == 1, cfg.target_col]
    non_mutati_age = df_model.loc[df_model["_match_variant"] == 0, cfg.target_col]
    onset_result = compute_onset_age_result(
        mutati_age, non_mutati_age,
        use_mann_whitney=cfg.use_mann_whitney,
        alpha=cfg.onset_alpha,
        min_group_size=cfg.onset_min_group_size,
        low_power_threshold=cfg.onset_low_power_threshold,
        n_boot=cfg.n_boot,
        seed=cfg.random_state,
    )
    onset_dict = onset_result.__dict__ if onset_result is not None else None

    matched_obs = match_control_units(
        df_model, "_match_variant", k=cfg.match_k, covariates_for_matching=Ecols + covariate_cols
    )
    if matched_obs is None or matched_obs.shape[0] < cfg.min_sample_size:
        return _empty(onset=onset_dict)

    smd_results = check_balance(matched_obs, "_match_variant", Ecols + covariate_cols)
    max_smd = max(smd_results.values()) if smd_results else 1

    if max_smd > cfg.max_smd:
        return _empty(max_smd=max_smd, onset=onset_dict)

    # covariate_cols (sex, PCs) enters here into the formula as an additive
    # term "+ sex + PC1 + PC2 + ..." OUTSIDE the multiplication with
    # variant -> corrects the model without introducing variant:covariate
    # interactions (see build_formula and fast_ols.py for the design
    # matrix structure). The same covariates are ALSO used in the matching
    # above (covariates_for_matching=Ecols + covariate_cols): matching
    # balances exposure, sex and population structure together between
    # carriers and non-carriers, not exposure alone.
    formula = build_formula(cfg.target_col, variant_col, Ecols, covariate_cols, matched_obs)
    mod = smf.ols(formula=formula, data=matched_obs).fit()
    interaction_name = _find_interaction_term(mod.params.index, variant_col)

    if interaction_name is None:
        return _empty(onset=onset_dict)

    obs_coef = float(mod.params[interaction_name])
    n_treated_matched = int(matched_obs["_match_variant"].sum())
    n_control_matched = int((matched_obs["_match_variant"] == 0).sum())

    if full_beta:
        col_names = design_column_names(variant_col, Ecols, covariate_cols)
        obs_vec = np.array([mod.params.get(name, np.nan) for name in col_names])

        X_scaled = precompute_scaled_covariates(df_model, Ecols + covariate_cols)
        assert_numeric_covariates(df_model[Ecols + covariate_cols])

        perm_matrix = _run_permutation_batch(
            df_model, variant_col, X_scaled, Ecols, covariate_cols, cfg,
            np.random.RandomState(_stable_seed(cfg.random_state, variant_col)),
            cfg.n_perm, log_prefix=f"[{variant_col}] FULL", full_beta=True,
        )

        if perm_matrix.shape[0] == 0:
            full_model = {name: {"obs": safe_val(o), "perm_mean": None, "perm_std": None, "p_emp": None}
                          for name, o in zip(col_names, obs_vec)}
        else:
            perm_mean = perm_matrix.mean(axis=0)
            perm_std = perm_matrix.std(axis=0)
            p_emp = (np.abs(perm_matrix) >= np.abs(obs_vec)).mean(axis=0)
            full_model = {
                name: {
                    "obs": safe_val(o), "perm_mean": safe_val(m),
                    "perm_std": safe_val(s), "p_emp": safe_val(p),
                }
                for name, o, m, s, p in zip(col_names, obs_vec, perm_mean, perm_std, p_emp)
            }

        interaction_name = _find_interaction_term(mod.params.index, variant_col)
        inter_stats = full_model.get(interaction_name, {})

        return {
            "variant": variant_original,
            "n_treated": int(matched_obs["_match_variant"].sum()),
            "n_control": int((matched_obs["_match_variant"] == 0).sum()),
            "obs_coef": inter_stats.get("obs"),
            "perm_mean": inter_stats.get("perm_mean"),
            "perm_std": inter_stats.get("perm_std"),
            "p_emp": inter_stats.get("p_emp"),
            "max_smd": max_smd,
            "iterations": cfg.n_perm,
            "onset": onset_dict,
            "full_model": full_model,
        }

    if abs(obs_coef) < cfg.min_obs_coef:
        return {
            "variant": variant_original,
            "n_treated": n_treated_matched,
            "n_control": n_control_matched,
            "obs_coef": obs_coef,
            "perm_mean": None,
            "perm_std": None,
            "p_emp": 1,
            "max_smd": max_smd,
            "iterations": cfg.n_perm,
            "onset": onset_dict,
        }

    rng = np.random.RandomState(_stable_seed(cfg.random_state, variant_col))

    # Scaler on the MATCHING covariates (Ecols + covariate_cols, i.e.
    # exposure + sex + PCs together) fitted ONCE per variant (not on every
    # permutation, see fast_ols.py/matching.py). Must use the SAME set of
    # columns used for the observed matching above, otherwise the scaled
    # matrix here wouldn't be comparable to the one used to get
    # matched_obs. Computed only here, after the min_obs_coef filter, to
    # avoid wasted work on variants that won't reach the permutation phase
    # anyway.
    assert_numeric_covariates(df_model[Ecols + covariate_cols])
    X_scaled = precompute_scaled_covariates(df_model, Ecols + covariate_cols)

    # ======================================================
    # LIGHT permutations, with adaptive futility check:
    # every `adaptive_perm_check_every` permutations we check whether the
    # partial p-value is already well beyond the threshold (futility), in
    # which case we stop before finishing all N_PERM: the variant won't be
    # promoted to HIGH permutations anyway.
    # ======================================================
    perm_betas_light = []
    check_every = max(1, cfg.adaptive_perm_check_every)
    stopped_early = False

    for start in range(0, cfg.n_perm, check_every):
        n_batch = min(check_every, cfg.n_perm - start)
        batch = _run_permutation_batch(
            df_model, variant_col, X_scaled, Ecols, covariate_cols, cfg, rng, n_batch,
            log_prefix=f"[{variant_col}] LIGHT",
        )
        perm_betas_light.extend(batch.tolist())

        done = start + n_batch
        if done < cfg.n_perm and len(perm_betas_light) > 0:
            partial = np.array(perm_betas_light)
            partial_p = float(np.mean(np.abs(partial) >= abs(obs_coef)))
            if partial_p >= cfg.adaptive_perm_futility_p:
                log.debug(
                    "[%s] futility stop after %d/%d permutations (partial p=%.3f >= %.3f)",
                    variant_col, done, cfg.n_perm, partial_p, cfg.adaptive_perm_futility_p,
                )
                stopped_early = True
                break

    perm_betas_light = np.array(perm_betas_light)
    p_emp_light = float(np.mean(np.abs(perm_betas_light) >= abs(obs_coef))) if perm_betas_light.size > 0 else None
    iterations_light = len(perm_betas_light) if stopped_early else cfg.n_perm

    # ======================================================
    # HIGH permutations -- only if LIGHT is significant and we didn't stop
    # for futility.
    # ======================================================
    if not stopped_early and p_emp_light is not None and p_emp_light <= cfg.pvalue_threshold:
        n_additional = cfg.n_perm_high - cfg.n_perm
        perm_betas_additional = _run_permutation_batch(
            df_model, variant_col, X_scaled, Ecols, covariate_cols, cfg, rng, n_additional,
            log_prefix=f"[{variant_col}] HIGH",
        )
        perm_betas_final = np.concatenate([perm_betas_light, perm_betas_additional])
        p_emp_final = float(np.mean(np.abs(perm_betas_final) >= abs(obs_coef))) if perm_betas_final.size > 0 else 1
        iterations_final = cfg.n_perm_high
    else:
        perm_betas_final = perm_betas_light
        p_emp_final = p_emp_light
        iterations_final = iterations_light

    return {
        "variant": variant_original,
        "n_treated": n_treated_matched,
        "n_control": n_control_matched,
        "obs_coef": obs_coef,
        "perm_mean": float(np.mean(perm_betas_final)) if perm_betas_final.size > 0 else None,
        "perm_std": float(np.std(perm_betas_final)) if perm_betas_final.size > 0 else None,
        "p_emp": p_emp_final,
        "max_smd": max_smd,
        "iterations": iterations_final,
        "onset": onset_dict,
    }