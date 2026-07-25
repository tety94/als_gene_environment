"""
Step 7 - robustness and permutations on the top loci.

1. Re-runs the Step 5 interaction test on the top loci with alternative
   phenotype transformations (log, rank-inverse-normal, outliers with
   |z|>3 removed) to assess the stability of beta_I.
2. Freedman-Lane permutation test: permutes the RESIDUALS of the reduced
   model (y ~ SNP + exposure + covariates, without the interaction term),
   adds them back to that same reduced model's fitted values, then refits
   the FULL model (with interaction) on the permuted outcome. This
   preserves the main-effect structure under H0 ("no interaction"), unlike
   directly permuting the raw phenotype, which would also destroy the
   covariate<->phenotype relationship (not part of H0).
3. IN THE SAME loop as point 2 (same per-locus iteration, not a separate
   step): a permutation-based Levene test on the variance of the
   residualized phenotype by genotype group, as an assumption-light
   confirmation of the variance effect detected by the Step 3 QUAIL scan
   for this locus. Procedure: the observed Levene statistic is computed on
   the REAL genotype groups (rounded 0/1/2 dosage) of the residualized
   phenotype; the genotype LABELS are permuted (not the residuals, unlike
   point 2) while keeping the residualized phenotype fixed; the statistic
   is recomputed on the permuted groups; this is repeated N_PERM times to
   build an empirical null distribution; the p-value is the fraction of
   permuted statistics >= the observed one. Any asymmetry/non-normality in
   the real phenotype distribution is automatically absorbed into the null
   (built from the same data), unlike a textbook Levene test, which assumes
   asymptotic normality. This is not a separate simulation: it uses the
   same parallelization infrastructure (same n_splits/joblib) as point 2,
   inside the same loop over the top loci. The result depends only on
   genotype (not on exposure): for a locus with several tested exposures,
   it is computed once and reused for the subsequent rows of the same SNP.

As in Step 5/6, dosage comes from a column of the DataFrame already built
(no VCF re-reading).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from scipy import stats

from gene_environment.logging_utils import get_logger

from vqtl.config import VqtlConfig
from vqtl.core.data import VqtlDataset, dosage_matrix
from vqtl.core.interaction import fit_interaction, is_binary
from vqtl.core.scan import residualize

log = get_logger(__name__)


def _fit_reduced_model(y, dosage, exposure_vals, covariates):
    df = pd.DataFrame({"y": y, "snp": dosage, "exposure": exposure_vals})
    for i, c in enumerate(covariates.T):
        df[f"cov{i}"] = c
    df = df.dropna()
    cov_cols = [c for c in df.columns if c.startswith("cov")]
    X = sm.add_constant(df[["snp", "exposure"] + cov_cols])
    model = sm.OLS(df["y"], X).fit()
    return model.fittedvalues.values, model.resid.values, df.index.values


def _levene_stat(genotype: np.ndarray, r: np.ndarray, min_group_size: int = 2) -> float:
    """Levene statistic (center='median', i.e. Brown-Forsythe: robust to
    non-normal phenotypes, the standard choice) on the 0/1/2 genotype
    groups of the residualized phenotype r. Groups with fewer than
    min_group_size observations are excluded (not enough data to estimate
    the variance in that group); if fewer than 2 groups remain, the
    statistic is undefined."""
    groups = [r[genotype == g] for g in (0, 1, 2) if np.sum(genotype == g) >= min_group_size]
    if len(groups) < 2:
        return np.nan
    try:
        stat, _p = stats.levene(*groups, center="median")
        return float(stat)
    except Exception:
        return np.nan


def _levene_perm_batch(genotype: np.ndarray, r: np.ndarray, n_perm_local: int, seed: int) -> np.ndarray:
    """Permutes the genotype LABELS (r stays fixed) and recomputes the
    Levene statistic at each iteration: builds the empirical null
    distribution for the locus's variance-by-genotype test."""
    rng = np.random.default_rng(seed)
    n = len(genotype)
    out = np.empty(n_perm_local)
    for i in range(n_perm_local):
        perm_genotype = genotype[rng.permutation(n)]
        out[i] = _levene_stat(perm_genotype, r)
    return out


def _run_levene_permutation_test(
    dosage: np.ndarray, y_orig: np.ndarray, covariates: np.ndarray, n_perm: int, n_jobs: int,
) -> dict:
    """Full permutation-based Levene test for one locus (one SNP): see
    point 3 of the module docstring. The phenotype is residualized on the
    SAME covariates used by the Step 3 scan (vqtl.core.scan.residualize),
    for consistency with the "variance effect" definition used there."""
    r_all, ok_mask = residualize(y_orig, covariates)
    ok = ok_mask & ~np.isnan(dosage)
    genotype = np.round(dosage[ok]).astype(int)
    r = r_all[ok]

    observed = _levene_stat(genotype, r)
    if np.isnan(observed):
        return {"levene_stat_observed": None, "levene_pval": None, "levene_n_perm_valid": None}

    n_jobs_eff = n_jobs if n_jobs > 0 else (os.cpu_count() or 1)
    n_splits = max(1, min(n_jobs_eff, 8))
    per_split = int(np.ceil(n_perm / n_splits))
    batches = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_levene_perm_batch)(genotype, r, per_split, seed=10_000 + s) for s in range(n_splits)
    )
    null_dist = np.concatenate(batches)[:n_perm]
    null_dist = null_dist[~np.isnan(null_dist)]
    n_valid = len(null_dist)
    # one-sided test: the Levene statistic is >= 0, "larger" is always in
    # the direction of "more heteroscedasticity" (unlike beta_I, which can
    # have either sign, so an absolute-value comparison would not make
    # sense here)
    pval = (1 + np.sum(null_dist >= observed)) / (n_valid + 1) if n_valid else np.nan

    return {
        "levene_stat_observed": observed,
        "levene_pval": float(pval) if not np.isnan(pval) else None,
        "levene_n_perm_valid": n_valid,
    }


def run_robustness_and_permutation(
    dataset: VqtlDataset, vcfg: VqtlConfig, interaction_df: pd.DataFrame, target_col: str, generation: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from vqtl.db import repository as repo

    if interaction_df.empty:
        log.warning("No interaction results: Step 7 skipped.")
        return pd.DataFrame(), pd.DataFrame()

    inv_mapping = {v: k for k, v in dataset.mapping.items()}
    covariates = dataset.df[dataset.covariate_cols].to_numpy(dtype=float) if dataset.covariate_cols else np.zeros((len(dataset.df), 0))
    binary_outcome = is_binary(dataset.df[target_col])

    top_loci = interaction_df.sort_values("pval").head(vcfg.perm_top_n_loci).copy()
    log.info("Step 7: top %d loci selected for robustness and permutations.", len(top_loci))

    y_orig = dataset.df[target_col].to_numpy(dtype=float)
    y_log = dataset.df[f"{target_col}_log"].to_numpy(dtype=float)
    y_rint = dataset.df[f"{target_col}_rint"].to_numpy(dtype=float)
    z_orig = (y_orig - np.nanmean(y_orig)) / np.nanstd(y_orig)
    outlier_mask = np.abs(z_orig) <= 3
    phenotype_variants = {
        "original": (y_orig, np.ones(len(y_orig), dtype=bool)),
        "log_transform": (y_log, np.ones(len(y_orig), dtype=bool)),
        "rank_inverse_normal": (y_rint, np.ones(len(y_orig), dtype=bool)),
        "outliers_removed": (y_orig, outlier_mask),
    }

    # ---- 1. Robustness across transformations / outliers ----
    robustness_placeholders = [
        {"variant": row["SNP"], "exposure": row["exposure"], "chromosome": row["CHR"], "position": row["POS"], "phenotype_variant": pv}
        for _, row in top_loci.iterrows() for pv in phenotype_variants
    ]
    repo.ensure_placeholders("robustness", generation, robustness_placeholders)
    done_robustness = repo.get_done_keys("robustness", generation)

    robustness_rows = []
    for _, row in top_loci.iterrows():
        snp_id, exp_raw = row["SNP"], row["exposure"]
        safe_col = inv_mapping.get(snp_id)
        exp_std_col = dataset.exposure_std_cols.get(exp_raw)
        if safe_col is None or exp_std_col is None:
            continue
        dosage = dosage_matrix(dataset, [safe_col])[:, 0]  # missing genotypes ('.') are treated as NaN, same as scan.py
        exposure_vals = dataset.df[exp_std_col].to_numpy(dtype=float)

        for variant_name, (yv, mask) in phenotype_variants.items():
            if (snp_id, exp_raw, variant_name) in done_robustness:
                continue
            res = fit_interaction(yv[mask], dosage[mask], exposure_vals[mask], covariates[mask], binary_outcome, robust=vcfg.robust_se)
            row_out = {"variant": snp_id, "exposure": exp_raw, "phenotype_variant": variant_name, "status": "done", "error_message": None}
            if res is None:
                row_out.update({"beta_i": None, "se": None, "pval": None, "n": None, "maf": None})
            else:
                row_out.update({"beta_i": res["beta_I"], "se": res["SE"], "pval": res["pval"], "n": res["N"], "maf": res["MAF"]})
            robustness_rows.append(row_out)
    if robustness_rows:
        repo.bulk_update_status("robustness", generation, robustness_rows)

    robustness_df = repo.fetch_results("robustness", generation)
    if not robustness_df.empty:
        robustness_df = robustness_df.rename(columns={"beta_i": "beta_I", "n": "N", "maf": "MAF", "se": "SE", "phenotype_variant": "variant"})
        robustness_df = robustness_df.dropna(subset=["pval"])

    # ---- 2. Freedman-Lane permutations ----
    n_perm = vcfg.n_perm
    log.info("Step 7: %d permutations for %d top loci (n_jobs=%s)", n_perm, len(top_loci), vcfg.n_jobs)

    perm_placeholders = [
        {"variant": row["SNP"], "exposure": row["exposure"], "chromosome": row["CHR"], "position": row["POS"]}
        for _, row in top_loci.iterrows()
    ]
    repo.ensure_placeholders("permutation", generation, perm_placeholders)
    done_perm = repo.get_done_keys("permutation", generation)
    levene_cache: dict[str, dict] = {}

    def _perm_batch(dosage, exposure_vals, resid_reduced, yhat_reduced, cov, n_perm_local, seed):
        rng = np.random.default_rng(seed)
        n = len(resid_reduced)
        estimates = np.empty(n_perm_local)
        for i in range(n_perm_local):
            perm_idx = rng.permutation(n)
            y_perm = yhat_reduced + resid_reduced[perm_idx]
            res = fit_interaction(y_perm, dosage, exposure_vals, cov, binary_outcome, robust=vcfg.robust_se)
            estimates[i] = res["beta_I"] if res is not None else np.nan
        return estimates

    for _, row in top_loci.iterrows():
        snp_id, exp_raw = row["SNP"], row["exposure"]
        if (snp_id, exp_raw) in done_perm:
            continue
        safe_col = inv_mapping.get(snp_id)
        exp_std_col = dataset.exposure_std_cols.get(exp_raw)
        if safe_col is None or exp_std_col is None:
            continue
        dosage = dosage_matrix(dataset, [safe_col])[:, 0]  # missing genotypes ('.') are treated as NaN, same as scan.py
        exposure_vals = dataset.df[exp_std_col].to_numpy(dtype=float)
        observed_beta = row["beta_I"]

        yhat_reduced, resid_reduced, kept_idx = _fit_reduced_model(y_orig, dosage, exposure_vals, covariates)
        dosage_kept = dosage[kept_idx]
        exposure_kept = exposure_vals[kept_idx]
        covariates_kept = covariates[kept_idx]

        n_jobs_eff = vcfg.n_jobs if vcfg.n_jobs > 0 else (os.cpu_count() or 1)
        n_splits = max(1, min(n_jobs_eff, 8))
        per_split = int(np.ceil(n_perm / n_splits))
        batches = Parallel(n_jobs=vcfg.n_jobs, backend="loky")(
            delayed(_perm_batch)(dosage_kept, exposure_kept, resid_reduced, yhat_reduced, covariates_kept, per_split, seed=s)
            for s in range(n_splits)
        )
        all_estimates = np.concatenate(batches)[:n_perm]
        all_estimates = all_estimates[~np.isnan(all_estimates)]
        n_valid = len(all_estimates)
        emp_p = (1 + np.sum(np.abs(all_estimates) >= abs(observed_beta))) / (n_valid + 1) if n_valid else np.nan

        # Permutation-based Levene test: depends only on genotype, not on
        # exposure -- computed once per SNP and reused if the same SNP
        # appears again in top_loci for another exposure.
        if snp_id not in levene_cache:
            levene_cache[snp_id] = _run_levene_permutation_test(dosage, y_orig, covariates, n_perm, vcfg.n_jobs)
        levene_result = levene_cache[snp_id]

        repo.bulk_update_status("permutation", generation, [{
            "variant": snp_id, "exposure": exp_raw, "status": "done", "error_message": None,
            "beta_i_observed": observed_beta, "n_perm_valid": n_valid, "empirical_pval": emp_p,
            "asymptotic_pval": row["pval"], **levene_result,
        }])
        log.info(
            "%s x %s: beta_I=%.4g, empirical p=%.4g (asymptotic=%.4g) | Levene stat=%s, empirical p=%s",
            snp_id, exp_raw, observed_beta, emp_p, row["pval"],
            levene_result["levene_stat_observed"], levene_result["levene_pval"],
        )

    perm_df = repo.fetch_results("permutation", generation)
    if not perm_df.empty:
        perm_df = perm_df.rename(columns={"beta_i_observed": "beta_I_observed"}).dropna(subset=["empirical_pval"])
    return robustness_df, perm_df
