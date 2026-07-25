"""
Step 3 - genome-wide vQTL scan (QUAIL-style):

  1. Residualize the standardized phenotype on the covariates (OLS) ->
     R = Y - covariates*b.
  2. For each SNP and each tau in cfg.taus: quantile regression of R on
     dosage at tau and at tau+0.5; beta_diff(tau) = beta_{tau+0.5} - beta_{tau}.
  3. beta_QI = mean_tau(beta_diff(tau)) (quantile-integrated effect).
  4. SE(beta_QI): two methods available via cfg.se_method (see below).

Dosage comes from a column of the DataFrame already built by
`core.data.load_vqtl_dataset` (already sample-aligned, already MAF/LD-
filtered upstream) instead of a per-chromosome VCF. Columns are processed
in chunks and parallelized with joblib (`cfg.n_jobs`, `cfg.chunk_size`),
passing each worker only the numpy sub-block for its chunk (not the whole
DataFrame) for efficiency.

=== SE method ===

The n_taus per-tau beta_diff(tau) estimates are NOT independent: they are
all fit on the exact same data (same dosage, same residual), just at
different -- and often closely spaced -- quantiles (default
0.05,0.10,...,0.45: 9 taus, step 0.05). There is no simple closed-form SE
for beta_QI without estimating the true covariance between the per-tau
estimates, which would require the joint asymptotic sandwich of the
quantile-regression process.

Both se_method options therefore estimate SE(beta_QI) via bootstrap
resampling (per-subject resampling, which makes no independence assumption
across taus), at two different costs:

- se_method="asymptotic" runs an internal mini-bootstrap
  (VQTL_ASYMPTOTIC_BOOTSTRAP_K replicates, default 50) -- cheap enough to
  run genome-wide, but with materially lower accuracy (see note below).
- se_method="bootstrap" (default) runs a fuller bootstrap
  (VQTL_BOOTSTRAP_K replicates, default 200) -- more accurate, at a
  substantially higher cost genome-wide.

Note on accuracy: with only VQTL_ASYMPTOTIC_BOOTSTRAP_K=50 replicates per
variant, se_method="asymptotic" underestimates SE(beta_QI) enough to
noticeably inflate the genomic-control factor (lambda_GC) on null data;
se_method="bootstrap" is well calibrated in the same setting. The
genomic-control correction applied in Step 4 (see core/filter_candidates.py)
partly compensates for this on the P_gc column, but assumes the inflation
factor is uniform across variants/MAF, which is not verified. Prefer
se_method="bootstrap" (the default) unless the reduced accuracy of
"asymptotic" is an acceptable trade-off for a faster genome-wide pass.
"""
from __future__ import annotations

import threading
import time
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from scipy import stats
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.tools.sm_exceptions import IterationLimitWarning

from gene_environment.logging_utils import get_logger

from vqtl.config import VqtlConfig
from vqtl.core.data import VqtlDataset, dosage_matrix, variant_chrom_pos

log = get_logger(__name__)

# Dosage is a discrete predictor (0/1/2): QuantReg's simplex solver can
# oscillate near the iteration limit on near-degenerate solutions. This cap
# is deliberately lower than statsmodels' default (500/1e-6): ~5x faster
# genome-wide for a negligible loss of precision.
QR_MAX_ITER = 100
QR_P_TOL = 1e-3

# ============================================================
# Convergence counters (diagnostics). WARNING: these are in-process --
# with Parallel(backend="loky") each worker is a separate process, so these
# counters in the main process do NOT reflect what happens in the parallel
# workers. Reliable only with n_jobs=1 (intended use: debug scripts), or if
# in the future they are aggregated explicitly from _process_chunk's return
# value instead of from global state.
# ============================================================

_convergence_lock = threading.Lock()
_convergence_stats = {"tau_fits_attempted": 0, "tau_fits_discarded": 0, "variants_all_nan": 0}


def reset_convergence_stats() -> None:
    with _convergence_lock:
        for k in _convergence_stats:
            _convergence_stats[k] = 0


def get_convergence_stats() -> dict:
    with _convergence_lock:
        return dict(_convergence_stats)


def _record_tau_fit(discarded: bool) -> None:
    with _convergence_lock:
        _convergence_stats["tau_fits_attempted"] += 1
        if discarded:
            _convergence_stats["tau_fits_discarded"] += 1


def _record_variant_all_nan() -> None:
    with _convergence_lock:
        _convergence_stats["variants_all_nan"] += 1


def residualize(y: np.ndarray, covariates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Residualizes y on the covariates (OLS). Public (not _residualize)
    because it is also reused by vqtl.core.permutation for the permutation
    Levene test on the locus (same covariate-adjustment logic as the Step 3
    scan, nothing to duplicate)."""
    X = sm.add_constant(covariates.astype(float))
    ok = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
    model = sm.OLS(y[ok], X[ok]).fit()
    resid = np.full_like(y, np.nan, dtype=float)
    resid[ok] = model.resid
    return resid, ok


def _beta_qi_single(dosage: np.ndarray, resid: np.ndarray, taus: list[float]) -> tuple[float, int]:
    """beta_QI only: one fit per tau and one per tau+0.5, mean of the diffs.
    Used both by the "full" bootstrap (per-subject resampling, K repeats of
    this function) and by the mini-bootstrap of se_method="asymptotic" (see
    _beta_qi_and_asymptotic_se) -- same fitting routine, different calling
    contexts, no duplicated logic."""
    ok = ~np.isnan(dosage) & ~np.isnan(resid)
    d, r = dosage[ok], resid[ok]
    if len(np.unique(d)) < 2 or len(d) < 20:
        return np.nan, int(ok.sum())
    X = sm.add_constant(d)
    diffs = []
    for tau in taus:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", IterationLimitWarning)
                fit_lo = QuantReg(r, X).fit(q=tau, max_iter=QR_MAX_ITER, p_tol=QR_P_TOL)
                fit_hi = QuantReg(r, X).fit(q=tau + 0.5, max_iter=QR_MAX_ITER, p_tol=QR_P_TOL)
            diffs.append(fit_hi.params[1] - fit_lo.params[1])
            _record_tau_fit(discarded=False)
        except (IterationLimitWarning, Exception):
            _record_tau_fit(discarded=True)
            continue
    if not diffs:
        _record_variant_all_nan()
        return np.nan, int(ok.sum())
    return float(np.mean(diffs)), int(ok.sum())


def _beta_qi_and_asymptotic_se(
    dosage: np.ndarray, resid: np.ndarray, taus: list[float], mini_bootstrap_k: int, seed: int,
) -> tuple[float, float, int]:
    """beta_QI (fit on the original data, one fit per tau and one per
    tau+0.5) + SE estimated via mini-bootstrap (mini_bootstrap_k replicates
    of _beta_qi_single on per-subject resampling) -- see the module
    docstring for why the per-tau estimates cannot simply be combined in
    closed form (they are correlated, not independent)."""
    ok = ~np.isnan(dosage) & ~np.isnan(resid)
    d, r = dosage[ok], resid[ok]
    if len(np.unique(d)) < 2 or len(d) < 20:
        return np.nan, np.nan, int(ok.sum())
    X = sm.add_constant(d)
    diffs = []
    for tau in taus:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", IterationLimitWarning)
                fit_lo = QuantReg(r, X).fit(q=tau, max_iter=QR_MAX_ITER, p_tol=QR_P_TOL)
                fit_hi = QuantReg(r, X).fit(q=tau + 0.5, max_iter=QR_MAX_ITER, p_tol=QR_P_TOL)
            diffs.append(fit_hi.params[1] - fit_lo.params[1])
            _record_tau_fit(discarded=False)
        except (IterationLimitWarning, Exception):
            _record_tau_fit(discarded=True)
            continue
    if not diffs:
        _record_variant_all_nan()
        return np.nan, np.nan, int(ok.sum())
    beta_qi = float(np.mean(diffs))

    # SE via mini-bootstrap (per-subject resampling, small K to stay fast
    # genome-wide).
    n = len(d)
    rng = np.random.default_rng(seed)
    boot = np.empty(mini_bootstrap_k)
    for b in range(mini_bootstrap_k):
        idx = rng.integers(0, n, n)
        bqi, _ = _beta_qi_single(d[idx], r[idx], taus)
        boot[b] = bqi if not np.isnan(bqi) else 0.0
    se = float(np.nanstd(boot, ddof=1))

    return beta_qi, se, int(ok.sum())


def _process_chunk(
    chunk_idx: int, dosage_chunk: np.ndarray, col_names: list[str], resid: np.ndarray, taus: list[float],
    se_method: str, bootstrap_k: int, asymptotic_bootstrap_k: int, min_maf: float, min_call_rate: float, seed: int,
) -> tuple[int, list[dict]]:
    """Always returns one row per variant in the chunk (never a "gap"): a
    variant discarded by the MAF/call-rate filters or for which QuantReg
    does not converge is still status='done' (statistics set to None), not
    missing -- otherwise it would stay 'pending' in the DB forever and a
    subsequent run would retry it indefinitely. status='failed' only for
    actual exceptions."""
    rng = np.random.default_rng(seed)
    rows = []
    for j, col in enumerate(col_names):
        try:
            dosage = dosage_chunk[:, j]

            call_rate = 1 - np.isnan(dosage).mean()
            maf = np.nanmean(dosage) / 2 if not np.isnan(np.nanmean(dosage)) else np.nan
            maf = min(maf, 1 - maf) if not np.isnan(maf) else np.nan
            if call_rate < min_call_rate or np.isnan(maf) or maf < min_maf:
                rows.append({"variant_safe": col, "status": "done", "n": None, "maf": None,
                             "beta_qi": None, "se": None, "z": None, "p": None, "error_message": None})
                continue

            ok = ~np.isnan(dosage) & ~np.isnan(resid)
            d, r = dosage[ok], resid[ok]

            if se_method == "bootstrap":
                beta_qi, n_used = _beta_qi_single(dosage, resid, taus)
                se = np.nan
                if not np.isnan(beta_qi):
                    n = len(d)
                    boot = np.empty(bootstrap_k)
                    for b in range(bootstrap_k):
                        idx = rng.integers(0, n, n)
                        bqi, _ = _beta_qi_single(d[idx], r[idx], taus)
                        boot[b] = bqi if not np.isnan(bqi) else 0.0
                    se = float(np.nanstd(boot, ddof=1))
            else:
                # "asymptotic": internal mini-bootstrap, see
                # _beta_qi_and_asymptotic_se. seed derived from (seed, j) to
                # stay deterministic per variant regardless of the chunk's
                # processing order.
                beta_qi, se, n_used = _beta_qi_and_asymptotic_se(
                    dosage, resid, taus, asymptotic_bootstrap_k, seed=seed * 100_003 + j,
                )

            if np.isnan(beta_qi) or se is None or np.isnan(se) or se == 0:
                rows.append({"variant_safe": col, "status": "done", "n": int(n_used) if n_used else None,
                             "maf": round(float(maf), 4), "beta_qi": None, "se": None, "z": None, "p": None,
                             "error_message": None})
                continue

            z = beta_qi / se
            p = 2 * (1 - stats.norm.cdf(abs(z)))
            rows.append({"variant_safe": col, "status": "done", "n": int(n_used), "maf": round(float(maf), 4),
                         "beta_qi": beta_qi, "se": se, "z": z, "p": p, "error_message": None})
        except Exception as e:
            rows.append({"variant_safe": col, "status": "failed", "n": None, "maf": None, "beta_qi": None,
                         "se": None, "z": None, "p": None, "error_message": str(e)[:500]})
    return chunk_idx, rows


def _scan_fingerprint(vcfg: VqtlConfig, cols: list[str]) -> dict:
    """Signature of this scan's configuration: if it changes (variants,
    statistical parameters...) the rows already saved to the DB for this
    generation are no longer valid and are discarded (vqtl_scan_results is
    cleared, computation restarts from scratch) instead of being reused by
    mistake."""
    return {
        "n_variants": len(cols),
        "first_variant": cols[0] if cols else None,
        "last_variant": cols[-1] if cols else None,
        "chunk_size": vcfg.chunk_size,
        "taus": vcfg.taus,
        "se_method": vcfg.se_method,
        "bootstrap_k": vcfg.bootstrap_k,
        "asymptotic_bootstrap_k": getattr(vcfg, "asymptotic_bootstrap_k", None),
        "min_maf": vcfg.min_maf,
        "min_call_rate": vcfg.min_call_rate,
    }


def run_vqtl_scan(
    dataset: VqtlDataset, vcfg: VqtlConfig, target_col: str, generation: int, force: bool = False,
    variant_subset: list[str] | None = None,
) -> pd.DataFrame:
    """Genome-wide vQTL scan, with state persisted to the DB
    (vqtl_scan_results, see vqtl/db/repository.py and db/schema.sql)
    instead of to files: a status='pending' placeholder is inserted for
    every variant before starting, and each completed chunk is updated
    (bulk UPDATE) IMMEDIATELY, not at the end of the scan. If the process
    is interrupted (Ctrl+C, OOM, a cluster job being killed, etc.),
    variants already marked 'done' are not repeated on restart -- execution
    resumes from where it left off, not from scratch. `force=True` (from
    --force in cli.py) clears everything and restarts from scratch even if
    the fingerprint matches.

    `variant_subset` ("safe" columns, see VqtlDataset.variant_cols):
    restricts the scan to an explicit list of variants instead of the full
    dataset.variant_cols -- this is the parameter cli.py passes from
    `--significant-only` (via select_variants_from_significant_results).
    None (default) = full genome-wide scan."""
    from vqtl.db import repository as repo

    y_col = f"{target_col}_z"
    cols = dataset.variant_cols if variant_subset is None else variant_subset
    if variant_subset is not None:
        log.info("vQTL scan restricted to an explicit subset of %d variants (--significant-only).", len(cols))

    asymptotic_bootstrap_k = getattr(vcfg, "asymptotic_bootstrap_k", 50)
    log.info(
        "Step 3 - vQTL scan: %d variants, taus=%s, se_method=%s (asymptotic_bootstrap_k=%s, bootstrap_k=%s), "
        "n_jobs=%s, chunk_size=%s",
        len(cols), vcfg.taus, vcfg.se_method, asymptotic_bootstrap_k, vcfg.bootstrap_k, vcfg.n_jobs, vcfg.chunk_size,
    )

    covariates = dataset.df[dataset.covariate_cols].to_numpy(dtype=float) if dataset.covariate_cols else np.zeros((len(dataset.df), 0))
    resid, ok_mask = residualize(dataset.df[y_col].to_numpy(dtype=float), covariates)
    log.info("Residualization on covariates %s: %d/%d complete samples", dataset.covariate_cols, int(ok_mask.sum()), len(dataset.df))

    # ---- Fingerprint: if it changed (or this is the first run), clear
    # vqtl_scan_results for this generation and reinsert the placeholders ----
    fingerprint = _scan_fingerprint(vcfg, cols)
    cached_fp = None if force else repo.get_scan_fingerprint(generation)
    if cached_fp != fingerprint:
        if force:
            log.info("--force: restarting from scratch (ignoring any saved fingerprint).")
        else:
            log.info("No valid fingerprint found (or parameters changed): scanning from scratch.")
        repo.reset_scan_run(generation, fingerprint)

    variants_meta = []
    for c in cols:
        real = dataset.mapping[c]
        chrom, pos = variant_chrom_pos(real)
        variants_meta.append({"variant": real, "chromosome": chrom, "position": pos})
    repo.ensure_scan_placeholders(generation, variants_meta)

    done_variants = repo.get_done_scan_variants(generation)
    inv_mapping = {v: k for k, v in dataset.mapping.items()}
    done_safe = {inv_mapping[v] for v in done_variants if v in inv_mapping}
    todo_cols = [c for c in cols if c not in done_safe]

    if done_safe:
        log.info(
            "Resuming from DB: %d/%d variants already completed in a previous run, "
            "recomputing only the remaining %d.", len(done_safe), len(cols), len(todo_cols),
        )

    chunks = [todo_cols[i:i + vcfg.chunk_size] for i in range(0, len(todo_cols), vcfg.chunk_size)]
    if todo_cols:
        # immediately mark all queued variants as 'in_progress': allows
        # live progress to be checked from another session with
        # "SELECT status, COUNT(*) FROM vqtl_scan_results WHERE generation=N GROUP BY status"
        # while the scan is still running, without waiting for chunks to finish.
        repo.mark_scan_in_progress(generation, [dataset.mapping[c] for c in todo_cols])
    n_chunks = len(chunks)
    if n_chunks:
        log.info("%d chunks of up to %d variants each", n_chunks, vcfg.chunk_size)

    t0 = time.monotonic()
    if chunks:
        gen = Parallel(n_jobs=vcfg.n_jobs, backend="loky", return_as="generator_unordered")(
            delayed(_process_chunk)(
                i, dosage_matrix(dataset, chunk_cols), chunk_cols, resid, vcfg.taus,
                vcfg.se_method, vcfg.bootstrap_k, asymptotic_bootstrap_k, vcfg.min_maf, vcfg.min_call_rate, seed=i,
            )
            for i, chunk_cols in enumerate(chunks)
        )
        n_done_this_run = 0
        log_every = max(1, n_chunks // 10)
        for chunk_idx, rows in gen:
            n_done_this_run += 1
            for r in rows:
                r["variant"] = dataset.mapping[r.pop("variant_safe")]
            repo.save_scan_chunk_results(generation, rows)
            if n_done_this_run % log_every == 0 or n_done_this_run == n_chunks:
                log.info(
                    "vQTL scan: %d/%d variants completed (%d chunks done in this run)",
                    len(done_safe) + n_done_this_run * vcfg.chunk_size, len(cols), n_done_this_run,
                )
    elapsed = time.monotonic() - t0
    log.info("vQTL scan: processing completed in %.1fs.", elapsed)

    df_res = repo.get_scan_results(generation, only_done=True)
    log.info("vQTL scan complete: %d variants with a valid result (out of %d tested).", len(df_res), len(cols))
    return df_res
