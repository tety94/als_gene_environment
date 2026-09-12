"""Statistics on the onset-age difference between mutant and non-mutant
patients. Usable both:
  1) INSIDE modeling.py, for each variant, so the result is saved to the
     database immediately, together with the model coefficient, instead of
     in a separate second script that recomputes everything from a CSV.
  2) In the reporting script (report_onset_age.py), to generate boxplots
     and forest plots from already-saved values.

The bootstrap is vectorized with numpy (a single matrix resample via
rng.choice, instead of a pure Python loop), which matters since it runs
n_boot=2000 times PER variant PER cohort across hundreds of variants.
Cases with n<2 are handled explicitly, to avoid silent errors/odd NaNs
from scipy with too-small samples.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import mannwhitneyu, ttest_ind


@dataclass
class OnsetAgeResult:
    n_mutati: int
    n_non_mutati: int
    median_mutati: float | None
    median_non_mutati: float | None
    delta_median: float | None
    ci_low: float | None
    ci_high: float | None
    statistic: float | None
    effect_size: float | None
    p_value: float | None
    low_power: bool
    method: str


def run_group_test(mutati, non_mutati, use_mann_whitney: bool = True):
    n1, n2 = len(mutati), len(non_mutati)
    if n1 < 2 or n2 < 2:
        return None, None, None

    mutati = np.asarray(mutati, dtype=float)
    non_mutati = np.asarray(non_mutati, dtype=float)

    if use_mann_whitney:
        stat, p = mannwhitneyu(mutati, non_mutati, alternative="two-sided")
        # rank-biserial correlation: 1 - 2U/(n1*n2)
        effect_size = 1 - (2 * stat) / (n1 * n2)
    else:
        stat, p = ttest_ind(mutati, non_mutati, equal_var=False, nan_policy="omit")
        pooled_sd = np.sqrt(
            ((n1 - 1) * mutati.std(ddof=1) ** 2 + (n2 - 1) * non_mutati.std(ddof=1) ** 2)
            / max(n1 + n2 - 2, 1)
        )
        effect_size = (mutati.mean() - non_mutati.mean()) / pooled_sd if pooled_sd > 0 else 0.0

    return float(stat), float(p), float(effect_size)


def bootstrap_median_diff_ci(
    mutati, non_mutati, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42
):
    """Bootstrap (percentile) CI for the median delta (mutant - non-mutant),
    vectorized: a single matrix resample instead of a Python loop."""
    mutati_arr = np.asarray(mutati, dtype=float)
    non_mutati_arr = np.asarray(non_mutati, dtype=float)
    if len(mutati_arr) == 0 or len(non_mutati_arr) == 0:
        return None, None

    rng = np.random.default_rng(seed)

    m_samples = rng.choice(mutati_arr, size=(n_boot, len(mutati_arr)), replace=True)
    nm_samples = rng.choice(non_mutati_arr, size=(n_boot, len(non_mutati_arr)), replace=True)

    diffs = np.median(m_samples, axis=1) - np.median(nm_samples, axis=1)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def compute_onset_age_result(
    mutati,
    non_mutati,
    *,
    use_mann_whitney: bool = True,
    alpha: float = 0.05,
    min_group_size: int = 5,
    low_power_threshold: int = 10,
    n_boot: int = 2000,
    seed: int = 42,
) -> OnsetAgeResult | None:
    """Compute all onset_age statistics for a single mutant vs non-mutant
    comparison. Returns None if the groups are too small."""
    n_mutati, n_non_mutati = len(mutati), len(non_mutati)
    if n_mutati < min_group_size or n_non_mutati < min_group_size:
        return None

    stat, p, effect_size = run_group_test(mutati, non_mutati, use_mann_whitney=use_mann_whitney)
    ci_low, ci_high = bootstrap_median_diff_ci(mutati, non_mutati, n_boot=n_boot, alpha=alpha, seed=seed)

    median_mutati = float(np.median(mutati))
    median_non_mutati = float(np.median(non_mutati))

    return OnsetAgeResult(
        n_mutati=n_mutati,
        n_non_mutati=n_non_mutati,
        median_mutati=median_mutati,
        median_non_mutati=median_non_mutati,
        delta_median=median_mutati - median_non_mutati,
        ci_low=ci_low,
        ci_high=ci_high,
        statistic=stat,
        effect_size=effect_size,
        p_value=p,
        low_power=(n_mutati < low_power_threshold) or (n_non_mutati < low_power_threshold),
        method="mannwhitney" if use_mann_whitney else "welch_ttest",
    )
