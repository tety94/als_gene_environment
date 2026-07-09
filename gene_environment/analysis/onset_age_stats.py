"""
Statistiche sulla differenza di età d'esordio (onset_age) fra mutati e non
mutati. Estratte da analyze_variant_onset_age.py in un modulo riusabile, in
modo da poter essere chiamate:
  1) DENTRO modeling.py, per ogni variante, così il risultato finisce a
     database SUBITO, insieme al coefficiente del modello (come richiesto:
     "vorrei che da subito a database si salvassero anche le info riguardo
     alle differenze di età di esordio"), invece che in un secondo script
     separato che ricalcola tutto da un CSV a parte.
  2) Nello script di reporting (report_onset_age.py) per generare boxplot e
     forest plot a partire dai valori già salvati.
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
    """IC bootstrap (percentile) per il delta di mediana (mutati - non_mutati),
    vettorizzato: un unico resample matriciale invece di un loop Python."""
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
    """Calcola tutte le statistiche onset_age per un singolo confronto
    mutati vs non mutati. Ritorna None se i gruppi sono troppo piccoli."""
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
