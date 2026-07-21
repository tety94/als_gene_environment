"""
Preparazione del fenotipo per lo scan vQTL (ex Step 2 del progetto originale).

Logica statistica invariata rispetto all'originale (z-score, log, rank-based
inverse-normal transform): e' genuinamente specifica del metodo vQTL/QUAIL e
non ha equivalente in gene_environment (che non trasforma onset_age, lo usa
cosi' com'e' nel matching+OLS). L'unica differenza e' che opera direttamente
sul DataFrame gia' unito da `core.data.load_vqtl_dataset` invece che su un
phenotype_clean.tsv scritto da uno step precedente.
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
    """Aggiunge <target_col>_z, _log, _rint. Ritorna una COPIA di df."""
    df = df.copy()
    y = pd.to_numeric(df[target_col], errors="coerce").values.astype(float)

    df[f"{target_col}_z"] = (y - np.nanmean(y)) / np.nanstd(y)

    shift = min(0.0, np.nanmin(y)) - 1.0
    with np.errstate(invalid="ignore"):
        df[f"{target_col}_log"] = np.log(y - shift)

    df[f"{target_col}_rint"] = rank_inverse_normal(y)

    log.info(
        "Fenotipo '%s' preparato: n=%d, media=%.3f, sd=%.3f (colonne aggiunte: _z, _log, _rint)",
        target_col, len(df), np.nanmean(y), np.nanstd(y),
    )
    return df
