"""
Matching (nearest-neighbor) fra pazienti mutati e non mutati sulle covariate.

Fix rispetto all'originale (matching.py):
  - Aggiunti type hints e logging (nessun comportamento cambia rispetto
    all'originale: la logica del matching era corretta).
  - `_prepare_matching_matrix`: se una colonna richiesta non esiste nel
    dataframe veniva silenziosamente saltata (`continue`); ora viene
    loggato un warning esplicito, così un typo nei nomi delle covariate
    (facile con EXPOSURE + "_std" costruito a runtime) non passa
    inosservato.
  - `check_balance`: se `matched_df` è None l'originale ritornava {} in
    silenzio; mantenuto per compatibilità ma ora con un log di debug.
"""
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
        raise ValueError("Nessuna colonna fornita per il matching")

    features = []
    for c in cols:
        if c not in df.columns:
            log.warning("Colonna di matching '%s' non trovata nel dataframe: saltata", c)
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            features.append(df[c].fillna(df[c].mean()))
        else:
            dummies = pd.get_dummies(df[c].astype(str), prefix=c, drop_first=True)
            features.append(dummies)

    if not features:
        raise ValueError("Nessuna feature di matching valida trovata nel dataframe")

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

    # Gestione dei pareggi: se più punti "other" sono alla STESSA distanza
    # del k-esimo vicino (es. una massa di valori identici nella covariata
    # di matching, come una categoria "non esposto" numericamente
    # consistente), li si include TUTTI, non solo i primi k trovati.
    # Comportamento standard nella letteratura sul matching (es. R MatchIt).
    # Senza questo, np.unique(indices.flatten()) troncava arbitrariamente i
    # pareggi a k rappresentanti, sbilanciando sistematicamente il campione
    # appaiato ogni volta che la covariata aveva molti valori ripetuti — e
    # peggiorando, non migliorando, all'aumentare del numero di pazienti.
    kth_dist = distances[:, -1]
    D_full = cdist(X_base.values, X_other.values)
    selected_other_pos = np.unique(np.where(D_full <= kth_dist[:, None] + 1e-9)[1])
    other_idx = X_other.index[selected_other_pos]

    matched_other = df_matching.loc[other_idx]
    matched_base = df_matching.loc[mask_base]

    return pd.concat([matched_base, matched_other], ignore_index=True)


def precompute_scaled_covariates(df: pd.DataFrame, covariates_for_matching: list[str]) -> np.ndarray:
    """Fitta lo StandardScaler UNA SOLA VOLTA sulle covariate di matching.

    Le covariate (Ecols) non cambiano mai tra una permutazione e l'altra —
    cambia solo l'etichetta trattato/controllo (_match_variant). L'originale
    rifaceva `_prepare_matching_matrix` (fit+transform dello scaler) dentro
    ogni singola chiamata a `match_control_units`, quindi ad ogni
    permutazione: puro lavoro ripetuto identico. Chiamare questa funzione
    una volta per variante e passare il risultato a
    `match_control_units_indices` elimina quella ridondanza.

    NB: `_prepare_matching_matrix` scala su `df_matching` (base+other, cioè
    l'intero dataframe passato), quindi statisticamente equivalente a
    scalare una volta sull'intero `df_model` come si fa qui: stesso insieme
    di righe, stessa media/std, indipendentemente dall'ordine con cui
    vengono concatenate base/other.
    """
    return _prepare_matching_matrix(df, covariates_for_matching).values


def match_control_units_indices(
    labels: np.ndarray, X_scaled: np.ndarray, k: int = 2
) -> tuple[np.ndarray, np.ndarray] | None:
    """Equivalente veloce di `match_control_units`, usato nel loop di
    permutazione: invece di ri-fittare `NearestNeighbors` (costruzione
    albero ad ogni chiamata) usa `cdist` + `argpartition` su una matrice di
    covariate GIA' scalata (vedi `precompute_scaled_covariates`).

    Gestisce i pareggi come `match_control_units` (vedi commento lì): se più
    punti "other" sono alla stessa distanza del k-esimo vicino, li include
    tutti, non solo i primi k trovati da argpartition.

    Ritorna (matched_base_idx, matched_other_idx): array di POSIZIONI
    intere in `X_scaled`/`labels` (non indici pandas), oppure None se un
    gruppo è vuoto o non ci sono vicini disponibili.
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
        log.debug("check_balance: matched_df è None, nessun balance calcolato")
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