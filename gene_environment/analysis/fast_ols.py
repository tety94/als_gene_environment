"""
OLS "fast path" per il loop di permutazione.

PROBLEMA MISURATO: `smf.ols(formula=...).fit()` (statsmodels + patsy) ri-parsa
la formula testuale e ricostruisce l'intero Model object (inclusi standard
error, T-stat, R^2 ecc. che nel loop di permutazione non servono: ci serve
SOLO il coefficiente di interazione) ad ogni singola chiamata. Nel loop di
permutazione questo viene ripetuto N_PERM (fino a N_PERM_HIGH) volte per
variante -> è la voce di costo singola più grande dell'intera pipeline
(misurato: ~10 ms/permutazione su un caso tipico, contro ~1.9 ms per
matching+scaler insieme).

Benchmark (vedi conversazione): con la formula realmente usata dalla pipeline
(`target ~ variant * (Ecols) [+ covariates]`) la design matrix ha una
struttura FISSA e nota a priori:

    [intercept, variant, E_1..E_p, variant*E_1..variant*E_p, C_1..C_q]

dove C_1..C_q sono covariate aggiuntive NON interagenti con variant (es. le
PC per la correzione di struttura di popolazione, vedi pca_utils.py) --
patsy le appende sempre DOPO il blocco di interazione perché in
`build_formula` vengono aggiunte con "+ covariate" fuori dalle parentesi
della moltiplicazione, non dentro. Verificato: coefficiente di interazione
confrontato con smf.ols (con e senza covariate aggiuntive), differenza
~1e-13, rumore in virgola mobile. Costruire la design matrix a mano con
numpy invece di richiamare patsy/statsmodels ad ogni permutazione dà
~22x di speedup sulla singola chiamata (misurato).

ASSUNZIONE (verificata su build_dataset.py): Ecols è sempre numerico
(pd.to_numeric + eventuale standardizzazione), MAI categoriale. Le PC sono
sempre numeriche per costruzione (output di plink2 --pca). Se in futuro
Ecols o le covariate aggiuntive dovessero includere colonne non numeriche
(es. `cfg.covariates` con una colonna categoriale tipo "sex" salvata come
stringa), questo modulo NON è più equivalente a smf.ols (patsy farebbe
one-hot encoding automatico) -> per questo `assert_numeric_covariates`
valida i dtype e solleva un errore esplicito invece di dare risultati
silenziosamente sbagliati.
"""
from __future__ import annotations

import numpy as np


def interaction_column_index(n_ecols: int) -> int:
    """Indice della PRIMA colonna di interazione variant:E_i nella design
    matrix [intercept, variant, E_1..E_p, variant*E_1..variant*E_p, C_1..C_q].

    Nella pipeline Ecols ha sempre un solo elemento (vedi build_dataset.py:
    Ecols = [exposure (+ "_std")]), quindi c'è un solo termine di
    interazione e questo indice coincide esattamente con quello che
    `_find_interaction_term` troverebbe come PRIMO match in mod.params.index
    (stesso ordine di patsy: intercept, main effects, interactions, poi le
    eventuali covariate aggiuntive C_1..C_q -- che infatti NON spostano
    questo indice, dato che vengono dopo il blocco di interazione, non
    prima)."""
    return 2 + n_ecols  # 0=intercept, 1=variant, [2 .. 2+p-1]=E_i, poi interazioni


def build_design_and_solve(
    variant_values: np.ndarray,
    E: np.ndarray,
    y: np.ndarray,
    C: np.ndarray | None = None,
) -> np.ndarray | None:
    """Costruisce la design matrix [1, variant, E, variant*E, C] e risolve i
    coefficienti OLS via lstsq (equivalente numerico a statsmodels, che usa
    anch'esso una pseudo-inversa sotto il cofano).

    variant_values: array (n,) — valori RAW della variante (dosaggio, non
        binarizzati: la binarizzazione serve solo per il matching, non per
        la regressione, esattamente come nell'originale).
    E: array (n, p) — covariate/esposizioni CHE INTERAGISCONO con variant,
        SEMPRE numeriche.
    y: array (n,) — target (onset_age).
    C: array (n, q) o None — covariate aggiuntive NON interagenti (es. le
        PC di struttura di popolazione), SEMPRE numeriche. None o array con
        q=0 colonne equivale a non avere covariate aggiuntive: la design
        matrix torna identica alla versione precedente [1, variant, E,
        variant*E], comportamento invariato quando le PCA sono disattivate
        (cfg.use_pca_covariates=False) o comunque covariate_cols è vuota.

    Ritorna None se il sistema è troppo piccolo/degenere per essere risolto
    (n < numero di colonne), altrimenti l'array dei coefficienti nello
    stesso ordine di [intercept, variant, E_1..E_p, variant:E_1..variant:E_p,
    C_1..C_q].
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
    """Guardia esplicita: il fast path assume covariate (esposizioni E E
    eventuali covariate aggiuntive C, es. le PC) numeriche. Se qualcuno in
    futuro passa una colonna categoriale, meglio un errore chiaro subito che
    un coefficiente silenziosamente sbagliato (patsy farebbe dummy-encoding,
    qui no).

    Il chiamante (modeling.py) passa qui sia Ecols sia covariate_cols
    insieme (Ecols + covariate_cols), cosi' la validazione copre entrambi i
    gruppi di colonne che finiscono nella design matrix del fast path."""
    import pandas as pd

    non_numeric = [c for c in E_df.columns if not pd.api.types.is_numeric_dtype(E_df[c])]
    if non_numeric:
        raise TypeError(
            f"fast_ols richiede covariate numeriche, trovate non numeriche: {non_numeric}. "
            "Il fast path non fa dummy-encoding come patsy: servirebbe il path smf.ols originale."
        )

def design_column_names(variant_col: str, Ecols: list[str], covariate_cols: list[str]) -> list[str]:
    """Nomi nello stesso ordine di build_design_and_solve:
    [intercept, variant, E_1..E_p, variant:E_1..variant:E_p, C_1..C_q]"""
    names = ["Intercept", variant_col]
    names += list(Ecols)
    names += [f"{variant_col}:{e}" for e in Ecols]
    names += list(covariate_cols)
    return names