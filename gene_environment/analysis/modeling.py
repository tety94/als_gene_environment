"""
Core del test per singola variante: matching + regressione + permutation test
+ statistiche di differenza onset_age, tutto salvato insieme.

NOVITA' (correzione per struttura di popolazione): il modello OLS include
ora, DI DEFAULT, le prime N componenti principali (PCA) come covariate di
correzione -- NON di interazione -- caricate per la generazione corrente
(cfg.generation) da gene_environment/utils/pca_utils.py. Le colonne PCA
sono popolate una volta per worker da `orchestrator.init_worker` in
`global_covariate_cols` (stesso pattern di `global_df`), cosi' non serve
ripassarle ad ogni submit al ProcessPoolExecutor.

Per disattivarle: cfg.use_pca_covariates = False (env USE_PCA_COVARIATES=
false) -> global_covariate_cols resta [] e il modello torna esattamente
quello di prima (nessuna covariata aggiuntiva, ne' nel path smf.ols ne' nel
fast path delle permutazioni).

BUG CRITICO trovato e corretto (invariato dalla versione precedente):
  `rng = np.random.RandomState(RANDOM_STATE + (abs(hash(variant_col)) % 2_000_000))`
  `hash()` su una stringa in Python NON è stabile fra esecuzioni diverse del
  processo (randomizzazione dell'hash attivata di default dal 2012, PEP 456)
  a meno di impostare esplicitamente PYTHONHASHSEED=0. Questo significa che
  gli stessi identici dati, con lo stesso RANDOM_STATE, potevano produrre
  permutazioni (e quindi p-value empirici) leggermente diversi ad ogni
  rilancio della pipeline: risultati non riproducibili, un problema serio
  per un'analisi statistica. Corretto usando hashlib.md5 (hash stabile,
  deterministico, indipendente da PYTHONHASHSEED).

ALTRE MODIFICHE (invariate dalla versione precedente):
  - tqdm dentro ai worker di ProcessPoolExecutor produceva output confuso
    (decine di processi che scrivono barre di progresso sullo stesso
    terminale). Sostituito con log periodici (ogni N permutazioni) tramite
    il logger centralizzato.
  - "Adaptive early stopping" per le permutazioni LIGHT: se dopo
    `adaptive_perm_check_every` permutazioni il numero di permutazioni con
    |beta_perm| >= |beta_oss| è già chiaramente troppo alto (futility check),
    ci si ferma prima di sprecare le permutazioni rimanenti su una variante
    che non risulterà comunque significativa. Ottimizzazione importante dato
    che il matching+OLS per permutazione è l'operazione più costosa della
    pipeline e viene ripetuta N_PERM (fino a N_PERM_HIGH) volte per
    variante.
  - Le statistiche di differenza onset_age (mutati vs non mutati, sullo
    stesso identico dataset usato per il modello) vengono calcolate qui,
    SUBITO, e restituite insieme al resto -> salvate a DB nella stessa riga,
    niente più script separato da rilanciare a posteriori.

NOTA (non affrontata qui, vedi conversazione): cfg.covariates (es. "sex")
esiste in config.py ma NON viene ancora usato in build_formula qui sotto
(passato come lista vuota nel path "osservato" prima di questa modifica).
Non l'ho unito automaticamente alle PCA in questo intervento perche' se
"sex" fosse salvato come stringa non numerica romperebbe
`assert_numeric_covariates` nel fast path per OGNI variante (il fast path
non fa dummy-encoding come patsy) -- va verificato il dtype reale prima di
aggiungerlo.
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
from gene_environment.logging_utils import get_logger

log = get_logger(__name__)

# Popolati dall'initializer del worker (vedi orchestrator.py) — evita di
# passare/pickle-are il dataframe intero (e la lista di covariate) ad ogni
# submit.
global_df = None
global_covariate_cols: list[str] = []


def _stable_seed(base_seed: int, variant_col: str) -> int:
    """Seed deterministico e riproducibile fra esecuzioni diverse, a
    differenza di hash() su stringa (vedi docstring del modulo)."""
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
    design_column_names,          # nuovo import
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
            log.debug("%s: %d/%d permutazioni completate", log_prefix, i + 1, n_perm)

    if full_beta:
        valid = ~np.isnan(betas[:, 0])
        return betas[valid]
    return betas[~np.isnan(betas)]

def process_single_variant(variant_col: str, variant_original: str, Ecols: list[str],  full_beta: bool = False) -> dict | None:
    cfg = get_config()
    df = global_df
    covariate_cols = global_covariate_cols  # es. le PC, popolate da init_worker; [] se disattivate

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

    # covariate_cols (le PC) entra qui nella selezione colonne: se un
    # campione non ha PCA (merge fallito per quel IID, vedi pca_utils.py)
    # viene scartato dal dropna() esattamente come per qualunque altra
    # covariata mancante -- stesso trattamento, nessuna gestione speciale.
    cols = [cfg.target_col, variant_col, "_match_variant"] + Ecols + covariate_cols
    df_model = df[cols].dropna()
    if df_model.shape[0] < cfg.min_sample_size:
        return _empty()

    # ---- Statistiche onset_age POOLED (mutati vs non mutati, esposizione
    # ignorata): calcolate qui, sullo stesso dataset usato per il modello,
    # così sono coerenti col resto del risultato e vengono salvate a DB
    # nella stessa riga/stessa transazione.
    # NB: la versione STRATIFICATA per esposizione (mutati/non mutati x
    # esposti/non esposti) NON viene calcolata qui — costerebbe lavoro su
    # milioni di varianti che nella stragrande maggioranza non sono
    # significative. Si calcola a valle, solo sulle varianti significative,
    # con lo script dedicato (vedi significant_variants/). ----
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

    matched_obs = match_control_units(df_model, "_match_variant", k=cfg.match_k, covariates_for_matching=Ecols)
    if matched_obs is None or matched_obs.shape[0] < cfg.min_sample_size:
        return _empty(onset=onset_dict)

    smd_results = check_balance(matched_obs, "_match_variant", Ecols)
    max_smd = max(smd_results.values()) if smd_results else 1

    if max_smd > cfg.max_smd:
        return _empty(max_smd=max_smd, onset=onset_dict)

    # covariate_cols (le PC) entra qui nella formula come termine additivo
    # "+ PC1 + PC2 + ..." FUORI dalla moltiplicazione con variant -> corregge
    # il modello senza introdurre interazioni variant:PCk (vedi
    # build_formula e fast_ols.py per la struttura della design matrix).
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

        X_scaled = precompute_scaled_covariates(df_model, Ecols)
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

    # Scaler sulle covariate di MATCHING (Ecols soltanto, non le PC) fittato
    # UNA VOLTA per variante (non ad ogni permutazione, vedi
    # fast_ols.py/matching.py). Calcolato solo qui, dopo il filtro
    # min_obs_coef, per non sprecare lavoro sulle varianti che non arrivano
    # comunque alla fase di permutazione.
    assert_numeric_covariates(df_model[Ecols + covariate_cols])
    X_scaled = precompute_scaled_covariates(df_model, Ecols)

    # ======================================================
    # PERMUTAZIONI LIGHT, con futility check adattivo:
    # ogni `adaptive_perm_check_every` permutazioni controlliamo se il
    # p-value parziale è già ben oltre la soglia (futility), nel qual caso
    # ci fermiamo prima di finire tutte le N_PERM: la variante non sarà
    # comunque promossa alle permutazioni HIGH.
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
                    "[%s] futility stop dopo %d/%d permutazioni (p parziale=%.3f >= %.3f)",
                    variant_col, done, cfg.n_perm, partial_p, cfg.adaptive_perm_futility_p,
                )
                stopped_early = True
                break

    perm_betas_light = np.array(perm_betas_light)
    p_emp_light = float(np.mean(np.abs(perm_betas_light) >= abs(obs_coef))) if perm_betas_light.size > 0 else None
    iterations_light = len(perm_betas_light) if stopped_early else cfg.n_perm

    # ======================================================
    # PERMUTAZIONI HIGH — solo se la LIGHT è significativa e non ci si è
    # fermati per futility.
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
