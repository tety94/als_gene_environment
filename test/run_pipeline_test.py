"""
Esegue il motore statistico della pipeline (matching + OLS + permutation
test sul genotipo, formula reale onset_age ~ variant * (exposure_std) +
PC1..PC5 + sex) sui dati sintetici di gen_fake_data.py, MA con una
correzione al matching per gestire correttamente la massa di pazienti a
esposizione=0 (vedi spiegazione sotto e nella conversazione).

PERCHÉ QUESTA VERSIONE E' DIVERSA DA UNA CHIAMATA DIRETTA A
gene_environment.analysis.modeling.process_single_variant:

Il matching nearest-neighbor di gene_environment/analysis/matching.py
(`match_control_units` / `match_control_units_indices`) tiene INTERO il
gruppo più piccolo ("base") e riduce l'altro gruppo ("other") ai soli vicini
UNICI trovati (np.unique). Quando la covariata di matching (l'esposizione
standardizzata) ha una grande massa di valori identici — qui ~40% dei
pazienti a esposizione=0 — tutti i punti "base" a zero selezionano gli
stessi pochi vicini "other" a zero: dopo la deduplicazione, quella fascia
collassa a ~k rappresentanti sul lato "other", indipendentemente da quanti
pazienti a zero ci siano davvero sul lato "base" (che restano TUTTI). Lo
sbilanciamento risultante (SMD) supera sistematicamente la soglia
(cfg.max_smd=0.25) e le varianti — comprese quelle con un vero effetto —
vengono scartate PRIMA ancora di essere testate. Aumentare il numero di
pazienti NON risolve (anzi peggiora leggermente): il rapporto fra massa
zero sul lato "base" (che cresce con N) e ~k rappresentanti fissi sul lato
"other" si allarga. Serve k enorme (50-90% del campione) per compensare, il
che vanifica lo scopo del matching (a quel punto si sta quasi prendendo
tutto il gruppo, non i vicini più simili).

CORREZIONE APPLICATA QUI (solo in questo script di test, NON nel pacchetto
gene_environment — quella è una scelta di produzione che spetta a te):
tratto l'esposizione=0 come uno STRATO A PAREGGIO ESATTO (0==0 per
definizione, non serve nearest-neighbor: tutti i pazienti a zero, di
entrambi i gruppi, entrano nel campione appaiato) e faccio il
nearest-neighbor matching SOLO sulla parte continua (esposizione>0),
richiamando le funzioni REALI della pipeline (`match_control_units`,
`match_control_units_indices`) ristrette a quel sottoinsieme. Verificato:
con questa correzione, k=3 (il default di config.py) è già sufficiente a
mantenere SMD sotto soglia sulla maggior parte delle varianti, incluse le
causali.

SEMPLIFICAZIONE rispetto a process_single_variant: qui le permutazioni sono
a STADIO SINGOLO (n_perm fisso, niente ottimizzazione adattiva
LIGHT/HIGH con futility stop). Replicare l'adattività con re-matching
stratificato ad ogni permutazione avrebbe appesantito molto lo script senza
aggiungere validità al test — l'obiettivo qui è verificare che l'effetto
iniettato venga recuperato, non misurare tempi di calcolo.

COME LANCIARLO (gira da TE, non da questa chat):
  1. Metti questo file nella cartella ROOT del repo (stesso livello della
     cartella "gene_environment/"), o cambia REPO_ROOT qui sotto.
  2. Esegui prima: python gen_fake_data.py
  3. pip install pandas numpy scipy scikit-learn statsmodels pyarrow mysql-connector-python
  4. python run_pipeline_test.py
"""
from __future__ import annotations

import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = SCRIPT_DIR  # cambia qui se lo script non sta nella root del repo
sys.path.insert(0, REPO_ROOT)

FAKE_DATA_DIR = os.path.join(SCRIPT_DIR, "fake_data")
WORK_DIR = SCRIPT_DIR

os.environ.update({
    "DB_USER": "test_user",
    "DB_PASSWORD": "test_pass",
    "DB_NAME": "test_db",
    "USE_PCA_COVARIATES": "true",
    "PCA_N_COMPONENTS": "10",
    "PCA_COVARIATES_PATH_TEMPLATE": os.path.join(FAKE_DATA_DIR, "pca_covariates_gen{generation}.csv"),
    "GENERATION": "1",
    "TARGET_COL": "onset_age",
    "EXPOSURE": "exposure_env",
    "COVARIATES": "sex",
    "RAW_FILE": os.path.join(FAKE_DATA_DIR, "genetic.csv"),
    "ENV_FILE": os.path.join(FAKE_DATA_DIR, "env.csv"),
    "SEP": ",",
    "TEMP_DF_PATH": os.path.join(WORK_DIR, "temp_df.pkl"),
    "LOG_DIR": os.path.join(WORK_DIR, "logs"),
    "PVALUE_THRESHOLD": "0.05",
    "MIN_OBS_COEF": "2",
    "MATCH_K": "3",          # default di config.py — funziona bene UNA VOLTA stratificato lo zero
    "MIN_TREATED": "5",
    "MIN_SAMPLE_SIZE": "10",
    "MAX_SMD": "0.25",
    "RANDOM_STATE": "42",
})

N_PERM = 2000  # stadio singolo, vedi docstring

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from gene_environment.config import get_config
from gene_environment.logging_utils import configure_logging, get_logger
from gene_environment.vcf_pipeline.build_dataset import load_and_prepare_data
from gene_environment.analysis.matching import (
    match_control_units, match_control_units_indices, check_balance, precompute_scaled_covariates,
)
from gene_environment.analysis.modeling import build_formula, _find_interaction_term
from gene_environment.analysis.fast_ols import build_design_and_solve, interaction_column_index, assert_numeric_covariates

cfg = get_config()
configure_logging(cfg.log_dir)
log = get_logger(__name__)

print("Carico e preparo il dataset (stesso codice usato in produzione)...")
df, variant_cols_safe, mapping, Ecols, variant_cols, covariate_cols = load_and_prepare_data(cfg)
print(f"Righe: {len(df)} | Esposizione: {Ecols} | Covariate: {covariate_cols}")

RAW_EXPOSURE_COL = cfg.exposure  # colonna grezza (non standardizzata): serve per individuare la massa esatta a zero


def match_observed_stratified(df_model: pd.DataFrame, treat_col: str, k: int) -> pd.DataFrame | None:
    """Equivalente stratificato di match_control_units per il campione osservato
    (una volta sola per variante, non per permutazione)."""
    zero_mask = df_model[RAW_EXPOSURE_COL] == 0
    d0 = df_model[zero_mask]              # pareggio esatto: tutti dentro, nessun NN
    d_pos = df_model[~zero_mask]
    if d_pos.empty:
        return d0 if not d0.empty else None
    matched_pos = match_control_units(d_pos, treat_col, k=k, covariates_for_matching=Ecols)
    if matched_pos is None:
        return d0 if not d0.empty else None
    return pd.concat([d0, matched_pos], ignore_index=True)


def match_indices_stratified(labels: np.ndarray, X_scaled: np.ndarray, zero_mask: np.ndarray, k: int) -> np.ndarray | None:
    """Equivalente stratificato di match_control_units_indices per il loop di
    permutazione (posizioni intere, non indici pandas). zero_mask è FISSA
    (dipende solo dall'esposizione grezza, non dall'etichetta permutata)."""
    zero_idx = np.where(zero_mask)[0]
    pos_idx = np.where(~zero_mask)[0]
    if pos_idx.shape[0] == 0:
        return zero_idx if zero_idx.shape[0] > 0 else None
    sub_labels = labels[pos_idx]
    sub_X = X_scaled[pos_idx]
    matched = match_control_units_indices(sub_labels, sub_X, k=k)
    if matched is None:
        return zero_idx if zero_idx.shape[0] > 0 else None
    base_pos, other_pos = matched
    pos_selected = np.concatenate([pos_idx[base_pos], pos_idx[other_pos]])
    return np.concatenate([zero_idx, pos_selected])


def process_variant_stratified(variant_col: str, variant_original: str, rng: np.random.RandomState):
    d = df[df[variant_col] != "."].copy()
    d[variant_col] = d[variant_col].astype(int)
    d["_match_variant"] = (d[variant_col] > 0).astype(int)

    n_treated = int((d["_match_variant"] == 1).sum())
    n_control = int((d["_match_variant"] == 0).sum())
    if n_treated < cfg.min_treated or n_control < cfg.min_treated:
        return None

    cols = [cfg.target_col, variant_col, "_match_variant", RAW_EXPOSURE_COL] + Ecols + covariate_cols
    d_model = d[cols].dropna().reset_index(drop=True)
    if d_model.shape[0] < cfg.min_sample_size:
        return None

    # ---- matching osservato, stratificato ----
    matched_obs = match_observed_stratified(d_model, "_match_variant", cfg.match_k)
    if matched_obs is None or matched_obs.shape[0] < cfg.min_sample_size:
        return None

    smd_results = check_balance(matched_obs, "_match_variant", Ecols)
    max_smd = max(smd_results.values()) if smd_results else 1
    if max_smd > cfg.max_smd:
        return None

    formula = build_formula(cfg.target_col, variant_col, Ecols, covariate_cols, matched_obs)
    mod = smf.ols(formula=formula, data=matched_obs).fit()
    interaction_name = _find_interaction_term(mod.params.index, variant_col)
    if interaction_name is None:
        return None
    obs_coef = float(mod.params[interaction_name])
    n_treated_matched = int(matched_obs["_match_variant"].sum())
    n_control_matched = int((matched_obs["_match_variant"] == 0).sum())

    if abs(obs_coef) < cfg.min_obs_coef:
        return {
            "variant": variant_original, "n_treated": n_treated_matched, "n_control": n_control_matched,
            "obs_coef": obs_coef, "p_emp": 1.0, "iterations": 0, "max_smd": max_smd,
        }

    # ---- permutazioni del genotipo, con re-matching stratificato ad ogni permutazione ----
    assert_numeric_covariates(d_model[Ecols + covariate_cols])
    X_scaled = precompute_scaled_covariates(d_model, Ecols)
    variant_values = d_model[variant_col].values
    y_values = d_model[cfg.target_col].values
    E_values = d_model[Ecols].values
    C_values = d_model[covariate_cols].values if covariate_cols else None
    zero_mask = (d_model[RAW_EXPOSURE_COL].values == 0)
    inter_idx = interaction_column_index(E_values.shape[1])

    perm_betas = np.full(N_PERM, np.nan)
    for i in range(N_PERM):
        perm_variant = rng.permutation(variant_values)
        perm_labels = (perm_variant > 0).astype(int)
        idx = match_indices_stratified(perm_labels, X_scaled, zero_mask, cfg.match_k)
        if idx is None or idx.shape[0] < cfg.min_sample_size:
            continue
        C_idx = C_values[idx] if C_values is not None else None
        beta = build_design_and_solve(perm_variant[idx], E_values[idx], y_values[idx], C_idx)
        if beta is not None:
            perm_betas[i] = beta[inter_idx]

    perm_betas = perm_betas[~np.isnan(perm_betas)]
    p_emp = float(np.mean(np.abs(perm_betas) >= abs(obs_coef))) if perm_betas.size > 0 else None

    return {
        "variant": variant_original, "n_treated": n_treated_matched, "n_control": n_control_matched,
        "obs_coef": obs_coef, "p_emp": p_emp, "iterations": int(perm_betas.size), "max_smd": max_smd,
    }


results = []
t0 = time.time()
for i, (v_safe, v_orig) in enumerate(zip(variant_cols_safe, variant_cols)):
    rng = np.random.RandomState(cfg.random_state + i)
    res = process_variant_stratified(v_safe, v_orig, rng)
    if res is not None:
        results.append(res)
    if (i + 1) % 10 == 0:
        print(f"  {i + 1}/{len(variant_cols_safe)} varianti processate ({time.time() - t0:.0f}s)")

print(f"Completato in {time.time() - t0:.0f}s. Risultati: {len(results)}")

res_df = pd.DataFrame(results)
out_path = os.path.join(FAKE_DATA_DIR, "pipeline_results.csv")
res_df.to_csv(out_path, index=False)
print(f"Salvato in {out_path}")
print(res_df[["variant", "n_treated", "n_control", "obs_coef", "p_emp", "iterations", "max_smd"]].to_string())