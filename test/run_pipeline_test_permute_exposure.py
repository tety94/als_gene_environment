"""
Stessa identica analisi di run_pipeline_test.py (matching + OLS sulla formula
reale della pipeline: onset_age ~ variant * (exposure_std) + PC1..PC5 + sex),
MA con un disegno di permutazione diverso: invece di permutare l'etichetta
genotipica (variant, con re-matching ad ogni permutazione) qui si permuta la
componente AMBIENTALE (exposure_std) tenendo fissi genotipo e gruppo
appaiato (matched_obs), e si ricalcola il coefficiente di interazione ad
ogni permutazione con lo stesso fast path (build_design_and_solve) usato
dalla pipeline.

CORREZIONE AL MATCHING (stessa di run_pipeline_test.py, vedi quel file per
la spiegazione completa del problema): il matching nearest-neighbor della
pipeline collassa quando la covariata di matching (esposizione
standardizzata) ha una grande massa di valori identici (qui ~40% dei
pazienti a esposizione=0), perché il gruppo "base" resta intero ma il
gruppo "other" viene deduplicato ai soli vicini unici. Qui, tratto
l'esposizione=0 come strato a pareggio esatto (tutti dentro, nessun NN
necessario) e faccio nearest-neighbor matching SOLO sulla parte continua
(>0), usando le funzioni REALI della pipeline (match_control_units)
ristrette a quel sottoinsieme. In questo script il matching si fa UNA SOLA
VOLTA per variante (il campione appaiato resta fisso, si permuta solo
l'esposizione al suo interno), quindi la correzione è più semplice che in
run_pipeline_test.py (lì va rifatta ad ogni permutazione).

IMPORTANTE (vedi discussione nella conversazione): le due strategie di
permutazione (genotipo vs esposizione) testano nulli leggermente diversi e
NON devono produrre p-value identici in generale — vedi docstring di
run_pipeline_test.py per il dettaglio.

** AVVISO — LIMITE STATISTICO VERIFICATO **: testando a N=1000 (vs N=600),
questo approccio (permutare l'esposizione dentro un campione appaiato
FISSO) ha mostrato un tasso di falsi positivi inflazionato: 8/47 varianti
nulle (17%) risultate "significative" a p<=0.05, contro un 5% atteso
(binomial test p=0.002 — statisticamente anomalo, non rumore campionario,
riprodotto identico su più run indipendenti con la stessa configurazione).
Nello stesso identico dataset, run_pipeline_test.py (che permuta il
GENOTIPO e rifà il matching ad ogni permutazione, cioè il disegno REALE
della pipeline) resta ben calibrato: 0/47 falsi positivi. Causa probabile:
il campione appaiato non è una selezione casuale rispetto all'esposizione
(il matching dipende congiuntamente da genotipo ed esposizione), quindi
permutare l'esposizione ENTRO quel campione fissato non garantisce
l'ipotesi di scambiabilità che il test di permutazione richiede — a N più
alti il matching seleziona sottoinsiemi più strutturati, amplificando il
problema. CONCLUSIONE: questo script NON è un test statisticamente valido
per trarre conclusioni — usalo solo a scopo esplorativo/diagnostico, non
come alternativa intercambiabile a run_pipeline_test.py. Per validare la
pipeline, fai riferimento a run_pipeline_test.py.
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
    "TEMP_DF_PATH": os.path.join(WORK_DIR, "temp_df_perm_e.pkl"),
    "LOG_DIR": os.path.join(WORK_DIR, "logs"),
    "MATCH_K": "3",   # default di config.py — funziona bene UNA VOLTA stratificato lo zero
    "MIN_TREATED": "5",
    "MIN_SAMPLE_SIZE": "10",
    "MAX_SMD": "0.25",
    "RANDOM_STATE": "42",
})

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from gene_environment.config import get_config
from gene_environment.logging_utils import configure_logging, get_logger
from gene_environment.vcf_pipeline.build_dataset import load_and_prepare_data
from gene_environment.analysis.matching import match_control_units, check_balance
from gene_environment.analysis.modeling import build_formula, _find_interaction_term
from gene_environment.analysis.fast_ols import build_design_and_solve, interaction_column_index

N_PERM_EXPOSURE = 2000  # permutazioni dell'esposizione (fast path: economiche)

cfg = get_config()
configure_logging(cfg.log_dir)
log = get_logger(__name__)

print("Carico e preparo il dataset (stesso codice usato in produzione)...")
df, variant_cols_safe, mapping, Ecols, variant_cols, covariate_cols = load_and_prepare_data(cfg)
print(f"Righe: {len(df)} | Esposizione: {Ecols} | Covariate: {covariate_cols}")

RAW_EXPOSURE_COL = cfg.exposure


def match_observed_stratified(d_model: pd.DataFrame, treat_col: str, k: int) -> pd.DataFrame | None:
    zero_mask = d_model[RAW_EXPOSURE_COL] == 0
    d0 = d_model[zero_mask]
    d_pos = d_model[~zero_mask]
    if d_pos.empty:
        return d0 if not d0.empty else None
    matched_pos = match_control_units(d_pos, treat_col, k=k, covariates_for_matching=Ecols)
    if matched_pos is None:
        return d0 if not d0.empty else None
    return pd.concat([d0, matched_pos], ignore_index=True)


def process_variant_permute_exposure(variant_col, variant_original, rng):
    d = df[df[variant_col] != "."].copy()
    d[variant_col] = d[variant_col].astype(int)
    d["_match_variant"] = (d[variant_col] > 0).astype(int)

    n_treated = int((d["_match_variant"] == 1).sum())
    n_control = int((d["_match_variant"] == 0).sum())
    if n_treated < cfg.min_treated or n_control < cfg.min_treated:
        return None

    cols = [cfg.target_col, variant_col, "_match_variant", RAW_EXPOSURE_COL] + Ecols + covariate_cols
    d_model = d[cols].dropna()
    if d_model.shape[0] < cfg.min_sample_size:
        return None

    # ---- matching stratificato, calcolato UNA VOLTA (osservato) ----
    matched_obs = match_observed_stratified(d_model, "_match_variant", cfg.match_k)
    if matched_obs is None or matched_obs.shape[0] < cfg.min_sample_size:
        return None

    smd_results = check_balance(matched_obs, "_match_variant", Ecols)
    max_smd = max(smd_results.values()) if smd_results else 1
    if max_smd > cfg.max_smd:
        return None

    # ---- stima osservata: stessa formula/smf.ols della pipeline reale ----
    formula = build_formula(cfg.target_col, variant_col, Ecols, covariate_cols, matched_obs)
    mod = smf.ols(formula=formula, data=matched_obs).fit()
    interaction_name = _find_interaction_term(mod.params.index, variant_col)
    if interaction_name is None:
        return None
    obs_coef = float(mod.params[interaction_name])

    # ---- permutazioni: si mescola SOLO l'esposizione, dentro il gruppo
    # appaiato osservato (fisso), non il genotipo. ----
    variant_values = matched_obs[variant_col].values.astype(float)
    y_values = matched_obs[cfg.target_col].values
    E_values = matched_obs[Ecols].values.astype(float)
    C_values = matched_obs[covariate_cols].values.astype(float) if covariate_cols else None
    inter_idx = interaction_column_index(E_values.shape[1])

    perm_betas = np.full(N_PERM_EXPOSURE, np.nan)
    for i in range(N_PERM_EXPOSURE):
        E_perm = rng.permutation(E_values, axis=0)
        beta = build_design_and_solve(variant_values, E_perm, y_values, C_values)
        if beta is not None:
            perm_betas[i] = beta[inter_idx]
    perm_betas = perm_betas[~np.isnan(perm_betas)]
    p_emp = float(np.mean(np.abs(perm_betas) >= abs(obs_coef))) if perm_betas.size > 0 else None

    return {
        "variant": variant_original,
        "n_treated_matched": int(matched_obs["_match_variant"].sum()),
        "n_control_matched": int((matched_obs["_match_variant"] == 0).sum()),
        "obs_coef": obs_coef,
        "p_emp_permuta_esposizione": p_emp,
        "iterations": int(perm_betas.size),
        "max_smd": max_smd,
    }


results = []
t0 = time.time()
for i, (v_safe, v_orig) in enumerate(zip(variant_cols_safe, variant_cols)):
    rng = np.random.default_rng(cfg.random_state + i)
    res = process_variant_permute_exposure(v_safe, v_orig, rng)
    if res is not None:
        results.append(res)
    if (i + 1) % 10 == 0:
        print(f"  {i + 1}/{len(variant_cols_safe)} varianti processate ({time.time() - t0:.0f}s)")

print(f"Completato in {time.time() - t0:.0f}s. Risultati: {len(results)}")

res_df = pd.DataFrame(results)
out_path = os.path.join(FAKE_DATA_DIR, "pipeline_results_permute_exposure.csv")
res_df.to_csv(out_path, index=False)
print(f"Salvato in {out_path}")
print(res_df.to_string())