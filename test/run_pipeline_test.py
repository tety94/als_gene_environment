"""
Esegue il motore statistico REALE della pipeline (nessuna reimplementazione:
chiama direttamente gene_environment.analysis.modeling.process_single_variant)
sui dati sintetici di gen_fake_data.py, con permutazione del genotipo e
matching adattivo LIGHT/HIGH -- esattamente come farebbe l'orchestratore in
produzione, tranne la persistenza su MySQL (bypassata) e il
ProcessPoolExecutor (qui i "worker" girano in sequenza nello stesso
processo, sugli stessi oggetti globali che orchestrator.init_worker()
popolerebbe).

*** RICHIEDE che il TUO gene_environment/analysis/matching.py e modeling.py
siano già stati aggiornati con la correzione discussa in conversazione
(gestione dei pareggi "tie-aware" nel k-NN + matching su Ecols +
covariate_cols invece di Ecols soltanto) -- vedi matching_PATCHED.py e
modeling_PATCHED.py. Senza quella correzione, questo script funziona
comunque (chiama le tue funzioni, qualunque esse siano), ma con l'esposizione
zero-inflazionata di gen_fake_data.py (~40% dei pazienti a esposizione=0) il
matching non deduplicato collassa e le varianti causali vengono scartate dal
filtro SMD PRIMA di essere testate -- è il problema diagnosticato nella
conversazione, non un problema di questo script. ***

Prima di questa versione, questo file conteneva una stratificazione manuale
(isolare l'esposizione=0 come pareggio esatto, NN-matching solo sul resto)
per aggirare il bug: non serve più, ora che la correzione è nella pipeline
stessa -- lo script torna a essere un chiamante diretto e fedele di
modeling.process_single_variant, senza logica di matching propria.

COME LANCIARLO (gira da TE, non da questa chat):
  1. Metti questo file nella cartella ROOT del repo (stesso livello della
     cartella "gene_environment/"), o cambia REPO_ROOT qui sotto.
  2. Esegui prima: python gen_fake_data.py
  3. pip install pandas numpy scipy scikit-learn statsmodels pyarrow mysql-connector-python
  4. python run_pipeline_test.py
"""
from __future__ import annotations
from test.report_utils  import generate_recap

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
    "N_PERM": "150",
    "N_PERM_HIGH": "2000",
    "ADAPTIVE_PERM_CHECK_EVERY": "50",
    "ADAPTIVE_PERM_FUTILITY_P": "0.5",
    "PVALUE_THRESHOLD": "0.05",
    "MIN_OBS_COEF": "2",
    "MATCH_K": "3",          # default di config.py -- ora sufficiente col matching corretto
    "MIN_TREATED": "5",
    "MIN_SAMPLE_SIZE": "10",
    "MAX_SMD": "0.25",
    "RANDOM_STATE": "42",
})

import pandas as pd

from gene_environment.config import get_config
from gene_environment.logging_utils import configure_logging, get_logger
from gene_environment.vcf_pipeline.build_dataset import load_and_prepare_data
from gene_environment.analysis import modeling

cfg = get_config()
configure_logging(cfg.log_dir)
log = get_logger(__name__)

print("Carico e preparo il dataset (stesso codice usato in produzione)...")
df, variant_cols_safe, mapping, Ecols, variant_cols, covariate_cols = load_and_prepare_data(cfg)
print(f"Righe: {len(df)} | Esposizione (Ecols): {Ecols} | Covariate: {covariate_cols}")
print(f"Varianti da testare: {len(variant_cols_safe)}")

# Simula orchestrator.init_worker() senza pickle/ProcessPoolExecutor: stessi
# oggetti globali, chiamata diretta e sequenziale a process_single_variant.
modeling.global_df = df
modeling.global_covariate_cols = covariate_cols

results = []
t0 = time.time()
for i, (v_safe, v_orig) in enumerate(zip(variant_cols_safe, variant_cols)):
    res = modeling.process_single_variant(v_safe, v_orig, Ecols, full_beta=False)
    if res is not None:
        res["variant"] = v_orig
        results.append(res)
    if (i + 1) % 10 == 0:
        print(f"  {i + 1}/{len(variant_cols_safe)} varianti processate ({time.time() - t0:.0f}s)")

print(f"Completato in {time.time() - t0:.0f}s. Risultati: {len(results)}")

res_df = pd.DataFrame(results)
out_path = os.path.join(FAKE_DATA_DIR, "pipeline_results.csv")
res_df.to_csv(out_path, index=False)
print(f"Salvato in {out_path}")
print(res_df[["variant", "n_treated", "n_control", "obs_coef", "p_emp", "iterations", "max_smd"]].to_string())


recap_summary = generate_recap(
    ground_truth_path=os.path.join(FAKE_DATA_DIR, "ground_truth.csv"),
    pipeline_results_path=out_path,          # e' gia' fake_data/pipeline_results.csv
    out_dir=os.path.join(FAKE_DATA_DIR, "recap"),
)
print(f"Potenza G×E complessiva: {recap_summary['gxe_interaction']['power_overall']*100:.1f}%")
print(f"Falsi positivi (nulle): {recap_summary['null_genomewide']['false_positive_rate']*100:.2f}%")