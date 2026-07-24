"""
Orchestratore multi-scenario (battery di robustezza).

Per ciascuno scenario definito in SCENARIOS:
  1) genera i dati sintetici       -> gen_fake_data.generate_dataset(**params)
  2) gira la parte gene-ambiente   -> run_ge_interaction() qui sotto
     (chiamata diretta e sequenziale a modeling.process_single_variant,
     nessuna reimplementazione)
  3) gira la parte vQTL in due modalita':
       - "debug"      -> run_vqtl_debug() qui sotto: confronto asymptotic
                          vs bootstrap su (varianti causali + 20 nulle a
                          caso), utile per un check rapido di divergenza
                          fra i due metodi di SE
       - "asymptotic" -> run_vqtl_asymptotic() qui sotto: pipeline vQTL
                          completa Step 3->7 (run_pipeline_for_method di
                          test_vqtl_pipeline.py, importata cosi' com'e', non
                          duplicata), solo se_method="asymptotic" (il
                          bootstrap e' gia' confrontato nel debug, farlo
                          girare anche qui raddoppierebbe il tempo per poco
                          valore aggiunto)

Ogni scenario scrive sotto scenarios/<nome>/ (fake_data/, vqtl_results/,
debug/, ge_pipeline_results.csv, scenario_summary.json). Alla fine viene
scritto un report riassuntivo unico in scenarios/all_scenarios_summary.csv
e .json, cosi' si vede a colpo d'occhio se qualche scenario e' andato
storto (has_failures, causali recuperate, lambda_GC, eventuali eccezioni),
piu' un report Word aggregato (scenarios/recap_all/all_scenarios_report.docx).

*** VARIANTI CAUSALI USATE QUI: un set PICCOLO E FISSO (SCENARIO_CAUSAL_
VARIANTS / SCENARIO_PURE_VARIANCE_VARIANTS sotto), NON i default "grandi"
di gen_fake_data.py (DEFAULT_CAUSAL_VARIANTS / DEFAULT_PURE_VARIANCE_VARIANTS,
46 G×E + 16 pure_variance).

PERCHE': gen_fake_data.py somma il contributo di OGNI variante causale
attiva alla STESSA onset_age, per lo stesso paziente:

    for lab, (beta_inter, beta_main) in causal_variants.items():
        onset_age += beta_main*dosage + beta_inter*dosage*exposure_std
    for lab, sd_by_dosage in pure_variance_variants.items():
        onset_age += rng.normal(0, sd_i, size=n)

Se se ne attivano 46+16 insieme (misurato su un run reale): il rumore
"nascosto" che ogni singolo test si trova davanti sale a ~4.5x il noise_sd
dichiarato (soprattutto per colpa delle pure_variance, da sole ~15x la
varianza base), perche' nessun test di una singola variante si aggiusta
per le altre varianti causali attive nello stesso dataset. Per testare la
ROBUSTEZZA della pipeline a condizioni diverse (stratificazione,
missingness, campione piccolo, esposizione zero-inflazionata) serve un
segnale pulito e comparabile fra scenari -- quindi qui si usa SEMPRE questo
set piccolo, indipendentemente dai default di gen_fake_data.py. La curva di
potenza per magnitudine/segno va invece misurata con
run_isolated_casual_test.py (1 sola variante attiva per dataset), non qui.
***

QUESTO FILE E' SIA UNA LIBRERIA CHE UNO SCRIPT:
  - run_isolated_casual_test.py lo importa (`import run_scenarios as rs`)
    per riusare _set_common_env / run_ge_interaction / run_vqtl_debug (test
    isolato per-variante) e run_all_scenarios (battery di scenari, come
    "fase 2" della batteria completa) -- vedi il docstring di quel file per
    il punto d'ingresso unico consigliato.
  - Puo' comunque essere lanciato da solo, se si vuole SOLO la battery di
    scenari senza il resto:
        python run_scenarios.py                          # sequenziale, tutti gli scenari
        python run_scenarios.py baseline small_sample     # sequenziale, solo alcuni
        python run_scenarios.py --workers 4               # parallelo, 4 scenari insieme
        python run_scenarios.py --workers 4 baseline small_sample

Metti questo file nella cartella ROOT del repo, insieme a gen_fake_data.py,
test_vqtl_pipeline.py, fake_vqtl_repository.py e report_utils.py.

get_config()/get_vqtl_config() del repo leggono gli env ogni volta (non
sono cachate con lru_cache) -- per questo qui basta aggiornare os.environ e
richiamarle ad ogni scenario, senza bisogno di cache_clear()/reload.

PARALLELIZZAZIONE (--workers > 1):
  os.environ, modeling.global_df e le tabelle in-memory di
  fake_vqtl_repository sono stato GLOBALE DI PROCESSO -- se due scenari
  girassero come thread nello stesso processo si sovrascriverebbero a
  vicenda le variabili d'ambiente e i dati caricati. Per questo qui si usa
  ProcessPoolExecutor (processi separati, ognuno col proprio spazio di
  stato): ogni worker importa da zero i moduli e imposta i propri env,
  senza interferenze fra scenari concorrenti. L'overhead di avvio
  processo/import e' trascurabile rispetto al tempo di scan/permutazioni.

  VQTL_N_JOBS (i job interni di joblib per lo scan vQTL, non gli scenari)
  viene ridotto automaticamente in base a --workers per non sovrasaturare
  la CPU: con N core disponibili e W worker di scenario, ogni scenario usa
  al piu' N//W job interni (minimo 1). Con --workers 1 il comportamento e'
  identico a prima (fino a 4 job interni, come negli script originali).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = SCRIPT_DIR  # cambia qui se questo file non sta nella root del repo
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from report_utils import generate_recap, generate_multi_scenario_recap, load_scenario_recap

SCENARIOS_ROOT = os.path.join(SCRIPT_DIR, "scenarios")

# ============================================================
# Varianti causali per gli scenari di ROBUSTEZZA: set piccolo e fisso,
# gli stessi 5 G×E + 2 pure_variance "storici" (quelli con cui la pipeline
# e' stata validata originariamente). Vedi spiegazione in cima al file sul
# perche' NON si usano i default grandi di gen_fake_data.py qui.
# ============================================================
SCENARIO_CAUSAL_VARIANTS: dict[str, tuple[float, float]] = {
    "1_1000001_A_G": (-4.5, -1.0),
    "2_2000002_C_T": (4.0, 0.5),
    "3_3000003_G_A": (-3.5, 0.0),
    "4_4000004_T_C": (3.2, -0.5),
    "5_5000005_A_T": (-5.0, 1.0),
}
SCENARIO_PURE_VARIANCE_VARIANTS: dict[str, dict[int, float]] = {
    "7_7000001_A_G": {0: 3.0, 1: 12.0},
    "7_7000002_C_T": {0: 12.0, 1: 3.0},
}

# ============================================================
# Definizione scenari: ogni dict e' passato a
# gen_fake_data.generate_dataset(out_dir=..., causal_variants=SCENARIO_CAUSAL_VARIANTS,
# pure_variance_variants=SCENARIO_PURE_VARIANCE_VARIANTS, **params).
# Tutti i parametri non specificati restano ai default del generatore.
# Aggiungine/modificane pure altri qui (NON mettere "causal_variants" o
# "pure_variance_variants" dentro questi dict: sono gia' fissati sopra e
# passati esplicitamente in run_scenario(), per evitare l'errore
# "got multiple values for keyword argument").
# ============================================================
SCENARIOS: dict[str, dict] = {
    "baseline": {},
    "population_stratification": {
        "subpop_frac": 0.30,
        "subpop_onset_shift": 3.0,
        "subpop_maf_shift": 0.15,
    },
    "nonrandom_missing_carriers": {
        "nonrandom_missing_carrier_rate": 0.15,
    },
    "small_sample": {
        "n_patients": 300,
    },
    "high_zero_inflation_exposure": {
        "prop_unexposed": 0.60,
    },
    "high_sample": {
        "n_patients": 10000,
    },
}

GENERATION = 1


def section(title: str) -> None:
    print("\n" + "#" * 88)
    print(title)
    print("#" * 88)


def _set_common_env(fake_dir: str, work_dir: str, vqtl_n_jobs: int | None = None) -> None:
    """Chiavi/valori di default per gene_environment.config.get_config() e
    vqtl.config.get_vqtl_config(), puntate alle cartelle dello scenario
    corrente. vqtl_n_jobs sovrascrive il default (4) -- usato per non
    sovrasaturare la CPU quando piu' scenari girano in parallelo (vedi
    run_scenario)."""
    n_jobs = vqtl_n_jobs if vqtl_n_jobs is not None else min(4, os.cpu_count() or 2)
    os.environ.update({
        "DB_USER": "test_user",
        "DB_PASSWORD": "test_pass",
        "DB_NAME": "test_db",
        "USE_PCA_COVARIATES": "true",
        "PCA_N_COMPONENTS": "10",
        "PCA_COVARIATES_PATH_TEMPLATE": os.path.join(fake_dir, "pca_covariates_gen{generation}.csv"),
        "GENERATION": str(GENERATION),
        "TARGET_COL": "onset_age",
        "EXPOSURE": "exposure_env",
        "COVARIATES": "sex",
        "RAW_FILE": os.path.join(fake_dir, "genetic.csv"),
        "ENV_FILE": os.path.join(fake_dir, "env.csv"),
        "SEP": ",",
        "TEMP_DF_PATH": os.path.join(work_dir, "temp_df.pkl"),
        "LOG_DIR": os.path.join(work_dir, "logs"),
        "N_PERM": "100",
        "N_PERM_HIGH": "1000",
        "ADAPTIVE_PERM_CHECK_EVERY": "50",
        "ADAPTIVE_PERM_FUTILITY_P": "0.5",
        "PVALUE_THRESHOLD": "0.05",
        "MIN_OBS_COEF": "2",
        "MATCH_K": "3",
        "MIN_TREATED": "5",
        "MIN_SAMPLE_SIZE": "10",
        "MAX_SMD": "0.25",
        "RANDOM_STATE": "42",
        "VQTL_N_PERM": "500",
        "VQTL_CHUNK_SIZE": "20",
        "VQTL_N_JOBS": str(n_jobs),
        "VQTL_FILTER_TOP_N": "15",
    })


def _force_cfg_overrides(cfg, overrides: dict, context: str):
    """Forza sui campi della dataclass di config (get_config()/get_vqtl_config())
    i valori passati in overrides, A PRESCINDERE da cosa quelle funzioni
    hanno effettivamente letto dagli env.

    Perche' serve: il sintomo osservato e' che, pur avendo messo
    RAW_FILE=.../fake_data/genetic.csv in os.environ PRIMA di chiamare
    get_config(), la pipeline ha comunque caricato un file completamente
    diverso (gen.parquet reale). Questo succede se get_config() dentro
    config.py ricarica un .env reale (es. via python-dotenv con
    override=True) OGNI volta che viene chiamata: in quel caso qualsiasi
    valore messo a mano in os.environ PRIMA della chiamata viene
    silenziosamente rimpiazzato DENTRO la chiamata stessa, e non c'e' modo
    di proteggersene lato env. L'unico modo robusto e' sovrascrivere i
    campi della dataclass risultante DOPO che get_config() e' gia' stata
    chiamata.

    Usa dataclasses.fields() per applicare solo ai nomi di campo che
    esistono davvero: se un nome indovinato qui non esiste sulla tua
    dataclass, lo stampa come "ignorato" invece di far crashare tutto --
    in quel caso il nome vero del campo in config.py e' diverso, correggi
    la mappa qui sotto (in _cfg_field_names) con quello giusto."""
    import dataclasses as _dc
    if not _dc.is_dataclass(cfg):
        print(f"[{context}] ATTENZIONE: cfg non è una dataclass, salto _force_cfg_overrides (verifica a mano i path usati).")
        return cfg
    valid_fields = {f.name for f in _dc.fields(cfg)}
    applied = {k: v for k, v in overrides.items() if k in valid_fields}
    skipped = {k: v for k, v in overrides.items() if k not in valid_fields}
    if skipped:
        print(f"[{context}] ATTENZIONE: campi non trovati sulla dataclass di config (nome errato? "
              f"correggi _cfg_field_names in run_scenarios.py), NON forzati: {skipped} "
              f"| campi disponibili: {sorted(valid_fields)}")
    if applied:
        cfg = _dc.replace(cfg, **applied)
        print(f"[{context}] Path forzati sulla config (ignorando qualunque .env caricato internamente): {applied}")
    return cfg


# ============================================================
# Step 2: parte gene-ambiente. Chiama direttamente modeling.process_single_
# variant su ogni variante del dataset, in sequenza -- nessuna
# reimplementazione della statistica, solo orchestrazione. Unica copia di
# questa logica nel repo: riusata sia da run_scenario() qui sotto sia da
# run_isolated_casual_test.py per il test isolato per-variante.
# ============================================================

def run_ge_interaction(fake_dir: str, work_dir: str) -> dict:
    from gene_environment.config import get_config
    from gene_environment.logging_utils import configure_logging, get_logger
    from gene_environment.vcf_pipeline.build_dataset import load_and_prepare_data
    from gene_environment.analysis import modeling

    cfg = get_config()
    cfg = _force_cfg_overrides(cfg, {
        "raw_file": os.path.join(fake_dir, "genetic.csv"),
        "env_file": os.path.join(fake_dir, "env.csv"),
        "log_dir": os.path.join(work_dir, "logs"),
        "target_col": "onset_age",
        "generation": GENERATION,
        "exposure": "exposure_env",
        "covariates": "sex",
        # Deve puntare a un path che NON esiste per i dati sintetici: se
        # lasciato al valore letto dal .env reale del progetto, build_dataset.
        # _build_narrow_covariates() userebbe la mappa id->generazione VERA
        # (pazienti reali), che non contiene nessuno degli id sintetici qui
        # -- ogni riga risulterebbe "generazione sconosciuta" e verrebbe
        # scartata (sintomo: "N -> 0 righe" nei log, poi crash a valle sullo
        # StandardScaler per array vuoto). Puntandolo a un path inesistente,
        # _build_narrow_covariates() prende il ramo "nessuna mappa trovata,
        # uso tutte le righe", corretto per un dataset sintetico mono-
        # generazione.
        "sample_generation_map": os.path.join(fake_dir, "__no_sample_generation_map__.csv"),
    }, context="gene-ambiente")
    configure_logging(cfg.log_dir)
    log = get_logger(__name__)

    print("Carico e preparo il dataset (stesso codice usato in produzione)...")
    df, variant_cols_safe, mapping, Ecols, variant_cols, covariate_cols = load_and_prepare_data(cfg)
    print(f"Righe: {len(df)} | Esposizione (Ecols): {Ecols} | Covariate: {covariate_cols}")
    print(f"Varianti da testare: {len(variant_cols_safe)}")

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

    elapsed = time.time() - t0
    print(f"Completato in {elapsed:.0f}s. Risultati: {len(results)}")

    res_df = pd.DataFrame(results)
    out_path = os.path.join(fake_dir, "pipeline_results.csv")
    res_df.to_csv(out_path, index=False)
    print(f"Salvato in {out_path}")

    n_sig = int((res_df["p_emp"] < 0.05).sum()) if "p_emp" in res_df.columns and not res_df.empty else 0

    recap_summary = generate_recap(
        ground_truth_path=os.path.join(fake_dir, "ground_truth.csv"),
        pipeline_results_path=os.path.join(fake_dir, "pipeline_results.csv"),
        out_dir=os.path.join(work_dir, "recap"),
    )
    recap_public = {k: v for k, v in recap_summary.items() if k != "_detail"}  # niente DataFrame nel JSON

    return {
        "n_variants_tested": len(variant_cols_safe),
        "n_results": len(results),
        "n_significant_p_emp_lt_05": n_sig,
        "elapsed_s": round(elapsed, 1),
        "results_csv": out_path,
        "recap": recap_public,
    }


# ============================================================
# Step 3a: parte vQTL "debug" -- confronto asymptotic vs bootstrap su un
# sottoinsieme piccolo (causali + 20 nulle a caso). Unica copia di questa
# logica nel repo: riusata sia da run_scenario() qui sotto sia da
# run_isolated_casual_test.py per il confronto per-variante.
# ============================================================

def run_vqtl_debug(fake_dir: str, work_dir: str) -> dict:
    import fake_vqtl_repository as fake_repo
    sys.modules["vqtl.db.repository"] = fake_repo

    from gene_environment.config import get_config
    from gene_environment.logging_utils import configure_logging
    from vqtl.config import VqtlConfig
    from vqtl.core.data import load_vqtl_dataset
    from vqtl.core.phenotype import prepare_phenotype
    from vqtl.core.scan import run_vqtl_scan, reset_convergence_stats, get_convergence_stats

    ge_cfg = get_config()
    ge_cfg = _force_cfg_overrides(ge_cfg, {
        "raw_file": os.path.join(fake_dir, "genetic.csv"),
        "env_file": os.path.join(fake_dir, "env.csv"),
        "log_dir": os.path.join(work_dir, "logs"),
        "target_col": "onset_age",
        "generation": GENERATION,
        "exposure": "exposure_env",
        "covariates": "sex",
        # Deve puntare a un path che NON esiste per i dati sintetici: se
        # lasciato al valore letto dal .env reale del progetto, build_dataset.
        # _build_narrow_covariates() userebbe la mappa id->generazione VERA
        # (pazienti reali), che non contiene nessuno degli id sintetici qui
        # -- ogni riga risulterebbe "generazione sconosciuta" e verrebbe
        # scartata (sintomo: "N -> 0 righe" nei log, poi crash a valle sullo
        # StandardScaler per array vuoto). Puntandolo a un path inesistente,
        # _build_narrow_covariates() prende il ramo "nessuna mappa trovata,
        # uso tutte le righe", corretto per un dataset sintetico mono-
        # generazione.
        "sample_generation_map": os.path.join(fake_dir, "__no_sample_generation_map__.csv"),
    }, context="vqtl-debug")
    configure_logging(ge_cfg.log_dir)

    truth = pd.read_csv(os.path.join(fake_dir, "ground_truth.csv"))
    causal = truth.loc[truth["effect_type"].isin(["gxe_meanshift", "pure_variance"]), "variant"].tolist()

    rng = np.random.default_rng(0)
    nulls_all = truth.loc[truth["effect_type"] == "no_effect", "variant"].tolist()
    n_sample = min(20, len(nulls_all))
    nulls_sample = rng.choice(nulls_all, size=n_sample, replace=False).tolist()

    wanted_labels = set(causal) | set(nulls_sample)
    print(f"Sottoinsieme: {len(causal)} causali + {len(nulls_sample)} nulle = {len(wanted_labels)} varianti")

    ds = load_vqtl_dataset(ge_cfg, VqtlConfig(), generation=GENERATION)
    ds.df = prepare_phenotype(ds.df, ge_cfg.target_col)

    inv_mapping = {v: k for k, v in ds.mapping.items()}
    variant_subset = [inv_mapping[lab] for lab in wanted_labels if lab in inv_mapping]
    missing = wanted_labels - set(inv_mapping)
    if missing:
        print(f"ATTENZIONE: {len(missing)} label non trovate nel dataset: {missing}")

    results = {}
    convergence = {}
    for method in ["asymptotic", "bootstrap"]:
        fake_repo.reset_all()
        reset_convergence_stats()
        vcfg = VqtlConfig(se_method=method, n_jobs=1)
        print(f"\n--- se_method={method} (n_jobs=1 per contatori affidabili) ---")
        t0 = time.time()
        df = run_vqtl_scan(ds, vcfg, ge_cfg.target_col, generation=GENERATION, variant_subset=variant_subset)
        print(f"Fatto in {time.time() - t0:.1f}s")
        df = df.merge(truth[["variant", "effect_type"]], left_on="SNP", right_on="variant", how="left")
        results[method] = df[["SNP", "effect_type", "N", "MAF", "beta_QI", "SE", "Z", "P"]].sort_values("SNP")

        stats = get_convergence_stats()
        attempted = stats["tau_fits_attempted"]
        discarded = stats["tau_fits_discarded"]
        pct = 100 * discarded / attempted if attempted else 0.0
        convergence[method] = {
            "tau_fits_attempted": attempted, "tau_fits_discarded": discarded,
            "pct_discarded": round(pct, 1), "variants_all_nan": stats["variants_all_nan"],
        }
        print(f"Convergenza fit tau: {discarded}/{attempted} scartati ({pct:.1f}%) | "
              f"varianti con beta_QI=NaN: {stats['variants_all_nan']}")

    merged = results["asymptotic"].merge(
        results["bootstrap"], on=["SNP", "effect_type"], suffixes=("_asym", "_boot")
    ).sort_values("effect_type")

    debug_dir = os.path.join(work_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    out_csv = os.path.join(debug_dir, "debug_se_method_comparison.csv")
    merged.to_csv(out_csv, index=False)
    print(f"[export] {out_csv}")

    lambda_gc_local = {}
    for method in ["asymptotic", "bootstrap"]:
        z = results[method]["Z"].dropna().to_numpy()
        if len(z):
            lambda_gc_local[method] = round(float(np.median(z ** 2) / 0.4549364), 3)

    return {
        "n_variants_subset": len(variant_subset),
        "lambda_gc_local": lambda_gc_local,
        "convergence": convergence,
        "comparison_csv": out_csv,
    }


# ============================================================
# Step 3b: parte vQTL "asymptotic" -- pipeline completa Step 3->7,
# riusando run_pipeline_for_method di test_vqtl_pipeline.py cosi' com'e'
# (nessuna duplicazione della logica statistica/di orchestrazione).
# ============================================================

def run_vqtl_asymptotic(fake_dir: str, work_dir: str) -> dict:
    import fake_vqtl_repository as fake_repo
    sys.modules["vqtl.db.repository"] = fake_repo

    from gene_environment.config import get_config
    from gene_environment.logging_utils import configure_logging
    from vqtl.config import get_vqtl_config
    import test_vqtl_pipeline as tvp

    ge_cfg = get_config()
    ge_cfg = _force_cfg_overrides(ge_cfg, {
        "raw_file": os.path.join(fake_dir, "genetic.csv"),
        "env_file": os.path.join(fake_dir, "env.csv"),
        "log_dir": os.path.join(work_dir, "logs"),
        "target_col": "onset_age",
        "generation": GENERATION,
        "exposure": "exposure_env",
        "covariates": "sex",
        # Deve puntare a un path che NON esiste per i dati sintetici: se
        # lasciato al valore letto dal .env reale del progetto, build_dataset.
        # _build_narrow_covariates() userebbe la mappa id->generazione VERA
        # (pazienti reali), che non contiene nessuno degli id sintetici qui
        # -- ogni riga risulterebbe "generazione sconosciuta" e verrebbe
        # scartata (sintomo: "N -> 0 righe" nei log, poi crash a valle sullo
        # StandardScaler per array vuoto). Puntandolo a un path inesistente,
        # _build_narrow_covariates() prende il ramo "nessuna mappa trovata,
        # uso tutte le righe", corretto per un dataset sintetico mono-
        # generazione.
        "sample_generation_map": os.path.join(fake_dir, "__no_sample_generation_map__.csv"),
    }, context="vqtl-asymptotic")
    configure_logging(ge_cfg.log_dir)
    vcfg_base = get_vqtl_config()

    truth = pd.read_csv(os.path.join(fake_dir, "ground_truth.csv"))
    causal_gxe = set(truth.loc[truth["effect_type"] == "gxe_meanshift", "variant"])
    causal_pure_var = set(truth.loc[truth["effect_type"] == "pure_variance", "variant"])
    all_causal = causal_gxe | causal_pure_var
    n_null_truth = int((truth["effect_type"] == "no_effect").sum())
    print(f"Ground truth: {len(causal_gxe)} G×E causali, {len(causal_pure_var)} vQTL pure, {n_null_truth} nulle")

    # work_dir e GENERATION passati esplicitamente (non piu' monkey-patchati
    # sul modulo tvp): sicuro da richiamare per piu' scenari di fila nello
    # stesso processo, vedi docstring di test_vqtl_pipeline.py.
    summary = tvp.run_pipeline_for_method(
        "asymptotic", ge_cfg, vcfg_base, truth, all_causal, n_null_truth,
        work_dir=work_dir, generation=GENERATION,
    )
    return summary


# ============================================================
# Orchestrazione per singolo scenario + main
# ============================================================

def run_scenario(name: str, gen_params: dict, n_workers: int = 1) -> dict:
    section(f"SCENARIO: {name}")
    scenario_dir = os.path.join(SCENARIOS_ROOT, name)
    fake_dir = os.path.join(scenario_dir, "fake_data")
    os.makedirs(fake_dir, exist_ok=True)

    result: dict = {"scenario": name, "params": gen_params, "status": "ok", "error": None}

    try:
        from gen_fake_data import generate_dataset
        # Set FISSO e piccolo (vedi spiegazione in cima al file): NON i
        # default grandi di gen_fake_data.py, per non contaminare il
        # rumore effettivo che ogni test vede rispetto al noise_sd
        # dichiarato -- qui vogliamo isolare l'effetto del PARAMETRO di
        # scenario (stratificazione, missingness, ecc.), non mischiarlo
        # con l'effetto di avere decine di varianti causali attive insieme.
        gen_summary = generate_dataset(
            out_dir=fake_dir, verbose=True,
            causal_variants=SCENARIO_CAUSAL_VARIANTS,
            pure_variance_variants=SCENARIO_PURE_VARIANCE_VARIANTS,
            **gen_params,
        )
        result["gen_summary"] = gen_summary

        vqtl_n_jobs = max(1, (os.cpu_count() or 2) // max(1, n_workers))
        _set_common_env(fake_dir, scenario_dir, vqtl_n_jobs=vqtl_n_jobs)

        section(f"[{name}] Parte gene-ambiente")
        result["ge_interaction"] = run_ge_interaction(fake_dir, scenario_dir)

        section(f"[{name}] Parte vQTL — debug (asymptotic vs bootstrap, sottoinsieme)")
        result["vqtl_debug"] = run_vqtl_debug(fake_dir, scenario_dir)

        section(f"[{name}] Parte vQTL — pipeline completa Step 3→7 (se_method=asymptotic)")
        result["vqtl_asymptotic"] = run_vqtl_asymptotic(fake_dir, scenario_dir)

    except Exception as exc:  # uno scenario che fallisce non deve bloccare gli altri
        result["status"] = "FAILED"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        print(f"\n*** SCENARIO '{name}' FALLITO: {result['error']} ***")
        traceback.print_exc()

    with open(os.path.join(scenario_dir, "scenario_summary.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


def _run_scenario_worker(name: str, n_workers: int) -> dict:
    """Wrapper top-level (necessario: ProcessPoolExecutor deve poter fare
    pickle della funzione sottomessa). Ogni chiamata gira in un processo
    Python nuovo -> nessuna condivisione di os.environ / stato dei moduli
    con gli altri scenari in corso."""
    return run_scenario(name, SCENARIOS[name], n_workers=n_workers)


def run_all_scenarios(names: list[str] | None = None, n_workers: int = 1) -> dict:
    """Gira la battery di scenari (tutti, o solo `names` se specificato),
    scrive i report aggregati sotto SCENARIOS_ROOT e ritorna un dict con:
      - "all_results":  lista dei result dict di ogni scenario (uno per
                         scenario, stesso schema di run_scenario())
      - "summary_df":   pd.DataFrame di riepilogo (stesso contenuto di
                         all_scenarios_summary.csv)
      - "failed":       nomi degli scenari falliti per eccezione
      - "vqtl_failed":  nomi degli scenari con has_failures=True sui
                         controlli automatici della pipeline vQTL
      - "has_failures": True se failed o vqtl_failed non sono vuoti
    Usata sia da main() (CLI standalone) sia da run_isolated_casual_test.py
    (come "fase scenari" della batteria di test completa)."""
    names = list(names) if names else list(SCENARIOS.keys())
    unknown = set(names) - set(SCENARIOS)
    if unknown:
        raise SystemExit(f"Scenari sconosciuti: {sorted(unknown)}. Disponibili: {list(SCENARIOS)}")

    n_workers = max(1, n_workers)
    os.makedirs(SCENARIOS_ROOT, exist_ok=True)
    all_results = []
    t0 = time.time()

    if n_workers == 1:
        for name in names:
            all_results.append(run_scenario(name, SCENARIOS[name], n_workers=1))
    else:
        print(f"Eseguo {len(names)} scenari con {n_workers} processi paralleli "
              f"(ognuno con al piu' {max(1, (os.cpu_count() or 2) // n_workers)} job interni vQTL)...")
        results_by_name = {}
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_run_scenario_worker, name, n_workers): name for name in names}
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    results_by_name[name] = fut.result()
                except Exception as exc:  # errore non catturato dentro run_scenario stesso (raro)
                    results_by_name[name] = {
                        "scenario": name, "status": "FAILED",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    print(f"\n*** SCENARIO '{name}' FALLITO nel processo worker: {exc} ***")
                print(f"[{name}] completato ({results_by_name[name]['status']}).")
        # riordino secondo l'ordine richiesto, per output deterministico
        all_results = [results_by_name[name] for name in names]

    section("RIEPILOGO FINALE — tutti gli scenari")
    rows = []
    for r in all_results:
        vqtl_a = r.get("vqtl_asymptotic", {}) or {}
        vqtl_d = r.get("vqtl_debug", {}) or {}
        ge = r.get("ge_interaction", {}) or {}
        ge_recap = ge.get("recap", {}) or {}
        ge_power = (ge_recap.get("gxe_interaction", {}) or {}).get("power_overall")
        rows.append({
            "scenario": r["scenario"],
            "status": r["status"],
            "error": r.get("error"),
            "ge_n_significant": ge.get("n_significant_p_emp_lt_05"),
            "ge_power_overall": ge_power,
            "vqtl_lambda_gc": vqtl_a.get("lambda_gc"),
            "vqtl_causal_found": vqtl_a.get("n_found_causal"),
            "vqtl_causal_total": vqtl_a.get("n_causal_total"),
            "vqtl_false_positives": vqtl_a.get("n_false_positives"),
            "vqtl_has_failures": vqtl_a.get("has_failures"),
            "debug_lambda_gc_asym": (vqtl_d.get("lambda_gc_local") or {}).get("asymptotic"),
            "debug_lambda_gc_boot": (vqtl_d.get("lambda_gc_local") or {}).get("bootstrap"),
        })
    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))

    summary_csv = os.path.join(SCENARIOS_ROOT, "all_scenarios_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    with open(os.path.join(SCENARIOS_ROOT, "all_scenarios_summary.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[export] {summary_csv}")
    print(f"[export] {os.path.join(SCENARIOS_ROOT, 'all_scenarios_summary.json')}")
    print(f"\nCompletato in {time.time() - t0:.0f}s.")

    section("REPORT FINALE UNICO (tutti gli scenari)")
    scenario_summaries_for_agg = {}
    for r in all_results:
        if r["status"] != "ok":
            continue
        recap_dir = os.path.join(SCENARIOS_ROOT, r["scenario"], "recap")
        if not os.path.isdir(recap_dir):
            print(f"[{r['scenario']}] nessuna cartella recap/, salto dal report aggregato")
            continue
        detail, recap_summary = load_scenario_recap(recap_dir)
        scenario_summaries_for_agg[r["scenario"]] = {**recap_summary, "_detail": detail}

    if scenario_summaries_for_agg:
        generate_multi_scenario_recap(
            scenario_summaries_for_agg,
            out_dir=os.path.join(SCENARIOS_ROOT, "recap_all"),
        )
        print(f"[export] {os.path.join(SCENARIOS_ROOT, 'recap_all', 'all_scenarios_report.docx')}")
    else:
        print("Nessuno scenario ok con recap disponibile: report aggregato saltato.")

    failed = [r["scenario"] for r in all_results if r["status"] == "FAILED"]
    vqtl_failed = [r["scenario"] for r in all_results
                   if r["status"] == "ok" and (r.get("vqtl_asymptotic") or {}).get("has_failures")]
    if failed:
        print(f"\n*** SCENARI FALLITI (eccezione): {failed} ***")
    if vqtl_failed:
        print(f"*** SCENARI CON CONTROLLI vQTL FALLITI (has_failures): {vqtl_failed} ***")
    if not failed and not vqtl_failed:
        print("\n*** Tutti gli scenari completati senza errori bloccanti. ***")

    return {
        "all_results": all_results,
        "summary_df": summary_df,
        "failed": failed,
        "vqtl_failed": vqtl_failed,
        "has_failures": bool(failed or vqtl_failed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Orchestratore multi-scenario")
    parser.add_argument("scenarios", nargs="*", help="Nomi scenari da lanciare (default: tutti)")
    parser.add_argument("--workers", type=int, default=1,
                         help="Numero di scenari da eseguire in parallelo (processi separati). Default 1 (sequenziale).")
    parser.add_argument("--output-dir", default=None,
                         help="Cartella dove scrivere scenarios/ (default: la cartella di questo script).")
    args = parser.parse_args()

    if args.output_dir:
        global SCENARIOS_ROOT
        SCENARIOS_ROOT = os.path.join(os.path.abspath(args.output_dir), "scenarios")
        print(f"[config] Output: {SCENARIOS_ROOT}")

    result = run_all_scenarios(names=args.scenarios or None, n_workers=args.workers)
    if result["has_failures"]:
        sys.exit(1)


if __name__ == "__main__":
    main()