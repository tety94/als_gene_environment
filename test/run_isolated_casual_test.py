"""
Test isolato di potenza: UNA variante causale alla volta, contro il pool di
varianti nulle del pannello, per misurare la curva di potenza per
magnitudine/segno SENZA la contaminazione che si ha quando piu' causali
sono attive insieme nello stesso dataset (vedi spiegazione in cima a
run_scenarios.py: con 46 G×E + 16 pure_variance insieme il rumore
"nascosto" per ogni singolo test saliva a ~4.5x il noise_sd dichiarato).

Qui, per ciascuna variante nei pool di default di gen_fake_data.py
(DEFAULT_CAUSAL_VARIANTS per la parte gene-ambiente, DEFAULT_PURE_VARIANCE_
VARIANTS per la parte vQTL), si genera un dataset con QUELLA SOLA variante
marcata come causale (nessun'altra causale/pure_variance attiva) e si
verifica se la pipeline la recupera. Tutte le altre varianti del pannello
restano nulle per costruzione di gen_fake_data.py, quindi questo isola
esattamente il contributo della singola variante.

*** ASSUNZIONI SU gen_fake_data.py (verificale, non ho il file in questa
chat) ***
  - DEFAULT_CAUSAL_VARIANTS: dict[str, tuple[float, float]]  (beta_inter, beta_main)
  - DEFAULT_PURE_VARIANCE_VARIANTS: dict[str, dict[int, float]]  (sd per dosage)
  - generate_dataset(out_dir=..., causal_variants=..., pure_variance_variants=..., **kw)
    con "tutto il resto nullo di default" se non elencato nei due dict sopra.
Se uno di questi nomi/comportamenti non corrisponde, aggiusta gli import e
la chiamata a generate_dataset() qui sotto -- il resto della logica non
cambia.

COSA TESTA:
  - parte G×E: per ogni variante in DEFAULT_CAUSAL_VARIANTS, gira
    run_ge_interaction() (importata da run_scenarios.py, stessa logica di
    run_pipeline_test.py) e controlla se quella variante risulta
    significativa (p_emp < PVALUE_THRESHOLD) nel risultato.
  - parte vQTL: per ogni variante in DEFAULT_PURE_VARIANCE_VARIANTS, gira
    run_vqtl_debug() (importata da run_scenarios.py) -- che confronta
    asymptotic vs bootstrap sulle causali + un campione di nulle lette dal
    ground_truth di QUESTO dataset (che qui contiene solo 1 causale) -- e
    controlla se quella variante risulta significativa.

OUTPUT: isolated/<tipo>/<variant_label>/... (dati + risultati grezzi) e un
riepilogo unico isolated/isolated_power_curve.csv + .json con, per ogni
variante: beta/sd dichiarati, se recuperata, p-value, SE.

COME LANCIARLO (gira da TE, non da questa chat, stessa cartella di
run_scenarios.py):
    python run_isolated_causal_test.py                 # tutte le varianti default, sequenziale
    python run_isolated_causal_test.py --workers 4      # parallelo, processi separati
    python run_isolated_causal_test.py --only-ge         # solo parte G×E
    python run_isolated_causal_test.py --only-vqtl       # solo parte vQTL/pure_variance
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Riuso diretto della logica gia' scritta in run_scenarios.py -- niente
# duplicazione: stessa run_ge_interaction/run_vqtl_debug/_set_common_env/
# _force_cfg_overrides usate li'.
import run_scenarios as rs
from gen_fake_data import (
    generate_dataset,
    DEFAULT_CAUSAL_VARIANTS,
    DEFAULT_PURE_VARIANCE_VARIANTS,
)

ISOLATED_ROOT = os.path.join(SCRIPT_DIR, "isolated")
PVALUE_THRESHOLD = 0.05


def section(title: str) -> None:
    print("\n" + "#" * 88)
    print(title)
    print("#" * 88)


# ============================================================
# Una variante G×E alla volta
# ============================================================

def run_isolated_ge_variant(label: str, beta_inter: float, beta_main: float,
                             n_workers: int = 1) -> dict:
    section(f"[GxE isolata] {label}  (beta_inter={beta_inter}, beta_main={beta_main})")
    var_dir = os.path.join(ISOLATED_ROOT, "ge", label)
    fake_dir = os.path.join(var_dir, "fake_data")
    os.makedirs(fake_dir, exist_ok=True)

    result: dict = {
        "variant": label, "kind": "gxe", "beta_inter": beta_inter, "beta_main": beta_main,
        "status": "ok", "error": None,
    }
    try:
        generate_dataset(
            out_dir=fake_dir, verbose=True,
            causal_variants={label: (beta_inter, beta_main)},
            pure_variance_variants={},
        )
        vqtl_n_jobs = max(1, (os.cpu_count() or 2) // max(1, n_workers))
        rs._set_common_env(fake_dir, var_dir, vqtl_n_jobs=vqtl_n_jobs)

        ge_res = rs.run_ge_interaction(fake_dir, var_dir)
        result["ge_result"] = ge_res

        res_df = pd.read_csv(ge_res["results_csv"])
        row = res_df.loc[res_df["variant"] == label]
        if row.empty:
            result["found_in_results"] = False
            result["recovered"] = False
        else:
            p_emp = float(row["p_emp"].iloc[0]) if "p_emp" in row.columns else None
            result["found_in_results"] = True
            result["p_emp"] = p_emp
            result["recovered"] = (p_emp is not None) and (p_emp < PVALUE_THRESHOLD)

    except Exception as exc:
        result["status"] = "FAILED"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        print(f"\n*** VARIANTE '{label}' (GxE) FALLITA: {result['error']} ***")
        traceback.print_exc()

    with open(os.path.join(var_dir, "isolated_summary.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


# ============================================================
# Una variante pure_variance (vQTL) alla volta
# ============================================================

def run_isolated_vqtl_variant(label: str, sd_by_dosage: dict, n_workers: int = 1) -> dict:
    section(f"[vQTL isolata] {label}  (sd_by_dosage={sd_by_dosage})")
    var_dir = os.path.join(ISOLATED_ROOT, "vqtl", label)
    fake_dir = os.path.join(var_dir, "fake_data")
    os.makedirs(fake_dir, exist_ok=True)

    result: dict = {
        "variant": label, "kind": "pure_variance", "sd_by_dosage": sd_by_dosage,
        "status": "ok", "error": None,
    }
    try:
        generate_dataset(
            out_dir=fake_dir, verbose=True,
            causal_variants={},
            pure_variance_variants={label: sd_by_dosage},
        )
        vqtl_n_jobs = max(1, (os.cpu_count() or 2) // max(1, n_workers))
        rs._set_common_env(fake_dir, var_dir, vqtl_n_jobs=vqtl_n_jobs)

        # run_vqtl_debug legge le causali/nulle dal ground_truth di QUESTO
        # dataset: qui contiene solo `label` come pure_variance, quindi il
        # confronto e' automaticamente isolato (1 causale + 20 nulle campionate).
        debug_res = rs.run_vqtl_debug(fake_dir, var_dir)
        result["vqtl_debug"] = debug_res

        comparison = pd.read_csv(debug_res["comparison_csv"])
        row = comparison.loc[comparison["SNP"] == label]
        if row.empty:
            result["found_in_results"] = False
            result["recovered"] = False
        else:
            p_asym = float(row["P_asym"].iloc[0]) if "P_asym" in row.columns else None
            result["found_in_results"] = True
            result["p_asymptotic"] = p_asym
            result["recovered"] = (p_asym is not None) and (p_asym < PVALUE_THRESHOLD)

    except Exception as exc:
        result["status"] = "FAILED"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        print(f"\n*** VARIANTE '{label}' (pure_variance) FALLITA: {result['error']} ***")
        traceback.print_exc()

    with open(os.path.join(var_dir, "isolated_summary.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


# ============================================================
# Wrapper top-level per ProcessPoolExecutor (deve essere picklabile)
# ============================================================

def _worker_ge(label: str, beta_inter: float, beta_main: float, n_workers: int) -> dict:
    return run_isolated_ge_variant(label, beta_inter, beta_main, n_workers=n_workers)


def _worker_vqtl(label: str, sd_by_dosage: dict, n_workers: int) -> dict:
    return run_isolated_vqtl_variant(label, sd_by_dosage, n_workers=n_workers)


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Test isolato: una causale alla volta")
    parser.add_argument("--workers", type=int, default=1,
                         help="Numero di varianti da testare in parallelo (processi separati). Default 1.")
    parser.add_argument("--only-ge", action="store_true", help="Testa solo le varianti G×E")
    parser.add_argument("--only-vqtl", action="store_true", help="Testa solo le varianti pure_variance")
    args = parser.parse_args()

    do_ge = not args.only_vqtl
    do_vqtl = not args.only_ge
    n_workers = max(1, args.workers)
    os.makedirs(ISOLATED_ROOT, exist_ok=True)

    jobs = []
    if do_ge:
        for label, (beta_inter, beta_main) in DEFAULT_CAUSAL_VARIANTS.items():
            jobs.append(("ge", label, beta_inter, beta_main))
    if do_vqtl:
        for label, sd_by_dosage in DEFAULT_PURE_VARIANCE_VARIANTS.items():
            jobs.append(("vqtl", label, sd_by_dosage, None))

    t0 = time.time()
    all_results = []

    if n_workers == 1:
        for job in jobs:
            if job[0] == "ge":
                _, label, beta_inter, beta_main = job
                all_results.append(run_isolated_ge_variant(label, beta_inter, beta_main, n_workers=1))
            else:
                _, label, sd_by_dosage, _ = job
                all_results.append(run_isolated_vqtl_variant(label, sd_by_dosage, n_workers=1))
    else:
        print(f"Eseguo {len(jobs)} varianti isolate con {n_workers} processi paralleli...")
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {}
            for job in jobs:
                if job[0] == "ge":
                    _, label, beta_inter, beta_main = job
                    fut = pool.submit(_worker_ge, label, beta_inter, beta_main, n_workers)
                else:
                    _, label, sd_by_dosage, _ = job
                    fut = pool.submit(_worker_vqtl, label, sd_by_dosage, n_workers)
                futures[fut] = (job[0], job[1])
            for fut in as_completed(futures):
                kind, label = futures[fut]
                try:
                    all_results.append(fut.result())
                except Exception as exc:
                    all_results.append({
                        "variant": label, "kind": kind, "status": "FAILED",
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    print(f"\n*** VARIANTE '{label}' FALLITA nel processo worker: {exc} ***")
                print(f"[{kind}/{label}] completato ({all_results[-1]['status']}).")

    section("CURVA DI POTENZA — riepilogo")
    rows = []
    for r in all_results:
        rows.append({
            "kind": r["kind"],
            "variant": r["variant"],
            "status": r["status"],
            "error": r.get("error"),
            "beta_inter": r.get("beta_inter"),
            "beta_main": r.get("beta_main"),
            "sd_by_dosage": r.get("sd_by_dosage"),
            "p_value": r.get("p_emp") if r["kind"] == "ge" else r.get("p_asymptotic"),
            "recovered": r.get("recovered"),
        })
    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))

    summary_csv = os.path.join(ISOLATED_ROOT, "isolated_power_curve.csv")
    summary_df.to_csv(summary_csv, index=False)
    with open(os.path.join(ISOLATED_ROOT, "isolated_power_curve.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[export] {summary_csv}")
    print(f"Completato in {time.time() - t0:.0f}s.")

    not_recovered = [r["variant"] for r in all_results if r["status"] == "ok" and not r.get("recovered")]
    failed = [r["variant"] for r in all_results if r["status"] == "FAILED"]
    if not_recovered:
        print(f"\n*** VARIANTI CAUSALI NON RECUPERATE (isolate): {not_recovered} ***")
    if failed:
        print(f"*** VARIANTI FALLITE (eccezione): {failed} ***")
    if not_recovered or failed:
        sys.exit(1)
    print("\n*** Tutte le varianti causali isolate recuperate correttamente. ***")


if __name__ == "__main__":
    main()