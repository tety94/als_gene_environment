"""
Script diagnostico: confronta se_method="asymptotic" vs "bootstrap" SOLO su
un piccolo sottoinsieme di varianti (le 7 causali + 20 nulle scelte a
caso), per capire in pochi secondi/minuti se il problema di lambda_GC che
oscilla (0.306 vs 2.754) e' davvero nella formula della SE asintotica dello
scan (vqtl/core/scan.py::_beta_qi_and_asymptotic_se) o e' altrove.

Metti questo file nella stessa cartella di test_vqtl_pipeline.py (quella
con fake_data/, fake_vqtl_repository.py, report_utils.py ecc.) e lancialo
con: python debug_se_method.py

Non tocca il DB finto persistente tra i due run: chiama fake_repo.reset_all()
tra un se_method e l'altro, cosi' non c'e' short-circuit da placeholder gia'
'done' del run precedente.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import replace

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCRIPT_DIR)

FAKE = os.path.join(SCRIPT_DIR, "fake_data")
if not os.path.isdir(FAKE):
    raise SystemExit(f"{FAKE} non esiste: lancia prima gen_fake_data_vqtl.py")

os.environ.setdefault("DB_USER", "unused")
os.environ.setdefault("DB_PASSWORD", "unused")
os.environ.setdefault("DB_NAME", "unused")
os.environ.update({
    "USE_PCA_COVARIATES": "true",
    "PCA_N_COMPONENTS": "10",
    "PCA_COVARIATES_PATH_TEMPLATE": os.path.join(FAKE, "pca_covariates_gen{generation}.csv"),
    "GENERATION": "1",
    "TARGET_COL": "onset_age",
    "EXPOSURE": "exposure_env",
    "COVARIATES": "sex",
    "RAW_FILE": os.path.join(FAKE, "genetic.csv"),
    "ENV_FILE": os.path.join(FAKE, "env.csv"),
    "SEP": ",",
    "LOG_DIR": os.path.join(SCRIPT_DIR, "logs"),
    "VQTL_CHUNK_SIZE": "10",
    "VQTL_N_JOBS": str(min(8, os.cpu_count() or 2)),
    # NOTA: se os.cpu_count() < 8, alzare N_JOBS oltre i core fisici non
    # velocizza granche' (context-switching). Il vero costo qui e' il
    # bootstrap stesso (K repliche x 9 tau x 2 fit QuantReg per variante).
    "VQTL_BOOTSTRAP_K": "200",  # come nel test principale
})

import fake_vqtl_repository as fake_repo  # noqa: E402
sys.modules["vqtl.db.repository"] = fake_repo

from gene_environment.config import get_config  # noqa: E402
from gene_environment.logging_utils import configure_logging  # noqa: E402
from vqtl.config import VqtlConfig  # noqa: E402
from vqtl.core.data import load_vqtl_dataset  # noqa: E402
from vqtl.core.phenotype import prepare_phenotype  # noqa: E402
from vqtl.core.scan import run_vqtl_scan, reset_convergence_stats, get_convergence_stats  # noqa: E402

GENERATION = 1


def main() -> None:
    ge_cfg = get_config()
    configure_logging(ge_cfg.log_dir)

    truth = pd.read_csv(os.path.join(FAKE, "ground_truth.csv"))
    causal = truth.loc[truth["effect_type"].isin(["gxe_meanshift", "pure_variance"]), "variant"].tolist()

    rng = np.random.default_rng(0)
    nulls_all = truth.loc[truth["effect_type"] == "no_effect", "variant"].tolist()
    nulls_sample = rng.choice(nulls_all, size=20, replace=False).tolist()

    wanted_labels = set(causal) | set(nulls_sample)
    print(f"Sottoinsieme: {len(causal)} causali + {len(nulls_sample)} nulle = {len(wanted_labels)} varianti")

    ds = load_vqtl_dataset(ge_cfg, VqtlConfig(), generation=GENERATION)
    ds.df = prepare_phenotype(ds.df, ge_cfg.target_col)

    # dataset.mapping: variant_safe -> label reale. Vogliamo il subset di
    # variant_safe corrispondente alle label che ci interessano.
    inv_mapping = {v: k for k, v in ds.mapping.items()}
    variant_subset = [inv_mapping[lab] for lab in wanted_labels if lab in inv_mapping]
    missing = wanted_labels - set(inv_mapping)
    if missing:
        print(f"ATTENZIONE: {len(missing)} label non trovate nel dataset: {missing}")

    results = {}
    for method in ["asymptotic", "bootstrap"]:
        fake_repo.reset_all()
        reset_convergence_stats()
        # NOTA: n_jobs=1 qui e' voluto -- i contatori di convergenza sono
        # in-process (dict globale del modulo scan.py); con backend="loky"
        # (processi separati) i worker avrebbero ciascuno la propria copia
        # e il conteggio nel processo principale resterebbe a zero. Per un
        # run "vero" parallelizzato i contatori non sono affidabili cosi'
        # come sono (andrebbero aggregati via joblib, es. ritornandoli dal
        # return di _process_chunk insieme alle righe).
        vcfg = VqtlConfig(se_method=method, n_jobs=1)
        print(f"\n--- se_method={method} (taus={vcfg.taus}, n_jobs=1 per contatori affidabili) ---")
        t0 = time.time()
        df = run_vqtl_scan(ds, vcfg, ge_cfg.target_col, generation=GENERATION, variant_subset=variant_subset)
        print(f"Fatto in {time.time() - t0:.1f}s")
        df = df.merge(truth[["variant", "effect_type"]], left_on="SNP", right_on="variant", how="left")
        results[method] = df[["SNP", "effect_type", "N", "MAF", "beta_QI", "SE", "Z", "P"]].sort_values("SNP")

        stats = get_convergence_stats()
        attempted = stats["tau_fits_attempted"]
        discarded = stats["tau_fits_discarded"]
        pct = 100 * discarded / attempted if attempted else 0.0
        print(
            f"Convergenza fit tau: {discarded}/{attempted} scartati ({pct:.1f}%) | "
            f"varianti finite con beta_QI=NaN (tutti i tau scartati): {stats['variants_all_nan']}"
        )

    merged = results["asymptotic"].merge(
        results["bootstrap"], on=["SNP", "effect_type"], suffixes=("_asym", "_boot")
    )
    merged = merged.sort_values("effect_type")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 100)
    print("\n" + "=" * 100)
    print("CONFRONTO asymptotic vs bootstrap (stesso identico dataset/dosaggio/residuo)")
    print("=" * 100)
    print(merged[[
        "SNP", "effect_type",
        "beta_QI_asym", "SE_asym", "Z_asym", "P_asym",
        "beta_QI_boot", "SE_boot", "Z_boot", "P_boot",
    ]].to_string(index=False))

    # lambda_GC "locale" sul sottoinsieme, solo indicativo (pool piccolo)
    for method in ["asymptotic", "bootstrap"]:
        z = results[method]["Z"].dropna().to_numpy()
        if len(z):
            lam = float(np.median(z ** 2) / 0.4549364)
            print(f"\nlambda_GC locale ({method}, n={len(z)}): {lam:.3f}  (indicativo, pool piccolo)")

    out_csv = os.path.join(SCRIPT_DIR, "debug_se_method_comparison.csv")
    merged.to_csv(out_csv, index=False)
    print(f"\n[export] {out_csv}")


if __name__ == "__main__":
    main()