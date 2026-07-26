"""
Multi-scenario orchestrator (robustness battery).

For each scenario defined in SCENARIOS:
  1) generate the synthetic data      -> gen_fake_data.generate_dataset(**params)
  2) run the gene-environment part    -> run_ge_interaction() below
     (direct, sequential call to modeling.process_single_variant,
     no reimplementation)
  3) run the vQTL part in two modes:
       - "debug"      -> run_vqtl_debug() below: asymptotic vs bootstrap
                          comparison on (causal variants + 20 random nulls),
                          useful for a quick check of divergence between
                          the two SE methods
       - "asymptotic" -> run_vqtl_asymptotic() below: full Step 3->7 vQTL
                          pipeline (run_pipeline_for_method from
                          test_vqtl_pipeline.py, imported as-is, not
                          duplicated), se_method="asymptotic" only (the
                          bootstrap is already compared in debug, running it
                          here too would double the time for little added
                          value)

Each scenario writes under scenarios/<name>/ (fake_data/, vqtl_results/,
debug/, ge_pipeline_results.csv, scenario_summary.json). At the end, a
single summary report is written to scenarios/all_scenarios_summary.csv
and .json, so it is immediately visible whether any scenario went wrong
(has_failures, causal variants recovered, lambda_GC, any exceptions), plus
an aggregate Word report (scenarios/recap_all/all_scenarios_report.docx)
and a manuscript-ready comparison table + figure
(scenarios/recap_all/manuscript_scenario_report.docx, see
report_utils.build_scenario_comparison_table /
write_manuscript_scenario_report).

*** CAUSAL VARIANTS USED HERE: a SMALL, FIXED set (SCENARIO_CAUSAL_
VARIANTS / SCENARIO_PURE_VARIANCE_VARIANTS below), NOT the "large" defaults
from gen_fake_data.py (DEFAULT_CAUSAL_VARIANTS / DEFAULT_PURE_VARIANCE_VARIANTS,
46 G×E + 16 pure_variance).

WHY: gen_fake_data.py adds the contribution of EVERY active causal variant
to the SAME onset_age, for the same patient:

    for lab, (beta_inter, beta_main) in causal_variants.items():
        onset_age += beta_main*dosage + beta_inter*dosage*exposure_std
    for lab, sd_by_dosage in pure_variance_variants.items():
        onset_age += rng.normal(0, sd_i, size=n)

If 46+16 are activated together (measured on a real run): the "hidden"
noise that every single test faces rises to ~4.5x the declared noise_sd
(mostly because of the pure_variance ones, alone ~15x the base variance),
because no single-variant test adjusts for the other active causal
variants in the same dataset. To test the ROBUSTNESS of the pipeline to
different conditions (stratification, missingness, small sample,
zero-inflated exposure), a clean signal comparable across scenarios is
needed -- so this small set is ALWAYS used here, regardless of
gen_fake_data.py's defaults. The power curve by magnitude/sign, instead,
should be measured with run_isolated_casual_test.py (1 single active
variant per dataset), not here.
***

THIS FILE IS BOTH A LIBRARY AND A SCRIPT:
  - run_isolated_casual_test.py imports it (`import run_scenarios as rs`)
    to reuse _set_common_env / run_ge_interaction / run_vqtl_debug (per-
    variant isolated test) and run_all_scenarios (scenario battery, as
    "phase 2" of the full battery) -- see that file's docstring for the
    recommended single entry point.
  - It can still be run standalone, if you want ONLY the scenario battery
    and nothing else:
        python run_scenarios.py                          # sequential, all scenarios
        python run_scenarios.py baseline small_sample     # sequential, only some
        python run_scenarios.py --workers 4               # parallel, 4 scenarios at once
        python run_scenarios.py --workers 4 baseline small_sample

Put this file in the repo ROOT folder, together with gen_fake_data.py,
test_vqtl_pipeline.py, fake_vqtl_repository.py and report_utils.py.

get_config()/get_vqtl_config() from the repo read the env every time (they
are not cached with lru_cache) -- which is why it is enough here to update
os.environ and call them again for every scenario, with no need for
cache_clear()/reload.

PARALLELIZATION (--workers > 1):
  os.environ, modeling.global_df, and the in-memory tables of
  fake_vqtl_repository are PROCESS-GLOBAL state -- if two scenarios ran as
  threads in the same process they would overwrite each other's
  environment variables and loaded data. This is why ProcessPoolExecutor
  is used here (separate processes, each with its own state): every worker
  imports the modules from scratch and sets its own env, with no
  interference between concurrent scenarios. The process-start/import
  overhead is negligible compared to the scan/permutation time.

  VQTL_N_JOBS (joblib's internal jobs for the vQTL scan, not the
  scenarios) is automatically reduced based on --workers so as not to
  oversaturate the CPU: with N available cores and W scenario workers,
  each scenario uses at most N//W internal jobs (minimum 1). With
  --workers 1 the behaviour is identical to before (up to 4 internal jobs,
  as in the original scripts).
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
REPO_ROOT = SCRIPT_DIR  # change this if this file is not in the repo root
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from report_utils import (
    generate_recap, generate_multi_scenario_recap, load_scenario_recap,
    build_scenario_comparison_table, write_manuscript_scenario_report,
)

SCENARIOS_ROOT = os.path.join(SCRIPT_DIR, "scenarios")

# ============================================================
# Causal variants for the ROBUSTNESS scenarios: small, fixed set, the same
# "historical" 5 G×E + 2 pure_variance ones (the ones the pipeline was
# originally validated with). See the explanation at the top of the file
# for why the large gen_fake_data.py defaults are NOT used here.
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
# Scenario definitions: each dict is passed to
# gen_fake_data.generate_dataset(out_dir=..., causal_variants=SCENARIO_CAUSAL_VARIANTS,
# pure_variance_variants=SCENARIO_PURE_VARIANCE_VARIANTS, **params).
# Any parameter not specified stays at the generator's default.
# Feel free to add/change others here (do NOT put "causal_variants" or
# "pure_variance_variants" inside these dicts: they are already fixed above
# and passed explicitly in run_scenario(), to avoid the "got multiple
# values for keyword argument" error).
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
    """Default keys/values for gene_environment.config.get_config() and
    vqtl.config.get_vqtl_config(), pointed at the current scenario's
    folders. vqtl_n_jobs overrides the default (4) -- used to avoid
    oversaturating the CPU when several scenarios run in parallel (see
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
    """Forces the values passed in `overrides` onto the config dataclass's
    fields (get_config()/get_vqtl_config()), REGARDLESS of what those
    functions actually read from the env.

    Why this is needed: the observed symptom is that, despite having set
    RAW_FILE=.../fake_data/genetic.csv in os.environ BEFORE calling
    get_config(), the pipeline still loaded a completely different file
    (a real gen.parquet). This happens if get_config() inside config.py
    reloads a real .env file (e.g. via python-dotenv with override=True)
    EVERY time it is called: in that case any value set by hand in
    os.environ BEFORE the call gets silently replaced INSIDE the call
    itself, and there is no way to protect against it from the env side.
    The only robust way is to overwrite the resulting dataclass's fields
    AFTER get_config() has already been called.

    Uses dataclasses.fields() to apply only to field names that actually
    exist: if a guessed name here does not exist on your dataclass, it is
    printed as "skipped" instead of crashing everything -- in that case the
    real field name in config.py is different, fix the map here
    (_cfg_field_names) with the right one."""
    import dataclasses as _dc
    if not _dc.is_dataclass(cfg):
        print(f"[{context}] WARNING: cfg is not a dataclass, skipping _force_cfg_overrides (check the paths used by hand).")
        return cfg
    valid_fields = {f.name for f in _dc.fields(cfg)}
    applied = {k: v for k, v in overrides.items() if k in valid_fields}
    skipped = {k: v for k, v in overrides.items() if k not in valid_fields}
    if skipped:
        print(f"[{context}] WARNING: fields not found on the config dataclass (wrong name? "
              f"fix _cfg_field_names in run_scenarios.py), NOT forced: {skipped} "
              f"| available fields: {sorted(valid_fields)}")
    if applied:
        cfg = _dc.replace(cfg, **applied)
        print(f"[{context}] Paths forced onto the config (ignoring any internally loaded .env): {applied}")
    return cfg


# ============================================================
# Step 2: gene-environment part. Calls modeling.process_single_variant
# directly on every variant of the dataset, sequentially -- no
# reimplementation of the statistics, orchestration only. Single copy of
# this logic in the repo: reused both by run_scenario() below and by
# run_isolated_casual_test.py for the per-variant isolated test.
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
        # Must point to a path that does NOT exist for the synthetic data:
        # if left at the value read from the project's real .env,
        # build_dataset._build_narrow_covariates() would use the REAL
        # id->generation map (real patients), which contains none of the
        # synthetic ids here -- every row would come out as "unknown
        # generation" and get dropped (symptom: "N -> 0 rows" in the logs,
        # then a downstream crash on StandardScaler for an empty array).
        # By pointing it at a nonexistent path, _build_narrow_covariates()
        # takes the "no map found, use all rows" branch, which is correct
        # for a single-generation synthetic dataset.
        "sample_generation_map": os.path.join(fake_dir, "__no_sample_generation_map__.csv"),
    }, context="gene-environment")
    configure_logging(cfg.log_dir)
    log = get_logger(__name__)

    print("Loading and preparing the dataset (same code used in production)...")
    df, variant_cols_safe, mapping, Ecols, variant_cols, covariate_cols = load_and_prepare_data(cfg)
    print(f"Rows: {len(df)} | Exposure (Ecols): {Ecols} | Covariates: {covariate_cols}")
    print(f"Variants to test: {len(variant_cols_safe)}")

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
            print(f"  {i + 1}/{len(variant_cols_safe)} variants processed ({time.time() - t0:.0f}s)")

    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.0f}s. Results: {len(results)}")

    res_df = pd.DataFrame(results)
    out_path = os.path.join(fake_dir, "pipeline_results.csv")
    res_df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

    n_sig = int((res_df["p_emp"] < 0.05).sum()) if "p_emp" in res_df.columns and not res_df.empty else 0

    recap_summary = generate_recap(
        ground_truth_path=os.path.join(fake_dir, "ground_truth.csv"),
        pipeline_results_path=os.path.join(fake_dir, "pipeline_results.csv"),
        out_dir=os.path.join(work_dir, "recap"),
    )
    recap_public = {k: v for k, v in recap_summary.items() if k != "_detail"}  # no DataFrame in the JSON

    return {
        "n_variants_tested": len(variant_cols_safe),
        "n_results": len(results),
        "n_significant_p_emp_lt_05": n_sig,
        "elapsed_s": round(elapsed, 1),
        "results_csv": out_path,
        "recap": recap_public,
    }


# ============================================================
# Step 3a: vQTL "debug" part -- asymptotic vs bootstrap comparison on a
# small subset (causal variants + 20 random nulls). Single copy of this
# logic in the repo: reused both by run_scenario() below and by
# run_isolated_casual_test.py for the per-variant comparison.
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
        # Must point to a path that does NOT exist for the synthetic data:
        # if left at the value read from the project's real .env,
        # build_dataset._build_narrow_covariates() would use the REAL
        # id->generation map (real patients), which contains none of the
        # synthetic ids here -- every row would come out as "unknown
        # generation" and get dropped (symptom: "N -> 0 rows" in the logs,
        # then a downstream crash on StandardScaler for an empty array).
        # By pointing it at a nonexistent path, _build_narrow_covariates()
        # takes the "no map found, use all rows" branch, which is correct
        # for a single-generation synthetic dataset.
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
    print(f"Subset: {len(causal)} causal + {len(nulls_sample)} null = {len(wanted_labels)} variants")

    ds = load_vqtl_dataset(ge_cfg, VqtlConfig(ge=ge_cfg, exposures=["exposure_env"]), generation=GENERATION)
    ds.df = prepare_phenotype(ds.df, ge_cfg.target_col)

    inv_mapping = {v: k for k, v in ds.mapping.items()}
    variant_subset = [inv_mapping[lab] for lab in wanted_labels if lab in inv_mapping]
    missing = wanted_labels - set(inv_mapping)
    if missing:
        print(f"WARNING: {len(missing)} labels not found in the dataset: {missing}")

    results = {}
    convergence = {}
    for method in ["asymptotic", "bootstrap"]:
        fake_repo.reset_all()
        reset_convergence_stats()
        vcfg = VqtlConfig(ge=ge_cfg, se_method=method, n_jobs=1)
        print(f"\n--- se_method={method} (n_jobs=1 for reliable counters) ---")
        t0 = time.time()
        df = run_vqtl_scan(ds, vcfg, ge_cfg.target_col, generation=GENERATION, variant_subset=variant_subset)
        print(f"Done in {time.time() - t0:.1f}s")
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
        print(f"Tau fit convergence: {discarded}/{attempted} discarded ({pct:.1f}%) | "
              f"variants with beta_QI=NaN: {stats['variants_all_nan']}")

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
# Step 3b: vQTL "asymptotic" part -- full Step 3->7 pipeline, reusing
# run_pipeline_for_method from test_vqtl_pipeline.py as-is (no duplication
# of the statistical/orchestration logic).
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
        # Must point to a path that does NOT exist for the synthetic data:
        # if left at the value read from the project's real .env,
        # build_dataset._build_narrow_covariates() would use the REAL
        # id->generation map (real patients), which contains none of the
        # synthetic ids here -- every row would come out as "unknown
        # generation" and get dropped (symptom: "N -> 0 rows" in the logs,
        # then a downstream crash on StandardScaler for an empty array).
        # By pointing it at a nonexistent path, _build_narrow_covariates()
        # takes the "no map found, use all rows" branch, which is correct
        # for a single-generation synthetic dataset.
        "sample_generation_map": os.path.join(fake_dir, "__no_sample_generation_map__.csv"),
    }, context="vqtl-asymptotic")
    configure_logging(ge_cfg.log_dir)
    vcfg_base = get_vqtl_config()
    vcfg_base = _force_cfg_overrides(vcfg_base, {
        "exposures": ["exposure_env"],
    }, context="vqtl-asymptotic-vcfg")

    truth = pd.read_csv(os.path.join(fake_dir, "ground_truth.csv"))
    causal_gxe = set(truth.loc[truth["effect_type"] == "gxe_meanshift", "variant"])
    causal_pure_var = set(truth.loc[truth["effect_type"] == "pure_variance", "variant"])
    all_causal = causal_gxe | causal_pure_var
    n_null_truth = int((truth["effect_type"] == "no_effect").sum())
    print(f"Ground truth: {len(causal_gxe)} G×E causal, {len(causal_pure_var)} pure vQTL, {n_null_truth} null")

    # work_dir and GENERATION passed explicitly (no longer monkey-patched
    # onto the tvp module): safe to call for several scenarios in a row in
    # the same process, see test_vqtl_pipeline.py's docstring.
    summary = tvp.run_pipeline_for_method(
        "asymptotic", ge_cfg, vcfg_base, truth, all_causal, n_null_truth,
        work_dir=work_dir, generation=GENERATION,
    )
    return summary


# ============================================================
# Cache: if a scenario already has a scenario_summary.json on disk with
# status "ok" (i.e. it completed without raising, whatever the automated
# pipeline checks concluded -- has_failures is a separate, non-fatal
# signal, see run_all_scenarios), skip regenerating the data and rerunning
# the whole pipeline for it. Bypassable with force=True (--force on the
# CLI). A scenario that previously FAILED (exception) is always retried,
# never served from cache.
# ============================================================

def _load_cached_scenario(scenario_dir: str, force: bool = False) -> dict | None:
    if force:
        return None
    summary_path = os.path.join(scenario_dir, "scenario_summary.json")
    if not os.path.isfile(summary_path):
        return None
    try:
        with open(summary_path) as f:
            cached = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None  # corrupted/incomplete file: recompute
    if cached.get("status") != "ok":
        return None  # a previously failed run should always be retried
    return cached


# ============================================================
# Per-scenario orchestration + main
# ============================================================

def run_scenario(name: str, gen_params: dict, n_workers: int = 1, force: bool = False) -> dict:
    scenario_dir = os.path.join(SCENARIOS_ROOT, name)

    cached = _load_cached_scenario(scenario_dir, force=force)
    if cached is not None:
        section(f"SCENARIO: {name} (cached result found, status ok, skipping recompute)")
        return cached

    section(f"SCENARIO: {name}")
    fake_dir = os.path.join(scenario_dir, "fake_data")
    os.makedirs(fake_dir, exist_ok=True)

    result: dict = {"scenario": name, "params": gen_params, "status": "ok", "error": None}

    try:
        from gen_fake_data import generate_dataset
        # FIXED, small set (see the explanation at the top of the file):
        # NOT the large gen_fake_data.py defaults, so as not to contaminate
        # the actual noise each test sees relative to the declared
        # noise_sd -- here we want to isolate the effect of the SCENARIO
        # PARAMETER (stratification, missingness, etc.), not mix it with
        # the effect of having dozens of causal variants active together.
        gen_summary = generate_dataset(
            out_dir=fake_dir, verbose=True,
            causal_variants=SCENARIO_CAUSAL_VARIANTS,
            pure_variance_variants=SCENARIO_PURE_VARIANCE_VARIANTS,
            **gen_params,
        )
        result["gen_summary"] = gen_summary

        vqtl_n_jobs = max(1, (os.cpu_count() or 2) // max(1, n_workers))
        _set_common_env(fake_dir, scenario_dir, vqtl_n_jobs=vqtl_n_jobs)

        section(f"[{name}] Gene-environment part")
        result["ge_interaction"] = run_ge_interaction(fake_dir, scenario_dir)

        section(f"[{name}] vQTL part — debug (asymptotic vs bootstrap, subset)")
        result["vqtl_debug"] = run_vqtl_debug(fake_dir, scenario_dir)

        section(f"[{name}] vQTL part — full Step 3→7 pipeline (se_method=asymptotic)")
        result["vqtl_asymptotic"] = run_vqtl_asymptotic(fake_dir, scenario_dir)

    except Exception as exc:  # a failed scenario must not block the others
        result["status"] = "FAILED"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        print(f"\n*** SCENARIO '{name}' FAILED: {result['error']} ***")
        traceback.print_exc()

    with open(os.path.join(scenario_dir, "scenario_summary.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


def _run_scenario_worker(name: str, n_workers: int, force: bool = False) -> dict:
    """Top-level wrapper (needed: ProcessPoolExecutor must be able to
    pickle the submitted function). Every call runs in a new Python
    process -> no sharing of os.environ / module state with the other
    scenarios in progress."""
    return run_scenario(name, SCENARIOS[name], n_workers=n_workers, force=force)


def run_all_scenarios(names: list[str] | None = None, n_workers: int = 1, force: bool = False) -> dict:
    """Runs the scenario battery (all of them, or only `names` if
    specified), writes the aggregate reports under SCENARIOS_ROOT and
    returns a dict with:
      - "all_results":  list of each scenario's result dict (one per
                         scenario, same schema as run_scenario())
      - "summary_df":   pd.DataFrame summary (same content as
                         all_scenarios_summary.csv)
      - "failed":       names of scenarios that failed with an exception
      - "vqtl_failed":  names of scenarios with has_failures=True on the
                         vQTL pipeline's automated checks
      - "has_failures": True if failed or vqtl_failed are non-empty
    force: if True, ignore any cached scenario_summary.json (status "ok")
        found on disk and recompute every scenario from scratch. Default
        False: a scenario whose scenario_summary.json already has
        status="ok" is served from cache instead of being rerun (data
        regeneration + gene-environment + vQTL Step 3-7 all skipped) --
        a scenario that previously FAILED is always retried regardless of
        this flag.
    Used both by main() (standalone CLI) and by run_isolated_casual_test.py
    (as the "scenarios phase" of the full test battery)."""
    names = list(names) if names else list(SCENARIOS.keys())
    unknown = set(names) - set(SCENARIOS)
    if unknown:
        raise SystemExit(f"Unknown scenarios: {sorted(unknown)}. Available: {list(SCENARIOS)}")

    n_workers = max(1, n_workers)
    os.makedirs(SCENARIOS_ROOT, exist_ok=True)
    all_results = []
    t0 = time.time()

    if n_workers == 1:
        for name in names:
            all_results.append(run_scenario(name, SCENARIOS[name], n_workers=1, force=force))
    else:
        print(f"Running {len(names)} scenarios with {n_workers} parallel processes "
              f"(each with at most {max(1, (os.cpu_count() or 2) // n_workers)} internal vQTL jobs)...")
        results_by_name = {}
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_run_scenario_worker, name, n_workers, force): name for name in names}
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    results_by_name[name] = fut.result()
                except Exception as exc:  # error not caught inside run_scenario itself (rare)
                    results_by_name[name] = {
                        "scenario": name, "status": "FAILED",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    print(f"\n*** SCENARIO '{name}' FAILED in the worker process: {exc} ***")
                print(f"[{name}] completed ({results_by_name[name]['status']}).")
        # reorder according to the requested order, for deterministic output
        all_results = [results_by_name[name] for name in names]

    section("FINAL SUMMARY — all scenarios")
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
    print(f"\nCompleted in {time.time() - t0:.0f}s.")

    section("SINGLE FINAL REPORT (all scenarios)")
    scenario_summaries_for_agg = {}
    for r in all_results:
        if r["status"] != "ok":
            continue
        recap_dir = os.path.join(SCENARIOS_ROOT, r["scenario"], "recap")
        if not os.path.isdir(recap_dir):
            print(f"[{r['scenario']}] no recap/ folder, skipping from the aggregate report")
            continue
        detail, recap_summary = load_scenario_recap(recap_dir)
        scenario_summaries_for_agg[r["scenario"]] = {**recap_summary, "_detail": detail}

    recap_all_dir = os.path.join(SCENARIOS_ROOT, "recap_all")
    if scenario_summaries_for_agg:
        generate_multi_scenario_recap(
            scenario_summaries_for_agg,
            out_dir=recap_all_dir,
        )
        print(f"[export] {os.path.join(recap_all_dir, 'all_scenarios_report.docx')}")
    else:
        print("No ok scenario with a recap available: aggregate report skipped.")

    section("MANUSCRIPT-READY SCENARIO COMPARISON TABLE")
    # Compact, one-row-per-scenario table (G×E + vQTL metrics together),
    # meant to go directly into the manuscript/supplementary -- see
    # report_utils.build_scenario_comparison_table /
    # write_manuscript_scenario_report. If isolated/plots/manuscript_power_curve.png
    # already exists (produced by run_isolated_casual_test.py's phase 1),
    # it is embedded in the same docx; otherwise the table is written on
    # its own and the figure can be added later.
    os.makedirs(recap_all_dir, exist_ok=True)
    comparison_table = build_scenario_comparison_table(all_results)
    comparison_table.to_csv(os.path.join(recap_all_dir, "manuscript_scenario_table.csv"), index=False)
    # Derived from SCENARIOS_ROOT's parent, NOT from SCRIPT_DIR: SCENARIOS_ROOT
    # can be overridden at runtime (this script's own --output-dir, or
    # run_isolated_casual_test.py setting rs.SCENARIOS_ROOT directly), and in
    # that case isolated/plots/ lives next to the overridden scenarios/, not
    # next to this file -- using SCRIPT_DIR here was a bug (the figure was
    # silently never found whenever --output-dir was used).
    output_root = os.path.dirname(os.path.normpath(SCENARIOS_ROOT))
    power_curve_path = os.path.join(output_root, "isolated", "plots", "manuscript_power_curve.png")
    write_manuscript_scenario_report(
        comparison_table,
        out_path=os.path.join(recap_all_dir, "manuscript_scenario_report.docx"),
        figure_path=power_curve_path if os.path.exists(power_curve_path) else None,
    )
    if not os.path.exists(power_curve_path):
        print(f"[note] {power_curve_path} not found: the manuscript report was written with the table "
              f"only. Run run_isolated_casual_test.py's phase 1 first to also get the power-curve figure.")

    failed = [r["scenario"] for r in all_results if r["status"] == "FAILED"]
    vqtl_failed = [r["scenario"] for r in all_results
                   if r["status"] == "ok" and (r.get("vqtl_asymptotic") or {}).get("has_failures")]
    if failed:
        print(f"\n*** FAILED SCENARIOS (exception): {failed} ***")
    if vqtl_failed:
        print(f"*** SCENARIOS WITH FAILED vQTL CHECKS (has_failures): {vqtl_failed} ***")
    if not failed and not vqtl_failed:
        print("\n*** All scenarios completed with no blocking errors. ***")

    return {
        "all_results": all_results,
        "summary_df": summary_df,
        "failed": failed,
        "vqtl_failed": vqtl_failed,
        "has_failures": bool(failed or vqtl_failed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-scenario orchestrator")
    parser.add_argument("scenarios", nargs="*", help="Scenario names to run (default: all)")
    parser.add_argument("--workers", type=int, default=1,
                         help="Number of scenarios to run in parallel (separate processes). Default 1 (sequential).")
    parser.add_argument("--force", action="store_true",
                         help="Ignore any cached scenario_summary.json (status ok) and recompute every "
                              "scenario from scratch, even if it already completed successfully before.")
    parser.add_argument("--output-dir", default=None,
                         help="Folder where scenarios/ is written (default: this script's folder).")
    args = parser.parse_args()

    if args.output_dir:
        global SCENARIOS_ROOT
        SCENARIOS_ROOT = os.path.join(os.path.abspath(args.output_dir), "scenarios")
        print(f"[config] Output: {SCENARIOS_ROOT}")

    result = run_all_scenarios(names=args.scenarios or None, n_workers=args.workers, force=args.force)
    if result["has_failures"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
