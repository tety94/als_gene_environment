#!/usr/bin/env python3
"""
run_pipeline.py
================
Single entry point for the whole QC pipeline: runs every step, for every
cohort in the config file, in order, with one command:

    python3 run_pipeline.py --config pipeline_config.yaml

Steps, per cohort (in this order):
  1. qc          00_run_plink_qc.sh          (merge, filter, prune, kinship, PCA)
  2. extra       01_run_extra_qc_checks.sh   (sex check, heterozygosity, MAF, metadata)
  3. attrition   qc_attrition_summary.py     (sample/variant counts per stage)
  4. kinship     qc_report.py                (kinship + PCA/batch-effect plots)
  5. diagnostics interpret_plink_output.py   (once per exposure in the config)
  6. plots       qc_supplementary_plots.py   (missingness/sex/het/MAF figures)
  7. covariates  extract_pca_covariates.py   (PCs -> CSV for the G x E model)
  8. docx        build_supplementary_report.py (assembles the Word report)

Steps 1-2 run on the server with plink2/bcftools; steps 3-8 are pure
Python (pandas/matplotlib/python-docx). All of it is chained here so you
do not have to copy-paste commands between them or track by hand which
cohort/exposure you've already run.

RESUME: steps 1-2 already skip work that is already done (see the
comments in 00_run_plink_qc.sh); steps 3-8 are fast and always recompute,
so re-running this script after steps 1-2 have completed just refreshes
the tables/figures/report cheaply. Use --force to redo steps 1-2 from
scratch as well.

CONFIGURATION: see pipeline_config.example.yaml for the full schema and
inline documentation of every field. Command-line flags below override
the corresponding config values without editing the file.

USAGE EXAMPLES
--------------
Run everything, both cohorts, using the config as written:
    python3 run_pipeline.py --config pipeline_config.yaml

Only re-run the Python reporting steps (e.g. after tweaking a threshold),
skipping the multi-hour genomic steps:
    python3 run_pipeline.py --config pipeline_config.yaml \\
        --only attrition,kinship,diagnostics,plots,covariates,docx

Only gen1, and see the exact commands without running them:
    python3 run_pipeline.py --config pipeline_config.yaml --cohorts gen1 --dry-run

Redo absolutely everything from scratch, both cohorts:
    python3 run_pipeline.py --config pipeline_config.yaml --force
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit(
        "ERROR: PyYAML is required (pip install pyyaml --break-system-packages, "
        "or install it in the conda environment used for the rest of the pipeline)."
    )

SCRIPT_DIR = Path(__file__).resolve().parent

ALL_STEPS = ["qc", "extra", "attrition", "kinship", "diagnostics", "plots", "covariates", "docx"]


# ---------------------------------------------------------------------------
# Logging: everything printed also goes to a run log file.
# ---------------------------------------------------------------------------

class Logger:
    def __init__(self, log_path: Path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(log_path, "a", encoding="utf-8")
        self.path = log_path

    def __call__(self, msg: str = ""):
        print(msg)
        self._fh.write(msg + "\n")
        self._fh.flush()

    def close(self):
        self._fh.close()


def run_step(log: Logger, name: str, cmd: list, dry_run: bool, env: dict | None = None) -> int:
    printable = " ".join(shlex.quote(str(c)) for c in cmd)
    log(f"\n{'-'*70}\n>>> [{name}] {printable}\n{'-'*70}")
    if dry_run:
        log(f">>> [{name}] (dry-run, not executed)")
        return 0
    t0 = time.time()
    result = subprocess.run(cmd, env=env)
    elapsed = time.time() - t0
    if result.returncode == 0:
        log(f">>> [{name}] OK in {elapsed:.1f}s")
    else:
        log(f">>> [{name}] FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
    return result.returncode


# ---------------------------------------------------------------------------
# Config loading + CLI overrides
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    if not path.exists():
        sys.exit(
            f"ERROR: config file not found: {path}\n"
            f"Copy pipeline_config.example.yaml to a real config first, e.g.:\n"
            f"  cp pipeline_config.example.yaml pipeline_config.yaml"
        )
    with open(path) as f:
        config = yaml.safe_load(f)

    if "cohorts" not in config or not config["cohorts"]:
        sys.exit("ERROR: config has no 'cohorts' entries.")
    if "metadata_csv" not in config:
        sys.exit("ERROR: config is missing 'metadata_csv'.")
    if "exposures" not in config or not config["exposures"]:
        sys.exit("ERROR: config has no 'exposures' entries.")
    primary = config.get("primary_exposure_for_report")
    if primary and primary not in config["exposures"]:
        sys.exit(
            f"ERROR: primary_exposure_for_report ('{primary}') is not in the "
            f"'exposures' list."
        )

    config.setdefault("qc", {})
    qc_defaults = {
        "use_filtered": True, "jobs": 16, "force": False,
        "maf_threshold": 0.01, "ld_window_size": 50, "ld_step": 5, "ld_r2_threshold": 0.5,
        "geno_thresh": 0.05, "mind_thresh": 0.05, "exclude_id_prefixes": "",
        "n_pcs": 10, "strip_doubled_id": True, "pi_hat_threshold": 0.125,
        "het_sd_threshold": 3.0,
    }
    for k, v in qc_defaults.items():
        config["qc"].setdefault(k, v)

    config.setdefault("gwas_results", {})
    config.setdefault("gwas_pvalue_col", "p")

    return config


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    if args.force:
        config["qc"]["force"] = True
    if args.jobs is not None:
        config["qc"]["jobs"] = args.jobs
    if args.cohorts:
        wanted = [c.strip() for c in args.cohorts.split(",") if c.strip()]
        unknown = [c for c in wanted if c not in config["cohorts"]]
        if unknown:
            sys.exit(f"ERROR: unknown cohort(s) in --cohorts: {unknown}. "
                      f"Cohorts in config: {list(config['cohorts'])}")
        config["cohorts"] = {k: v for k, v in config["cohorts"].items() if k in wanted}
    return config


def resolve_steps(args: argparse.Namespace) -> list:
    if args.only:
        wanted = [s.strip() for s in args.only.split(",") if s.strip()]
        unknown = [s for s in wanted if s not in ALL_STEPS]
        if unknown:
            sys.exit(f"ERROR: unknown step(s) in --only: {unknown}. Valid steps: {ALL_STEPS}")
        return [s for s in ALL_STEPS if s in wanted]
    if args.skip:
        skip = {s.strip() for s in args.skip.split(",") if s.strip()}
        unknown = skip - set(ALL_STEPS)
        if unknown:
            sys.exit(f"ERROR: unknown step(s) in --skip: {unknown}. Valid steps: {ALL_STEPS}")
        return [s for s in ALL_STEPS if s not in skip]
    return list(ALL_STEPS)


# ---------------------------------------------------------------------------
# Per-cohort step commands
# ---------------------------------------------------------------------------

def run_cohort(log: Logger, cohort_key: str, cohort_cfg: dict, config: dict,
                steps: list, dry_run: bool) -> bool:
    """Runs the requested steps for one cohort. Returns True if all requested steps succeeded."""
    qc = config["qc"]
    out_dir = Path(cohort_cfg["out_dir"])
    vcf_dirs = [Path(d) for d in cohort_cfg["vcf_dirs"]]
    role = cohort_cfg.get("role", "")
    label = f"{cohort_key} ({role})" if role else cohort_key

    log(f"\n{'='*70}")
    log(f"COHORT: {label}")
    log(f"out_dir: {out_dir}")
    log(f"{'='*70}")

    def ok(code):
        return code == 0

    if "qc" in steps:
        cmd = ["bash", str(SCRIPT_DIR / "00_run_plink_qc.sh")]
        if qc["use_filtered"]:
            cmd.append("--use-filtered")
        cmd += ["--jobs", str(qc["jobs"])]
        if qc["force"]:
            cmd.append("--force")
        cmd += [str(d) for d in vcf_dirs] + [str(out_dir)]
        log(f"(env overrides in effect: MAF_THRESHOLD={qc['maf_threshold']} "
            f"LD_WINDOW_SIZE={qc['ld_window_size']} LD_STEP={qc['ld_step']} "
            f"LD_R2_THRESHOLD={qc['ld_r2_threshold']} GENO_THRESH={qc['geno_thresh']} "
            f"MIND_THRESH={qc['mind_thresh']} EXCLUDE_ID_PREFIXES='{qc['exclude_id_prefixes']}')")
        env = {
            **os.environ,
            "MAF_THRESHOLD": str(qc["maf_threshold"]),
            "LD_WINDOW_SIZE": str(qc["ld_window_size"]),
            "LD_STEP": str(qc["ld_step"]),
            "LD_R2_THRESHOLD": str(qc["ld_r2_threshold"]),
            "GENO_THRESH": str(qc["geno_thresh"]),
            "MIND_THRESH": str(qc["mind_thresh"]),
            "EXCLUDE_ID_PREFIXES": str(qc["exclude_id_prefixes"]),
        }
        if not ok(run_step(log, "qc", cmd, dry_run, env=env)):
            return False

    if "extra" in steps:
        cmd = ["bash", str(SCRIPT_DIR / "01_run_extra_qc_checks.sh")]
        if qc["force"]:
            cmd.append("--force")
        cmd.append(str(out_dir))
        if not ok(run_step(log, "extra", cmd, dry_run)):
            return False

    if "attrition" in steps:
        cmd = ["python3", str(SCRIPT_DIR / "qc_attrition_summary.py"),
               "--qc-dir", str(out_dir),
               "--out", str(out_dir / "qc_attrition.csv")]
        if not ok(run_step(log, "attrition", cmd, dry_run)):
            return False

    if "kinship" in steps:
        cmd = ["python3", str(SCRIPT_DIR / "qc_report.py"),
               "--kin", str(out_dir / "king.kin0"),
               "--eigenvec", str(out_dir / "pca.eigenvec"),
               "--eigenval", str(out_dir / "pca.eigenval"),
               "--vcf-dirs", *[str(d) for d in vcf_dirs],
               "--out-dir", str(out_dir / "qc_report")]
        if qc["use_filtered"]:
            cmd.append("--use-filtered")
        if not ok(run_step(log, "kinship", cmd, dry_run)):
            return False

    if "diagnostics" in steps:
        gwas_path = (config.get("gwas_results") or {}).get(cohort_key)
        for exposure in config["exposures"]:
            cmd = ["python3", str(SCRIPT_DIR / "interpret_plink_output.py"),
                   "--kin0", str(out_dir / "king.kin0"),
                   "--eigenvec", str(out_dir / "pca.eigenvec"),
                   "--metadata", str(config["metadata_csv"]),
                   "--exposure-col", exposure,
                   "--pi-hat-threshold", str(qc["pi_hat_threshold"]),
                   "--n-pcs", str(qc["n_pcs"]),
                   "--out-dir", str(out_dir / f"diagnostics_output_{exposure}")]
            if qc["strip_doubled_id"]:
                cmd.append("--strip-doubled-id")
            if gwas_path:
                cmd += ["--pvalues", str(gwas_path), "--pvalue-col", str(config["gwas_pvalue_col"])]
            if not ok(run_step(log, f"diagnostics[{exposure}]", cmd, dry_run)):
                return False

    if "plots" in steps:
        cmd = ["python3", str(SCRIPT_DIR / "qc_supplementary_plots.py"),
               "--qc-dir", str(out_dir),
               "--out-dir", str(out_dir / "supplementary_plots"),
               "--geno-thresh", str(qc["geno_thresh"]),
               "--mind-thresh", str(qc["mind_thresh"]),
               "--het-sd-threshold", str(qc["het_sd_threshold"])]
        if not ok(run_step(log, "plots", cmd, dry_run)):
            return False

    if "covariates" in steps:
        cmd = ["python3", str(SCRIPT_DIR / "extract_pca_covariates.py"),
               "--eigenvec", str(out_dir / "pca.eigenvec"),
               "--n-pcs", str(qc["n_pcs"]),
               "--out", str(out_dir / "pca_covariates.csv")]
        if qc["strip_doubled_id"]:
            cmd.append("--strip-doubled-id")
        if not ok(run_step(log, "covariates", cmd, dry_run)):
            return False

    if "docx" in steps:
        primary = config.get("primary_exposure_for_report") or config["exposures"][0]
        cmd = ["python3", str(SCRIPT_DIR / "build_supplementary_report.py"),
               "--qc-dir", str(out_dir),
               "--kinship-report-dir", str(out_dir / "qc_report"),
               "--diagnostics-dir", str(out_dir / f"diagnostics_output_{primary}"),
               "--attrition-csv", str(out_dir / "qc_attrition.csv"),
               "--supp-plots-dir", str(out_dir / "supplementary_plots"),
               "--cohort-label", label,
               "--out", str(out_dir / f"Supplementary_QC_Report_{cohort_key}.docx")]
        if not ok(run_step(log, "docx", cmd, dry_run)):
            return False

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run the entire QC pipeline (all cohorts, all steps) with one command.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "pipeline_config.yaml",
                         help="path to the YAML config (default: pipeline_config.yaml next to this script)")
    parser.add_argument("--force", action="store_true",
                         help="ignore existing output and redo every step from scratch (overrides qc.force)")
    parser.add_argument("--jobs", type=int, default=None,
                         help="override qc.jobs (parallel workers for 00_run_plink_qc.sh)")
    parser.add_argument("--cohorts", type=str, default=None,
                         help="comma-separated subset of cohort keys to run (default: all cohorts in config)")
    parser.add_argument("--only", type=str, default=None,
                         help=f"comma-separated subset of steps to run, in {ALL_STEPS}")
    parser.add_argument("--skip", type=str, default=None,
                         help="comma-separated steps to skip (alternative to --only)")
    parser.add_argument("--dry-run", action="store_true",
                         help="print every command that would run, without running it")
    parser.add_argument("--keep-going", action="store_true",
                         help="if a cohort fails, continue with the remaining cohorts instead of aborting")
    args = parser.parse_args()

    if args.only and args.skip:
        parser.error("--only and --skip are mutually exclusive")

    config = load_config(args.config)
    config = apply_cli_overrides(config, args)
    steps = resolve_steps(args)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = Logger(SCRIPT_DIR / "logs" / f"run_pipeline_{timestamp}.log")

    log(f"run_pipeline.py starting: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Config: {args.config}")
    log(f"Cohorts: {list(config['cohorts'])}")
    log(f"Steps: {steps}")
    log(f"Dry run: {args.dry_run} | Force: {config['qc']['force']} | Keep going on failure: {args.keep_going}")
    log(f"Full log: {log.path}")

    results = {}
    for cohort_key, cohort_cfg in config["cohorts"].items():
        try:
            success = run_cohort(log, cohort_key, cohort_cfg, config, steps, args.dry_run)
        except KeyboardInterrupt:
            log("\nInterrupted by user (Ctrl-C).")
            log.close()
            sys.exit(130)
        results[cohort_key] = success
        if not success:
            log(f"\n>>> COHORT {cohort_key} FAILED.")
            if not args.keep_going:
                log(">>> Aborting (use --keep-going to run remaining cohorts despite a failure).")
                break
            else:
                log(">>> --keep-going set, continuing with the next cohort.")

    log(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for cohort_key in config["cohorts"]:
        status = results.get(cohort_key, "not run")
        status_str = "OK" if status is True else ("FAILED" if status is False else "not run")
        log(f"  {cohort_key}: {status_str}")
    log(f"\nFull log saved to: {log.path}")

    log.close()
    if not all(v is True for v in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
