#!/usr/bin/env python3
"""
qc_supplementary_plots.py
===========================
Supplementary figures for the paper, read from files already produced by
00_run_plink_qc.sh (missingness) and by 01_run_extra_qc_checks.sh
(sex_check.sexcheck, heterozygosity.het, maf.afreq). None of these files
are recomputed here: the script only reads and plots them. Each figure
is skipped individually (with a warning) if its input file is missing,
instead of failing the whole script.

USAGE:
  python3 qc_supplementary_plots.py \
      --qc-dir /mnt/genome_datasets/qc_output_cohortA \
      --out-dir /mnt/genome_datasets/qc_output_cohortA/supplementary_plots \
      --geno-thresh 0.05 --mind-thresh 0.05
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from plink_io import read_plink_table


def plot_missingness(qc_dir: Path, out_dir: Path, geno_thresh: float, mind_thresh: float) -> None:
    vmiss_path = qc_dir / "missingness.vmiss"
    smiss_path = qc_dir / "missingness.smiss"

    if not vmiss_path.exists() or not smiss_path.exists():
        print(f"  [skipping missingness: {vmiss_path.name} or {smiss_path.name} not found]")
        return

    vmiss = read_plink_table(vmiss_path)
    smiss = read_plink_table(smiss_path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].hist(vmiss["F_MISS"], bins=100, color="steelblue")
    axes[0].axvline(geno_thresh, color="red", linestyle="--", label=f"--geno threshold {geno_thresh}")
    axes[0].set_xlabel("F_MISS per variant")
    axes[0].set_ylabel("N variants")
    axes[0].set_yscale("log")
    axes[0].set_title("Per-variant missingness (pre-filter)")
    axes[0].legend()

    axes[1].hist(smiss["F_MISS"], bins=100, color="slategray")
    axes[1].axvline(mind_thresh, color="red", linestyle="--", label=f"--mind threshold {mind_thresh}")
    axes[1].set_xlabel("F_MISS per sample")
    axes[1].set_ylabel("N samples")
    axes[1].set_title("Per-sample missingness (pre-filter)")
    axes[1].legend()

    fig.tight_layout()
    out_path = out_dir / "missingness_distributions.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # Note: a bimodal distribution in vmiss (a block around F_MISS
    # ~1/n_batches) is the signature of sites present in only some
    # batches -- see the Step 5 comment in 00_run_plink_qc.sh.
    frac_above_geno = (vmiss["F_MISS"] > geno_thresh).mean()
    frac_above_mind = (smiss["F_MISS"] > mind_thresh).mean()
    print(f"  Variants above --geno threshold: {100*frac_above_geno:.2f}%")
    print(f"  Samples above --mind threshold: {100*frac_above_mind:.2f}%")


def plot_sex_check(qc_dir: Path, out_dir: Path) -> None:
    path = qc_dir / "sex_check.sexcheck"
    if not path.exists():
        print(f"  [skipping sex-check: {path.name} not found -- run 01_run_extra_qc_checks.sh first]")
        return

    df = read_plink_table(path)
    if "F" not in df.columns or "STATUS" not in df.columns:
        print(f"  [skipping sex-check: expected columns (F, STATUS) not found in {path}]")
        return

    problem = df["STATUS"] == "PROBLEM"
    n_problem = int(problem.sum())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(df.loc[~problem, "F"], bins=60, color="steelblue", alpha=0.8, label="OK")
    if n_problem:
        ax.hist(df.loc[problem, "F"], bins=60, color="red", alpha=0.8, label="PROBLEM (sex mismatch)")
    ax.set_xlabel("F-statistic (chrX heterozygosity)")
    ax.set_ylabel("N samples")
    ax.set_title(f"Sex check -- {n_problem} samples with discordant genetic sex")
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / "sex_check_distribution.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}  ({n_problem} PROBLEM out of {len(df)} samples)")

    if n_problem:
        flagged_path = out_dir / "sex_check_flagged_samples.csv"
        df.loc[problem].to_csv(flagged_path, index=False)
        print(f"  Flagged samples saved to: {flagged_path}")


def plot_heterozygosity(qc_dir: Path, out_dir: Path, sd_threshold: float = 3.0) -> None:
    path = qc_dir / "heterozygosity.het"
    if not path.exists():
        print(f"  [skipping heterozygosity: {path.name} not found -- run 01_run_extra_qc_checks.sh first]")
        return

    df = read_plink_table(path)
    if "F" not in df.columns:
        print(f"  [skipping heterozygosity: column F not found in {path}]")
        return

    mean_f = df["F"].mean()
    sd_f = df["F"].std()
    lower = mean_f - sd_threshold * sd_f
    upper = mean_f + sd_threshold * sd_f
    outliers = df[(df["F"] < lower) | (df["F"] > upper)]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(df["F"], bins=80, color="steelblue")
    ax.axvline(lower, color="red", linestyle="--", label=f"+/-{sd_threshold:.0f} SD")
    ax.axvline(upper, color="red", linestyle="--")
    ax.set_xlabel("F-statistic (heterozygosity, LD-pruned SNPs)")
    ax.set_ylabel("N samples")
    ax.set_title(f"Heterozygosity check -- {len(outliers)} outliers beyond {sd_threshold:.0f} SD")
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / "heterozygosity_distribution.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}  ({len(outliers)} outliers out of {len(df)} samples)")

    if len(outliers) > 0:
        outliers_path = out_dir / "heterozygosity_outlier_samples.csv"
        outliers.to_csv(outliers_path, index=False)
        print(f"  Outlier samples saved to: {outliers_path}")


def plot_maf_spectrum(qc_dir: Path, out_dir: Path) -> None:
    path = qc_dir / "maf.afreq"
    if not path.exists():
        print(f"  [skipping MAF spectrum: {path.name} not found -- run 01_run_extra_qc_checks.sh first]")
        return

    df = read_plink_table(path)
    freq_col = "ALT_FREQS" if "ALT_FREQS" in df.columns else None
    if freq_col is None:
        print(f"  [skipping MAF spectrum: column ALT_FREQS not found in {path}. Columns: {list(df.columns)}]")
        return

    maf = np.minimum(df[freq_col], 1 - df[freq_col])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(maf, bins=100, color="slategray")
    ax.set_xlabel("MAF (minor allele frequency)")
    ax.set_ylabel("N variants")
    ax.set_title(f"Post-filter MAF spectrum -- {len(maf):,} variants")
    fig.tight_layout()
    out_path = out_dir / "maf_spectrum.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Supplementary QC figures (missingness, sex-check, heterozygosity, MAF spectrum)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--qc-dir", required=True, type=Path, help="pipeline out_dir (00_run_plink_qc.sh)")
    parser.add_argument("--out-dir", required=True, type=Path, help="directory where figures are saved")
    parser.add_argument("--geno-thresh", type=float, default=0.05, help="--geno threshold used in the pipeline (default 0.05)")
    parser.add_argument("--mind-thresh", type=float, default=0.05, help="--mind threshold used in the pipeline (default 0.05)")
    parser.add_argument("--het-sd-threshold", type=float, default=3.0, help="SD threshold for heterozygosity outliers (default 3)")
    args = parser.parse_args()

    if plt is None:
        print("ERROR: matplotlib not available, cannot generate figures.")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("==> Missingness (per variant and per sample)")
    plot_missingness(args.qc_dir, args.out_dir, args.geno_thresh, args.mind_thresh)

    print("\n==> Sex check")
    plot_sex_check(args.qc_dir, args.out_dir)

    print("\n==> Heterozygosity check")
    plot_heterozygosity(args.qc_dir, args.out_dir, args.het_sd_threshold)

    print("\n==> MAF spectrum")
    plot_maf_spectrum(args.qc_dir, args.out_dir)

    print(f"\n==> DONE. Output in: {args.out_dir}")


if __name__ == "__main__":
    main()
