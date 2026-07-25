#!/usr/bin/env python3
"""
qc_attrition_summary.py
========================
Reconstructs the "samples/variants at each step" table by reading the
intermediate files 00_run_plink_qc.sh leaves on disk (it does not
recompute anything, does not touch the bash pipeline). Useful as a
supplementary table for the paper: shows where and how much is lost at
each QC stage.

STAGES RECONSTRUCTED (from which files):
  merge (pre-QC)   <- merged_all.psam / merged_all.pvar
  post --geno      <- merged_geno.psam / merged_geno.pvar
  post --mind      <- merged_qc.psam   / merged_qc.pvar
  post LD pruning  <- merged_pruned.psam / pruned.prune.in
                       (pruned variants are the ones in pruned.prune.in,
                       not in the .pvar, because --maf is reapplied in
                       that same step)

If a file is missing (e.g. --force not re-run yet, or step not yet
executed), that stage is simply omitted from the table with a warning
instead of failing.

USAGE:
  python3 qc_attrition_summary.py \
      --qc-dir /mnt/genome_datasets/qc_output_cohortA \
      --out /mnt/genome_datasets/qc_output_cohortA/qc_attrition.csv
"""

import argparse
from pathlib import Path

import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def count_lines(path: Path) -> int:
    with open(path) as f:
        return sum(1 for _ in f)


def n_samples_from_psam(path: Path) -> int:
    # .psam has one header line (#FID IID ...) plus one line per sample.
    return count_lines(path) - 1


def n_variants_from_pvar(path: Path) -> int:
    # .pvar has header lines starting with '#' (including the column
    # header) plus one line per variant.
    with open(path) as f:
        return sum(1 for line in f if not line.startswith("#"))


def n_variants_from_prune_in(path: Path) -> int:
    return count_lines(path)


STAGES = [
    # (label, psam_file_or_none, variant_file_or_none, variant_count_function)
    ("post-merge (pre-QC)", "merged_all.psam", "merged_all.pvar", n_variants_from_pvar),
    ("post --geno", "merged_geno.psam", "merged_geno.pvar", n_variants_from_pvar),
    ("post --mind", "merged_qc.psam", "merged_qc.pvar", n_variants_from_pvar),
    ("post LD pruning", "merged_pruned.psam", "pruned.prune.in", n_variants_from_prune_in),
]


def build_attrition_table(qc_dir: Path) -> pd.DataFrame:
    rows = []
    for label, psam_name, var_name, var_fn in STAGES:
        psam_path = qc_dir / psam_name
        var_path = qc_dir / var_name
        if not psam_path.exists() or not var_path.exists():
            print(f"  [skipping '{label}': missing {psam_path.name} or {var_path.name}]")
            continue
        rows.append({
            "stage": label,
            "n_samples": n_samples_from_psam(psam_path),
            "n_variants": var_fn(var_path),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["samples_dropped_step"] = df["n_samples"].shift(1) - df["n_samples"]
    df["variants_dropped_step"] = df["n_variants"].shift(1) - df["n_variants"]
    df.loc[0, ["samples_dropped_step", "variants_dropped_step"]] = 0
    df["samples_dropped_step"] = df["samples_dropped_step"].astype(int)
    df["variants_dropped_step"] = df["variants_dropped_step"].astype(int)

    n0_samples = df["n_samples"].iloc[0]
    n0_variants = df["n_variants"].iloc[0]
    df["pct_samples_remaining"] = (100 * df["n_samples"] / n0_samples).round(2)
    df["pct_variants_remaining"] = (100 * df["n_variants"] / n0_variants).round(2)

    return df


def plot_attrition(df: pd.DataFrame, out_path: Path) -> None:
    if plt is None or df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].bar(df["stage"], df["n_samples"], color="steelblue")
    axes[0].set_ylabel("N samples")
    axes[0].set_title("Samples per step")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(df["stage"], df["n_variants"], color="slategray")
    axes[1].set_ylabel("N variants")
    axes[1].set_title("Variants per step")
    axes[1].tick_params(axis="x", rotation=30)

    for ax in axes:
        for label in ax.get_xticklabels():
            label.set_ha("right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstructs the sample/variant attrition table from 00_run_plink_qc.sh's intermediate files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--qc-dir", required=True, type=Path, help="out_dir passed to 00_run_plink_qc.sh")
    parser.add_argument("--out", required=True, type=Path, help="output CSV path")
    args = parser.parse_args()

    print(f"==> Reading intermediate files from: {args.qc_dir}")
    df = build_attrition_table(args.qc_dir)

    if df.empty:
        print("No stage could be reconstructed: check that 00_run_plink_qc.sh has run in this folder.")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nTable saved to: {args.out}\n")
    print(df.to_string(index=False))

    plot_path = args.out.with_suffix(".png")
    plot_attrition(df, plot_path)
    if plt is not None:
        print(f"\nChart saved to: {plot_path}")
    else:
        print("\n(matplotlib not available, chart skipped)")


if __name__ == "__main__":
    main()
