#!/usr/bin/env python3
"""
qc_report.py
============
QC checks on kinship (KING) and PCA produced by 00_run_plink_qc.sh:

  1. Kinship distribution (king.kin0): histogram + classification of
     pairs by degree of relatedness using the standard KING thresholds,
     with a separate table of suspicious pairs (>= 3rd-degree relatives),
     specifically flagging cross-batch cases (possible duplicates/shared
     patients between batches that slipped past the ID check in Step 0
     of the bash pipeline).

  2. PCA (pca.eigenvec) colored by batch of origin, as a visual batch-
     effect check. The sample -> batch map is derived by querying
     bcftools on the original VCFs (the same criterion used in Step 0 of
     00_run_plink_qc.sh), so it requires bcftools in PATH and the batch
     directories to still be accessible.

     Note: if this cohort was genotyped in a single batch (the common
     case -- see 00_run_plink_qc.sh's note on "batch" vs. "cohort"), the
     batch-effect check is not meaningful (everything will be one color)
     and can be skipped with --no-batch-plot; the kinship section (within-
     cohort duplicate/relatedness check) and a plain PCA scatter still run
     regardless.

REQUIREMENTS: python3 with pandas, numpy, matplotlib. bcftools in PATH if
you want the batch-colored PCA plot (otherwise use --no-batch-plot for a
plain PC1 vs PC2 scatter with no color).

USAGE:
  python3 qc_report.py \
      --kin /mnt/genome_datasets/qc_output_cohortA/king.kin0 \
      --eigenvec /mnt/genome_datasets/qc_output_cohortA/pca.eigenvec \
      --eigenval /mnt/genome_datasets/qc_output_cohortA/pca.eigenval \
      --vcf-dirs /mnt/genome_datasets/cohortA_batch1 /mnt/genome_datasets/cohortA_batch2 \
      --use-filtered \
      --out-dir /mnt/genome_datasets/qc_output_cohortA/qc_report

If you don't want/can't derive batches (e.g. the VCF directories are no
longer accessible from here), omit --vcf-dirs: the script still runs the
full kinship section and a plain PCA scatter with no batch coloring.
"""

import argparse
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display: this runs on a headless server
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plink_io import load_eigenval, load_eigenvec, load_kinship

# Standard KING thresholds for degree of relatedness (kinship
# coefficient). Source: KING / plink2 documentation (Manichaikul et al.
# 2010).
KING_THRESHOLDS = [
    (0.354, "duplicate/monozygotic twin"),
    (0.177, "1st-degree relative"),
    (0.0884, "2nd-degree relative"),
    (0.0442, "3rd-degree relative"),
]


def classify_kinship(k: float) -> str:
    for threshold, label in KING_THRESHOLDS:
        if k >= threshold:
            return label
    return "unrelated"


# ---------------------------------------------------------------------------
# Kinship
# ---------------------------------------------------------------------------

def summarize_kinship(df: pd.DataFrame, batch_map: dict, out_dir: Path) -> None:
    df = df.copy()
    df["category"] = df["KINSHIP"].apply(classify_kinship)

    if batch_map:
        df["batch1"] = df["IID1"].map(batch_map)
        df["batch2"] = df["IID2"].map(batch_map)
        df["same_batch"] = df["batch1"] == df["batch2"]
        n_missing_batch = df["batch1"].isna().sum() + df["batch2"].isna().sum()
        if n_missing_batch:
            print(
                f"  WARNING: {n_missing_batch} sample references in king.kin0 "
                f"were not found in the batch map (mismatched IDs?)."
            )

    counts = df["category"].value_counts()
    print("\n--- Pair distribution by degree of relatedness (KING) ---")
    for _, label in KING_THRESHOLDS + [(0, "unrelated")]:
        n = counts.get(label, 0)
        print(f"  {label:35s}: {n}")

    # Supplementary table: same distribution, on file, ready for the paper.
    order = [label for _, label in KING_THRESHOLDS] + ["unrelated"]
    counts_df = pd.DataFrame(
        {"category": order, "n_pairs": [int(counts.get(c, 0)) for c in order]}
    )
    counts_df["pct_pairs"] = 100 * counts_df["n_pairs"] / counts_df["n_pairs"].sum()
    counts_path = out_dir / "kinship_category_counts.csv"
    counts_df.to_csv(counts_path, index=False)
    print(f"  Distribution table saved to: {counts_path}")

    # Table of suspicious pairs: 3rd-degree relatives or closer.
    flagged = df[df["category"] != "unrelated"].sort_values(
        "KINSHIP", ascending=False
    )
    flagged_path = out_dir / "kinship_flagged_pairs.csv"
    flagged.to_csv(flagged_path, index=False)
    print(f"\n  Pairs with relatedness >= 3rd degree saved to: {flagged_path}")
    print(f"  Total flagged pairs: {len(flagged)}")

    if batch_map:
        dup_or_close = flagged[
            flagged["category"].isin(
                ["duplicate/monozygotic twin", "1st-degree relative"]
            )
        ]
        cross_batch_suspect = dup_or_close[dup_or_close["same_batch"] == False]
        if len(cross_batch_suspect) > 0:
            print(
                f"\n  >>> WARNING: {len(cross_batch_suspect)} pairs with duplicate/1st-degree "
                f"kinship BELONG TO DIFFERENT BATCHES."
            )
            print(
                "  >>> Step 0 of the bash pipeline only checks for identical IDs: "
                "these cases have different IDs but near-identical DNA -- likely "
                "the same patient genotyped in two batches under different IDs. "
                "These need manual verification before proceeding with the analysis."
            )
            cross_path = out_dir / "kinship_cross_batch_duplicates_suspect.csv"
            cross_batch_suspect.to_csv(cross_path, index=False)
            print(f"  >>> Details saved to: {cross_path}")
        else:
            print(
                "\n  No cross-batch duplicate/1st-degree suspect pairs found."
            )

    # Histogram of the kinship distribution, with thresholds marked.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df["KINSHIP"], bins=200, color="steelblue", edgecolor="none")
    ax.set_yscale("log")
    ax.set_xlabel("KINSHIP (KING-robust coefficient)")
    ax.set_ylabel("N pairs (log scale)")
    ax.set_title(f"Kinship distribution -- {len(df):,} total pairs")
    colors = ["red", "orange", "goldenrod", "green"]
    for (threshold, label), color in zip(KING_THRESHOLDS, colors):
        ax.axvline(threshold, color=color, linestyle="--", linewidth=1)
        ax.text(
            threshold, ax.get_ylim()[1] * 0.9, label, rotation=90,
            color=color, fontsize=8, ha="right", va="top",
        )
    fig.tight_layout()
    hist_path = out_dir / "kinship_distribution.png"
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    print(f"\n  Histogram saved to: {hist_path}")


# ---------------------------------------------------------------------------
# Sample -> batch map (same criterion as Step 0 of the bash pipeline)
# ---------------------------------------------------------------------------

def find_chr1_vcf(vcf_dir: Path, use_filtered: bool) -> Path | None:
    if use_filtered:
        search_dir = vcf_dir / "vcf_filtered"
        matches = sorted(search_dir.glob("*chr1_filtered.vcf.gz"))
    else:
        search_dir = vcf_dir
        matches = sorted(search_dir.glob("*chr1.vcf.gz"))
    return matches[0] if matches else None


def get_batch_sample_map(vcf_dirs: list[Path], use_filtered: bool) -> dict:
    if not vcf_dirs:
        return {}

    try:
        subprocess.run(
            ["bcftools", "--version"], capture_output=True, check=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        print(
            "  WARNING: bcftools not found in PATH -- cannot derive the "
            "sample->batch map. The PCA plot will be produced without batch "
            "coloring. Activate the right conda environment and re-run if "
            "you want the batch-effect check."
        )
        return {}

    mapping: dict[str, str] = {}
    for d in vcf_dirs:
        batch = d.name
        vcf = find_chr1_vcf(d, use_filtered)
        if vcf is None:
            print(f"  WARNING: no chr1 VCF found for batch {batch} in {d}, skipping.")
            continue
        result = subprocess.run(
            ["bcftools", "query", "-l", str(vcf)],
            capture_output=True, text=True, check=True,
        )
        samples = [s for s in result.stdout.splitlines() if s]
        for s in samples:
            if s in mapping and mapping[s] != batch:
                print(
                    f"  WARNING: sample {s} is present in both batch "
                    f"{mapping[s]} and {batch} (duplicate ID across batches)."
                )
            mapping[s] = batch
        print(f"  {batch}: {len(samples)} samples mapped")
    return mapping


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def plot_pca(df: pd.DataFrame, batch_map: dict, eigenval: list[float] | None, out_dir: Path) -> None:
    if "PC1" not in df.columns or "PC2" not in df.columns:
        print("  WARNING: PC1/PC2 not found in eigenvec, skipping the PCA plot.")
        return

    if eigenval:
        total = sum(eigenval)
        pct1 = 100 * eigenval[0] / total if total else float("nan")
        pct2 = 100 * eigenval[1] / total if total else float("nan")
        xlabel = f"PC1 ({pct1:.1f}% variance)"
        ylabel = f"PC2 ({pct2:.1f}% variance)"
    else:
        xlabel, ylabel = "PC1", "PC2"

    fig, ax = plt.subplots(figsize=(8, 7))

    if batch_map:
        df = df.copy()
        df["batch"] = df["IID"].map(batch_map)
        n_unmapped = df["batch"].isna().sum()
        if n_unmapped:
            print(
                f"  WARNING: {n_unmapped} samples in eigenvec have no matching "
                f"batch (IDs not found in the map)."
            )
        for batch, group in df.groupby("batch", dropna=False):
            label = batch if pd.notna(batch) else "unknown batch"
            ax.scatter(group["PC1"], group["PC2"], s=8, alpha=0.6, label=f"{label} (n={len(group)})")
        ax.legend(fontsize=8, loc="best")
        title = "PCA colored by batch -- tight clusters indicate a batch effect"
    else:
        ax.scatter(df["PC1"], df["PC2"], s=8, alpha=0.6, color="steelblue")
        title = "PCA (no batch map available)"

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    scatter_path = out_dir / "pca_scatter_by_batch.png"
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)
    print(f"\n  PCA scatter saved to: {scatter_path}")

    if eigenval:
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        pcs = np.arange(1, len(eigenval) + 1)
        total = sum(eigenval)
        pct = [100 * e / total for e in eigenval] if total else eigenval
        ax2.bar(pcs, pct, color="slategray")
        ax2.set_xlabel("principal component")
        ax2.set_ylabel("% variance explained")
        ax2.set_title("Scree plot")
        ax2.set_xticks(pcs)
        fig2.tight_layout()
        scree_path = out_dir / "pca_scree_plot.png"
        fig2.savefig(scree_path, dpi=150)
        plt.close(fig2)
        print(f"  Scree plot saved to: {scree_path}")

    # Simple numeric batch-effect check: how much of PC1/PC2's variance is
    # "explained" by batch membership (eta-squared, one-way ANOVA-like).
    if batch_map and "batch" in df.columns:
        eta_rows = []
        for pc in ["PC1", "PC2"]:
            valid = df.dropna(subset=[pc, "batch"])
            if valid["batch"].nunique() < 2:
                continue
            grand_mean = valid[pc].mean()
            ss_total = ((valid[pc] - grand_mean) ** 2).sum()
            ss_between = sum(
                len(g) * (g[pc].mean() - grand_mean) ** 2
                for _, g in valid.groupby("batch")
            )
            eta_sq = ss_between / ss_total if ss_total else float("nan")
            print(
                f"  Fraction of {pc} variance explained by batch (eta^2): {eta_sq:.3f} "
                f"({'HIGH -- possible batch effect to correct for' if eta_sq > 0.1 else 'low'})"
            )
            eta_rows.append({
                "PC": pc,
                "eta_squared": round(float(eta_sq), 4),
                "high_batch_effect": bool(eta_sq > 0.1),
            })
        if eta_rows:
            eta_path = out_dir / "pca_batch_eta2.csv"
            pd.DataFrame(eta_rows).to_csv(eta_path, index=False)
            print(f"  eta^2 table saved to: {eta_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="QC on kinship (KING) and PCA produced by 00_run_plink_qc.sh",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--kin", required=True, type=Path, help="path to king.kin0")
    parser.add_argument("--eigenvec", required=True, type=Path, help="path to pca.eigenvec")
    parser.add_argument("--eigenval", type=Path, default=None, help="path to pca.eigenval (optional)")
    parser.add_argument(
        "--vcf-dirs", nargs="+", type=Path, default=None,
        help="original batch VCF directories for THIS cohort (to derive the sample->batch map)",
    )
    parser.add_argument(
        "--use-filtered", action="store_true",
        help="use vcf_filtered/*_filtered.vcf.gz for chr1, as in 00_run_plink_qc.sh --use-filtered",
    )
    parser.add_argument(
        "--out-dir", required=True, type=Path,
        help="directory where charts and tables are saved",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("==> Loading kinship")
    kin_df = load_kinship(args.kin)
    print(f"  {len(kin_df):,} pairs loaded from {args.kin}")

    batch_map = {}
    if args.vcf_dirs:
        print("\n==> Deriving the sample -> batch map (via bcftools, as in Step 0 of the bash pipeline)")
        batch_map = get_batch_sample_map(args.vcf_dirs, args.use_filtered)
        if not batch_map:
            print("  No batch map available, proceeding without it.")

    print("\n==> Kinship analysis")
    summarize_kinship(kin_df, batch_map, args.out_dir)

    print("\n==> Loading PCA")
    eigen_df = load_eigenvec(args.eigenvec)
    eigenval = load_eigenval(args.eigenval)
    print(f"  {len(eigen_df):,} samples loaded from {args.eigenvec}")

    print("\n==> PCA plot")
    plot_pca(eigen_df, batch_map, eigenval, args.out_dir)

    print(f"\n==> DONE. Output in: {args.out_dir}")


if __name__ == "__main__":
    main()
