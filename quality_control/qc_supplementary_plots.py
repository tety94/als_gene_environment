#!/usr/bin/env python3
"""
qc_supplementary_plots.py
===========================
Grafici supplementari per il paper, letti dai file gia' prodotti da
00_run_plink_qc.sh (missingness) e da 01_run_extra_qc_checks.sh
(sex_check.sexcheck, heterozygosity.het, maf.afreq). Nessuno di questi
file viene ricalcolato qui: lo script si limita a leggerli e plottarli.
Ogni grafico viene saltato singolarmente (con avviso) se il file di input
non e' presente, invece di far fallire l'intero script.

USO:
  python3 qc_supplementary_plots.py \
      --qc-dir /mnt/cresla_prod/genome_datasets/qc_output \
      --out-dir /mnt/cresla_prod/genome_datasets/qc_output/supplementary_plots \
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


def _read_plink_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+")
    df.columns = [c.lstrip("#") for c in df.columns]
    return df


def plot_missingness(qc_dir: Path, out_dir: Path, geno_thresh: float, mind_thresh: float) -> None:
    vmiss_path = qc_dir / "missingness.vmiss"
    smiss_path = qc_dir / "missingness.smiss"

    if not vmiss_path.exists() or not smiss_path.exists():
        print(f"  [salto missingness: {vmiss_path.name} o {smiss_path.name} non trovati]")
        return

    vmiss = _read_plink_table(vmiss_path)
    smiss = _read_plink_table(smiss_path)

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
    print(f"  Salvato: {out_path}")

    # Nota: una distribuzione bimodale in vmiss (blocco intorno a F_MISS
    # ~1/n_batch) e' la firma di siti presenti solo in alcuni batch -- vedi
    # commento nello Step 5 di 00_run_plink_qc.sh.
    frac_above_geno = (vmiss["F_MISS"] > geno_thresh).mean()
    frac_above_mind = (smiss["F_MISS"] > mind_thresh).mean()
    print(f"  Varianti sopra soglia --geno: {100*frac_above_geno:.2f}%")
    print(f"  Campioni sopra soglia --mind: {100*frac_above_mind:.2f}%")


def plot_sex_check(qc_dir: Path, out_dir: Path) -> None:
    path = qc_dir / "sex_check.sexcheck"
    if not path.exists():
        print(f"  [salto sex-check: {path.name} non trovato -- lancia prima 01_run_extra_qc_checks.sh]")
        return

    df = _read_plink_table(path)
    if "F" not in df.columns or "STATUS" not in df.columns:
        print(f"  [salto sex-check: colonne attese (F, STATUS) non trovate in {path}]")
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
    print(f"  Salvato: {out_path}  ({n_problem} PROBLEM su {len(df)} campioni)")

    if n_problem:
        flagged_path = out_dir / "sex_check_flagged_samples.csv"
        df.loc[problem].to_csv(flagged_path, index=False)
        print(f"  Campioni flaggati salvati in: {flagged_path}")


def plot_heterozygosity(qc_dir: Path, out_dir: Path, sd_threshold: float = 3.0) -> None:
    path = qc_dir / "heterozygosity.het"
    if not path.exists():
        print(f"  [salto heterozygosity: {path.name} non trovato -- lancia prima 01_run_extra_qc_checks.sh]")
        return

    df = _read_plink_table(path)
    if "F" not in df.columns:
        print(f"  [salto heterozygosity: colonna F non trovata in {path}]")
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
    print(f"  Salvato: {out_path}  ({len(outliers)} outlier su {len(df)} campioni)")

    if len(outliers) > 0:
        outliers_path = out_dir / "heterozygosity_outlier_samples.csv"
        outliers.to_csv(outliers_path, index=False)
        print(f"  Campioni outlier salvati in: {outliers_path}")


def plot_maf_spectrum(qc_dir: Path, out_dir: Path) -> None:
    path = qc_dir / "maf.afreq"
    if not path.exists():
        print(f"  [salto MAF spectrum: {path.name} non trovato -- lancia prima 01_run_extra_qc_checks.sh]")
        return

    df = _read_plink_table(path)
    freq_col = "ALT_FREQS" if "ALT_FREQS" in df.columns else None
    if freq_col is None:
        print(f"  [salto MAF spectrum: colonna ALT_FREQS non trovata in {path}. Colonne: {list(df.columns)}]")
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
    print(f"  Salvato: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grafici supplementari QC (missingness, sex-check, heterozygosity, MAF spectrum)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--qc-dir", required=True, type=Path, help="out_dir della pipeline (00_run_plink_qc.sh)")
    parser.add_argument("--out-dir", required=True, type=Path, help="directory dove salvare i grafici")
    parser.add_argument("--geno-thresh", type=float, default=0.05, help="soglia --geno usata nella pipeline (default 0.05)")
    parser.add_argument("--mind-thresh", type=float, default=0.05, help="soglia --mind usata nella pipeline (default 0.05)")
    parser.add_argument("--het-sd-threshold", type=float, default=3.0, help="soglia in SD per outlier di eterozigosita' (default 3)")
    args = parser.parse_args()

    if plt is None:
        print("ERRORE: matplotlib non disponibile, impossibile generare grafici.")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("==> Missingness (per variante e per campione)")
    plot_missingness(args.qc_dir, args.out_dir, args.geno_thresh, args.mind_thresh)

    print("\n==> Sex check")
    plot_sex_check(args.qc_dir, args.out_dir)

    print("\n==> Heterozygosity check")
    plot_heterozygosity(args.qc_dir, args.out_dir, args.het_sd_threshold)

    print("\n==> MAF spectrum")
    plot_maf_spectrum(args.qc_dir, args.out_dir)

    print(f"\n==> FATTO. Output in: {args.out_dir}")


if __name__ == "__main__":
    main()
