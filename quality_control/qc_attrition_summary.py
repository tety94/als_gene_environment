#!/usr/bin/env python3
"""
qc_attrition_summary.py
========================
Ricostruisce la tabella "campioni/varianti a ogni step" leggendo i file
intermedi che 00_run_plink_qc.sh lascia sul disco (non li ricalcola, non
tocca la pipeline bash). Utile come tabella supplementare per il paper:
mostra dove e quanto si perde a ogni stage del QC.

STEP RICOSTRUITI (da quali file):
  merge (pre-QC)          <- merged_all.psam / merged_all.pvar
  post --geno             <- merged_geno.psam / merged_geno.pvar
  post --mind             <- merged_qc.psam   / merged_qc.pvar
  post LD pruning          <- merged_pruned.psam / pruned.prune.in
                               (le varianti pruned sono quelle in
                               pruned.prune.in, non nel .pvar, perche'
                               --maf viene riapplicato nello stesso step)

Se qualche file manca (es. --force non ancora rilanciato, o step non
ancora eseguito), lo step viene semplicemente omesso dalla tabella con un
avviso, invece di fallire.

USO:
  python3 qc_attrition_summary.py \
      --qc-dir /mnt/cresla_prod/genome_datasets/qc_output \
      --out /mnt/cresla_prod/genome_datasets/qc_output/qc_attrition.csv
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
    # .psam ha una riga di header (#FID IID ...) piu' una riga per campione.
    return count_lines(path) - 1


def n_variants_from_pvar(path: Path) -> int:
    # .pvar ha righe di header che iniziano con '#' (incluso l'header
    # colonne) piu' una riga per variante.
    with open(path) as f:
        return sum(1 for line in f if not line.startswith("#"))


def n_variants_from_prune_in(path: Path) -> int:
    return count_lines(path)


STAGES = [
    # (etichetta, file_psam_o_none, file_varianti_o_none, funzione_conteggio_varianti)
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
            print(f"  [salto '{label}': manca {psam_path.name} o {var_path.name}]")
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
        description="Ricostruisce la tabella di attrition campioni/varianti dai file intermedi di 00_run_plink_qc.sh",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--qc-dir", required=True, type=Path, help="out_dir passato a 00_run_plink_qc.sh")
    parser.add_argument("--out", required=True, type=Path, help="path del CSV di output")
    args = parser.parse_args()

    print(f"==> Leggo file intermedi da: {args.qc_dir}")
    df = build_attrition_table(args.qc_dir)

    if df.empty:
        print("Nessuno step ricostruibile: controlla che 00_run_plink_qc.sh sia stato eseguito in questa cartella.")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nTabella salvata in: {args.out}\n")
    print(df.to_string(index=False))

    plot_path = args.out.with_suffix(".png")
    plot_attrition(df, plot_path)
    if plt is not None:
        print(f"\nGrafico salvato in: {plot_path}")
    else:
        print("\n(matplotlib non disponibile, grafico saltato)")


if __name__ == "__main__":
    main()
