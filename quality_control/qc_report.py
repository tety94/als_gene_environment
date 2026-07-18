#!/usr/bin/env python3
"""
qc_report.py
============
Controlli QC su kinship (KING) e PCA prodotti da 00_run_plink_qc.sh:

  1. Distribuzione del kinship (king.kin0): istogramma + classificazione
     delle coppie per grado di parentela usando le soglie standard KING,
     con una tabella separata delle coppie sospette (>= parenti di 3o
     grado), evidenziando in particolare i casi cross-batch (possibili
     duplicati/pazienti condivisi tra batch sfuggiti al controllo ID
     dello Step 0 della pipeline bash).

  2. PCA (pca.eigenvec) colorata per batch di provenienza, per un check
     visivo di batch effect. La mappa campione -> batch viene derivata
     interrogando bcftools sui VCF originali (stesso identico criterio
     usato nello Step 0 di 00_run_plink_qc.sh), quindi richiede che
     bcftools sia nel PATH e che le directory dei batch siano ancora
     accessibili.

REQUISITI: python3 con pandas, numpy, matplotlib. bcftools nel PATH se
si vuole il grafico PCA colorato per batch (altrimenti usare
--no-batch-plot per un semplice scatter PC1 vs PC2 senza colore).

USO:
  python3 qc_report.py \
      --kin /mnt/cresla_prod/genome_datasets/qc_output/king.kin0 \
      --eigenvec /mnt/cresla_prod/genome_datasets/qc_output/pca.eigenvec \
      --eigenval /mnt/cresla_prod/genome_datasets/qc_output/pca.eigenval \
      --vcf-dirs /mnt/cresla_prod/genome_datasets/gen1 \
                 /mnt/cresla_prod/genome_datasets/gen2 \
                 /mnt/cresla_prod/genome_datasets/gen3 \
      --use-filtered \
      --out-dir /mnt/cresla_prod/genome_datasets/qc_output/qc_report

Se non vuoi/puoi derivare i batch (es. le directory VCF non sono piu'
accessibili da qui), ometti --vcf-dirs: lo script fa comunque tutta la
parte di kinship e un semplice scatter PCA senza colore per batch.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # nessun display: si lavora su un server headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Soglie standard KING per il grado di parentela (kinship coefficient).
# Fonte: documentazione KING / plink2 (Manichaikul et al. 2010).
KING_THRESHOLDS = [
    (0.354, "duplicato/gemello monozigote"),
    (0.177, "parente di 1o grado"),
    (0.0884, "parente di 2o grado"),
    (0.0442, "parente di 3o grado"),
]

# Etichette inglesi parallele, usate SOLO per il testo del grafico (la
# classificazione interna/CSV resta in italiano per non rompere i confronti
# di stringa gia' usati altrove in questo file).
KING_THRESHOLDS_LABELS_EN = {
    "duplicato/gemello monozigote": "duplicate/monozygotic twin",
    "parente di 1o grado": "1st-degree relative",
    "parente di 2o grado": "2nd-degree relative",
    "parente di 3o grado": "3rd-degree relative",
}


def classify_kinship(k: float) -> str:
    for threshold, label in KING_THRESHOLDS:
        if k >= threshold:
            return label
    return "non imparentati"


# ---------------------------------------------------------------------------
# Kinship
# ---------------------------------------------------------------------------

def load_kinship(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+")
    df.columns = [c.lstrip("#") for c in df.columns]
    # Nomi colonna storicamente un po' incoerenti tra versioni plink2:
    # a volte "ID1"/"ID2", a volte "IID1"/"IID2". Normalizziamo.
    rename = {"ID1": "IID1", "ID2": "IID2"}
    df = df.rename(columns=rename)
    required = {"IID1", "IID2", "NSNP", "KINSHIP"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Colonne attese mancanti in {path}: {missing}. "
            f"Colonne trovate: {list(df.columns)}"
        )
    return df


def summarize_kinship(df: pd.DataFrame, batch_map: dict, out_dir: Path) -> None:
    df = df.copy()
    df["categoria"] = df["KINSHIP"].apply(classify_kinship)

    if batch_map:
        df["batch1"] = df["IID1"].map(batch_map)
        df["batch2"] = df["IID2"].map(batch_map)
        df["stesso_batch"] = df["batch1"] == df["batch2"]
        n_missing_batch = df["batch1"].isna().sum() + df["batch2"].isna().sum()
        if n_missing_batch:
            print(
                f"  ATTENZIONE: {n_missing_batch} riferimenti campione in king.kin0 "
                f"non trovati nella mappa batch (ID non corrispondenti?)."
            )

    counts = df["categoria"].value_counts()
    print("\n--- Distribuzione coppie per grado di parentela (KING) ---")
    for _, label in KING_THRESHOLDS + [(0, "non imparentati")]:
        n = counts.get(label, 0)
        print(f"  {label:35s}: {n}")

    # Tabella supplementare: stessa distribuzione ma su file, pronta per il paper.
    order = [label for _, label in KING_THRESHOLDS] + ["non imparentati"]
    counts_df = pd.DataFrame(
        {"categoria": order, "n_coppie": [int(counts.get(c, 0)) for c in order]}
    )
    counts_df["pct_coppie"] = 100 * counts_df["n_coppie"] / counts_df["n_coppie"].sum()
    counts_path = out_dir / "kinship_category_counts.csv"
    counts_df.to_csv(counts_path, index=False)
    print(f"  Tabella distribuzione salvata in: {counts_path}")

    # Tabella delle coppie sospette: parenti di 3o grado o piu' stretti.
    flagged = df[df["categoria"] != "non imparentati"].sort_values(
        "KINSHIP", ascending=False
    )
    flagged_path = out_dir / "kinship_flagged_pairs.csv"
    flagged.to_csv(flagged_path, index=False)
    print(f"\n  Coppie con parentela >= 3o grado salvate in: {flagged_path}")
    print(f"  Totale coppie segnalate: {len(flagged)}")

    if batch_map:
        dup_or_close = flagged[
            flagged["categoria"].isin(
                ["duplicato/gemello monozigote", "parente di 1o grado"]
            )
        ]
        cross_batch_suspect = dup_or_close[dup_or_close["stesso_batch"] == False]
        if len(cross_batch_suspect) > 0:
            print(
                f"\n  >>> ATTENZIONE: {len(cross_batch_suspect)} coppie con kinship da "
                f"duplicato/1o grado APPARTENGONO A BATCH DIVERSI."
            )
            print(
                "  >>> Lo Step 0 della pipeline bash controlla solo ID identici: "
                "questi casi hanno ID diversi ma DNA quasi identico -- probabile "
                "stesso paziente genotipizzato in due batch con ID differenti. "
                "Vanno verificati manualmente prima di procedere con l'analisi."
            )
            cross_path = out_dir / "kinship_cross_batch_duplicates_suspect.csv"
            cross_batch_suspect.to_csv(cross_path, index=False)
            print(f"  >>> Dettaglio salvato in: {cross_path}")
        else:
            print(
                "\n  Nessuna coppia sospetta di duplicato/1o grado tra batch diversi."
            )

    # Istogramma della distribuzione del kinship, con le soglie segnate.
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
            threshold, ax.get_ylim()[1] * 0.9, KING_THRESHOLDS_LABELS_EN.get(label, label), rotation=90,
            color=color, fontsize=8, ha="right", va="top",
        )
    fig.tight_layout()
    hist_path = out_dir / "kinship_distribution.png"
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    print(f"\n  Istogramma salvato in: {hist_path}")


# ---------------------------------------------------------------------------
# Mappa campione -> batch (stesso criterio dello Step 0 della pipeline bash)
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
            "  ATTENZIONE: bcftools non trovato nel PATH -- impossibile derivare "
            "la mappa campione->batch. Il grafico PCA verra' fatto senza colore "
            "per batch. Attiva l'ambiente conda giusto (es. 'conda activate "
            "geneenv') e rilancia se vuoi il check per batch effect."
        )
        return {}

    mapping: dict[str, str] = {}
    for d in vcf_dirs:
        batch = d.name
        vcf = find_chr1_vcf(d, use_filtered)
        if vcf is None:
            print(f"  ATTENZIONE: nessun VCF chr1 trovato per il batch {batch} in {d}, salto.")
            continue
        result = subprocess.run(
            ["bcftools", "query", "-l", str(vcf)],
            capture_output=True, text=True, check=True,
        )
        samples = [s for s in result.stdout.splitlines() if s]
        for s in samples:
            if s in mapping and mapping[s] != batch:
                print(
                    f"  ATTENZIONE: campione {s} presente sia nel batch "
                    f"{mapping[s]} che in {batch} (ID duplicato tra batch)."
                )
            mapping[s] = batch
        print(f"  {batch}: {len(samples)} campioni mappati")
    return mapping


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def load_eigenvec(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+")
    df.columns = [c.lstrip("#") for c in df.columns]
    if "IID" not in df.columns:
        raise ValueError(
            f"Colonna IID non trovata in {path}. Colonne trovate: {list(df.columns)}"
        )
    return df


def load_eigenval(path: Path | None) -> list[float] | None:
    if path is None or not path.exists():
        return None
    with open(path) as f:
        return [float(line.strip()) for line in f if line.strip()]


def plot_pca(df: pd.DataFrame, batch_map: dict, eigenval: list[float] | None, out_dir: Path) -> None:
    if "PC1" not in df.columns or "PC2" not in df.columns:
        print("  ATTENZIONE: PC1/PC2 non trovate in eigenvec, salto il grafico PCA.")
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
                f"  ATTENZIONE: {n_unmapped} campioni in eigenvec senza batch "
                f"corrispondente (ID non trovati nella mappa)."
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
    print(f"\n  Scatter PCA salvato in: {scatter_path}")

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
        print(f"  Scree plot salvato in: {scree_path}")

    # Check numerico semplice di batch effect: quanta varianza di PC1/PC2 e'
    # "spiegata" dall'appartenenza al batch (eta-squared, one-way ANOVA-like).
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
                f"  Frazione di varianza di {pc} spiegata dal batch (eta^2): {eta_sq:.3f} "
                f"({'ALTA -- possibile batch effect da correggere' if eta_sq > 0.1 else 'bassa'})"
            )
            eta_rows.append({
                "PC": pc,
                "eta_squared": round(float(eta_sq), 4),
                "batch_effect_alto": bool(eta_sq > 0.1),
            })
        if eta_rows:
            eta_path = out_dir / "pca_batch_eta2.csv"
            pd.DataFrame(eta_rows).to_csv(eta_path, index=False)
            print(f"  Tabella eta^2 salvata in: {eta_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="QC su kinship (KING) e PCA prodotti da 00_run_plink_qc.sh",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--kin", required=True, type=Path, help="path a king.kin0")
    parser.add_argument("--eigenvec", required=True, type=Path, help="path a pca.eigenvec")
    parser.add_argument("--eigenval", type=Path, default=None, help="path a pca.eigenval (opzionale)")
    parser.add_argument(
        "--vcf-dirs", nargs="+", type=Path, default=None,
        help="directory dei batch VCF originali (per derivare la mappa campione->batch)",
    )
    parser.add_argument(
        "--use-filtered", action="store_true",
        help="usa vcf_filtered/*_filtered.vcf.gz per il chr1, come in 00_run_plink_qc.sh --use-filtered",
    )
    parser.add_argument(
        "--out-dir", required=True, type=Path,
        help="directory dove salvare grafici e tabelle",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("==> Carico kinship")
    kin_df = load_kinship(args.kin)
    print(f"  {len(kin_df):,} coppie caricate da {args.kin}")

    batch_map = {}
    if args.vcf_dirs:
        print("\n==> Derivo la mappa campione -> batch (via bcftools, come Step 0 della pipeline)")
        batch_map = get_batch_sample_map(args.vcf_dirs, args.use_filtered)
        if not batch_map:
            print("  Nessuna mappa batch disponibile, procedo senza.")

    print("\n==> Analisi kinship")
    summarize_kinship(kin_df, batch_map, args.out_dir)

    print("\n==> Carico PCA")
    eigen_df = load_eigenvec(args.eigenvec)
    eigenval = load_eigenval(args.eigenval)
    print(f"  {len(eigen_df):,} campioni caricati da {args.eigenvec}")

    print("\n==> Grafico PCA")
    plot_pca(eigen_df, batch_map, eigenval, args.out_dir)

    print(f"\n==> FATTO. Output in: {args.out_dir}")


if __name__ == "__main__":
    main()