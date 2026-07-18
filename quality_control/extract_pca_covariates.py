#!/usr/bin/env python3
"""
extract_pca_covariates.py
==========================
Estrae le prime N componenti principali da pca.eigenvec (calcolata da
00_run_plink_qc.sh sull'intera coorte fusa, gen1+gen2 insieme) e le salva
in un CSV pronto per il merge con i dati di esposizione/fenotipo, da usare
come covariate nel modello di interazione gene-ambiente.

La PCA e' gia' stata calcolata su TUTTA la coorte insieme (non batch per
batch): questo e' l'input corretto, non serve rifare nulla a monte. Vedi
pca.eigenvec prodotto dallo Step 8 della pipeline bash.

USO:
  python3 extract_pca_covariates.py \
      --eigenvec /mnt/cresla_prod/genome_datasets/qc_output/pca.eigenvec \
      --n-pcs 10 \
      --out /mnt/cresla_prod/genome_datasets/qc_output/pca_covariates.csv

Il CSV di output ha una riga per campione, colonne: IID, PC1 ... PC10
(rinominale in "sample_id" con --id-column-name se il tuo dataframe di
esposizione usa un altro nome per la chiave di merge).
"""

import argparse
from pathlib import Path

import pandas as pd


def load_eigenvec(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+")
    df.columns = [c.lstrip("#") for c in df.columns]
    if "IID" not in df.columns:
        raise ValueError(
            f"Colonna IID non trovata in {path}. Colonne trovate: {list(df.columns)}"
        )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estrae le prime N PC da pca.eigenvec e le salva come covariate CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--eigenvec", required=True, type=Path, help="path a pca.eigenvec")
    parser.add_argument(
        "--n-pcs", type=int, default=10,
        help="numero di componenti principali da estrarre (default 10, il massimo calcolato dalla pipeline)",
    )
    parser.add_argument("--out", required=True, type=Path, help="path del CSV di output")
    parser.add_argument(
        "--id-column-name", default="IID",
        help="nome da dare alla colonna identificativo campione nel CSV di output (default: IID)",
    )
    args = parser.parse_args()

    df = load_eigenvec(args.eigenvec)

    pc_cols = [f"PC{i}" for i in range(1, args.n_pcs + 1)]
    missing_pcs = [c for c in pc_cols if c not in df.columns]
    if missing_pcs:
        available = sorted(
            (c for c in df.columns if c.startswith("PC")),
            key=lambda c: int(c[2:]),
        )
        raise ValueError(
            f"Richieste {args.n_pcs} PC ma mancano: {missing_pcs}. "
            f"PC disponibili in {args.eigenvec}: {available}"
        )

    out_df = df[["IID"] + pc_cols].copy()
    if args.id_column_name != "IID":
        out_df = out_df.rename(columns={"IID": args.id_column_name})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    print(f"Campioni: {len(out_df):,}")
    print(f"Componenti estratte: PC1 ... PC{args.n_pcs}")
    print(f"Salvato in: {args.out}")
    print("\nAnteprima:")
    print(out_df.head().to_string(index=False))


if __name__ == "__main__":
    main()