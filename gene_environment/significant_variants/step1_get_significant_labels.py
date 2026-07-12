"""
SCRIPT 1/3 — estrae la lista di varianti significative (stored procedure
`get_significant_results`, coorte 1 e 2 affiancate) e la salva su un file
di testo, una variante per riga (label originale CHROM_POS_MUTATION).

Nessuna scrittura a DB, nessun genotipo: solo la lista di nomi. Input per
lo script 2 (extract_significant_genetic_columns.py).
"""
from __future__ import annotations

import argparse
import os

from gene_environment.config import get_config
from gene_environment.db.repository import get_significant_results
from gene_environment.logging_utils import configure_logging, get_logger

log = get_logger(__name__)


def get_significant_labels(exposure: str | None = None) -> list[str]:
    df = get_significant_results(exposure=exposure)
    if df.empty:
        return []
    return sorted(df["variant"].unique().tolist())


def run(exposure: str | None, out_path: str) -> str:
    cfg = get_config()
    configure_logging(cfg.log_dir)

    labels = get_significant_labels(exposure)
    log.info("%d varianti significative trovate (exposure=%s)", len(labels), exposure or cfg.exposure)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(labels) + ("\n" if labels else ""))
    log.info("Lista salvata in %s", out_path)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exposure", type=str, default=None, help="Default: config.exposure")
    parser.add_argument("--out", type=str, default="./output/significant_variants.txt")
    args = parser.parse_args()
    run(args.exposure, args.out)