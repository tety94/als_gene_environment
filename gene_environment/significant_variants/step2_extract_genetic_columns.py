"""
SCRIPT 2/3 — estrae SOLO le colonne delle varianti significative (lista
prodotta dallo script 1) dal file genetico grande (cfg.raw_file), senza
caricare in memoria l'intera matrice (milioni di colonne). Scrive un file
piccolo (id + N varianti significative), generation-agnostico esattamente
come cfg.raw_file (la generazione si applica dopo, al merge con l'ambiente
— stesso schema di vcf_pipeline/build_dataset.py).

ATTENZIONE SULLA LETTURA COLONNE: pyarrow seleziona colonne per NOME dallo
schema del parquet. build_dataset.py legge oggi l'intero file e poi
recupera l'id campione dall'INDICE del DataFrame (non da una colonna
"id" esplicita nello schema) — per una lettura selettiva dobbiamo sapere
come si chiama quella colonna/indice nello schema fisico del file, che
varia a seconda di come è stato scritto (indice con nome vs
"__index_level_0__" generato automaticamente da pandas per indici senza
nome). Usa `--inspect-schema` PRIMA di lanciare l'estrazione vera, su
un file di cui ti fidi, per verificare che l'euristica sotto trovi la
colonna giusta — se non la trova, fallisce con un errore esplicito
invece di produrre un file silenziosamente senza id.
"""
from __future__ import annotations

import argparse
import os

import pandas as pd
import pyarrow.parquet as pq

from gene_environment.config import get_config
from gene_environment.logging_utils import configure_logging, get_logger
from gene_environment.utils.id_utils import clean_sample_id
from gene_environment.vcf_pipeline.build_dataset import NON_GEN_COLS

log = get_logger(__name__)

# Nomi tipici che pandas/pyarrow usano per materializzare l'indice come
# colonna fisica nel parquet quando l'indice non ha un nome esplicito.
_INDEX_LIKE_PREFIXES = ("__index_level_",)


def _detect_id_columns(schema_names: list[str]) -> list[str]:
    candidates = [c for c in schema_names if c in NON_GEN_COLS]
    candidates += [c for c in schema_names if c.startswith(_INDEX_LIKE_PREFIXES)]
    return candidates


def inspect_schema(raw_file: str) -> None:
    pf = pq.ParquetFile(raw_file, thrift_string_size_limit=2_000_000_000, thrift_container_size_limit=2_000_000_000)
    names = pf.schema.names
    id_cols = _detect_id_columns(names)
    print(f"Totale colonne nello schema: {len(names)}")
    print(f"Prime 10 colonne: {names[:10]}")
    print(f"Colonne id/indice rilevate euristicamente: {id_cols}")
    if not id_cols:
        print("ATTENZIONE: nessuna colonna id/indice riconosciuta -> l'estrazione fallirebbe. "
              "Guarda 'Prime 10 colonne' sopra e capisci a mano quale contiene l'id campione.")


def load_significant_labels(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def run(labels_path: str, out_path: str, raw_file: str | None = None) -> str:
    cfg = get_config()
    configure_logging(cfg.log_dir)
    raw_file = raw_file or cfg.raw_file

    labels = load_significant_labels(labels_path)
    if not labels:
        log.warning("Nessuna variante nella lista %s. Esco.", labels_path)
        return ""
    log.info("%d varianti da estrarre da %s", len(labels), raw_file)

    pf = pq.ParquetFile(raw_file, thrift_string_size_limit=2_000_000_000, thrift_container_size_limit=2_000_000_000)
    schema_names = set(pf.schema.names)
    id_cols = _detect_id_columns(pf.schema.names)
    if not id_cols:
        raise RuntimeError(
            f"Non trovo la colonna id/indice nello schema di {raw_file}. "
            f"Rilancia con --inspect-schema per vedere i nomi delle colonne ed eventualmente "
            f"aggiusta _detect_id_columns()."
        )

    present = [v for v in labels if v in schema_names]
    missing = [v for v in labels if v not in schema_names]
    if missing:
        log.warning("%d/%d varianti significative NON presenti in %s (filtrate a monte da MAF/LD-prune?): %s",
                    len(missing), len(labels), raw_file, missing[:20] if len(missing) > 20 else missing)
    if not present:
        log.warning("Nessuna variante significativa presente nel raw_file. Esco.")
        return ""

    columns_to_read = id_cols + present
    log.info("Leggo %d colonne (di cui %d id/indice) da %s", len(columns_to_read), len(id_cols), raw_file)
    table = pf.read(columns=columns_to_read, use_pandas_metadata=True)
    df = table.to_pandas()

    # normalizza l'id come fa build_dataset.py, qualunque sia il nome fisico
    # della colonna id trovata
    id_col = next((c for c in id_cols if c in df.columns), None)
    if id_col is None and df.index.name in id_cols:
        df = df.reset_index()
        id_col = df.columns[0]
    if id_col is None:
        raise RuntimeError(
            f"Le colonne id rilevate ({id_cols}) non sono finite né tra le colonne né nell'indice "
            f"del DataFrame risultante — controllo manuale necessario."
        )
    df = df.rename(columns={id_col: "id"})
    df["id"] = df["id"].astype(str).map(clean_sample_id)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_parquet(out_path, engine="pyarrow", index=False)
    log.info("Estrazione completata: %d campioni x %d varianti -> %s", df.shape[0], len(present), out_path)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=str, default="./output/significant_variants.txt")
    parser.add_argument("--out", type=str, default="./output/significant_genetic_matrix.parquet")
    parser.add_argument("--raw-file", type=str, default=None, help="Default: config.raw_file")
    parser.add_argument("--inspect-schema", action="store_true", help="Solo ispeziona lo schema ed esce, non estrae nulla")
    args = parser.parse_args()

    if args.inspect_schema:
        cfg = get_config()
        inspect_schema(args.raw_file or cfg.raw_file)
    else:
        run(args.labels, args.out, args.raw_file)