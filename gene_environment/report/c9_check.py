"""
build_c9_check.py

Pipeline:
1. Chiama la stored procedure `get_annotated_results()` -> ottiene
   (exposure, gene_name, variant, gna.*)
2. Filtra le colonne del parquet RAW_FILE tenendo solo quelle che
   corrispondono a `variant` o a `"char_" + variant` presenti in (1),
   più le colonne identificative dei campioni.
3. Salva un CSV "ristretto" (solo genotipi delle varianti annotate).
4. Fa il join con ENV_FILE (componenti ambientali) sulla chiave paziente/campione.
5. Salva il risultato finale in OUT_DIR.

"""

import logging
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

# Import della funzione già esistente nel tuo codebase.
# Aggiusta il path del modulo se get_annotated_results vive altrove
# (es. `from db_utils import get_annotated_results`)
from gene_environment.db.repository import get_annotated_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
RAW_FILE = "/mnt/cresla_prod/genome_datasets/merged_csv/gen.parquet"
ENV_FILE = "/srv/python-projects/gene_environment_v2/data/componenti_ambientali_full.csv"
OUT_DIR = Path("/mnt/cresla_prod/stefano_ge/c9_check")

RESTRICTED_CSV = OUT_DIR / "gen_restricted_variants.csv"
MERGED_CSV = OUT_DIR / "c9_check_merged.csv"

# Colonna identificativa: si usa la PRIMA colonna di ciascun file.
def first_column_name(path: str, is_parquet: bool) -> str:
    if is_parquet:
        return pq.ParquetFile(path,
                              thrift_string_size_limit=2_000_000_000,
                              thrift_container_size_limit=2_000_000_000,
                              ).schema.names[0]
    else:
        return pd.read_csv(path, nrows=0).columns[0]


ID_COLS_RAW = [first_column_name(RAW_FILE, is_parquet=True)]
ID_COLS_ENV = [first_column_name(ENV_FILE, is_parquet=False)]

# Se i nomi delle colonne id differiscono tra RAW e ENV, mappa qui:
# {"nome_in_env": "nome_in_raw"}
JOIN_RENAME_ENV_TO_RAW = {}


def get_target_columns(annotated_df: pd.DataFrame) -> set:
    """Costruisce l'insieme di nomi colonna da cercare nel parquet:
    sia `variant` sia `char_<variant>`."""
    variants = annotated_df["variant"].dropna().unique().tolist()
    targets = set(variants) | {f"char_{v}" for v in variants}
    return targets


def filter_parquet_columns(raw_file: str, target_variants: set, id_cols: list) -> list:
    """Legge solo lo schema del parquet (senza caricarlo tutto in memoria)
    e restituisce la lista di colonne da leggere davvero: id_cols + varianti
    che matchano target_variants."""
    schema_cols = pq.ParquetFile(raw_file,
                                 thrift_string_size_limit=2_000_000_000,
                                 thrift_container_size_limit=2_000_000_000,
                                 ).schema.names

    matched = [c for c in schema_cols if c in target_variants]
    missing_ids = [c for c in id_cols if c not in schema_cols]
    if missing_ids:
        raise KeyError(
            f"Le colonne ID {missing_ids} non esistono nel parquet. "
            f"Colonne disponibili (prime 20): {schema_cols[:20]}"
        )

    log.info("Colonne varianti trovate nel parquet: %d / %d target", len(matched), len(target_variants))
    if not matched:
        log.warning("Nessuna colonna variante ha fatto match. Controlla il formato dei nomi (char_ prefix?).")

    return id_cols + matched


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("ID colonna rilevata nel parquet (RAW): %s", ID_COLS_RAW[0])
    log.info("ID colonna rilevata in ENV_FILE: %s", ID_COLS_ENV[0])

    # 1. Stored procedure
    log.info("Chiamo get_annotated_results()...")
    annotated_df = get_annotated_results()
    if annotated_df.empty:
        raise RuntimeError("get_annotated_results() ha restituito 0 righe, controlla il DB.")
    log.info("Annotazioni ricevute: %d righe, %d colonne", *annotated_df.shape)

    # 2. Colonne target nel parquet
    target_variants = get_target_columns(annotated_df)
    selected_cols = filter_parquet_columns(RAW_FILE, target_variants, ID_COLS_RAW)

    # 3. Lettura ristretta del parquet e salvataggio CSV
    log.info("Leggo il parquet limitandomi a %d colonne...", len(selected_cols))
    raw_df = pd.read_parquet(RAW_FILE, columns=selected_cols)
    raw_df.to_csv(RESTRICTED_CSV, index=False)
    log.info("CSV ristretto salvato in %s (%d righe, %d colonne)", RESTRICTED_CSV, *raw_df.shape)

    # 4. Join con il file ambientale
    log.info("Carico ENV_FILE...")
    env_df = pd.read_csv(ENV_FILE)
    if JOIN_RENAME_ENV_TO_RAW:
        env_df = env_df.rename(columns=JOIN_RENAME_ENV_TO_RAW)

    missing_env_ids = [c for c in ID_COLS_ENV if c not in env_df.columns]
    if missing_env_ids:
        raise KeyError(
            f"Le colonne ID {missing_env_ids} non esistono in ENV_FILE. "
            f"Colonne disponibili: {list(env_df.columns)}"
        )

    merged = raw_df.merge(
        env_df,
        left_on=ID_COLS_RAW,
        right_on=ID_COLS_ENV,
        how="inner",  # cambia in "left" se vuoi tenere tutti i campioni del parquet
    )
    log.info("Join completato: %d righe, %d colonne", *merged.shape)

    # 5. Salvataggio finale
    merged.to_csv(MERGED_CSV, index=False)
    log.info("File finale salvato in %s", MERGED_CSV)


if __name__ == "__main__":
    main()