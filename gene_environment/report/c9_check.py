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

NOTE DA VERIFICARE (marcate anche sotto con TODO):
- ID_COLS_RAW: nome/i colonna/e identificative nel parquet (es. sample id).
- ID_COLS_ENV: nome/i colonna/e identificative in ENV_FILE per il join.
  Se i nomi non coincidono tra i due file, usa JOIN_MAP per rinominare.
"""

import logging
import os
import subprocess
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

# Import della funzione già esistente nel tuo codebase.
# Aggiusta il path del modulo se get_annotated_results vive altrove
# (es. `from db_utils import get_annotated_results`)
from gene_environment.db.repository import get_annotated_results
from gene_environment.utils.id_utils import clean_sample_id

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
CODICE_GEN_FILE = OUT_DIR / "codice_gen.csv"
VCF_FILE_GEN1 = "/mnt/cresla_prod/genome_datasets/gen1/gen1_onlycases_vcf_chr22.vcf.gz"
VCF_FILE_GEN2 = "/mnt/cresla_prod/genome_datasets/gen2/gen2_vcf_chr22.vcf.gz"
VARIANT_COLS_FILE = OUT_DIR / "variant_columns.json"  # lista colonne-varianti, riusata da generation_stats.py

# Nel parquet l'id campione NON è una colonna: è l'indice del DataFrame
# (come in _load_genetic_data). Nel CSV ambientale invece si usa la prima
# colonna, come colonna vera e propria.
ID_COL_RAW = "id"  # nome che diamo all'indice dopo il reset_index()


def first_column_name(path: str) -> str:
    return pd.read_csv(path, nrows=0).columns[0]


ID_COLS_RAW = [ID_COL_RAW]
ID_COLS_ENV = [first_column_name(ENV_FILE)]

# Se i nomi delle colonne id differiscono tra RAW e ENV, mappa qui:
# {"nome_in_env": "nome_in_raw"}
JOIN_RENAME_ENV_TO_RAW = {}


def get_target_columns(annotated_df: pd.DataFrame) -> set:
    """Costruisce l'insieme di nomi colonna da cercare nel parquet:
    sia `variant` sia `char_<variant>`."""
    variants = annotated_df["variant"].dropna().unique().tolist()
    targets = set(variants) | {f"char_{v}" for v in variants}
    return targets


def filter_parquet_columns(raw_file: str, target_variants: set) -> list:
    """Legge solo lo schema del parquet (senza caricarlo tutto in memoria)
    e restituisce la lista di colonne varianti che matchano target_variants.
    L'id campione NON è tra queste colonne: è l'indice, gestito a parte
    in fase di lettura (use_pandas_metadata=True)."""
    schema_cols = pq.ParquetFile(
        raw_file,
        thrift_string_size_limit=2_000_000_000,
        thrift_container_size_limit=2_000_000_000,
    ).schema.names

    matched = [c for c in schema_cols if c in target_variants]

    log.info("Colonne varianti trovate nel parquet: %d / %d target", len(matched), len(target_variants))
    if not matched:
        log.warning("Nessuna colonna variante ha fatto match. Controlla il formato dei nomi (char_ prefix?).")

    return matched


def get_vcf_sample_ids(vcf_path: str) -> set:
    """Legge SOLO l'header del VCF (bcftools query -l), niente scrittura,
    niente modifica del file. Applica clean_sample_id per uniformare gli id
    doppi (es. ACH10008_ACH10008 -> ACH10008)."""
    if not os.path.exists(vcf_path):
        raise FileNotFoundError(f"VCF non trovato: {vcf_path}")

    result = subprocess.run(
        ["bcftools", "query", "-l", vcf_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"bcftools query -l fallito su {vcf_path} (exit {result.returncode}).\n"
            f"stderr: {result.stderr.strip()}"
        )

    raw_ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    cleaned = {clean_sample_id(rid) for rid in raw_ids}
    log.info("Sample id trovati nel VCF: %d (esempio raw: %s)", len(raw_ids), raw_ids[:1])
    return cleaned


def json_dump_list(path: Path, items: list) -> None:
    import json
    with open(path, "w") as f:
        json.dump(items, f, indent=2)



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
    selected_cols = filter_parquet_columns(RAW_FILE, target_variants)

    # 3. Lettura ristretta del parquet e salvataggio CSV
    # NB: l'indice (sample id) viene ricostruito automaticamente da
    # use_pandas_metadata=True anche se non è tra le `columns` richieste,
    # perché pyarrow lo traccia separatamente dai dati.
    log.info("Leggo il parquet limitandomi a %d colonne (+ indice)...", len(selected_cols))
    pf = pq.ParquetFile(
        RAW_FILE,
        thrift_string_size_limit=2_000_000_000,
        thrift_container_size_limit=2_000_000_000,
    )
    raw_df = pf.read(columns=selected_cols, use_pandas_metadata=True).to_pandas()
    raw_df.index = raw_df.index.astype(str).map(clean_sample_id)
    raw_df.index.name = ID_COL_RAW
    raw_df = raw_df.reset_index()

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

    # 5. Join con codice_gen.csv per attaccare la colonna parals_codals
    log.info("Carico CODICE_GEN_FILE (%s)...", CODICE_GEN_FILE)
    codice_gen_df = pd.read_csv(CODICE_GEN_FILE)
    if "corretto" not in codice_gen_df.columns:
        raise KeyError(
            f"La colonna 'corretto' non esiste in {CODICE_GEN_FILE}. "
            f"Colonne disponibili: {list(codice_gen_df.columns)}"
        )
    if "parals_codals" not in codice_gen_df.columns:
        raise KeyError(
            f"La colonna 'parals_codals' non esiste in {CODICE_GEN_FILE}. "
            f"Colonne disponibili: {list(codice_gen_df.columns)}"
        )

    codice_gen_df["corretto"] = codice_gen_df["corretto"].astype(str)
    merged[ID_COL_RAW] = merged[ID_COL_RAW].astype(str)

    merged = merged.merge(
        codice_gen_df[["corretto", "parals_codals"]],
        left_on=ID_COL_RAW,
        right_on="corretto",
        how="left",  # tiene tutte le righe di merged anche senza match
    )
    merged = merged.drop(columns=["corretto"])
    log.info("Aggiunta parals_codals: %d righe, %d colonne", *merged.shape)

    # 6. Colonna 'generation': 1 se l'id è nel VCF gen1, 2 se è nel VCF gen2.
    # Le righe il cui id non compare in NESSUNO dei due VCF vengono scartate.
    log.info("Leggo sample id dal VCF gen1 (%s)...", VCF_FILE_GEN1)
    vcf1_ids = get_vcf_sample_ids(VCF_FILE_GEN1)
    log.info("Leggo sample id dal VCF gen2 (%s)...", VCF_FILE_GEN2)
    vcf2_ids = get_vcf_sample_ids(VCF_FILE_GEN2)

    both = vcf1_ids & vcf2_ids
    if both:
        log.warning("%d id compaiono in ENTRAMBI i VCF (gen1 e gen2): %s", len(both), sorted(both)[:5])

    def assign_generation(x):
        in1 = x in vcf1_ids
        in2 = x in vcf2_ids
        if in1 and not in2:
            return 1
        if in2 and not in1:
            return 2
        if in1 and in2:
            return 2  # ambiguo: presente in entrambi, priorità a gen2 (vedi warning sopra)
        return None  # non trovato in nessuno dei due VCF

    merged[ID_COL_RAW] = merged[ID_COL_RAW].astype(str)
    merged["generation"] = merged[ID_COL_RAW].apply(assign_generation)

    n_before = len(merged)
    n_dropped = merged["generation"].isna().sum()
    merged = merged.dropna(subset=["generation"]).copy()
    merged["generation"] = merged["generation"].astype(int)

    log.info(
        "Righe scartate perché l'id non è in nessun VCF: %d / %d",
        n_dropped, n_before,
    )
    log.info(
        "Generation assegnata: gen1=%d, gen2=%d",
        (merged["generation"] == 1).sum(),
        (merged["generation"] == 2).sum(),
    )

    # Salvo la lista delle colonne-varianti per riuso in generation_stats.py
    json_dump_list(VARIANT_COLS_FILE, selected_cols)
    log.info("Lista colonne-varianti salvata in %s (%d colonne)", VARIANT_COLS_FILE, len(selected_cols))

    # 7. Salvataggio finale
    merged.to_csv(MERGED_CSV, index=False)
    log.info("File finale salvato in %s", MERGED_CSV)


if __name__ == "__main__":
    main()