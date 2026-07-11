"""
Prepara il dataset finale (merge genetica + ambientale) usato dal modeling.

Fix rispetto all'originale (data_loader.py):
  - RIMOSSA la riga `df_env['id'] = df_env['id'] + '_' + df_env['id']`.
    Era un workaround per far combaciare gli id del file ambientale con
    quelli (duplicati) del file genetico. Ora la normalizzazione avviene con
    `clean_sample_id` (utils/id_utils.py) applicata al file genetico, quindi
    il file ambientale resta con i suoi id originali, senza trasformazioni
    "magiche" difficili da ricordare/motivare a distanza di mesi.
  - Log invece di print(), con gli stessi checkpoint temporali dell'originale.
  - Validazione esplicita: se dopo il merge il numero di righe è 0, o la
    percentuale di id non matchati è alta, viene loggato un WARNING chiaro
    (prima si andava avanti in silenzio con un dataframe vuoto o quasi).
  - `SettingWithCopyWarning` potenziale su `df_gen.rename` risolto passando
    sempre per assegnazione esplicita.
"""
from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler

from gene_environment.config import Config, get_config
from gene_environment.logging_utils import get_logger
from gene_environment.utils.id_utils import clean_sample_id

log = get_logger(__name__)

NON_GEN_COLS = ["FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE", "id"]


def load_and_prepare_data(cfg: Config | None = None):
    cfg = cfg or get_config()

    fmt = cfg.raw_file_format
    if fmt == "auto":
        fmt = "parquet" if cfg.raw_file.endswith(".parquet") else "csv"

    log.info("Carico file genetica da %s (formato=%s)", cfg.raw_file, fmt)
    if fmt == "parquet":
        # Output diretto di build-matrix (vcf_to_parquet.py): l'id campione è
        # già l'indice del DataFrame, non una colonna.
        df_gen = pq.ParquetFile(
            cfg.raw_file,
            thrift_string_size_limit=2_000_000_000,
            thrift_container_size_limit=2_000_000_000,
        ).read(use_pandas_metadata=True).to_pandas()
        df_gen.index = df_gen.index.astype(str).map(clean_sample_id)
        df_gen.index.name = "id"
        df_gen = df_gen.reset_index()
    else:
        df_gen = pd.read_csv(cfg.raw_file, sep=cfg.sep, decimal=cfg.decimal, low_memory=False)
        if "IID" in df_gen.columns:
            df_gen = df_gen.rename(columns={"IID": "id"})
        if "id" in df_gen.columns:
            df_gen["id"] = df_gen["id"].astype(str).map(clean_sample_id)

    variant_cols = [c for c in df_gen.columns if c not in NON_GEN_COLS]
    log.info("Colonne varianti individuate: %d", len(variant_cols))

    log.info("Carico file ambientale da %s", cfg.env_file)
    df_env = pd.read_csv(cfg.env_file, sep=cfg.sep, decimal=cfg.decimal)
    df_env["id"] = df_env["id"].astype(str)
    if "sex" in df_env.columns:
        df_env["sex"] = df_env["sex"].astype("category")
    if "onset_site" in df_env.columns:
        df_env["onset_site"] = df_env["onset_site"].astype("category")

    log.info("Merge genetica <-> ambiente su 'id'")
    df = pd.merge(df_env, df_gen, on="id", how="inner")

    n_env, n_gen, n_merged = len(df_env), len(df_gen), len(df)
    log.info("Righe ambiente=%d, genetica=%d, dopo merge (inner)=%d", n_env, n_gen, n_merged)
    if n_merged == 0:
        log.warning(
            "Il merge ha prodotto 0 righe: nessun id in comune fra file ambientale e genetico. "
            "Controlla il formato degli id (prefissi genN_, duplicazioni XXX_XXX ecc.)."
        )
    elif n_merged < 0.5 * min(n_env, n_gen):
        log.warning(
            "Il merge ha 'perso' più del 50%% delle righe attese (%d su min(%d,%d)): "
            "verifica la coerenza degli id fra i due file.", n_merged, n_env, n_gen
        )

    # ---- Filtro per generazione, DOPO il join per id ----
    # Il file ambientale può non avere alcuna colonna di generazione (caso comune:
    # un unico ENV_FILE con tutti i pazienti, id come unica chiave di join). L'unica
    # fonte affidabile della coorte di un paziente è allora il VCF da cui proviene il
    # suo genotipo: build-matrix (vcf_to_parquet.py) produce una mappa id->generazione
    # che qui viene usata per tenere i run per gen1/gen2/gen3 indipendenti.
    map_path = cfg.sample_generation_map or os.path.join(cfg.output_folder, "sample_generation_map.csv")
    if os.path.exists(map_path):
        gen_map = pd.read_csv(map_path, dtype={"id": str})
        n_before = len(df)
        df = df.merge(gen_map, on="id", how="left")
        n_missing_map = df["generation"].isna().sum()
        if n_missing_map:
            log.warning(
                "%d pazienti dopo il merge non sono presenti nella mappa id->generazione (%s): "
                "verranno esclusi dal run (generazione sconosciuta).", n_missing_map, map_path,
            )
        df = df[df["generation"] == cfg.generation].drop(columns=["generation"])
        log.info(
            "Filtro per generazione=%s (mappa id->generazione da build-matrix): %d -> %d righe",
            cfg.generation, n_before, len(df),
        )
    elif cfg.env_generation_col and cfg.env_generation_col in df_env.columns:
        n_before = len(df)
        df = df[df[cfg.env_generation_col].astype(str) == str(cfg.generation)]
        log.info(
            "Filtro per generazione=%s (colonna '%s' nel file ambientale): %d -> %d righe",
            cfg.generation, cfg.env_generation_col, n_before, len(df),
        )
    else:
        log.warning(
            "Nessuna mappa id->generazione trovata (%s) e nessuna ENV_GENERATION_COL configurata: "
            "uso TUTTE le righe senza filtro per generazione. Se stai processando più coorti nello "
            "stesso pool genotipico, esegui prima 'build-matrix' (genera la mappa automaticamente) "
            "o imposta ENV_GENERATION_COL.", map_path,
        )

    if df.empty:
        log.warning("Dataset vuoto dopo il filtro per generazione=%s: controlla mappa/colonna generazione.", cfg.generation)

    log.info("Id unici post-merge/filtro: %d (righe totali: %d)", df["id"].nunique(), len(df))
    df = df.drop_duplicates("id")

    df[cfg.target_col] = pd.to_numeric(df[cfg.target_col], errors="coerce")

    log.info("Standardizzazione dell'esposizione '%s' (standardize=%s)", cfg.exposure, cfg.standardize)
    Ecols = []
    df[cfg.exposure] = pd.to_numeric(df[cfg.exposure], errors="coerce")
    if cfg.standardize:
        df[cfg.exposure + "_std"] = StandardScaler().fit_transform(df[[cfg.exposure]])
        Ecols.append(cfg.exposure + "_std")
    else:
        Ecols.append(cfg.exposure)

    log.info("Creo nomi 'safe' per le varianti (variant_i) per compatibilità con formule statsmodels")
    safe = {g: f"variant_{i}" for i, g in enumerate(variant_cols)}
    df = df.rename(columns=safe)
    variant_cols_safe = list(safe.values())
    mapping = {v: k for k, v in safe.items()}

    return df, variant_cols_safe, mapping, Ecols, variant_cols
