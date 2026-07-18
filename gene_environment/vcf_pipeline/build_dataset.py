"""
Prepara il dataset finale (merge genetica + ambientale + PCA) usato dal modeling.

RISTRUTTURAZIONE (fix performance): il dataframe genetico (df_gen) ha
~1.3M colonne. Ogni merge/rename che lo tocca ricostruisce l'intero
BlockManager di pandas, quindi il costo NON dipende solo dalle righe ma
da quante volte l'oggetto largo viene "rimescolato". La versione
precedente lo toccava 3 volte (merge con env, merge con gen_map, merge
con PCA in orchestrator.py) + un rename. Qui invece:

  1. Tutte le parti "strette" (env, gen_map, PCA) vengono unite fra loro
     PRIMA -- sono piccole, quindi economico farlo quante volte serve.
  2. df_gen viene rinominato (safe names) una volta, subito dopo il
     caricamento.
  3. Un SOLO merge finale unisce il blocco di covariate strette con
     df_gen.

La logica di caricamento PCA (load_pca_covariates) resta in
pca_utils.py ma viene chiamata da qui invece che dall'orchestrator, cosi'
il merge con le PCA avviene sul dataframe stretto e non su quello largo.
merge_pca_covariates (in pca_utils.py) non e' piu' usata.
"""
from __future__ import annotations

import os

import pandas as pd
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler

from gene_environment.config import Config, get_config
from gene_environment.logging_utils import get_logger
from gene_environment.utils.id_utils import clean_sample_id
from gene_environment.utils.pca_utils import load_pca_covariates, PCA_ID_COLUMN

log = get_logger(__name__)

NON_GEN_COLS = ["FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE", "id"]


def _load_genetic_data(cfg: Config) -> tuple[pd.DataFrame, list[str], dict, list[str]]:
    fmt = cfg.raw_file_format
    if fmt == "auto":
        fmt = "parquet" if cfg.raw_file.endswith(".parquet") else "csv"

    log.info("Carico file genetica da %s (formato=%s)", cfg.raw_file, fmt)
    if fmt == "parquet":
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

    # Rename a nomi "safe" (variant_i) subito: unico touch del frame largo
    # per questa operazione, invece di farlo dopo altri merge.
    safe = {g: f"variant_{i}" for i, g in enumerate(variant_cols)}
    df_gen = df_gen.rename(columns=safe)
    variant_cols_safe = list(safe.values())
    mapping = {v: k for k, v in safe.items()}

    return df_gen, variant_cols_safe, mapping, variant_cols


def _build_narrow_covariates(cfg: Config, gen_ids: pd.Series) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Costruisce il blocco 'stretto' (env + mappa generazione + PCA),
    tutte operazioni economiche perche' i frame coinvolti sono piccoli.
    gen_ids e' passato solo a scopo diagnostico (log di quanti id
    combaciano), non viene mai fatto merge diretto con df_gen qui."""

    log.info("Carico file ambientale da %s", cfg.env_file)
    df_env = pd.read_csv(cfg.env_file, sep=cfg.sep, decimal=cfg.decimal)
    df_env["id"] = df_env["id"].astype(str)
    if "sex" in df_env.columns:
        df_env["sex"] = df_env["sex"].astype("category")
    if "onset_site" in df_env.columns:
        df_env["onset_site"] = df_env["onset_site"].astype("category")

    df = df_env

    # ---- Mappa id -> generazione ----
    map_path = cfg.sample_generation_map or os.path.join(cfg.output_folder, "sample_generation_map.csv")
    if os.path.exists(map_path):
        gen_map = pd.read_csv(map_path, dtype={"id": str})
        n_before = len(df)
        df = df.merge(gen_map, on="id", how="left")
        n_missing_map = df["generation"].isna().sum()
        if n_missing_map:
            log.warning(
                "%d pazienti non sono presenti nella mappa id->generazione (%s): "
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
            "uso TUTTE le righe senza filtro per generazione.", map_path,
        )

    df = df.drop_duplicates("id")

    # ---- Standardizzazione esposizione (sul frame stretto) ----
    df[cfg.target_col] = pd.to_numeric(df[cfg.target_col], errors="coerce")
    log.info("Standardizzazione dell'esposizione '%s' (standardize=%s)", cfg.exposure, cfg.standardize)
    Ecols = []
    df[cfg.exposure] = pd.to_numeric(df[cfg.exposure], errors="coerce")
    if cfg.standardize:
        df[cfg.exposure + "_std"] = StandardScaler().fit_transform(df[[cfg.exposure]])
        Ecols.append(cfg.exposure + "_std")
    else:
        Ecols.append(cfg.exposure)

    # ---- PCA (frame stretto, merge economico) ----
    covariate_cols: list[str] = []
    if cfg.use_pca_covariates:
        pca_df = load_pca_covariates(cfg.pca_covariates_path_template, cfg.generation, cfg.pca_n_components)

        if "id" in df.columns and PCA_ID_COLUMN not in df.columns:
            n_match = df["id"].isin(pca_df[PCA_ID_COLUMN]).sum()
            log.info("PCA: %d/%d id del blocco covariate trovano corrispondenza in IID.", n_match, len(df))
            df = df.merge(pca_df, left_on="id", right_on=PCA_ID_COLUMN, how="left")
        else:
            df = df.merge(pca_df, on=PCA_ID_COLUMN, how="left")

        covariate_cols = [c for c in pca_df.columns if c != PCA_ID_COLUMN]
        n_missing = int(df[covariate_cols[0]].isna().sum())
        if n_missing:
            pct = 100 * n_missing / len(df)
            log.warning(
                "PCA: %d/%d campioni (%.1f%%) senza corrispondenza dopo il merge (blocco stretto).",
                n_missing, len(df), pct,
            )
        else:
            log.info("PCA: merge completato, tutti i %d campioni hanno le PC.", len(df))
    else:
        log.info("PCA disattivate (cfg.use_pca_covariates=False): nessuna covariata di popolazione.")

    return df, Ecols, covariate_cols


def load_and_prepare_data(cfg: Config | None = None):
    cfg = cfg or get_config()

    df_gen, variant_cols_safe, mapping, variant_cols = _load_genetic_data(cfg)

    covariates, Ecols, covariate_cols = _build_narrow_covariates(cfg, df_gen["id"])

    log.info("Merge finale (unico touch del dataframe genetico largo) su 'id'")
    df = pd.merge(covariates, df_gen, on="id", how="inner")

    n_cov, n_gen, n_merged = len(covariates), len(df_gen), len(df)
    log.info("Righe covariate=%d, genetica=%d, dopo merge finale (inner)=%d", n_cov, n_gen, n_merged)
    if n_merged == 0:
        log.warning(
            "Il merge finale ha prodotto 0 righe: nessun id in comune fra covariate e genetica. "
            "Controlla il formato degli id (prefissi genN_, duplicazioni XXX_XXX ecc.)."
        )
    elif n_merged < 0.5 * min(n_cov, n_gen):
        log.warning(
            "Il merge finale ha 'perso' più del 50%% delle righe attese (%d su min(%d,%d)): "
            "verifica la coerenza degli id fra i file.", n_merged, n_cov, n_gen
        )

    log.info("Id unici post-merge: %d (righe totali: %d)", df["id"].nunique(), len(df))

    return df, variant_cols_safe, mapping, Ecols, variant_cols, covariate_cols