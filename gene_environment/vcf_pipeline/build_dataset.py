"""
Prepara il dataset finale (merge genetica + ambientale) usato dal modeling.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
from sklearn.preprocessing import StandardScaler

from gene_environment.config import Config, get_config
from gene_environment.logging_utils import get_logger
from gene_environment.utils.id_utils import clean_sample_id

log = get_logger(__name__)

NON_GEN_COLS = ["FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE", "id"]


def load_and_prepare_data(cfg: Config | None = None):
    cfg = cfg or get_config()

    log.info("Carico file genetica da %s", cfg.raw_file)
    df_gen = pd.read_csv(cfg.raw_file, sep=cfg.sep, decimal=cfg.decimal, low_memory=False)

    variant_cols = [c for c in df_gen.columns if c not in NON_GEN_COLS]
    log.info("Colonne varianti individuate: %d", len(variant_cols))

    if "IID" in df_gen.columns:
        df_gen = df_gen.rename(columns={"IID": "id"})

    if "id" in df_gen.columns:
        df_gen["id"] = df_gen["id"].astype(str).map(clean_sample_id)

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

    log.info("Id unici post-merge: %d (righe totali: %d)", df["id"].nunique(), len(df))
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
