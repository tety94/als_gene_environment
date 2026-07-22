"""
Step 5 - test di interazione SNP x esposizione, per ogni SNP candidato
(Step 4) e ogni esposizione in vcfg.exposures:
    fenotipo ~ SNP + esposizione + SNP*esposizione + covariate
via OLS (o logit se il fenotipo e' binario, mantenuto generico come
nell'originale anche se onset_age e' quantitativo). SE robuste
all'eteroschedasticita' (HC3/HC1) di default -- vedi audit nel README storico.

UNICA differenza rispetto all'originale: il dosaggio del SNP candidato e la
covariata sono gia' colonne del DataFrame costruito da
`core.data.load_vqtl_dataset` (join fatto una volta sola a monte da
gene_environment). Non serve piu' `extract_snp_dosage` (rilettura dei VCF
per estrarre solo le SNP candidate, duplicata identica in step5/6/7/8 del
progetto originale) -- e' la semplificazione piu' grande resa possibile dal
riuso della matrice genotipica gia' costruita.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed

from gene_environment.logging_utils import get_logger

from vqtl.config import VqtlConfig
from vqtl.core.data import VqtlDataset, dosage_matrix

log = get_logger(__name__)


def is_binary(series: pd.Series) -> bool:
    vals = series.dropna().unique()
    return set(np.round(vals, 6)).issubset({0.0, 1.0}) and len(vals) <= 2


def fit_interaction(
    y: np.ndarray, snp: np.ndarray, exposure: np.ndarray, covariates: np.ndarray,
    binary_outcome: bool, robust: bool = True,
) -> dict | None:
    df = pd.DataFrame({"y": y, "snp": snp, "exposure": exposure})
    for i, c in enumerate(covariates.T):
        df[f"cov{i}"] = c
    df["snp_x_exp"] = df["snp"] * df["exposure"]
    df = df.dropna()
    if len(df) < 20 or df["snp"].nunique() < 2:
        return None

    cov_cols = [c for c in df.columns if c.startswith("cov")]
    X = sm.add_constant(df[["snp", "exposure", "snp_x_exp"] + cov_cols])
    y_fit = df["y"]
    try:
        if binary_outcome:
            model = sm.Logit(y_fit, X).fit(disp=0, maxiter=200, cov_type="HC1" if robust else "nonrobust")
        else:
            model = sm.OLS(y_fit, X).fit(cov_type="HC3" if robust else "nonrobust")
    except Exception:
        return None

    if "snp_x_exp" not in model.params:
        return None
    maf = np.nanmean(df["snp"]) / 2
    maf = min(maf, 1 - maf)
    return {
        "beta_I": float(model.params["snp_x_exp"]),
        "SE": float(model.bse["snp_x_exp"]),
        "pval": float(model.pvalues["snp_x_exp"]),
        "N": len(df),
        "MAF": round(float(maf), 4),
    }


def run_interaction_tests(
    dataset: VqtlDataset, vcfg: VqtlConfig, candidates: pd.DataFrame, target_col: str, generation: int,
) -> pd.DataFrame:
    from vqtl.db import repository as repo

    if candidates.empty:
        log.warning("Nessun candidato dallo Step 4: nessun test di interazione da eseguire.")
        return pd.DataFrame(columns=["SNP", "CHR", "POS", "exposure", "beta_I", "SE", "pval", "N", "MAF"])

    inv_mapping = {v: k for k, v in dataset.mapping.items()}
    y = dataset.df[target_col].to_numpy(dtype=float)
    covariates = dataset.df[dataset.covariate_cols].to_numpy(dtype=float) if dataset.covariate_cols else np.zeros((len(dataset.df), 0))
    binary_outcome = is_binary(dataset.df[target_col])
    log.info("Fenotipo '%s' rilevato come %s", target_col, "binario (logit)" if binary_outcome else "quantitativo (OLS)")

    # ---- placeholder per ogni coppia SNP candidata x esposizione ----
    placeholder_rows = [
        {"variant": row["SNP"], "exposure": exp, "chromosome": row["CHR"], "position": row["POS"]}
        for _, row in candidates.iterrows() for exp in dataset.exposure_std_cols
    ]
    repo.ensure_placeholders("interaction", generation, placeholder_rows)
    done_keys = repo.get_done_keys("interaction", generation)

    tasks = []
    for _, row in candidates.iterrows():
        snp_id = row["SNP"]
        safe_col = inv_mapping.get(snp_id)
        if safe_col is None:
            continue
        dosage = dosage_matrix(dataset, [safe_col])[:, 0]  # bugfix: gestisce i "." (genotipo mancante) come NaN, come fa scan.py
        for exp_raw, exp_std_col in dataset.exposure_std_cols.items():
            if (snp_id, exp_raw) in done_keys:
                continue
            exposure_vals = dataset.df[exp_std_col].to_numpy(dtype=float)
            tasks.append((snp_id, row["CHR"], row["POS"], exp_raw, dosage, exposure_vals))

    log.info(
        "Step 5 - test di interazione: %d combinazioni da calcolare (%d gia' fatte in un run precedente), n_jobs=%s",
        len(tasks), len(done_keys), vcfg.n_jobs,
    )

    def _run_one(snp_id, chrom, pos, exp_raw, dosage, exposure_vals):
        res = fit_interaction(y, dosage, exposure_vals, covariates, binary_outcome, robust=vcfg.robust_se)
        row = {"variant": snp_id, "exposure": exp_raw, "status": "done", "error_message": None}
        if res is None:
            row.update({"beta_i": None, "se": None, "pval": None, "n": None, "maf": None})
        else:
            row.update({"beta_i": res["beta_I"], "se": res["SE"], "pval": res["pval"], "n": res["N"], "maf": res["MAF"]})
        return row

    if tasks:
        new_rows = Parallel(n_jobs=vcfg.n_jobs, backend="loky")(delayed(_run_one)(*t) for t in tasks)
        repo.bulk_update_status("interaction", generation, new_rows)

    repo.sync_interaction_significant(generation, vcfg.interaction_sig_threshold)

    out_df = repo.fetch_results("interaction", generation)
    if not out_df.empty:
        out_df = out_df.rename(columns={"beta_i": "beta_I", "pval": "pval", "n": "N", "maf": "MAF", "se": "SE"})
        out_df = out_df.dropna(subset=["pval"])
        out_df = out_df[["SNP", "CHR", "POS", "exposure", "beta_I", "SE", "pval", "N", "MAF"]]
        out_df = out_df.sort_values("pval").reset_index(drop=True)
    log.info("Step 5 completato: %d risultati.", len(out_df))
    return out_df
