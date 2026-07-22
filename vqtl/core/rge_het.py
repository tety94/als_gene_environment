"""
Step 6 - rGE (gene-environment correlation) e test di eteroschedasticita',
per ogni SNP candidato x esposizione:
  1. rGE: SNP_dosage ~ esposizione + covariate (OLS, HC3). Un coefficiente
     dell'esposizione significativo suggerisce che genotipo ed esposizione
     non sono indipendenti in questo campione -- i SNP flaggati NON vengono
     scartati, solo segnalati (come nell'originale).
  2. Eteroschedasticita': fenotipo ~ SNP + esposizione + covariate (SENZA
     interazione), test di Breusch-Pagan sui residui. Un BP significativo
     segnala che le SE del modello di interazione (Step 5) vanno lette con
     le versioni robuste (gia' l'impostazione di default li' -- vedi
     `interaction.py`).

Come per lo Step 5, il dosaggio e' gia' una colonna del DataFrame: niente
piu' rilettura dei VCF (`extract_snp_dosage` duplicato nell'originale).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from statsmodels.stats.diagnostic import het_breuschpagan

from gene_environment.logging_utils import get_logger

from vqtl.config import VqtlConfig
from vqtl.core.data import VqtlDataset, dosage_matrix

log = get_logger(__name__)


def _rge_test(dosage, exposure, covariates):
    df = pd.DataFrame({"snp": dosage, "exposure": exposure})
    for i, c in enumerate(covariates.T):
        df[f"cov{i}"] = c
    df = df.dropna()
    if len(df) < 20 or df["snp"].nunique() < 2:
        return None
    cov_cols = [c for c in df.columns if c.startswith("cov")]
    X = sm.add_constant(df[["exposure"] + cov_cols])
    try:
        model = sm.OLS(df["snp"], X).fit(cov_type="HC3")
    except Exception:
        return None
    return {
        "beta_exposure_on_snp": float(model.params["exposure"]),
        "SE": float(model.bse["exposure"]),
        "pval": float(model.pvalues["exposure"]),
        "N": len(df),
    }


def _het_test(y, dosage, exposure, covariates):
    df = pd.DataFrame({"y": y, "snp": dosage, "exposure": exposure})
    for i, c in enumerate(covariates.T):
        df[f"cov{i}"] = c
    df = df.dropna()
    if len(df) < 20 or df["snp"].nunique() < 2:
        return None
    cov_cols = [c for c in df.columns if c.startswith("cov")]
    X = sm.add_constant(df[["snp", "exposure"] + cov_cols])
    try:
        model = sm.OLS(df["y"], X).fit()
        lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(model.resid, model.model.exog)
    except Exception:
        return None
    return {"BP_lm_stat": lm_stat, "BP_lm_pvalue": lm_pvalue, "BP_f_stat": f_stat, "BP_f_pvalue": f_pvalue, "N": len(df)}


def run_rge_het(
    dataset: VqtlDataset, vcfg: VqtlConfig, candidates: pd.DataFrame, target_col: str, generation: int,
) -> pd.DataFrame:
    from vqtl.db import repository as repo

    if candidates.empty:
        log.warning("Nessun candidato: nessun test rGE/eteroschedasticita' da eseguire.")
        return pd.DataFrame()

    inv_mapping = {v: k for k, v in dataset.mapping.items()}
    y = dataset.df[target_col].to_numpy(dtype=float)
    covariates = dataset.df[dataset.covariate_cols].to_numpy(dtype=float) if dataset.covariate_cols else np.zeros((len(dataset.df), 0))

    placeholder_rows = [
        {"variant": row["SNP"], "exposure": exp, "chromosome": row["CHR"], "position": row["POS"]}
        for _, row in candidates.iterrows() for exp in dataset.exposure_std_cols
    ]
    repo.ensure_placeholders("rge_het", generation, placeholder_rows)
    done_keys = repo.get_done_keys("rge_het", generation)

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
            tasks.append((snp_id, exp_raw, dosage, exposure_vals))

    log.info(
        "Step 6 - rGE/eteroschedasticita': %d combinazioni da calcolare (%d gia' fatte)",
        len(tasks), len(done_keys),
    )

    def _run_one(snp_id, exp_raw, dosage, exposure_vals):
        rge = _rge_test(dosage, exposure_vals, covariates)
        het = _het_test(y, dosage, exposure_vals, covariates)
        rge_flag = (rge is not None) and (rge["pval"] < vcfg.rge_het_alpha)
        het_flag = (het is not None) and (het["BP_lm_pvalue"] < vcfg.rge_het_alpha)
        return {
            "variant": snp_id, "exposure": exp_raw, "status": "done", "error_message": None,
            "rge_beta_exposure_on_snp": rge["beta_exposure_on_snp"] if rge else None,
            "rge_se": rge["SE"] if rge else None,
            "rge_pval": rge["pval"] if rge else None,
            "rge_flag": rge_flag,
            "het_bp_lm_stat": het["BP_lm_stat"] if het else None,
            "het_bp_lm_pvalue": het["BP_lm_pvalue"] if het else None,
            "het_bp_f_stat": het["BP_f_stat"] if het else None,
            "het_bp_f_pvalue": het["BP_f_pvalue"] if het else None,
            "heteroscedasticity_flag": het_flag,
        }

    if tasks:
        new_rows = Parallel(n_jobs=vcfg.n_jobs, backend="loky")(delayed(_run_one)(*t) for t in tasks)
        repo.bulk_update_status("rge_het", generation, new_rows)

    out_df = repo.fetch_results("rge_het", generation)
    if not out_df.empty:
        out_df = out_df.rename(columns={
            "rge_beta_exposure_on_snp": "rGE_beta_exposure_on_snp", "rge_se": "rGE_SE", "rge_pval": "rGE_pval",
            "rge_flag": "rGE_flag", "het_bp_lm_stat": "het_BP_lm_stat", "het_bp_lm_pvalue": "het_BP_lm_pvalue",
            "het_bp_f_stat": "het_BP_f_stat", "het_bp_f_pvalue": "het_BP_f_pvalue",
        })
        n_rge = int(out_df["rGE_flag"].fillna(False).astype(bool).sum())
        n_het = int(out_df["heteroscedasticity_flag"].fillna(False).astype(bool).sum())
        log.info("rGE flaggati (p<%.2g): %d/%d", vcfg.rge_het_alpha, n_rge, len(out_df))
        log.info("Eteroschedasticita' flaggata (BP p<%.2g): %d/%d", vcfg.rge_het_alpha, n_het, len(out_df))
        if n_rge:
            log.warning("I SNP flaggati per rGE NON vanno letti come prova definitiva di G x E senza ulteriori analisi.")

    return out_df
