"""
Step 7 - robustezza e permutazioni sui top loci.

1. Ri-esegue il test di interazione dello Step 5 sui top loci con
   trasformazioni alternative del fenotipo (log, rank-inverse-normal,
   outlier rimossi |z|>3) per valutare la stabilita' di beta_I.
2. Test di permutazione Freedman-Lane (invariato dall'audit del progetto
   originale, vedi README storico punto 3): permuta i RESIDUI del modello
   ridotto (y ~ SNP + esposizione + covariate, senza interazione), li
   riaggiunge ai valori stimati dello stesso modello ridotto, poi rifitta il
   modello COMPLETO (con interazione) sull'esito permutato. Preserva la
   struttura di effetto principale sotto H0 ("nessuna interazione"), a
   differenza di permutare direttamente il fenotipo grezzo (distrugge anche
   la relazione covariate<->fenotipo, che non fa parte di H0).
3. NELLO STESSO loop del punto 2 (stessa iterazione per locus, non uno step
   separato): test di Levene permutazionale sulla varianza del fenotipo
   residualizzato per gruppo genotipico, a conferma assumption-light
   dell'effetto di varianza rilevato dallo scan QUAIL del Step 3 per questo
   locus. Procedura: si calcola la statistica di Levene osservata sui gruppi
   genotipo REALI (dosaggio arrotondato 0/1/2) del fenotipo residualizzato;
   si permutano le ETICHETTE di genotipo (non i residui, a differenza del
   punto 2) tenendo fisso il fenotipo residualizzato; si ricalcola la
   statistica sui gruppi permutati; si ripete N_PERM volte per costruire una
   distribuzione nulla empirica; la p-value e' la frazione di statistiche
   permutate >= quella osservata. Qualsiasi asimmetria/non-normalita' della
   distribuzione reale del fenotipo e' automaticamente assorbita nel null
   (costruito dagli stessi dati), a differenza del test di Levene "da
   manuale" che assume normalita' asintotica. Non e' una simulazione a
   parte: usa la stessa infrastruttura di parallelizzazione (stessi
   n_splits/joblib) del punto 2, dentro lo stesso ciclo sui top loci. Il
   risultato dipende solo dal genotipo (non dall'esposizione): per un locus
   con piu' esposizioni testate, viene calcolato una volta sola e riusato
   per le righe successive dello stesso SNP.

Come per Step 5/6, il dosaggio arriva da una colonna del DataFrame gia'
costruito (niente rilettura VCF).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from scipy import stats

from gene_environment.logging_utils import get_logger

from vqtl.config import VqtlConfig
from vqtl.core.data import VqtlDataset
from vqtl.core.interaction import fit_interaction, is_binary
from vqtl.core.scan import residualize

log = get_logger(__name__)


def _fit_reduced_model(y, dosage, exposure_vals, covariates):
    df = pd.DataFrame({"y": y, "snp": dosage, "exposure": exposure_vals})
    for i, c in enumerate(covariates.T):
        df[f"cov{i}"] = c
    df = df.dropna()
    cov_cols = [c for c in df.columns if c.startswith("cov")]
    X = sm.add_constant(df[["snp", "exposure"] + cov_cols])
    model = sm.OLS(df["y"], X).fit()
    return model.fittedvalues.values, model.resid.values, df.index.values


def _levene_stat(genotype: np.ndarray, r: np.ndarray, min_group_size: int = 2) -> float:
    """Statistica di Levene (center='median', cioe' Brown-Forsythe: robusta
    a fenotipi non normali, la scelta standard) sui gruppi genotipici 0/1/2
    del fenotipo residualizzato r. Gruppi con meno di min_group_size
    osservazioni vengono esclusi (non abbastanza dati per stimare la
    varianza in quel gruppo); se restano meno di 2 gruppi, non e' definita."""
    groups = [r[genotype == g] for g in (0, 1, 2) if np.sum(genotype == g) >= min_group_size]
    if len(groups) < 2:
        return np.nan
    try:
        stat, _p = stats.levene(*groups, center="median")
        return float(stat)
    except Exception:
        return np.nan


def _levene_perm_batch(genotype: np.ndarray, r: np.ndarray, n_perm_local: int, seed: int) -> np.ndarray:
    """Permuta le ETICHETTE di genotipo (r resta fisso) e ricalcola la
    statistica di Levene ad ogni iterazione: costruisce la distribuzione
    nulla empirica per il test di varianza-per-genotipo del locus."""
    rng = np.random.default_rng(seed)
    n = len(genotype)
    out = np.empty(n_perm_local)
    for i in range(n_perm_local):
        perm_genotype = genotype[rng.permutation(n)]
        out[i] = _levene_stat(perm_genotype, r)
    return out


def _run_levene_permutation_test(
    dosage: np.ndarray, y_orig: np.ndarray, covariates: np.ndarray, n_perm: int, n_jobs: int,
) -> dict:
    """Test di Levene permutazionale completo per un locus (un SNP): vedi
    punto 3 del docstring di modulo. Il fenotipo viene residualizzato sulle
    STESSE covariate usate dallo scan Step 3 (vqtl.core.scan.residualize),
    per coerenza con la definizione di "effetto di varianza" li' usata."""
    r_all, ok_mask = residualize(y_orig, covariates)
    ok = ok_mask & ~np.isnan(dosage)
    genotype = np.round(dosage[ok]).astype(int)
    r = r_all[ok]

    observed = _levene_stat(genotype, r)
    if np.isnan(observed):
        return {"levene_stat_observed": None, "levene_pval": None, "levene_n_perm_valid": None}

    n_jobs_eff = n_jobs if n_jobs > 0 else (os.cpu_count() or 1)
    n_splits = max(1, min(n_jobs_eff, 8))
    per_split = int(np.ceil(n_perm / n_splits))
    batches = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_levene_perm_batch)(genotype, r, per_split, seed=10_000 + s) for s in range(n_splits)
    )
    null_dist = np.concatenate(batches)[:n_perm]
    null_dist = null_dist[~np.isnan(null_dist)]
    n_valid = len(null_dist)
    # test one-sided: la statistica di Levene e' >= 0, "piu' grande" e' sempre
    # nella direzione di "piu' eteroschedasticita'" (a differenza di beta_I,
    # che puo' avere segno, qui non ha senso un confronto in valore assoluto)
    pval = (1 + np.sum(null_dist >= observed)) / (n_valid + 1) if n_valid else np.nan

    return {
        "levene_stat_observed": observed,
        "levene_pval": float(pval) if not np.isnan(pval) else None,
        "levene_n_perm_valid": n_valid,
    }


def run_robustness_and_permutation(
    dataset: VqtlDataset, vcfg: VqtlConfig, interaction_df: pd.DataFrame, target_col: str, generation: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from vqtl.db import repository as repo

    if interaction_df.empty:
        log.warning("Nessun risultato di interazione: Step 7 saltato.")
        return pd.DataFrame(), pd.DataFrame()

    inv_mapping = {v: k for k, v in dataset.mapping.items()}
    covariates = dataset.df[dataset.covariate_cols].to_numpy(dtype=float) if dataset.covariate_cols else np.zeros((len(dataset.df), 0))
    binary_outcome = is_binary(dataset.df[target_col])

    top_loci = interaction_df.sort_values("pval").head(vcfg.perm_top_n_loci).copy()
    log.info("Step 7: top %d loci selezionati per robustezza e permutazioni.", len(top_loci))

    y_orig = dataset.df[target_col].to_numpy(dtype=float)
    y_log = dataset.df[f"{target_col}_log"].to_numpy(dtype=float)
    y_rint = dataset.df[f"{target_col}_rint"].to_numpy(dtype=float)
    z_orig = (y_orig - np.nanmean(y_orig)) / np.nanstd(y_orig)
    outlier_mask = np.abs(z_orig) <= 3
    phenotype_variants = {
        "original": (y_orig, np.ones(len(y_orig), dtype=bool)),
        "log_transform": (y_log, np.ones(len(y_orig), dtype=bool)),
        "rank_inverse_normal": (y_rint, np.ones(len(y_orig), dtype=bool)),
        "outliers_removed": (y_orig, outlier_mask),
    }

    # ---- 1. Robustezza su trasformazioni / outlier ----
    robustness_placeholders = [
        {"variant": row["SNP"], "exposure": row["exposure"], "chromosome": row["CHR"], "position": row["POS"], "phenotype_variant": pv}
        for _, row in top_loci.iterrows() for pv in phenotype_variants
    ]
    repo.ensure_placeholders("robustness", generation, robustness_placeholders)
    done_robustness = repo.get_done_keys("robustness", generation)

    robustness_rows = []
    for _, row in top_loci.iterrows():
        snp_id, exp_raw = row["SNP"], row["exposure"]
        safe_col = inv_mapping.get(snp_id)
        exp_std_col = dataset.exposure_std_cols.get(exp_raw)
        if safe_col is None or exp_std_col is None:
            continue
        dosage = dataset.df[safe_col].to_numpy(dtype=float)
        exposure_vals = dataset.df[exp_std_col].to_numpy(dtype=float)

        for variant_name, (yv, mask) in phenotype_variants.items():
            if (snp_id, exp_raw, variant_name) in done_robustness:
                continue
            res = fit_interaction(yv[mask], dosage[mask], exposure_vals[mask], covariates[mask], binary_outcome, robust=vcfg.robust_se)
            row_out = {"variant": snp_id, "exposure": exp_raw, "phenotype_variant": variant_name, "status": "done", "error_message": None}
            if res is None:
                row_out.update({"beta_i": None, "se": None, "pval": None, "n": None, "maf": None})
            else:
                row_out.update({"beta_i": res["beta_I"], "se": res["SE"], "pval": res["pval"], "n": res["N"], "maf": res["MAF"]})
            robustness_rows.append(row_out)
    if robustness_rows:
        repo.bulk_update_status("robustness", generation, robustness_rows)

    robustness_df = repo.fetch_results("robustness", generation)
    if not robustness_df.empty:
        robustness_df = robustness_df.rename(columns={"beta_i": "beta_I", "n": "N", "maf": "MAF", "se": "SE", "phenotype_variant": "variant"})
        robustness_df = robustness_df.dropna(subset=["pval"])

    # ---- 2. Permutazioni Freedman-Lane ----
    n_perm = vcfg.n_perm
    log.info("Step 7: %d permutazioni per %d top loci (n_jobs=%s)", n_perm, len(top_loci), vcfg.n_jobs)

    perm_placeholders = [
        {"variant": row["SNP"], "exposure": row["exposure"], "chromosome": row["CHR"], "position": row["POS"]}
        for _, row in top_loci.iterrows()
    ]
    repo.ensure_placeholders("permutation", generation, perm_placeholders)
    done_perm = repo.get_done_keys("permutation", generation)
    levene_cache: dict[str, dict] = {}

    def _perm_batch(dosage, exposure_vals, resid_reduced, yhat_reduced, cov, n_perm_local, seed):
        rng = np.random.default_rng(seed)
        n = len(resid_reduced)
        estimates = np.empty(n_perm_local)
        for i in range(n_perm_local):
            perm_idx = rng.permutation(n)
            y_perm = yhat_reduced + resid_reduced[perm_idx]
            res = fit_interaction(y_perm, dosage, exposure_vals, cov, binary_outcome, robust=vcfg.robust_se)
            estimates[i] = res["beta_I"] if res is not None else np.nan
        return estimates

    for _, row in top_loci.iterrows():
        snp_id, exp_raw = row["SNP"], row["exposure"]
        if (snp_id, exp_raw) in done_perm:
            continue
        safe_col = inv_mapping.get(snp_id)
        exp_std_col = dataset.exposure_std_cols.get(exp_raw)
        if safe_col is None or exp_std_col is None:
            continue
        dosage = dataset.df[safe_col].to_numpy(dtype=float)
        exposure_vals = dataset.df[exp_std_col].to_numpy(dtype=float)
        observed_beta = row["beta_I"]

        yhat_reduced, resid_reduced, kept_idx = _fit_reduced_model(y_orig, dosage, exposure_vals, covariates)
        dosage_kept = dosage[kept_idx]
        exposure_kept = exposure_vals[kept_idx]
        covariates_kept = covariates[kept_idx]

        n_jobs_eff = vcfg.n_jobs if vcfg.n_jobs > 0 else (os.cpu_count() or 1)
        n_splits = max(1, min(n_jobs_eff, 8))
        per_split = int(np.ceil(n_perm / n_splits))
        batches = Parallel(n_jobs=vcfg.n_jobs, backend="loky")(
            delayed(_perm_batch)(dosage_kept, exposure_kept, resid_reduced, yhat_reduced, covariates_kept, per_split, seed=s)
            for s in range(n_splits)
        )
        all_estimates = np.concatenate(batches)[:n_perm]
        all_estimates = all_estimates[~np.isnan(all_estimates)]
        n_valid = len(all_estimates)
        emp_p = (1 + np.sum(np.abs(all_estimates) >= abs(observed_beta))) / (n_valid + 1) if n_valid else np.nan

        # Test di Levene permutazionale: dipende solo dal genotipo, non
        # dall'esposizione -- calcolato una volta per SNP e riusato se lo
        # stesso SNP compare di nuovo in top_loci per un'altra esposizione.
        if snp_id not in levene_cache:
            levene_cache[snp_id] = _run_levene_permutation_test(dosage, y_orig, covariates, n_perm, vcfg.n_jobs)
        levene_result = levene_cache[snp_id]

        repo.bulk_update_status("permutation", generation, [{
            "variant": snp_id, "exposure": exp_raw, "status": "done", "error_message": None,
            "beta_i_observed": observed_beta, "n_perm_valid": n_valid, "empirical_pval": emp_p,
            "asymptotic_pval": row["pval"], **levene_result,
        }])
        log.info(
            "%s x %s: beta_I=%.4g, p empirica=%.4g (asintotica=%.4g) | Levene stat=%s, p empirica=%s",
            snp_id, exp_raw, observed_beta, emp_p, row["pval"],
            levene_result["levene_stat_observed"], levene_result["levene_pval"],
        )

    perm_df = repo.fetch_results("permutation", generation)
    if not perm_df.empty:
        perm_df = perm_df.rename(columns={"beta_i_observed": "beta_I_observed"}).dropna(subset=["empirical_pval"])
    return robustness_df, perm_df
