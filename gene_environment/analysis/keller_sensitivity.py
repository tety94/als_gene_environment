"""Sensitivity alla Keller (2014, PMID 24135711) + diagnostica numerica dei fit.

INPUT  varianti.csv con colonne `variant` (CHROM_POS_MUTATION) ed `exposure`
       (nome della colonna nel file ambientale). Una riga = una coppia
       variante x esposizione; i duplicati vengono eliminati. Se manca la
       colonna `exposure` si usa cfg.exposure (con un warning).

Il file genetico viene letto UNA volta sola; per ogni (generazione, esposizione)
si ricostruisce il blocco ambientale + PC esattamente come fa la pipeline
(standardizzazione dell'esposizione sulla coorte, PC della generazione).

Per ogni (generazione, esposizione, variante), sullo STESSO campione matchato
della pipeline, si stimano:
    nocov   : onset ~ G*E                 (nessuna covariata)
    base    : onset ~ G*E + C             (modello della pipeline; C = sex + PC)
    cxe     : onset ~ G*E + C + C:E       (quanto chiesto da Keller)
    [opz.] cxe_gxc : cxe + G:C            (--with-gxc; instabile con varianti rare)

Le regressioni sono OLS in forma chiusa: non esiste "convergenza" nel senso
iterativo. Quello che si controlla e' se il fit e' ben posto:
  * rango vs numero di colonne, condizionamento (colonne standardizzate),
    leva massima, SE HC3 finito  -> colonna `ok_<modello>`
  * concordanza tra smf.ols (percorso osservato della pipeline), fast path
    (build_design_and_solve, quello delle permutazioni) e il design di questo
    script, sullo stesso campione. NB: la concordanza NON rileva il rango
    deficiente (lstsq e pinv danno entrambi la stessa soluzione a norma
    minima): per quello servono rango e condizionamento.
  * con --perm B: salute delle prime B permutazioni (stesso seed della
    pipeline): frazione valide, con rango deficiente, errore Monte Carlo.

OUTPUT (default ./output/keller_sensitivity, cfg.keller_sensitivity_dir):
    results_all.csv, results_gen<N>.csv, summary_by_gen_exposure.csv,
    keller_sensitivity_report.docx, figures/*.png, run_info.json

Uso:
    python -m gene_environment.analysis.keller_sensitivity varianti.csv
    python -m gene_environment.analysis.keller_sensitivity varianti.csv --perm 500
    python -m gene_environment.analysis.keller_sensitivity varianti.csv --generations 1 2 \
        --out-dir output/keller_sensitivity --with-gxc
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gc
import io
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

# ---- soglie (modificabili) -------------------------------------------------
COND_MAX = 1e6            # oltre questo (colonne standardizzate) il fit e' inaffidabile
LEV_WARN = 0.5            # leva massima: una sola osservazione domina il fit
CONC_RTOL = 1e-6          # tolleranza relativa per la concordanza tra percorsi di calcolo
CONC_ATOL = 1e-8
SHIFT_WARN = 0.5          # |shift relativo| del beta (nocov->base o base->cxe) da segnalare
PERM_VALID_WARN = 0.95    # frazione di permutazioni valide sotto cui si segnala
PERM_VALID_FAIL = 0.50
PERM_RANKDEF_WARN = 0.05  # frazione di permutazioni a rango deficiente

MODEL_TAGS = ("nocov", "base", "cxe", "cxe_gxc")


# ============================================================================
# Fit e diagnostica numerica
# ============================================================================
def _design(df, variant_col, Ecols, Ccols, cxe: bool, gxc: bool):
    v = df[variant_col].to_numpy(float)
    E = df[Ecols].to_numpy(float)
    C = df[Ccols].to_numpy(float) if Ccols else np.empty((len(df), 0))
    cols, names = [np.ones(len(df))], ["Intercept"]
    cols.append(v); names.append("G")
    for j, e in enumerate(Ecols):
        cols.append(E[:, j]); names.append(e)
    for j, e in enumerate(Ecols):
        cols.append(v * E[:, j]); names.append(f"G:{e}")
    for j, c in enumerate(Ccols):
        cols.append(C[:, j]); names.append(c)
    if cxe:
        for j, c in enumerate(Ccols):
            for k, e in enumerate(Ecols):
                cols.append(C[:, j] * E[:, k]); names.append(f"{c}:{e}")
    if gxc:
        for j, c in enumerate(Ccols):
            cols.append(v * C[:, j]); names.append(f"G:{c}")
    return np.column_stack(cols), names


def _max_leverage(X: np.ndarray) -> float:
    """Leva massima via SVD (corretta anche con rango deficiente)."""
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    tol = s.max() * max(X.shape) * np.finfo(float).eps if s.size else 0.0
    keep = s > tol
    return float((U[:, keep] ** 2).sum(axis=1).max()) if keep.any() else float("nan")


def _fit(df, target, variant_col, Ecols, Ccols, cxe, gxc):
    y = df[target].to_numpy(float)
    X, names = _design(df, variant_col, Ecols, Ccols, cxe, gxc)
    Z = X[:, 1:]
    sd = Z.std(axis=0)
    Zs = (Z - Z.mean(axis=0)) / np.where(sd > 0, sd, 1)
    rank = int(np.linalg.matrix_rank(X))
    cond = float(np.linalg.cond(Zs)) if (sd > 0).all() else np.inf
    lev = _max_leverage(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # i fit singolari vengono marcati da ok_*
        res = sm.OLS(y, X).fit(cov_type="HC3")
    i = names.index(f"G:{Ecols[0]}")
    se = float(res.bse[i])
    return {
        "beta": float(res.params[i]), "se": se, "p": float(res.pvalues[i]),
        "n_params": X.shape[1], "rank": rank, "cond": cond, "max_lev": lev,
        "ok": bool(rank == X.shape[1] and cond < COND_MAX and np.isfinite(se)),
    }


def fit_models(df, target, variant_col, Ecols, Ccols, with_gxc=False) -> dict:
    specs = {"nocov": ([], False, False), "base": (Ccols, False, False), "cxe": (Ccols, True, False)}
    if with_gxc:
        specs["cxe_gxc"] = (Ccols, True, True)
    out = {}
    for tag, (cc, cxe, gxc) in specs.items():
        r = _fit(df, target, variant_col, Ecols, cc, cxe, gxc)
        out.update({f"{k}_{tag}": v for k, v in r.items()})
    b0, b1, b2 = out["beta_nocov"], out["beta_base"], out["beta_cxe"]
    out["delta_beta_cov"] = b1 - b0
    out["rel_shift_cov"] = (b1 - b0) / abs(b0) if b0 else np.nan
    out["same_sign_cov"] = bool(np.sign(b0) == np.sign(b1))
    out["delta_beta_cxe"] = b2 - b1
    out["rel_shift_cxe"] = (b2 - b1) / abs(b1) if b1 else np.nan
    out["same_sign_cxe"] = bool(np.sign(b1) == np.sign(b2))
    return out


def _product_smd(df, treat_col, Ecols, Ccols) -> float:
    """SMD massimo sui PRODOTTI C x E: il matching bilancia i main effect, non i prodotti."""
    worst, t = 0.0, df[treat_col] == 1
    for c in Ccols:
        for e in Ecols:
            p = df[c] * df[e]
            sd = np.sqrt((p[t].var() + p[~t].var()) / 2)
            if sd > 0:
                worst = max(worst, abs(p[t].mean() - p[~t].mean()) / sd)
    return float(worst)


def _carrier_support(m, treat_col, Ecols) -> dict:
    """Quanta informazione c'e' sui portatori per identificare G:E."""
    e = m[Ecols[0]]
    car = m[treat_col] == 1
    out = {
        "n_carriers_matched": int(car.sum()), "n_noncarriers_matched": int((~car).sum()),
        "n_distinct_E_carriers": int(e[car].nunique()),
        "sd_E_carriers": float(e[car].std()) if car.sum() > 1 else np.nan,
    }
    binary = e.nunique() <= 2
    out["E_binary"] = bool(binary)
    if binary:
        hi = e.max()
        out["n_carriers_E_high"] = int((car & (e == hi)).sum())
        out["n_carriers_E_low"] = int((car & (e != hi)).sum())
    return out


def _concordance(d, m, target, col, Ecols, Ccols, cfg) -> dict:
    """smf.ols (percorso osservato) vs fast path (percorso permutazioni)."""
    import statsmodels.formula.api as smf
    from gene_environment.analysis.fast_ols import build_design_and_solve, interaction_column_index
    from gene_environment.analysis.matching import match_control_units_indices, precompute_scaled_covariates
    from gene_environment.analysis.modeling import _find_interaction_term, build_formula

    out = {"beta_smf": np.nan, "beta_fast": np.nan, "same_matched_set": False}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = smf.ols(build_formula(target, col, Ecols, Ccols, m), data=m).fit()
    name = _find_interaction_term(mod.params.index, col)
    if name is not None:
        out["beta_smf"] = float(mod.params[name])

    Xs = precompute_scaled_covariates(d, Ecols + Ccols)
    mi = match_control_units_indices(d["_match_variant"].to_numpy(), Xs, k=cfg.match_k)
    if mi is not None:
        idx = np.concatenate(mi)
        y = d[target].to_numpy(float)
        C = d[Ccols].to_numpy(float)[idx] if Ccols else None
        b = build_design_and_solve(d[col].to_numpy(float)[idx], d[Ecols].to_numpy(float)[idx], y[idx], C)
        if b is not None:
            out["beta_fast"] = float(b[interaction_column_index(len(Ecols))])
        out["same_matched_set"] = bool(
            len(idx) == len(m) and np.allclose(np.sort(y[idx]), np.sort(m[target].to_numpy(float)))
        )
    return out


def _perm_health(d, obs_coef, col, Ecols, Ccols, cfg, B) -> dict:
    """Prime B permutazioni con lo stesso seed e la stessa logica della pipeline
    (modeling._run_permutation_batch), ma contando cosa va storto."""
    from gene_environment.analysis.fast_ols import build_design_and_solve, interaction_column_index
    from gene_environment.analysis.matching import match_control_units_indices, precompute_scaled_covariates
    from gene_environment.analysis.modeling import _stable_seed

    vv = d[col].to_numpy()
    y = d[cfg.target_col].to_numpy(float)
    E = d[Ecols].to_numpy(float)
    C = d[Ccols].to_numpy(float) if Ccols else None
    Xs = precompute_scaled_covariates(d, Ecols + Ccols)
    inter = interaction_column_index(E.shape[1])
    q = 0 if C is None else C.shape[1]
    n_cols = 2 + 2 * E.shape[1] + q
    rng = np.random.RandomState(_stable_seed(cfg.random_state, col))

    cnt = {"matching": 0, "min_sample": 0, "design": 0}
    betas, rankdef = [], []
    for _ in range(B):
        pv = rng.permutation(vv)
        mi = match_control_units_indices((pv > 0).astype(int), Xs, k=cfg.match_k)
        if mi is None:
            cnt["matching"] += 1
            continue
        idx = np.concatenate(mi)
        if idx.shape[0] < cfg.min_sample_size:
            cnt["min_sample"] += 1
            continue
        Ci = C[idx] if C is not None else None
        beta = build_design_and_solve(pv[idx], E[idx], y[idx], Ci)
        if beta is None:
            cnt["design"] += 1
            continue
        v = pv[idx].astype(float).reshape(-1, 1)
        X = np.column_stack([np.ones(len(idx)), v, E[idx], v * E[idx]] + ([Ci] if Ci is not None else []))
        rankdef.append(int(np.linalg.matrix_rank(X)) < n_cols)
        betas.append(float(beta[inter]))

    n_valid = len(betas)
    out = {
        "perm_B": B, "perm_valid": n_valid,
        "perm_valid_frac": n_valid / B if B else np.nan,
        "perm_fail_matching": cnt["matching"], "perm_fail_min_sample": cnt["min_sample"],
        "perm_fail_design": cnt["design"],
    }
    if n_valid:
        b = np.array(betas)
        rd = np.array(rankdef)
        p = float(np.mean(np.abs(b) >= abs(obs_coef)))
        out.update({
            "perm_rankdef_frac": float(rd.mean()),
            "perm_p_emp": p,
            "perm_p_emp_fullrank": float(np.mean(np.abs(b[~rd]) >= abs(obs_coef))) if (~rd).any() else np.nan,
            "perm_mc_se": float(np.sqrt(max(p * (1 - p), 0) / n_valid)),
            "perm_mean": float(b.mean()), "perm_std": float(b.std()),
            "perm_z_obs": float((obs_coef - b.mean()) / b.std()) if b.std() > 0 else np.nan,
            "below_min_obs_coef": bool(abs(obs_coef) < cfg.min_obs_coef),
        })
    return out


# ============================================================================
# Classificazione OK / WARN / FAIL
# ============================================================================
def classify(r: dict, has_perm: bool, with_gxc: bool) -> tuple[str, str]:
    fail, warn = [], []
    if not r.get("ok_base", False):
        fail.append("fit base non affidabile (rango/condizionamento/SE)")
    if r.get("concordant") is False:
        fail.append("smf.ols e fast path non concordano")
    if r.get("same_matched_set") is False:
        fail.append("matching osservato != matching fast path")
    if r.get("ok_base", False):
        if not r.get("ok_nocov", True):
            warn.append("fit senza covariate non affidabile")
        if not r.get("ok_cxe", True):
            warn.append("fit con C x E non affidabile")
        if with_gxc and not r.get("ok_cxe_gxc", True):
            warn.append("fit con G x C non affidabile")
    if r.get("max_lev_base", 0) > LEV_WARN:
        warn.append(f"leva massima {r['max_lev_base']:.2f} > {LEV_WARN}")
    if r.get("max_smd_main", 0) > r.get("cfg_max_smd", np.inf):
        fail.append("SMD > MAX_SMD: la pipeline scarterebbe questa variante (nessun risultato in DB)")
    if r.get("ok_nocov", False) and r.get("ok_base", False):
        if not r.get("same_sign_cov", True):
            warn.append("segno del beta cambia aggiungendo le covariate")
        elif abs(r.get("rel_shift_cov", 0) or 0) > SHIFT_WARN:
            warn.append(f"beta cambia >{SHIFT_WARN:.0%} aggiungendo le covariate")
    if r.get("ok_base", False) and r.get("ok_cxe", False):
        if not r.get("same_sign_cxe", True):
            warn.append("segno del beta cambia con C x E")
        elif abs(r.get("rel_shift_cxe", 0) or 0) > SHIFT_WARN:
            warn.append(f"beta cambia >{SHIFT_WARN:.0%} con C x E")
    if has_perm and "perm_valid_frac" in r:
        vf = r["perm_valid_frac"]
        if vf < PERM_VALID_FAIL:
            fail.append(f"solo {vf:.0%} di permutazioni valide")
        elif vf < PERM_VALID_WARN:
            warn.append(f"{vf:.0%} di permutazioni valide")
        if (r.get("perm_rankdef_frac") or 0) > PERM_RANKDEF_WARN:
            warn.append(f"{r['perm_rankdef_frac']:.0%} di permutazioni a rango deficiente")
    status = "FAIL" if fail else ("WARN" if warn else "OK")
    return status, "; ".join(fail + warn)


# ============================================================================
# Una coppia (variante, esposizione) su una generazione
# ============================================================================
def analyze_one(df, col, label, exposure, generation, Ecols, Ccols, cfg, with_gxc, perm_B) -> dict:
    from gene_environment.analysis.matching import check_balance, match_control_units

    row = {"generation": generation, "exposure": exposure, "variant": label}
    d = df[df[col].notna() & (df[col] != ".")].copy()
    d[col] = pd.to_numeric(d[col]).astype(int)
    d["_match_variant"] = (d[col] > 0).astype(int)
    n_t, n_c = int(d["_match_variant"].sum()), int((d["_match_variant"] == 0).sum())
    row.update(n_carriers_all=n_t, n_noncarriers_all=n_c)
    if n_t < cfg.min_treated or n_c < cfg.min_treated:
        return {**row, "status": "SKIP", "reasons": "sotto MIN_TREATED: la pipeline non lo testa"}

    d = d[[cfg.target_col, col, "_match_variant"] + Ecols + Ccols].dropna()
    if d.shape[0] < cfg.min_sample_size:
        return {**row, "status": "SKIP", "reasons": "campione sotto MIN_SAMPLE_SIZE dopo dropna"}
    m = match_control_units(d, "_match_variant", k=cfg.match_k, covariates_for_matching=Ecols + Ccols)
    if m is None or m.shape[0] < cfg.min_sample_size:
        return {**row, "status": "SKIP", "reasons": "matching fallito"}

    row["n_matched"] = int(m.shape[0])
    row.update(_carrier_support(m, "_match_variant", Ecols))
    row["max_smd_main"] = float(max(check_balance(m, "_match_variant", Ecols + Ccols).values() or [0]))
    row["cfg_max_smd"] = float(cfg.max_smd)
    row["max_smd_CxE"] = _product_smd(m, "_match_variant", Ecols, Ccols)

    row.update(fit_models(m, cfg.target_col, col, Ecols, Ccols, with_gxc))

    conc = _concordance(d, m, cfg.target_col, col, Ecols, Ccols, cfg)
    row.update(conc)
    betas = [conc["beta_smf"], conc["beta_fast"], row["beta_base"]]
    if np.all(np.isfinite(betas)):
        ref = betas[0]
        row["conc_maxdiff"] = float(max(abs(b - ref) for b in betas))
        row["concordant"] = bool(row["conc_maxdiff"] <= CONC_ATOL + CONC_RTOL * abs(ref))
    else:
        row["conc_maxdiff"], row["concordant"] = np.nan, False

    if perm_B and np.isfinite(conc["beta_smf"]):
        row.update(_perm_health(d, conc["beta_smf"], col, Ecols, Ccols, cfg, perm_B))

    row["status"], row["reasons"] = classify(row, bool(perm_B), with_gxc)
    return row


# ============================================================================
# Caricamento dati (genetica una volta sola)
# ============================================================================
def _quiet(fn, *a, **k):
    """La pipeline stampa DEBUG/df.columns su stdout: qui li silenzio."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def _read_input(path: str, cfg) -> pd.DataFrame:
    from gene_environment.logging_utils import get_logger
    log = get_logger(__name__)
    raw = pd.read_csv(path)
    if "variant" not in raw.columns:
        raise ValueError(f"{path}: manca la colonna 'variant' (colonne: {list(raw.columns)})")
    if "exposure" not in raw.columns:
        if not cfg.exposure:
            raise ValueError(f"{path}: manca la colonna 'exposure' e EXPOSURE non e' configurata")
        log.warning("Colonna 'exposure' assente: uso cfg.exposure=%s per tutte le varianti", cfg.exposure)
        raw["exposure"] = cfg.exposure
    raw["variant"] = raw["variant"].astype(str).str.strip()
    raw["exposure"] = raw["exposure"].astype(str).str.strip()
    raw = raw[(raw["variant"] != "") & (raw["exposure"] != "")]
    return raw[["variant", "exposure"]].drop_duplicates().reset_index(drop=True)


# ============================================================================
# Riepilogo, figure, report Word
# ============================================================================
def summarize(res: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (g, e), s in res.groupby(["generation", "exposure"], sort=True):
        fit = s[s["status"] != "SKIP"]
        r = {"generation": g, "exposure": e, "n_pairs": len(s), "n_skip": int((s["status"] == "SKIP").sum()),
             "n_ok": int((s["status"] == "OK").sum()), "n_warn": int((s["status"] == "WARN").sum()),
             "n_fail": int((s["status"] == "FAIL").sum())}
        if len(fit):
            for t in ("nocov", "base", "cxe"):
                r[f"frac_ok_{t}"] = float(fit[f"ok_{t}"].mean())
            r["frac_concordant"] = float(fit["concordant"].mean())
            r["median_cond_base"] = float(fit["cond_base"].median())
            good = fit[fit["ok_base"] & fit["ok_nocov"]]
            r["sign_kept_cov"] = float(good["same_sign_cov"].mean()) if len(good) else np.nan
            r["median_abs_shift_cov"] = float(good["rel_shift_cov"].abs().median()) if len(good) else np.nan
            good = fit[fit["ok_base"] & fit["ok_cxe"]]
            r["sign_kept_cxe"] = float(good["same_sign_cxe"].mean()) if len(good) else np.nan
            r["median_abs_shift_cxe"] = float(good["rel_shift_cxe"].abs().median()) if len(good) else np.nan
            r["median_se_ratio_cxe_base"] = float((good["se_cxe"] / good["se_base"]).median()) if len(good) else np.nan
            sig = good[good["p_base"] < 0.05]
            r["n_sig_base"] = len(sig)
            r["n_sig_kept_cxe"] = int((sig["p_cxe"] < 0.05).sum())
            if "perm_valid_frac" in fit:
                r["median_perm_valid_frac"] = float(fit["perm_valid_frac"].median())
                r["max_perm_rankdef_frac"] = float(fit["perm_rankdef_frac"].max())
        rows.append(r)
    return pd.DataFrame(rows)


def make_figures(res: pd.DataFrame, fig_dir: Path, has_perm: bool) -> dict[str, Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = {"OK": "#2a9d8f", "WARN": "#e9c46a", "FAIL": "#e63946", "SKIP": "#999999"}
    gens = sorted(res["generation"].unique())
    fit = res[res["status"] != "SKIP"]
    out: dict[str, Path] = {}

    # 1) stati per generazione x esposizione
    fig, ax = plt.subplots(figsize=(max(5, 1.3 * res.groupby(["generation", "exposure"]).ngroups), 4))
    tab = res.groupby(["generation", "exposure", "status"]).size().unstack(fill_value=0)
    tab = tab.reindex(columns=[c for c in colors if c in tab.columns])
    tab.plot(kind="bar", stacked=True, ax=ax, color=[colors[c] for c in tab.columns], width=0.75)
    ax.set_xlabel("(generazione, esposizione)"); ax.set_ylabel("n. varianti")
    ax.set_xticklabels([f"g{a}\n{b}" for a, b in tab.index], rotation=0, fontsize=8)
    ax.legend(title="stato", frameon=False); fig.tight_layout()
    out["status"] = fig_dir / "status_counts.png"; fig.savefig(out["status"], dpi=150); plt.close(fig)

    # 2) beta con/senza covariate e con C x E
    fig, axes = plt.subplots(len(gens), 2, figsize=(9, 4 * len(gens)), squeeze=False)
    for i, g in enumerate(gens):
        s = fit[(fit["generation"] == g) & fit["ok_base"] & fit["ok_nocov"] & fit["ok_cxe"]]
        for j, (xc, lab) in enumerate((("beta_nocov", "senza covariate"), ("beta_cxe", "con C x E"))):
            ax = axes[i, j]
            if len(s):
                for st, c in colors.items():
                    q = s[s["status"] == st]
                    ax.scatter(q[xc], q["beta_base"], s=22, color=c, label=st, alpha=0.85)
                lo = float(min(s[xc].min(), s["beta_base"].min())); hi = float(max(s[xc].max(), s["beta_base"].max()))
                ax.plot([lo, hi], [lo, hi], color="k", lw=0.8, ls="--")
                ax.legend(frameon=False, fontsize=7)
            else:
                ax.text(0.5, 0.5, "nessun fit affidabile", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel(f"beta G:E {lab}"); ax.set_ylabel("beta G:E modello base")
            ax.set_title(f"generazione {g}", fontsize=10)
    fig.tight_layout()
    out["beta_scatter"] = fig_dir / "beta_base_vs_alternatives.png"; fig.savefig(out["beta_scatter"], dpi=150); plt.close(fig)

    # 3) condizionamento
    fig, ax = plt.subplots(figsize=(6, 4))
    for g in gens:
        c = fit.loc[fit["generation"] == g, "cond_base"].replace(np.inf, np.nan).dropna()
        if len(c):
            ax.hist(np.log10(c), bins=20, histtype="step", lw=1.6, label=f"gen {g}")
    ax.axvline(np.log10(COND_MAX), color="r", ls="--", lw=1)
    ax.set_xlabel("log10 numero di condizionamento (modello base)"); ax.set_ylabel("n. varianti")
    ax.legend(frameon=False); fig.tight_layout()
    out["cond"] = fig_dir / "condition_number.png"; fig.savefig(out["cond"], dpi=150); plt.close(fig)

    # 4) forest della scala di modelli (peggiori 40 per shift) per generazione
    for g in gens:
        s = fit[(fit["generation"] == g) & fit["ok_base"] & fit["ok_nocov"] & fit["ok_cxe"]].copy()
        if s.empty:
            continue
        s["_k"] = s["rel_shift_cov"].abs().fillna(0) + s["rel_shift_cxe"].abs().fillna(0)
        s = s.sort_values("_k", ascending=False).head(40).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, max(3, 0.32 * len(s) + 1.5)))
        y = np.arange(len(s))
        for off, tag, mk, c in ((-0.22, "nocov", "s", "#8d99ae"), (0, "base", "o", "#264653"), (0.22, "cxe", "^", "#e76f51")):
            ax.errorbar(s[f"beta_{tag}"], y + off, xerr=1.96 * s[f"se_{tag}"], fmt=mk, ms=4, lw=0.8, color=c, label=tag)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_yticks(y); ax.set_yticklabels([f"{v} | {e}" for v, e in zip(s["variant"], s["exposure"])], fontsize=6)
        ax.set_xlabel("beta G:E (IC95% HC3)"); ax.set_title(f"generazione {g} - varianti con shift maggiore", fontsize=10)
        ax.legend(frameon=False, fontsize=7); fig.tight_layout()
        out[f"forest_g{g}"] = fig_dir / f"beta_ladder_gen{g}.png"; fig.savefig(out[f"forest_g{g}"], dpi=150); plt.close(fig)

    # 5) permutazioni
    if has_perm and "perm_valid_frac" in fit and fit["perm_valid_frac"].notna().any():
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].hist(fit["perm_valid_frac"].dropna(), bins=20, color="#264653")
        axes[0].axvline(PERM_VALID_WARN, color="r", ls="--", lw=1)
        axes[0].set_xlabel("frazione di permutazioni valide"); axes[0].set_ylabel("n. varianti")
        axes[1].hist(fit["perm_rankdef_frac"].dropna(), bins=20, color="#e76f51")
        axes[1].axvline(PERM_RANKDEF_WARN, color="r", ls="--", lw=1)
        axes[1].set_xlabel("frazione di permutazioni a rango deficiente")
        fig.tight_layout()
        out["perm"] = fig_dir / "permutation_health.png"; fig.savefig(out["perm"], dpi=150); plt.close(fig)
    return out


def build_docx(res, summ, figs, path: Path, info: dict) -> None:
    from docx import Document
    from docx.shared import Pt
    from gene_environment.report.word_utils import (
        add_figure_to_doc, repeat_header_row, set_cell_bg, set_cell_text, set_landscape, set_table_borders,
    )

    doc = Document()
    set_landscape(doc)
    doc.styles["Normal"].font.size = Pt(10)
    doc.add_heading("Sensitivity Keller e diagnostica dei fit", level=0)
    doc.add_paragraph(
        f"Generato il {info['timestamp']}. Generazioni: {', '.join(map(str, info['generations']))}. "
        f"Coppie variante x esposizione: {info['n_pairs']}. Permutazioni diagnostiche: "
        f"{info['perm_B'] if info['perm_B'] else 'non eseguite'}."
    )

    doc.add_heading("Cosa si controlla", level=1)
    for t in (
        "Le regressioni sono OLS in forma chiusa: non ci sono iterazioni che possano non convergere. "
        "Si verifica invece che ogni fit sia ben posto: rango pieno, condizionamento sotto "
        f"{COND_MAX:.0e}, SE HC3 finito, leva massima sotto {LEV_WARN}.",
        "Concordanza tra smf.ols (percorso osservato della pipeline), fast path (percorso delle permutazioni) "
        "e il design usato qui, sullo stesso campione matchato. La concordanza non rileva il rango deficiente "
        "(lstsq e pinv restituiscono la stessa soluzione a norma minima): per quello contano rango e condizionamento.",
        "Scala di modelli sullo stesso campione: senza covariate, base (pipeline), con C x E (richiesta Keller).",
        "Stato per riga: FAIL = fit base non affidabile o percorsi discordanti; WARN = problemi nei modelli "
        "alternativi, leva alta, SMD sopra soglia, shift del beta oltre "
        f"{SHIFT_WARN:.0%} o cambio di segno; SKIP = la pipeline non testerebbe la variante.",
    ):
        doc.add_paragraph(t, style="List Bullet")

    doc.add_heading("Riepilogo per generazione ed esposizione", level=1)
    cols = [("generation", "gen"), ("exposure", "esposizione"), ("n_pairs", "coppie"), ("n_ok", "OK"),
            ("n_warn", "WARN"), ("n_fail", "FAIL"), ("n_skip", "SKIP"), ("frac_concordant", "concordanti"),
            ("median_cond_base", "cond. mediano"), ("sign_kept_cov", "segno = (cov)"),
            ("median_abs_shift_cov", "|shift| cov"), ("sign_kept_cxe", "segno = (CxE)"),
            ("median_abs_shift_cxe", "|shift| CxE"), ("median_se_ratio_cxe_base", "SE CxE/base")]
    cols = [c for c in cols if c[0] in summ.columns]

    def fmt(k, v):
        if pd.isna(v):
            return "-"
        if k == "median_cond_base":
            return f"{v:.1e}"
        if k.startswith(("frac_", "sign_", "median_abs")):
            return f"{v:.0%}"
        if k == "median_se_ratio_cxe_base":
            return f"{v:.2f}"
        return str(int(v)) if isinstance(v, (int, np.integer, float, np.floating)) and float(v).is_integer() else str(v)

    def table(df_, columns, widths_pt=8):
        t = doc.add_table(rows=1, cols=len(columns))
        set_table_borders(t)
        for i, (_, h) in enumerate(columns):
            set_cell_text(t.rows[0].cells[i], h, bold=True); set_cell_bg(t.rows[0].cells[i])
        repeat_header_row(t)
        for _, r in df_.iterrows():
            cells = t.add_row().cells
            for i, (k, _) in enumerate(columns):
                set_cell_text(cells[i], fmt(k, r[k]) if not isinstance(r[k], str) else r[k])
        for row in t.rows:
            for c in row.cells:
                for p in c.paragraphs:
                    for run in p.runs:
                        run.font.size = Pt(widths_pt)
        return t

    table(summ, cols)

    doc.add_heading("Figure", level=1)
    caps = {"status": "Stato dei fit per generazione ed esposizione.",
            "beta_scatter": "Beta G:E del modello base contro senza covariate (sinistra) e con C x E (destra); solo fit affidabili.",
            "cond": "Numero di condizionamento del modello base; linea rossa = soglia.",
            "perm": "Salute delle permutazioni diagnostiche."}
    for k in ("status", "beta_scatter", "cond", "perm"):
        if k in figs:
            add_figure_to_doc(doc, figs[k], caps[k], width_in=6.5)
    for k, p in figs.items():
        if k.startswith("forest_g"):
            add_figure_to_doc(doc, p, f"Scala di modelli, generazione {k[8:]}: varianti con shift maggiore.", width_in=7.0)

    prob = res[res["status"].isin(["FAIL", "WARN", "SKIP"])].copy()
    doc.add_heading("Righe da guardare (FAIL, WARN, SKIP)", level=1)
    if prob.empty:
        doc.add_paragraph("Nessuna.")
    else:
        order = {"FAIL": 0, "WARN": 1, "SKIP": 2}
        prob = prob.sort_values(["status", "generation", "exposure"], key=lambda s: s.map(order) if s.name == "status" else s)
        cut = prob.head(80)
        table(cut, [("generation", "gen"), ("exposure", "esposizione"), ("variant", "variante"),
                    ("status", "stato"), ("reasons", "motivi")])
        if len(prob) > len(cut):
            doc.add_paragraph(f"Mostrate {len(cut)} righe su {len(prob)}: elenco completo in results_all.csv.")

    doc.add_heading("Note", level=1)
    for t in (
        "Il campione e' quello matchato con la stessa procedura e gli stessi parametri della pipeline "
        "(MATCH_K, matching su esposizione + sesso + PC); l'esposizione e' standardizzata sulla coorte della generazione.",
        "Le prime B permutazioni usano lo stesso seed della pipeline, quindi coincidono con le prime B permutazioni reali.",
        "Le soglie sono costanti in cima a keller_sensitivity.py.",
    ):
        doc.add_paragraph(t, style="List Bullet")
    doc.save(str(path))


# ============================================================================
# Orchestrazione
# ============================================================================
def run(variants_csv: str, out_dir: str | None = None, generations=(1, 2), with_gxc: bool = False,
        perm_B: int = 0) -> pd.DataFrame:
    from gene_environment.config import get_config
    from gene_environment.logging_utils import get_logger
    from gene_environment.vcf_pipeline.build_dataset import _build_narrow_covariates, _load_genetic_data

    log = get_logger(__name__)
    cfg = get_config()
    pairs = _read_input(variants_csv, cfg)
    out = Path(out_dir or getattr(cfg, "keller_sensitivity_dir", "./output/keller_sensitivity"))
    (out / "figures").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log.info("Input: %d coppie variante x esposizione, %d esposizioni, generazioni %s",
             len(pairs), pairs["exposure"].nunique(), list(generations))

    df_gen, _, mapping, _ = _load_genetic_data(cfg)
    orig_to_safe = {v: k for k, v in mapping.items()}
    needed = [orig_to_safe[v] for v in dict.fromkeys(pairs["variant"]) if v in orig_to_safe]
    df_small = df_gen[["id"] + needed].copy()
    del df_gen, mapping
    gc.collect()

    rows = []
    for g in generations:
        for exposure, sub in pairs.groupby("exposure", sort=False):
            log.info("Generazione %s, esposizione %s: %d varianti", g, exposure, len(sub))
            cfg_ge = dataclasses.replace(cfg, generation=g, exposure=exposure)
            try:
                cov, Ecols, Ccols = _quiet(_build_narrow_covariates, cfg_ge, df_small["id"])
                df = pd.merge(cov, df_small, on="id", how="inner")
            except Exception as exc:  # esposizione assente, PC mancanti, ecc.
                log.error("g%s %s: dataset non costruibile: %s", g, exposure, exc)
                rows += [{"generation": g, "exposure": exposure, "variant": v, "status": "SKIP",
                          "reasons": f"dataset non costruibile: {exc}"} for v in sub["variant"]]
                continue
            for k, lab in enumerate(sub["variant"], 1):
                col = orig_to_safe.get(lab)
                if col is None:
                    rows.append({"generation": g, "exposure": exposure, "variant": lab, "status": "SKIP",
                                 "reasons": "variante non presente nel file genetico"})
                    continue
                try:
                    rows.append(analyze_one(df, col, lab, exposure, g, Ecols, Ccols, cfg_ge, with_gxc, perm_B))
                except Exception as exc:
                    log.exception("g%s %s %s: errore", g, exposure, lab)
                    rows.append({"generation": g, "exposure": exposure, "variant": lab, "status": "FAIL",
                                 "reasons": f"errore: {type(exc).__name__}: {exc}"})
                if k % 25 == 0:
                    log.info("  %d/%d", k, len(sub))
            del df, cov
            gc.collect()

    res = pd.DataFrame(rows)
    lead = ["generation", "exposure", "variant", "status", "reasons"]
    res = res[lead + [c for c in res.columns if c not in lead]]
    res = res.drop(columns=[c for c in ("cfg_max_smd",) if c in res.columns])
    res.to_csv(out / "results_all.csv", index=False)
    for g in generations:
        res[res["generation"] == g].to_csv(out / f"results_gen{g}.csv", index=False)
    summ = summarize(res)
    summ.to_csv(out / "summary_by_gen_exposure.csv", index=False)

    figs = make_figures(res, out / "figures", bool(perm_B))
    info = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), "generations": list(generations),
            "n_pairs": int(len(pairs)), "exposures": sorted(pairs["exposure"].unique().tolist()),
            "perm_B": perm_B, "with_gxc": with_gxc, "variants_csv": str(variants_csv),
            "thresholds": {"COND_MAX": COND_MAX, "LEV_WARN": LEV_WARN, "SHIFT_WARN": SHIFT_WARN,
                           "CONC_RTOL": CONC_RTOL, "PERM_VALID_WARN": PERM_VALID_WARN,
                           "PERM_RANKDEF_WARN": PERM_RANKDEF_WARN},
            "pipeline": {"match_k": cfg.match_k, "min_treated": cfg.min_treated, "max_smd": cfg.max_smd,
                         "pca_n_components": cfg.pca_n_components, "use_pca": cfg.use_pca_covariates,
                         "random_state": cfg.random_state},
            "status_counts": res["status"].value_counts().to_dict(),
            "seconds": round(time.time() - t0, 1)}
    (out / "run_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    build_docx(res, summ, figs, out / "keller_sensitivity_report.docx", info)

    print(f"\nOutput in {out.resolve()}")
    print(summ.to_string(index=False))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sensitivity Keller + diagnostica dei fit")
    ap.add_argument("variants_csv", help="CSV con colonne variant ed exposure")
    ap.add_argument("--out-dir", default=None, help="default: cfg.keller_sensitivity_dir (./output/keller_sensitivity)")
    ap.add_argument("--generations", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--perm", type=int, default=0, metavar="B",
                    help="controlla la salute delle prime B permutazioni per variante (0 = salta; es. 500)")
    ap.add_argument("--with-gxc", action="store_true", help="aggiunge il modello con G x C (instabile con varianti rare)")
    a = ap.parse_args(argv)
    run(a.variants_csv, a.out_dir, tuple(a.generations), a.with_gxc, a.perm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
