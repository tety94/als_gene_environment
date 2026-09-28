"""Sensitivity analysis alla Keller (2014, PMID 24135711) sulle sole varianti replicate.

Confronta, sullo STESSO campione matchato:
    base : onset ~ G*E + C                     (modello della pipeline; C = sex + PC)
    cxe  : onset ~ G*E + C + C:E               (quanto chiesto dal revisore: PC×E e sex×E)
    [opz.] cxe_gxc : cxe + G:C                 (solo con --with-gxc; instabile con varianti rare)

Per ogni fit si riportano rango e condizionamento della matrice: un fit con
rango deficiente o condizionamento enorme NON è interpretabile e va escluso
(colonna `ok_<modello>`), non commentato.

Uso (GENERATION = coorte su cui rifittare):
    GENERATION=1 python -m gene_environment.analysis.keller_sensitivity varianti.csv out_gen1.csv
    GENERATION=1 python -m gene_environment.analysis.keller_sensitivity varianti.csv out_gen1.csv --with-gxc
varianti.csv: colonna `variant` (CHROM_POS_MUTATION); i duplicati vengono eliminati.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

COND_MAX = 1e6  # oltre questo (colonne standardizzate) il fit è considerato inaffidabile


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


def _fit(df, target, variant_col, Ecols, Ccols, cxe, gxc):
    y = df[target].to_numpy(float)
    X, names = _design(df, variant_col, Ecols, Ccols, cxe, gxc)
    Z = X[:, 1:]
    sd = Z.std(axis=0)
    Zs = (Z - Z.mean(axis=0)) / np.where(sd > 0, sd, 1)
    rank = int(np.linalg.matrix_rank(X))
    cond = float(np.linalg.cond(Zs)) if (sd > 0).all() else np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # i fit singolari vengono marcati da ok_*, non serve il warning
        res = sm.OLS(y, X).fit(cov_type="HC3")
    i = names.index(f"G:{Ecols[0]}")
    return {
        "beta": float(res.params[i]), "se": float(res.bse[i]), "p": float(res.pvalues[i]),
        "n_params": X.shape[1], "rank": rank, "cond": cond,
        "ok": bool(rank == X.shape[1] and cond < COND_MAX and np.isfinite(res.bse[i])),
    }


def fit_models(df, target, variant_col, Ecols, Ccols, with_gxc=False) -> dict:
    specs = {"base": (False, False), "cxe": (True, False)}
    if with_gxc:
        specs["cxe_gxc"] = (True, True)
    out = {}
    for tag, (cxe, gxc) in specs.items():
        r = _fit(df, target, variant_col, Ecols, Ccols, cxe, gxc)
        out.update({f"{k}_{tag}": v for k, v in r.items()})
    out["delta_beta_cxe"] = out["beta_cxe"] - out["beta_base"]
    out["rel_shift_cxe"] = out["delta_beta_cxe"] / abs(out["beta_base"]) if out["beta_base"] else np.nan
    out["same_sign_cxe"] = bool(np.sign(out["beta_base"]) == np.sign(out["beta_cxe"]))
    return out


def _product_smd(df, treat_col, Ecols, Ccols) -> float:
    """SMD massimo sui PRODOTTI C×E: il matching bilancia i main effect, non i prodotti."""
    worst, t = 0.0, df[treat_col] == 1
    for c in Ccols:
        for e in Ecols:
            p = df[c] * df[e]
            sd = np.sqrt((p[t].var() + p[~t].var()) / 2)
            if sd > 0:
                worst = max(worst, abs(p[t].mean() - p[~t].mean()) / sd)
    return float(worst)


def run(variants_csv: str, out_csv: str, with_gxc: bool = False) -> pd.DataFrame:
    from gene_environment.analysis.matching import check_balance, match_control_units
    from gene_environment.config import get_config
    from gene_environment.vcf_pipeline.build_dataset import load_and_prepare_data

    cfg = get_config()
    df, _, mapping, Ecols, _, covariate_cols = load_and_prepare_data(cfg)
    orig_to_safe = {v: k for k, v in mapping.items()}
    labels = list(dict.fromkeys(pd.read_csv(variants_csv)["variant"].tolist()))  # dedup, ordine mantenuto

    rows = []
    for lab in labels:
        col = orig_to_safe.get(lab)
        if col is None:
            rows.append({"variant": lab, "note": "non genotipizzata in questa coorte"})
            continue
        d = df[df[col] != "."].copy()
        d[col] = d[col].astype(int)
        d["_match_variant"] = (d[col] > 0).astype(int)
        d = d[[cfg.target_col, col, "_match_variant"] + Ecols + covariate_cols].dropna()
        m = match_control_units(d, "_match_variant", k=cfg.match_k,
                                covariates_for_matching=Ecols + covariate_cols)
        if m is None or m.shape[0] < cfg.min_sample_size:
            rows.append({"variant": lab, "note": "matching fallito"})
            continue
        r = fit_models(m, cfg.target_col, col, Ecols, covariate_cols, with_gxc)
        r.update(variant=lab, n_matched=int(m.shape[0]),
                 n_carriers=int((m["_match_variant"] == 1).sum()),
                 max_smd_main=max(check_balance(m, "_match_variant", Ecols + covariate_cols).values()),
                 max_smd_CxE=_product_smd(m, "_match_variant", Ecols, covariate_cols))
        rows.append(r)

    res = pd.DataFrame(rows)
    res.to_csv(out_csv, index=False)
    if "ok_cxe" in res:
        ok = res[res["ok_base"].fillna(False) & res["ok_cxe"].fillna(False)]
        print(f"{len(res)} varianti | fit affidabili (base e cxe): {len(ok)}")
        if len(ok):
            print(f"  segno conservato: {ok['same_sign_cxe'].mean():.0%} | "
                  f"|shift relativo| mediano: {ok['rel_shift_cxe'].abs().median():.1%} | "
                  f"SE cxe/base mediano: {(ok['se_cxe']/ok['se_base']).median():.2f}")
            sig = ok[ok["p_base"] < 0.05]
            print(f"  con p_base<0.05: {len(sig)} | ancora p<0.05 con cxe: {int((sig['p_cxe'] < 0.05).sum())}")
    return res


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], with_gxc="--with-gxc" in sys.argv)