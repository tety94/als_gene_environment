"""Sensitivity analysis alla Keller (2014, PMID 24135711) sulle sole varianti replicate.

Modello di base della pipeline (modeling.py):
    onset ~ 1 + G + E + G:E + sex + PC1..PC5          (covariate solo additive)

Modello Keller (stesso campione matchato, stessi termini + prodotti delle covariate):
    onset ~ 1 + G + E + G:E + C + C:E + G:C           con C = {sex, PC1..PC5}

C:E è quello chiesto dal revisore (PC×E e S×E); G:C è l'altra metà della
raccomandazione di Keller. Si riportano entrambi i modelli sullo stesso
campione, così lo spostamento di beta è attribuibile solo ai termini aggiunti.

Uso (da sostituire GENERATION con la coorte su cui si vuole rifittare):
    GENERATION=2 python -m gene_environment.analysis.keller_sensitivity \
        replicated_variants.csv keller_gen2.csv
dove replicated_variants.csv ha una colonna `variant` (formato CHROM_POS_MUTATION).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm


def _design(df, variant_col, Ecols, Ccols, keller: bool):
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
    if keller:
        for j, c in enumerate(Ccols):
            for k, e in enumerate(Ecols):
                cols.append(C[:, j] * E[:, k]); names.append(f"{c}:{e}")
            cols.append(v * C[:, j]); names.append(f"G:{c}")
    return np.column_stack(cols), names


def fit_both(df, target, variant_col, Ecols, Ccols) -> dict:
    """Fit base e Keller sullo stesso df; ritorna beta/SE(HC3)/p di G:E."""
    y = df[target].to_numpy(float)
    out = {}
    for tag, keller in (("base", False), ("keller", True)):
        X, names = _design(df, variant_col, Ecols, Ccols, keller)
        res = sm.OLS(y, X).fit(cov_type="HC3")
        i = names.index(f"G:{Ecols[0]}")
        out[f"beta_{tag}"] = float(res.params[i])
        out[f"se_{tag}"] = float(res.bse[i])
        out[f"p_{tag}"] = float(res.pvalues[i])
    out["delta_beta"] = out["beta_keller"] - out["beta_base"]
    out["rel_shift"] = out["delta_beta"] / abs(out["beta_base"]) if out["beta_base"] else np.nan
    out["same_sign"] = bool(np.sign(out["beta_base"]) == np.sign(out["beta_keller"]))
    return out


def _product_smd(df, treat_col, Ecols, Ccols) -> float:
    """SMD massimo sui PRODOTTI C×E: il matching bilancia i main effect,
    non necessariamente i loro prodotti."""
    worst = 0.0
    t = df[treat_col] == 1
    for c in Ccols:
        for e in Ecols:
            p = df[c] * df[e]
            sd = np.sqrt((p[t].var() + p[~t].var()) / 2)
            if sd > 0:
                worst = max(worst, abs(p[t].mean() - p[~t].mean()) / sd)
    return float(worst)


def run(variants_csv: str, out_csv: str) -> pd.DataFrame:
    # import pesanti qui, così fit_both resta testabile da solo
    from gene_environment.analysis.matching import check_balance, match_control_units
    from gene_environment.config import get_config
    from gene_environment.vcf_pipeline.build_dataset import load_and_prepare_data

    cfg = get_config()
    df, _, mapping, Ecols, _, covariate_cols = load_and_prepare_data(cfg)
    orig_to_safe = {v: k for k, v in mapping.items()}
    labels = pd.read_csv(variants_csv)["variant"].tolist()

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
        r = fit_both(m, cfg.target_col, col, Ecols, covariate_cols)
        r["variant"] = lab
        r["n_matched"] = int(m.shape[0])
        r["max_smd_main"] = max(check_balance(m, "_match_variant", Ecols + covariate_cols).values())
        r["max_smd_CxE"] = _product_smd(m, "_match_variant", Ecols, covariate_cols)
        rows.append(r)

    res = pd.DataFrame(rows)
    res.to_csv(out_csv, index=False)
    ok = res.dropna(subset=["beta_base"]) if "beta_base" in res else res.iloc[0:0]
    if len(ok):
        print(f"{len(ok)} varianti fittate | segno conservato: {ok['same_sign'].mean():.0%} | "
              f"|shift relativo| mediano: {ok['rel_shift'].abs().median():.1%} | "
              f"massimo: {ok['rel_shift'].abs().max():.1%}")
    return res


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])