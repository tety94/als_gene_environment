from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests

from gene_environment.logging_utils import get_logger

log = get_logger(__name__)


def add_fdr(df, p_col: str = "empirical_p", fdr_col: str = "fdr"):
    df = df.copy()
    pvals = df[p_col].astype(float).values
    if len(pvals) == 0:
        df[fdr_col] = []
        return df
    df[fdr_col] = multipletests(pvals, method="fdr_bh")[1]
    return df


def volcano_plot(
    df,
    beta_col: str = "obs_coef",
    p_col: str = "empirical_p",
    variant_col: str = "variant",
    p_thresh: float = 1e-5,
    fdr_col: str = "fdr",
    fdr_thresh: float = 0.05,
    save_path: str = None,
    min_p_for_log: float = 1e-300,
):
    if not save_path:
        raise ValueError("save_path è obbligatorio: niente plt.show() in job non interattivi.")

    df = df.copy()
    safe_p = df[p_col].astype(float).clip(lower=min_p_for_log)
    df["neglog10p"] = -np.log10(safe_p)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(df[beta_col], df["neglog10p"], alpha=0.6, label="tutte le varianti")
    ax.axhline(-np.log10(p_thresh), linestyle="--", color="red", label=f"p = {p_thresh}")

    if fdr_col in df.columns:
        sig_fdr = df[df[fdr_col] < fdr_thresh]
        ax.scatter(
            sig_fdr[beta_col], sig_fdr["neglog10p"],
            s=50, edgecolor="black", label=f"FDR < {fdr_thresh}", color="orange",
        )

    ax.set_xlabel("Beta dell'interazione")
    ax.set_ylabel("-log10(p)")
    ax.set_title("Volcano Plot: Interazioni Gene x Ambiente")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Volcano plot salvato in %s", save_path)
