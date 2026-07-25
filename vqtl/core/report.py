"""
Step 8 - figures and report.md for the top loci (per-genotype boxplot,
phenotype x exposure scatter colored by genotype, beta_I forest plot,
executive summary with methodological caveats). Dosage comes from columns
of the DataFrame already built (no VCF re-reading).
"""
from __future__ import annotations

import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from gene_environment.logging_utils import get_logger

from vqtl.config import VqtlConfig
from vqtl.core.data import VqtlDataset
from vqtl.core.filter_candidates import genomic_inflation

log = get_logger(__name__)

N_TOP_FIGURES = 10


def _boxplot_by_genotype(dosage, y, snp_id, exp_col, out_path):
    df = pd.DataFrame({"genotype": pd.array(np.round(dosage), dtype="Int64"), "phenotype": y}).dropna()
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.boxplot(data=df, x="genotype", y="phenotype", hue="genotype", ax=ax, palette="Blues", legend=False)
    sns.stripplot(data=df, x="genotype", y="phenotype", ax=ax, color="black", alpha=0.3, size=3)
    ax.set_title(f"{snp_id}\n(exposure: {exp_col})", fontsize=9)
    ax.set_xlabel("Genotype (ALT dosage)")
    ax.set_ylabel("Phenotype")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _scatter_by_exposure(dosage, y, exposure, snp_id, exp_col, out_path):
    df = pd.DataFrame({"genotype": pd.array(np.round(dosage), dtype="Int64"), "phenotype": y, "exposure": exposure}).dropna()
    fig, ax = plt.subplots(figsize=(4, 4))
    for geno, color in zip([0, 1, 2], ["#c6dbef", "#6baed6", "#08306b"]):
        sub = df[df["genotype"] == geno]
        if len(sub) == 0:
            continue
        ax.scatter(sub["exposure"], sub["phenotype"], s=12, color=color, label=f"genotype={geno}", alpha=0.7)
        if len(sub) > 2:
            z = np.polyfit(sub["exposure"], sub["phenotype"], 1)
            xs = np.linspace(sub["exposure"].min(), sub["exposure"].max(), 20)
            ax.plot(xs, np.polyval(z, xs), color=color, linewidth=1.5)
    ax.set_xlabel(exp_col)
    ax.set_ylabel("Phenotype")
    ax.set_title(f"{snp_id} x {exp_col}", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _forest_plot(top_df, out_path):
    fig, ax = plt.subplots(figsize=(6, max(3, 0.4 * len(top_df))))
    y_pos = np.arange(len(top_df))
    ci = 1.96 * top_df["SE"]
    ax.errorbar(top_df["beta_I"], y_pos, xerr=ci, fmt="o", color="#08306b", ecolor="#6baed6", capsize=3)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    labels = [f"{row.SNP} x {row.exposure}" for row in top_df.itertuples()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("\u03b2_interaction (95% CI)")
    ax.set_title("Top interaction loci - forest plot")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_report(
    dataset: VqtlDataset, vcfg: VqtlConfig, cohort_dir: str, generation: int,
    vqtl_df: pd.DataFrame, candidates: pd.DataFrame, interaction_df: pd.DataFrame,
    rge_df: pd.DataFrame, perm_df: pd.DataFrame, robustness_df: pd.DataFrame,
    target_col: str,
) -> str:
    fig_dir = os.path.join(cohort_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    inv_mapping = {v: k for k, v in dataset.mapping.items()}
    top_interactions = interaction_df.sort_values("pval").head(N_TOP_FIGURES).reset_index(drop=True) if not interaction_df.empty else interaction_df
    y = dataset.df[target_col].to_numpy(dtype=float)

    fig_paths = []
    for i, row in top_interactions.iterrows():
        snp_id, exp_raw = row["SNP"], row["exposure"]
        safe_col = inv_mapping.get(snp_id)
        exp_std_col = dataset.exposure_std_cols.get(exp_raw)
        if safe_col is None or exp_std_col is None:
            continue
        dosage = dataset.df[safe_col].to_numpy(dtype=float)
        exposure_vals = dataset.df[exp_std_col].to_numpy(dtype=float)

        box_path = os.path.join(fig_dir, f"locus{i + 1}_{snp_id}_boxplot.png")
        _boxplot_by_genotype(dosage, y, snp_id, exp_raw, box_path)
        scatter_path = os.path.join(fig_dir, f"locus{i + 1}_{snp_id}_scatter.png")
        _scatter_by_exposure(dosage, y, exposure_vals, snp_id, exp_raw, scatter_path)
        fig_paths.append((snp_id, exp_raw, box_path, scatter_path))

    forest_path = os.path.join(fig_dir, "forest_top_interactions.png")
    if len(top_interactions) > 0:
        _forest_plot(top_interactions, forest_path)
    log.info("Generated %d per-locus figure pairs + forest plot.", len(fig_paths))

    lam_note = ""
    if not vqtl_df.empty and "Z" in vqtl_df.columns:
        lam_note = f"{genomic_inflation(vqtl_df['Z'].values):.3f}"

    p_col_used = vcfg.filter_p_column if (not vqtl_df.empty and vcfg.filter_p_column in vqtl_df.columns) else "P"

    md = []
    md.append(f"# vQTL / G x E report -- generation (cohort): {generation}\n")
    md.append("## Executive summary\n")
    md.append(f"- Analysis-ready samples: **{len(dataset.df)}**\n")
    md.append(f"- Exposures tested: **{', '.join(vcfg.exposures)}**\n")
    md.append(f"- SNPs scanned (Step 3): **{len(vqtl_df)}**\n")
    md.append(
        f"- Candidate SNPs after the Step 4 filter: **{len(candidates)}** (filtered on `{p_col_used}`"
        f"{' < ' + str(vcfg.filter_p_threshold) if not vcfg.filter_top_n else ', top_n=' + str(vcfg.filter_top_n)})\n"
    )
    md.append(f"- Genomic inflation (lambda_GC): **{lam_note}**\n")
    if lam_note and float(lam_note) > 1.5:
        md.append(
            "  > **Note:** lambda_GC is markedly > 1. The asymptotic p-values from Step 3 are "
            "anti-conservative for a discrete dosage predictor (0/1/2) and should be treated as "
            "screening. Prefer `P_gc` and the empirical permutation p-values (Step 7) for inference "
            "on the top loci.\n"
        )
    md.append(f"- Interaction tests run: **{len(interaction_df)}** (candidate-SNP x exposure pairs)\n")
    if not rge_df.empty:
        n_rge = int(rge_df.get("rGE_flag", pd.Series(dtype=bool)).sum())
        n_het = int(rge_df.get("heteroscedasticity_flag", pd.Series(dtype=bool)).sum())
        md.append(f"- SNP x exposure pairs flagged for rGE (p<{vcfg.rge_het_alpha}): **{n_rge}** / {len(rge_df)} (flagged, not excluded)\n")
        md.append(f"- Pairs flagged for heteroscedasticity (Breusch-Pagan p<{vcfg.rge_het_alpha}): **{n_het}** / {len(rge_df)}\n")

    md.append("\n## Top vQTL loci (genome-wide scan)\n")
    if not vqtl_df.empty:
        md.append(vqtl_df.sort_values("P").head(10).to_markdown(index=False))
    md.append("\n")

    md.append("\n## Top SNP x exposure interaction loci\n")
    if not top_interactions.empty:
        md.append(top_interactions.to_markdown(index=False))
    md.append("\n")

    if not perm_df.empty:
        md.append("\n## Permutation-based robustness of the top loci (interaction + variance by genotype)\n")
        md.append(perm_df.to_markdown(index=False))
        md.append(
            "\n> Interaction: empirical p-value = (1 + #permutations with |beta_I_perm| >= |beta_I_observed|) / (n_perm + 1), "
            "Freedman-Lane permutation on the reduced model's residuals. "
            "Variance by genotype (levene_stat_observed/levene_pval): permutation-based Levene "
            "(Brown-Forsythe) test -- the genotype LABELS are permuted (not the residuals) on the "
            "residualized phenotype, using the same permutation infrastructure as the interaction test "
            "but a different statistic; provides an assumption-light confirmation of the variance effect "
            "detected by the Step 3 scan for this locus, without the asymptotic assumptions of quantile "
            "regression. Independent of exposure (same value across rows for the same SNP). "
            "Large discrepancies between the asymptotic and empirical p-values (in either test) indicate "
            "that the asymptotic SEs/p-values are not reliable for that locus.\n"
        )

    if not robustness_df.empty:
        md.append("\n## Sensitivity to phenotype transformations / outliers (top loci)\n")
        md.append(robustness_df.to_markdown(index=False))
        md.append(
            "\n> A locus whose direction/significance of beta_I is stable across `original`, "
            "`log_transform`, `rank_inverse_normal`, and `outliers_removed` is more likely to be a real "
            "effect than an artifact of the phenotype's distribution or of a few influential "
            "observations.\n"
        )

    md.append("\n## Known methodological limitations (read before drawing conclusions)\n")
    md.append(
        "- **Blind spot of the two-step design:** Step 3 is a *screening* step for SNPs whose dosage "
        "modulates the phenotype's dispersion; Step 5 only tests the interaction for SNPs that pass this "
        "filter. A SNP with a true G x E interaction but **no marginal effect on variance** can be missed "
        "entirely by this design. If you have a priori candidate SNPs/genes, test them directly with "
        "Step 5 regardless of their Step 3 p-value.\n"
    )
    md.append("- **Asymptotic SE inflation:** see the lambda_GC note above. Use the permutation p-values (Step 7) as the final word on any locus intended for publication/follow-up.\n")
    md.append("- **Loci flagged for rGE:** a significant SNP~exposure association does not by itself invalidate a G x E result, but it complicates causal interpretation and should be discussed explicitly.\n")
    md.append("- **Multiple testing:** the interaction/rGE/heteroscedasticity tables report nominal p-values, not corrected for the number of tested combinations (use Bonferroni or FDR).\n")

    md.append("\n## Figures\n")
    md.append("![Manhattan plot](figures/manhattan_vqtl.png)\n")
    md.append("![QQ plot](figures/qq_vqtl.png)\n")
    if os.path.exists(forest_path):
        md.append("![Forest plot](figures/forest_top_interactions.png)\n")
    for snp_id, exp_raw, box_path, scatter_path in fig_paths:
        md.append(f"\n### {snp_id} x {exp_raw}\n")
        md.append(f"![boxplot](figures/{os.path.basename(box_path)}) ![scatter](figures/{os.path.basename(scatter_path)})\n")

    report_path = os.path.join(cohort_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(md))
    log.info("Wrote %s", report_path)
    return report_path
