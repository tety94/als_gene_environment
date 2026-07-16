#!/usr/bin/env python3
"""
Generate boxplot+violin+beeswarm and forest plots for the onset-age
difference (onset_age) of significant variants.

The main statistics (median, delta, bootstrap CI, p-value, FDR) are NOT
recomputed here: they were already computed once in modeling.py and saved
to the DB for every variant (see analysis/modeling.py and
db/repository.py). This script only: (1) reads those statistics from the
DB, (2) reads the raw per-patient data (genotype + onset_age + exposure) to
draw the plots, (3) produces the figures.

Design notes
------------
- Parallelized at the exposure level (each environmental component runs in
  its own process, since the runs are independent: different DB queries,
  different output directories).
- `single_exposure` parameter:
    * False (default) -> fetches all distinct exposures available in
      variant_results from the DB and processes all of them, in parallel.
    * True -> only computes the current exposure (cfg.exposure), useful
      for quick debugging on a single environmental component.
- Each figure now shows BOTH cohort 1 and cohort 2 side by side, in two
  blocks of 4 groups each: WT-Far, WT-Near, Mutant-Far, Mutant-Near.
  "Near" is defined as the continuous exposure column in the csv being > 0
  (see load_onset_age / the "near" column built in _process_exposure).
- Each box is drawn as violin + boxplot + beeswarm (individual points),
  overlaid on the same x position.
- An exploratory Mann-Whitney U p-value between Mutant-Near and WT-Near is
  computed ON THE FLY per cohort (NOT saved to the DB, NOT FDR-corrected —
  it is additional to the global onset_p_value already computed in
  modeling.py) and annotated as a bracket above those two groups.
- At the start of every run, all previously generated images/tables for
  that exposure are deleted and regenerated from scratch (no stale files
  left behind from earlier runs).

IMPORTANT NOTE on get_config(): it is a singleton read once from env vars
(see config.py), so it does NOT accept (and cannot accept) an `exposure`
parameter to build a different config per environmental component.
cfg.onset_age_out_dir is therefore a FIXED path, identical for every
worker. To prevent parallel processes on different exposures from writing
into the same folder and overwriting each other, every worker creates its
own subfolder <onset_age_out_dir>/<exposure>/.

Dependency note: this version uses seaborn (>=0.12, for swarmplot's
native_scale support) in addition to matplotlib. Install with
`pip install seaborn` if not already present in the environment.
"""
from __future__ import annotations

import argparse
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib
from scipy.stats import mannwhitneyu

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from gene_environment.config import get_config
from gene_environment.db.connection import get_connection
from gene_environment.logging_utils import configure_logging, get_logger
from gene_environment.utils.id_utils import clean_sample_id

log = get_logger(__name__)

GROUP_ORDER = ["WT-Far", "WT-Near", "Mutant-Far", "Mutant-Near"]
GROUP_COLORS = {
    "WT-Far": "#B0B0B0",
    "WT-Near": "#6FA8DC",
    "Mutant-Far": "#F6B26B",
    "Mutant-Near": "#E06666",
}


# --------------------------------------------------------------------------
# Config helper
# --------------------------------------------------------------------------

def _get_config_for_exposure(exposure: str | None = None):
    """Return the global config.

    get_config() is a singleton read once from env vars: there is no (and
    no need for a) different config per exposure. The `exposure` parameter
    here is only kept for call-signature compatibility, but it is not
    passed to get_config(). Per-exposure differentiation instead happens
    at the output-directory level, inside _process_exposure.
    """
    return get_config()


# --------------------------------------------------------------------------
# Discovering available exposures
# --------------------------------------------------------------------------

def get_available_exposures(cfg) -> list[str]:
    query = """
        SELECT DISTINCT exposure
        FROM variant_results_significant
        WHERE completed = 1 AND onset_p_value IS NOT NULL
    """
    with get_connection() as conn:
        df = pd.read_sql(query, conn)
    return sorted(df["exposure"].dropna().unique().tolist())


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------

def load_onset_age(cfg, exposure: str) -> pd.DataFrame:
    """Load onset_age + covariates + the current exposure column (needed to
    stratify near/far). dict.fromkeys avoids duplicate columns in case
    exposure is already among the covariates."""
    cols = list(dict.fromkeys(
        [cfg.sample_id_col, cfg.target_col, exposure] + cfg.covariates
    ))
    df = pd.read_csv(cfg.env_file, usecols=cols)
    return df.set_index(cfg.sample_id_col)


def load_genotype_matrix_for_cohort(cfg, cohort: int) -> pd.DataFrame | None:
    combined_path = os.path.join(cfg.significant_matrix_dir, "combined_significant_variants.csv")
    if not os.path.exists(combined_path):
        log.warning("Missing %s (run extract_matrix first)", combined_path)
        return None

    df_all = pd.read_csv(combined_path, index_col="id")
    df_cohort = df_all[df_all["generation"] == cohort].drop(columns=["generation"])
    if df_cohort.empty:
        log.warning("No patients for cohort %d in %s", cohort, combined_path)
        return None

    df_cohort.index = [clean_sample_id(idx) for idx in df_cohort.index]
    df_cohort.index.name = "id"
    return df_cohort


def load_db_stats(exposure: str, generation: int) -> pd.DataFrame:
    query = """
        SELECT variant, onset_n_mutati, onset_n_non_mutati, onset_median_mutati,
               onset_median_non_mutati, onset_delta_median, onset_ci_low, onset_ci_high,
               onset_p_value, onset_low_power, empirical_p
        FROM variant_results_significant
        WHERE exposure = %s AND generation = %s AND completed = 1 AND onset_p_value IS NOT NULL
    """
    with get_connection() as conn:
        return pd.read_sql(query, conn, params=(exposure, generation))


# --------------------------------------------------------------------------
# Exploratory stat: mutant vs WT ONLY among the "near" patients
# --------------------------------------------------------------------------

def compute_near_stratified_stat(mut_near: pd.Series, wt_near: pd.Series) -> dict:
    """Mann-Whitney U between mutant-near and wt-near. Not saved to the DB,
    not FDR-corrected: it's an exploratory comparison to understand what
    happens specifically within the exposed subgroup."""
    if len(mut_near) < 2 or len(wt_near) < 2:
        return {
            "near_n_mutant": len(mut_near),
            "near_n_wt": len(wt_near),
            "near_median_mutant": mut_near.median() if len(mut_near) else np.nan,
            "near_median_wt": wt_near.median() if len(wt_near) else np.nan,
            "near_p_value": np.nan,
        }
    _, p = mannwhitneyu(mut_near, wt_near, alternative="two-sided")
    return {
        "near_n_mutant": len(mut_near),
        "near_n_wt": len(wt_near),
        "near_median_mutant": mut_near.median(),
        "near_median_wt": wt_near.median(),
        "near_p_value": p,
    }


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def _add_significance_bracket(ax, x1: float, x2: float, y: float, text: str) -> None:
    bar_height = y * 0.02
    ax.plot([x1, x1, x2, x2], [y, y + bar_height, y + bar_height, y],
            lw=1.2, color="black")
    ax.text((x1 + x2) / 2, y + bar_height, text, ha="center", va="bottom", fontsize=8)


def make_combined_boxplot(cohort_groups: dict[int, dict[str, pd.Series]],
                           near_pvalues: dict[int, float],
                           variant: str, out_dir: str) -> None:
    """One figure per variant, with one block of 4 groups (WT-Far, WT-Near,
    Mutant-Far, Mutant-Near) per available cohort, side by side. Each group
    is drawn as violin + boxplot + beeswarm points overlaid at the same x
    position. A significance bracket (Mann-Whitney U, computed on the fly)
    is drawn between WT-Near and Mutant-Near for each cohort block."""
    cohorts = sorted(cohort_groups.keys())

    positions: list[int] = []
    labels: list[str] = []
    colors: list[str] = []
    rows = []
    block_bounds: dict[int, tuple[int, int]] = {}
    x = 0
    for cohort in cohorts:
        start = x
        for grp in GROUP_ORDER:
            series = cohort_groups[cohort].get(grp, pd.Series(dtype=float))
            positions.append(x)
            labels.append(grp)
            colors.append(GROUP_COLORS[grp])
            for v in series:
                rows.append({"x": x, "group": grp, "value": v})
            x += 1
        block_bounds[cohort] = (start, x - 1)
        x += 1  # gap between cohort blocks

    if not rows:
        return
    long_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(2.2 * len(positions) + 1.5, 5.5))

    # Violin layer (background)
    for pos, color in zip(positions, colors):
        vals = long_df.loc[long_df["x"] == pos, "value"]
        if len(vals) < 2:
            continue
        parts = ax.violinplot([vals], positions=[pos], widths=0.8,
                               showmeans=False, showmedians=False, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.35)
            body.set_edgecolor("none")

    # Boxplot layer (narrow, on top of the violin)
    box_data = [long_df.loc[long_df["x"] == pos, "value"] for pos in positions]
    non_empty = [(p, d) for p, d in zip(positions, box_data) if len(d) > 0]
    if non_empty:
        bp = ax.boxplot([d for _, d in non_empty], positions=[p for p, _ in non_empty],
                         widths=0.18, patch_artist=True, showfliers=False, zorder=3)
        for patch, (pos, _) in zip(bp["boxes"], non_empty):
            patch.set_facecolor(colors[positions.index(pos)])
            patch.set_alpha(0.9)

    # Beeswarm layer (individual points, non-overlapping along x)
    for pos in positions:
        vals = long_df.loc[long_df["x"] == pos, "value"]
        if vals.empty:
            continue
        sns.swarmplot(x=[pos] * len(vals), y=vals, ax=ax, size=2.5,
                       color="black", alpha=0.5, native_scale=True)

    # Significance brackets: WT-Near vs Mutant-Near, per cohort block
    for cohort in cohorts:
        start, end = block_bounds[cohort]
        wt_near_pos = start + GROUP_ORDER.index("WT-Near")
        mut_near_pos = start + GROUP_ORDER.index("Mutant-Near")
        block_vals = long_df.loc[(long_df["x"] >= start) & (long_df["x"] <= end), "value"]
        if block_vals.empty:
            continue
        y_bracket = block_vals.max() * 1.05
        p = near_pvalues.get(cohort, np.nan)
        p_text = "p = n/a" if pd.isna(p) else f"p = {p:.3g}"
        _add_significance_bracket(ax, wt_near_pos, mut_near_pos, y_bracket, p_text)

    # Cohort block separators and labels
    for cohort in cohorts:
        start, end = block_bounds[cohort]
        ax.text((start + end) / 2, ax.get_ylim()[1], f"Cohort {cohort}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
        if end + 1 < len(positions):
            ax.axvline(end + 1, color="gray", linestyle=":", linewidth=1)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Onset age")
    ax.set_title(variant)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{variant}.png"), dpi=150)
    plt.close(fig)


def make_forest_plot(stats_df: pd.DataFrame, out_path: str):
    stats_df = stats_df.sort_values(["variant", "cohort"]).reset_index(drop=True)
    if stats_df.empty:
        log.warning("No data available for the forest plot")
        return

    y_labels = [f"{r.variant}  (cohort {r.cohort})" for r in stats_df.itertuples()]
    y_pos = np.arange(len(stats_df))[::-1]

    xerr_low = stats_df["onset_delta_median"] - stats_df["onset_ci_low"]
    xerr_high = stats_df["onset_ci_high"] - stats_df["onset_delta_median"]
    palette = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
    colors = stats_df["cohort"].map(palette).fillna("gray")

    fig, ax = plt.subplots(figsize=(7, max(4, 0.35 * len(stats_df))))
    ax.errorbar(stats_df["onset_delta_median"], y_pos, xerr=[xerr_low, xerr_high],
                fmt="none", ecolor="gray", capsize=3, zorder=1)
    ax.scatter(stats_df["onset_delta_median"], y_pos, c=colors, zorder=2)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Median onset_age delta (mutant - WT) [95% bootstrap CI]")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=palette.get(c, "gray"), label=f"Cohort {c}")
        for c in sorted(stats_df["cohort"].unique())
    ]
    ax.legend(handles=handles, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Forest plot saved to %s", out_path)


# --------------------------------------------------------------------------
# Core: report for a single exposure (top-level function -> picklable for
# ProcessPoolExecutor)
# --------------------------------------------------------------------------

def _process_exposure(exposure: str) -> str:
    cfg = _get_config_for_exposure(exposure)
    configure_logging(cfg.log_dir)

    # cfg.onset_age_out_dir is FIXED (same for every process, since cfg is a
    # singleton). Every worker writes to its own per-exposure subfolder to
    # avoid parallel runs on different environmental components
    # overwriting each other.
    exp_out_dir = os.path.join(cfg.onset_age_out_dir, exposure)

    # Wipe previous outputs for this exposure before regenerating anything,
    # so stale images/tables from earlier runs never linger.
    if os.path.isdir(exp_out_dir):
        shutil.rmtree(exp_out_dir)
    box_dir = os.path.join(exp_out_dir, "boxplots")
    os.makedirs(box_dir, exist_ok=True)

    df_onset = load_onset_age(cfg, exposure)
    df_onset["near"] = (df_onset[exposure] > 0).astype(int)

    all_stats = []
    # variant -> {cohort: {group_name: series}}
    variant_plot_data: dict[str, dict[int, dict[str, pd.Series]]] = {}
    # variant -> {cohort: near_p_value}
    variant_near_pvalues: dict[str, dict[int, float]] = {}

    for cohort in cfg.report_cohorts:
        df_geno = load_genotype_matrix_for_cohort(cfg, cohort)
        db_stats = load_db_stats(exposure, cohort)
        if df_geno is None or db_stats.empty:
            continue

        df = df_geno.join(df_onset, how="inner")
        variants = [v for v in df_geno.columns if v in set(db_stats["variant"])]
        log.info("[%s] Cohort %d: %d patients, %d variants with DB statistics",
                  exposure, cohort, len(df), len(variants))

        for variant in variants:
            row = db_stats.loc[db_stats["variant"] == variant].iloc[0]
            sub = df[[variant, cfg.target_col, "near"]].dropna()
            sub[variant] = pd.to_numeric(sub[variant], errors="coerce")

            is_mut = sub[variant] == 1
            is_wt = sub[variant] == 0
            is_near = sub["near"] == 1
            is_far = sub["near"] == 0

            mut_near = sub.loc[is_mut & is_near, cfg.target_col]
            mut_far = sub.loc[is_mut & is_far, cfg.target_col]
            wt_near = sub.loc[is_wt & is_near, cfg.target_col]
            wt_far = sub.loc[is_wt & is_far, cfg.target_col]

            # need at least one mutant and one WT overall, otherwise there
            # is nothing to plot
            if (len(mut_near) + len(mut_far)) == 0 or (len(wt_near) + len(wt_far)) == 0:
                continue

            variant_plot_data.setdefault(variant, {})[cohort] = {
                "WT-Far": wt_far,
                "WT-Near": wt_near,
                "Mutant-Far": mut_far,
                "Mutant-Near": mut_near,
            }

            near_stat = compute_near_stratified_stat(mut_near, wt_near)
            variant_near_pvalues.setdefault(variant, {})[cohort] = near_stat["near_p_value"]

            all_stats.append({"variant": variant, "cohort": cohort, **row.to_dict(), **near_stat})

    # One combined figure per variant, with both cohorts side by side
    for variant, cohort_groups in variant_plot_data.items():
        make_combined_boxplot(cohort_groups, variant_near_pvalues[variant], variant, box_dir)

    if not all_stats:
        log.info("[%s] No onset_age statistics available for the report. Exiting.", exposure)
        return exposure

    stats_df = pd.DataFrame(all_stats)
    stats_df.to_csv(os.path.join(exp_out_dir, "onset_age_report_table.csv"), index=False)
    make_forest_plot(stats_df, os.path.join(exp_out_dir, "forest_plot.png"))
    log.info("[%s] onset_age report completed in %s", exposure, exp_out_dir)
    return exposure


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def run_report_onset_age(single_exposure: bool = False, n_workers: int | None = None) -> None:
    """
    Parameters
    ----------
    single_exposure : bool
        If False (default), fetches all available exposures from the DB
        (DISTINCT exposure from variant_results) and processes all of
        them, in parallel.
        If True, only computes the current exposure (cfg.exposure) —
        useful for quick debugging on a single environmental component.
    n_workers : int | None
        Number of parallel processes. Default: min(n. exposures, n. cpus).
    """
    base_cfg = get_config()
    configure_logging(base_cfg.log_dir)

    if single_exposure:
        exposures = [base_cfg.exposure]
    else:
        exposures = get_available_exposures(base_cfg)
        log.info("Found %d exposures to process: %s", len(exposures), exposures)
        if not exposures:
            log.warning("No exposure found in variant_results. Exiting.")
            return

    if len(exposures) == 1:
        # No benefit in spawning a process for a single exposure
        _process_exposure(exposures[0])
        return

    n_workers = n_workers or min(len(exposures), os.cpu_count() or 1)
    log.info("Starting pool with %d workers for %d exposures", n_workers, len(exposures))

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_process_exposure, exp): exp for exp in exposures}
        for future in as_completed(futures):
            exp = futures[future]
            try:
                future.result()
            except Exception:
                log.exception("Error processing exposure %s", exp)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Onset_age report (boxplot + forest plot)")
    parser.add_argument(
        "--single-exposure",
        action="store_true",
        help="If set, only compute the current exposure (cfg.exposure) "
             "instead of all those available in the DB.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel processes (default: min(n. exposures, n. cpus)).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_report_onset_age(single_exposure=args.single_exposure, n_workers=args.workers)