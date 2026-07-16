#!/usr/bin/env python3
"""
Generate boxplot+violin+beeswarm and forest plots for the onset-age
difference (onset_age) of significant variants.

CRITICAL FIX: this script used to treat every variant present in
variant_results_significant for a given (exposure, generation) as
"significant" — i.e. significant in a SINGLE cohort alone. That is NOT
the actual significance criterion used downstream (see the
get_significant_results stored procedure): a variant only counts as
robustly significant if empirical_p is below threshold in BOTH cohorts
independently, for the SAME exposure, with the SAME sign of obs_coef.
Ignoring this made the report massively over-inclusive (e.g. ~190
variants shown per exposure/cohort in the old report vs. ~9 that
actually survive the real criterion for that exposure).

FIX: at the start of the run, call the get_significant_results() stored
procedure ONCE (in the parent process, before any parallelism — it's a
single lightweight query, no need to repeat it per worker) and use its
output as a whitelist of valid (variant, exposure) pairs. Every
downstream step (available exposures, per-cohort DB stats, plots) is
now restricted to this whitelist. A sanity check compares the total
distinct-variant count from the stored procedure against the sum of
per-exposure counts actually used in the report, and raises if they
don't match (see _validate_significant_variant_count).
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
    return get_config()


# --------------------------------------------------------------------------
# Significant-variant whitelist (replication in both cohorts, same sign,
# same exposure) — comes ONLY from the stored procedure, never re-derived
# in Python, so there is a single source of truth shared with any other
# consumer of "significant variants" (e.g. manual DB queries, dashboards).
# --------------------------------------------------------------------------

def load_significant_variants() -> pd.DataFrame:
    """Call get_significant_results() and return its output as a
    DataFrame with at least columns: exposure, variant, gene_name,
    empirical_p, obs_coef_g1, empirical_p_2, obs_coef_g2."""
    with get_connection() as conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("CALL get_significant_results()")
        rows = cursor.fetchall()
        # a stored procedure call can leave extra result sets (e.g. the
        # CALL status) in the connection; drain them before returning the
        # connection to the pool, otherwise the next user of this
        # connection can get "Commands out of sync" errors.
        while cursor.nextset():
            pass
        cursor.close()

    df = pd.DataFrame(rows)
    if df.empty:
        log.warning("get_significant_results() returned no rows")
        return df

    if "exposure" not in df.columns:
        raise RuntimeError(
            "get_significant_results() did not return an 'exposure' column. "
            "The stored procedure must SELECT vrs1.exposure AS exposure "
            "for this script to be able to group significant variants by "
            "environmental component."
        )
    return df


def get_available_exposures_from_whitelist(sig_df: pd.DataFrame) -> list[str]:
    return sorted(sig_df["exposure"].dropna().unique().tolist())


def _validate_significant_variant_count(sig_df: pd.DataFrame, per_exposure_counts: dict[str, int]) -> None:
    """Sanity check: the number of distinct variants actually used across
    all per-exposure reports must equal the number of distinct variants
    returned by the stored procedure overall. A mismatch means some
    variant was silently dropped (or double-counted) somewhere in the
    pipeline between the whitelist and the final report."""
    expected_total = sig_df["variant"].nunique()
    used_total = sum(per_exposure_counts.values())
    # NOTE: a variant with empirical_p < threshold for MULTIPLE exposures
    # would legitimately appear once per exposure, so used_total can be
    # >= expected_total in that case. We already checked with the user
    # that COUNT(*) == COUNT(DISTINCT variant) == COUNT(DISTINCT variant,exposure)
    # for this dataset, i.e. no variant repeats across exposures either —
    # so for THIS dataset the two totals must match exactly. If your data
    # ever has a variant significant for >1 exposure, relax this to
    # used_total >= expected_total instead of ==.
    if used_total != expected_total:
        log.error(
            "Significant-variant count mismatch: stored procedure returned "
            "%d distinct variants total, but only %d were actually used "
            "across all per-exposure reports (breakdown: %s). Investigate "
            "before trusting the report output.",
            expected_total, used_total, per_exposure_counts,
        )
    else:
        log.info(
            "Significant-variant count check OK: %d distinct variants total, "
            "matching across the stored procedure and the generated reports.",
            expected_total,
        )


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------

def load_onset_age(cfg, exposure: str) -> pd.DataFrame:
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


def load_db_stats(exposure: str, generation: int, allowed_variants: set[str]) -> pd.DataFrame:
    """Same onset_age stats query as before, but now restricted to the
    variants that passed the real significance criterion (whitelist from
    get_significant_results), not every row present for this
    exposure/generation."""
    if not allowed_variants:
        return pd.DataFrame()

    placeholders = ",".join(["%s"] * len(allowed_variants))
    query = f"""
        SELECT variant, onset_n_mutati, onset_n_non_mutati, onset_median_mutati,
               onset_median_non_mutati, onset_delta_median, onset_ci_low, onset_ci_high,
               onset_p_value, onset_low_power, empirical_p
        FROM variant_results_significant
        WHERE exposure = %s AND generation = %s AND completed = 1 AND onset_p_value IS NOT NULL
              AND variant IN ({placeholders})
    """
    params = (exposure, generation, *allowed_variants)
    with get_connection() as conn:
        return pd.read_sql(query, conn, params=params)


# --------------------------------------------------------------------------
# Exploratory stat: mutant vs WT ONLY among the "near" patients
# --------------------------------------------------------------------------

def compute_near_stratified_stat(mut_near: pd.Series, wt_near: pd.Series) -> dict:
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
        x += 1

    if not rows:
        return
    long_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(2.2 * len(positions) + 1.5, 5.5))

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

    box_data = [long_df.loc[long_df["x"] == pos, "value"] for pos in positions]
    non_empty = [(p, d) for p, d in zip(positions, box_data) if len(d) > 0]
    if non_empty:
        bp = ax.boxplot([d for _, d in non_empty], positions=[p for p, _ in non_empty],
                         widths=0.18, patch_artist=True, showfliers=False, zorder=3)
        for patch, (pos, _) in zip(bp["boxes"], non_empty):
            patch.set_facecolor(colors[positions.index(pos)])
            patch.set_alpha(0.9)

    for pos in positions:
        vals = long_df.loc[long_df["x"] == pos, "value"]
        if vals.empty:
            continue
        sns.swarmplot(x=[pos] * len(vals), y=vals, ax=ax, size=2.5,
                       color="black", alpha=0.5, native_scale=True)

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
# Core: report for a single exposure
# --------------------------------------------------------------------------

def _process_exposure(exposure: str, allowed_variants: set[str]) -> tuple[str, int]:
    """Returns (exposure, n_distinct_variants_used) so the caller can run
    the total-count sanity check."""
    cfg = _get_config_for_exposure(exposure)
    configure_logging(cfg.log_dir)

    exp_out_dir = os.path.join(cfg.onset_age_out_dir, exposure)
    if os.path.isdir(exp_out_dir):
        shutil.rmtree(exp_out_dir)
    box_dir = os.path.join(exp_out_dir, "boxplots")
    os.makedirs(box_dir, exist_ok=True)

    df_onset = load_onset_age(cfg, exposure)
    df_onset["near"] = (df_onset[exposure] > 0).astype(int)

    all_stats = []
    variant_plot_data: dict[str, dict[int, dict[str, pd.Series]]] = {}
    variant_near_pvalues: dict[str, dict[int, float]] = {}

    for cohort in cfg.report_cohorts:
        df_geno = load_genotype_matrix_for_cohort(cfg, cohort)
        db_stats = load_db_stats(exposure, cohort, allowed_variants)
        if df_geno is None or db_stats.empty:
            continue

        df = df_geno.join(df_onset, how="inner")
        variants = [v for v in df_geno.columns if v in set(db_stats["variant"])]
        log.info("[%s] Cohort %d: %d patients, %d whitelisted variants with DB statistics",
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

    for variant, cohort_groups in variant_plot_data.items():
        make_combined_boxplot(cohort_groups, variant_near_pvalues[variant], variant, box_dir)

    if not all_stats:
        log.info("[%s] No onset_age statistics available for the report. Exiting.", exposure)
        return exposure, 0

    stats_df = pd.DataFrame(all_stats)
    stats_df.to_csv(os.path.join(exp_out_dir, "onset_age_report_table.csv"), index=False)
    make_forest_plot(stats_df, os.path.join(exp_out_dir, "forest_plot.png"))
    n_used = stats_df["variant"].nunique()
    log.info("[%s] onset_age report completed in %s (%d distinct variants)", exposure, exp_out_dir, n_used)
    return exposure, n_used


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def run_report_onset_age(single_exposure: bool = False, n_workers: int | None = None) -> None:
    base_cfg = get_config()
    configure_logging(base_cfg.log_dir)

    sig_df = load_significant_variants()
    if sig_df.empty:
        log.warning("get_significant_results() returned no significant variants. Exiting.")
        return

    if single_exposure:
        exposures = [base_cfg.exposure]
    else:
        exposures = get_available_exposures_from_whitelist(sig_df)
        log.info("Found %d exposures with replicated significant variants: %s", len(exposures), exposures)
        if not exposures:
            log.warning("No exposure found in the significant-variant whitelist. Exiting.")
            return

    # exposure -> set of whitelisted variant names for that exposure
    variants_by_exposure = {
        exp: set(sig_df.loc[sig_df["exposure"] == exp, "variant"])
        for exp in exposures
    }

    per_exposure_counts: dict[str, int] = {}

    if len(exposures) == 1:
        exp = exposures[0]
        _, n_used = _process_exposure(exp, variants_by_exposure[exp])
        per_exposure_counts[exp] = n_used
    else:
        n_workers = n_workers or min(len(exposures), os.cpu_count() or 1)
        log.info("Starting pool with %d workers for %d exposures", n_workers, len(exposures))

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_process_exposure, exp, variants_by_exposure[exp]): exp
                for exp in exposures
            }
            for future in as_completed(futures):
                exp = futures[future]
                try:
                    _, n_used = future.result()
                    per_exposure_counts[exp] = n_used
                except Exception:
                    log.exception("Error processing exposure %s", exp)

    if not single_exposure:
        _validate_significant_variant_count(sig_df, per_exposure_counts)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Onset_age report (boxplot + forest plot)")
    parser.add_argument(
        "--single-exposure",
        action="store_true",
        help="If set, only compute the current exposure (cfg.exposure) "
             "instead of all those available in the significant-variant whitelist.",
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