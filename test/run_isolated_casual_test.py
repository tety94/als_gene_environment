"""
Isolated power test: EVERY variant (from either default pool) is tested with
BOTH the G×E mechanism and the vQTL (pure_variance) mechanism, one variant at
a time against an otherwise-null panel.

WHY BOTH TESTS FOR EVERY VARIANT
---------------------------------
The previous version of this script only ran the G×E pipeline on variants
from DEFAULT_CAUSAL_VARIANTS and the vQTL pipeline on variants from
DEFAULT_PURE_VARIANCE_VARIANTS. That answers "does the pipeline recover a
variant on the mechanism it was designed for", but not "does a variant
designed for one mechanism spuriously show up on the other" (crosstalk /
specificity), and not "what is the combined detection rate across both
mechanisms for the full panel". This version answers both questions.

DESIGN
------
For every variant label in the union of DEFAULT_CAUSAL_VARIANTS and
DEFAULT_PURE_VARIANCE_VARIANTS, we generate ONE dataset in which that single
variant is simultaneously declared:
  - causal for G×E, using its real (beta_inter, beta_main) if the label is in
    DEFAULT_CAUSAL_VARIANTS, otherwise a null (0.0, 0.0) -- i.e. no G×E or
    main effect at all.
  - causal for vQTL, using its real sd_by_dosage if the label is in
    DEFAULT_PURE_VARIANCE_VARIANTS, otherwise a "flat" null sd_by_dosage
    (same sd at every dosage -> no heteroscedasticity, no true vQTL effect),
    using the same dosage keys as the rest of the pool.
All other variants in the panel remain null by construction of
gen_fake_data.py, so this isolates exactly the contribution of the single
variant, on both mechanisms simultaneously, in a single dataset (i.e. no
extra dataset generations vs. before -- one dataset per variant, not one per
variant per mechanism).

Both run_ge_interaction() and run_vqtl_debug() are then run on that same
dataset, and the label is looked up in both result tables.

*** ASSUMPTIONS ON gen_fake_data.py (verify -- file not available in this
chat) ***
  - DEFAULT_CAUSAL_VARIANTS: dict[str, tuple[float, float]]  (beta_inter, beta_main)
  - DEFAULT_PURE_VARIANCE_VARIANTS: dict[str, dict[int, float]]  (sd per dosage,
    dosage keys are presumably {0, 1, 2})
  - generate_dataset(out_dir=..., causal_variants=..., pure_variance_variants=..., **kw)
    with "everything else null by default" if not listed in the two dicts above.
  - run_ge_interaction() returns a results CSV with a "variant" column and a
    "p_emp" column, and tests every variant in the panel (not just causal
    ones) -- this is required for the crosstalk check to work, since a
    variant not marked causal for G×E still needs a p_emp entry.
  - run_vqtl_debug() reads causal/null SNPs from THIS dataset's ground_truth
    and compares asymptotic vs bootstrap; because every variant here IS
    explicitly declared in pure_variance_variants (real or flat-null), it is
    guaranteed to appear in the comparison table (not left to random null
    sampling).
  - The "flat null" sd_by_dosage default below assumes dosage keys {0,1,2}
    and sd=1.0 at each. If DEFAULT_PURE_VARIANCE_VARIANTS uses different
    dosage keys or a different natural noise scale, adjust FLAT_NULL_SD and
    FLAT_NULL_DOSAGES below.
If any of this doesn't match, adjust the generate_dataset() call and the
result-table column names -- the rest of the logic doesn't change.

WHAT IT TESTS, PER VARIANT
---------------------------
  ge_role:
    "power"              -> beta_inter != 0 (designed G×E effect): expect significant
    "fpr_control"        -> beta_inter == 0 but beta_main != 0 (designed
                             main-effect-only control): expect NOT significant
    "crosstalk_check"    -> not in DEFAULT_CAUSAL_VARIANTS at all (a vQTL-only
                             variant riding along with a null G×E config):
                             expect NOT significant; if significant it's a
                             crosstalk / specificity failure
  vqtl_role:
    "power"               -> label in DEFAULT_PURE_VARIANCE_VARIANTS: expect significant
    "crosstalk_check"     -> not in DEFAULT_PURE_VARIANCE_VARIANTS (a G×E-only
                              variant riding along with a flat/null vQTL
                              config): expect NOT significant; if significant
                              it's a crosstalk / specificity failure

CACHING / RESUME BEHAVIOUR
---------------------------
Each variant's outcome is persisted to isolated/<variant>/isolated_summary.json
as soon as it is computed. Before (re-)computing a variant, the script checks
for this file: if it exists and has status "ok", the variant is SKIPPED
entirely (no dataset regeneration, no re-running of either test) and the
saved result is reused as-is. Use --force to ignore this cache and recompute
everything. A variant whose previous run FAILED is always retried regardless
of --force, since a failed run has nothing useful to reuse.

Because individual variants can be skipped (cached) across runs, the final
combined outputs (isolated_power_curve.csv/.json, plots, Word report) are
NOT simply built from whatever this particular process run happened to
compute in memory. Instead, once all variants in the current plan have been
processed (freshly computed or reused from cache), the script re-reads every
isolated/<variant>/isolated_summary.json from disk to assemble the combined
table. This guarantees the combined recap always reflects exactly what is
saved on disk -- including results from earlier runs/processes -- rather than
only what happened to be computed in this particular invocation.

OUTPUT: isolated/<variant_label>/... (data + raw results) and a single combined
recap isolated/isolated_power_curve.csv + .json with, for every variant: both
mechanisms' declared parameters, recovery flags, p-values, plus
isolated/plots/*.png and isolated/isolated_causal_test_report.docx with
tables and summary plots for the paper (all in English).

HOW TO RUN IT (run from YOU, not from this chat, same folder as
run_scenarios.py):
    python run_isolated_causal_test.py                 # all default variants, sequential
    python run_isolated_causal_test.py --workers 4      # parallel, separate processes
    python run_isolated_causal_test.py --force          # ignore cached isolated_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless: no display required
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Direct reuse of the logic already written in run_scenarios.py -- no
# duplication: same run_ge_interaction/run_vqtl_debug/_set_common_env used
# there.
import run_scenarios as rs
from gen_fake_data import (
    generate_dataset,
    DEFAULT_CAUSAL_VARIANTS,
    DEFAULT_PURE_VARIANCE_VARIANTS,
)

ISOLATED_ROOT = os.path.join(SCRIPT_DIR, "isolated")
PLOTS_DIR = os.path.join(ISOLATED_ROOT, "plots")
PVALUE_THRESHOLD = 0.05

# Fallback "flat null" sd_by_dosage used for variants that are NOT in
# DEFAULT_PURE_VARIANCE_VARIANTS, so they can still be explicitly declared
# causal-for-vQTL-with-null-effect and therefore still appear in the vQTL
# comparison table. Derived from an existing pool entry if possible so the
# dosage keys match; otherwise falls back to {0,1,2}: 1.0.
def _derive_flat_null_sd() -> dict:
    if DEFAULT_PURE_VARIANCE_VARIANTS:
        sample = next(iter(DEFAULT_PURE_VARIANCE_VARIANTS.values()))
        dosages = list(sample.keys())
    else:
        dosages = [0, 1, 2]
    return {d: 1.0 for d in dosages}


FLAT_NULL_SD = _derive_flat_null_sd()
NULL_GE_BETAS = (0.0, 0.0)


def section(title: str) -> None:
    print("\n" + "#" * 88)
    print(title)
    print("#" * 88)


def build_variant_plan() -> list[dict]:
    """Union of both default pools -> one plan entry per unique label, with
    both mechanisms' parameters (real or null) and both roles."""
    labels = sorted(set(DEFAULT_CAUSAL_VARIANTS) | set(DEFAULT_PURE_VARIANCE_VARIANTS))
    plan = []
    for label in labels:
        in_ge_pool = label in DEFAULT_CAUSAL_VARIANTS
        in_vqtl_pool = label in DEFAULT_PURE_VARIANCE_VARIANTS

        beta_inter, beta_main = DEFAULT_CAUSAL_VARIANTS.get(label, NULL_GE_BETAS)
        sd_by_dosage = DEFAULT_PURE_VARIANCE_VARIANTS.get(label, FLAT_NULL_SD)

        if not in_ge_pool:
            ge_role = "crosstalk_check"
        elif beta_inter == 0.0:
            ge_role = "fpr_control"
        else:
            ge_role = "power"

        vqtl_role = "power" if in_vqtl_pool else "crosstalk_check"

        plan.append({
            "variant": label,
            "in_ge_pool": in_ge_pool,
            "in_vqtl_pool": in_vqtl_pool,
            "beta_inter": beta_inter,
            "beta_main": beta_main,
            "ge_role": ge_role,
            "sd_by_dosage": sd_by_dosage,
            "vqtl_role": vqtl_role,
        })
    return plan


# ============================================================
# Cache: if a variant already has an isolated_summary.json with status "ok",
# skip re-running it -- useful to resume after an error/interruption without
# redoing every variant. Bypassable with --force.
# ============================================================

def _summary_path(label: str) -> str:
    return os.path.join(ISOLATED_ROOT, label, "isolated_summary.json")


def _load_cached_result(var_dir: str, force: bool = False) -> dict | None:
    if force:
        return None
    summary_path = os.path.join(var_dir, "isolated_summary.json")
    if not os.path.isfile(summary_path):
        return None
    try:
        with open(summary_path) as f:
            cached = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None  # corrupted/incomplete file: recompute
    if cached.get("status") != "ok":
        return None  # a previously failed run should always be retried
    return cached


# ============================================================
# One variant at a time, both mechanisms
# ============================================================

def run_isolated_variant(plan_entry: dict, n_workers: int = 1, force: bool = False) -> dict:
    label = plan_entry["variant"]
    var_dir = os.path.join(ISOLATED_ROOT, label)
    fake_dir = os.path.join(var_dir, "fake_data")

    cached = _load_cached_result(var_dir, force=force)
    if cached is not None:
        print(f"[isolated] {label}: cached result found (status ok), skipping recompute.")
        return cached

    section(
        f"[isolated] {label}  "
        f"(GxE: beta_inter={plan_entry['beta_inter']}, beta_main={plan_entry['beta_main']}, "
        f"role={plan_entry['ge_role']} | vQTL: sd_by_dosage={plan_entry['sd_by_dosage']}, "
        f"role={plan_entry['vqtl_role']})"
    )
    os.makedirs(fake_dir, exist_ok=True)

    result: dict = {
        "variant": label,
        "in_ge_pool": plan_entry["in_ge_pool"],
        "in_vqtl_pool": plan_entry["in_vqtl_pool"],
        "beta_inter": plan_entry["beta_inter"],
        "beta_main": plan_entry["beta_main"],
        "ge_role": plan_entry["ge_role"],
        "sd_by_dosage": plan_entry["sd_by_dosage"],
        "vqtl_role": plan_entry["vqtl_role"],
        "status": "ok", "error": None,
    }
    try:
        generate_dataset(
            out_dir=fake_dir, verbose=True,
            causal_variants={label: (plan_entry["beta_inter"], plan_entry["beta_main"])},
            pure_variance_variants={label: plan_entry["sd_by_dosage"]},
        )
        vqtl_n_jobs = max(1, (os.cpu_count() or 2) // max(1, n_workers))
        rs._set_common_env(fake_dir, var_dir, vqtl_n_jobs=vqtl_n_jobs)

        # ---- G×E side ----
        ge_res = rs.run_ge_interaction(fake_dir, var_dir)
        result["ge_result"] = ge_res
        res_df = pd.read_csv(ge_res["results_csv"])
        row = res_df.loc[res_df["variant"] == label]
        if row.empty:
            result["ge_found_in_results"] = False
            result["p_ge"] = None
            result["significant_ge"] = False
        else:
            p_emp = float(row["p_emp"].iloc[0]) if "p_emp" in row.columns else None
            result["ge_found_in_results"] = True
            result["p_ge"] = p_emp
            result["significant_ge"] = (p_emp is not None) and (p_emp < PVALUE_THRESHOLD)

        if not result["ge_found_in_results"]:
            result["recovered_ge"] = False
        elif plan_entry["ge_role"] == "power":
            result["recovered_ge"] = result["significant_ge"]
        else:  # fpr_control or crosstalk_check: expect NOT significant
            result["recovered_ge"] = not result["significant_ge"]

        # ---- vQTL side ----
        debug_res = rs.run_vqtl_debug(fake_dir, var_dir)
        result["vqtl_debug"] = debug_res
        comparison = pd.read_csv(debug_res["comparison_csv"])
        vrow = comparison.loc[comparison["SNP"] == label]
        if vrow.empty:
            result["vqtl_found_in_results"] = False
            result["p_vqtl"] = None
            result["significant_vqtl"] = False
        else:
            p_asym = float(vrow["P_asym"].iloc[0]) if "P_asym" in vrow.columns else None
            result["vqtl_found_in_results"] = True
            result["p_vqtl"] = p_asym
            result["significant_vqtl"] = (p_asym is not None) and (p_asym < PVALUE_THRESHOLD)

        if not result["vqtl_found_in_results"]:
            result["recovered_vqtl"] = False
        elif plan_entry["vqtl_role"] == "power":
            result["recovered_vqtl"] = result["significant_vqtl"]
        else:  # crosstalk_check: expect NOT significant
            result["recovered_vqtl"] = not result["significant_vqtl"]

    except Exception as exc:
        result["status"] = "FAILED"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        print(f"\n*** VARIANT '{label}' FAILED: {result['error']} ***")
        traceback.print_exc()

    with open(os.path.join(var_dir, "isolated_summary.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


def _worker(plan_entry: dict, n_workers: int, force: bool) -> dict:
    return run_isolated_variant(plan_entry, n_workers=n_workers, force=force)


# ============================================================
# Final aggregation, ALWAYS read back from disk.
#
# Individual variants may have been skipped this run (cache hit) or computed
# by a different process (--workers) or in a previous invocation of the
# script entirely. Rather than trusting the in-memory list accumulated
# during this particular run, we re-read every isolated_summary.json from
# disk for the variants in the current plan. This is the single source of
# truth for the combined recap (isolated_power_curve.csv/.json + report),
# and makes the combined output correct even if:
#   - some variants were skipped via cache in this run,
#   - the process crashed/was interrupted partway through a previous run,
#   - variants were computed across multiple separate invocations.
# ============================================================

def collect_results_from_disk(plan: list[dict]) -> pd.DataFrame:
    rows = []
    missing = []
    for entry in plan:
        label = entry["variant"]
        summary_path = _summary_path(label)
        if not os.path.isfile(summary_path):
            missing.append(label)
            continue
        try:
            with open(summary_path) as f:
                rows.append(json.load(f))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[warn] could not read {summary_path}: {exc}")
            missing.append(label)

    if missing:
        print(
            f"\n[warn] {len(missing)} variant(s) have no readable isolated_summary.json on disk "
            f"and are excluded from the combined recap: {missing}"
        )

    return pd.DataFrame(rows)


# ============================================================
# Reporting: plots (matplotlib) + Word report (python-docx), always written
# under ISOLATED_ROOT/plots/ and ISOLATED_ROOT/isolated_causal_test_report.docx.
# Requires: pip install matplotlib python-docx
# ============================================================

def _fmt_p(p) -> str:
    if p is None or pd.isna(p):
        return "n/a"
    return f"{p:.4f}" if p >= 0.0005 else f"{p:.2e}"


def _sd_ratio(d):
    if not isinstance(d, dict) or not d:
        return None
    vals = list(d.values())
    return max(vals) / min(vals) if min(vals) > 0 else None


def generate_plots(df: pd.DataFrame) -> dict:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    paths = {}

    ge_power = df[df["ge_role"] == "power"].copy()
    ge_fpr = df[df["ge_role"] == "fpr_control"].copy()
    vqtl_power = df[df["vqtl_role"] == "power"].copy()
    ge_crosstalk = df[df["ge_role"] == "crosstalk_check"].copy()   # vQTL-only variants tested on GxE
    vqtl_crosstalk = df[df["vqtl_role"] == "crosstalk_check"].copy()  # GxE-only variants tested on vQTL

    # ---- 1) GxE power scatter: |beta_inter| vs outcome ----
    if not ge_power.empty:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        abs_beta = ge_power["beta_inter"].abs()
        colors = ge_power["recovered_ge"].map({True: "#1D9E75", False: "#D85A30"})
        pos = ge_power["beta_inter"] > 0
        ax.scatter(abs_beta[pos], [1] * pos.sum(), c=colors[pos], marker="o", s=70,
                   edgecolor="white", linewidth=0.5, label="positive sign")
        ax.scatter(abs_beta[~pos], [0] * (~pos).sum(), c=colors[~pos], marker="s", s=70,
                   edgecolor="white", linewidth=0.5, label="negative sign")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["negative beta_inter", "positive beta_inter"])
        ax.set_xlabel("|beta_inter|")
        ax.set_title("GxE isolated power test — outcome by effect magnitude and sign")
        legend_elems = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#1D9E75", markersize=9, label="recovered (significant)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#D85A30", markersize=9, label="missed (not significant)"),
        ]
        ax.legend(handles=legend_elems, loc="lower right", frameon=False)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        p = os.path.join(PLOTS_DIR, "ge_power_scatter.png")
        fig.savefig(p, dpi=150); plt.close(fig)
        paths["ge_power_scatter"] = p

        bins = list(range(0, 10))
        ge_power["bin"] = pd.cut(abs_beta, bins=bins, right=True)
        by_bin = ge_power.groupby("bin", observed=True)["recovered_ge"].agg(["sum", "count"])
        by_bin["power_pct"] = 100 * by_bin["sum"] / by_bin["count"]
        by_bin = by_bin[by_bin["count"] > 0]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar([str(b) for b in by_bin.index], by_bin["power_pct"], color="#2a78d6")
        for i in range(len(by_bin)):
            ax.text(i, by_bin["power_pct"].iloc[i] + 2,
                     f"{int(by_bin['sum'].iloc[i])}/{int(by_bin['count'].iloc[i])}", ha="center", fontsize=8)
        ax.set_ylim(0, 110)
        ax.set_ylabel("power (%)")
        ax.set_xlabel("|beta_inter| bin")
        ax.set_title("GxE detection rate by effect-size bin")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        p = os.path.join(PLOTS_DIR, "ge_power_by_bin.png")
        fig.savefig(p, dpi=150); plt.close(fig)
        paths["ge_power_by_bin"] = p

    # ---- 2) FPR controls (main-effect only) ----
    if not ge_fpr.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ge_fpr["recovered_ge"].map({True: "#1D9E75", False: "#E24B4A"})
        ax.bar(ge_fpr["variant"], ge_fpr["p_ge"].fillna(1.0), color=colors)
        ax.axhline(0.05, color="#898781", linestyle="--", linewidth=1, label="p = 0.05 threshold")
        ax.set_ylabel("p_emp (interaction test)")
        ax.set_title("Main-effect-only controls (beta_inter = 0) — false-positive check")
        ax.tick_params(axis="x", rotation=60, labelsize=7)
        ax.legend(frameon=False, loc="upper right")
        fig.tight_layout()
        p = os.path.join(PLOTS_DIR, "ge_fpr_controls.png")
        fig.savefig(p, dpi=150); plt.close(fig)
        paths["ge_fpr_controls"] = p

    # ---- 3) vQTL power scatter ----
    if not vqtl_power.empty:
        vqtl_power["sd_ratio"] = vqtl_power["sd_by_dosage"].apply(_sd_ratio)
        fig, ax = plt.subplots(figsize=(7, 4.2))
        colors = vqtl_power["recovered_vqtl"].map({True: "#1D9E75", False: "#D85A30"})
        ax.scatter(vqtl_power["sd_ratio"], vqtl_power["p_vqtl"].fillna(1.0), c=colors, s=70,
                   edgecolor="white", linewidth=0.5)
        ax.axhline(0.05, color="#898781", linestyle="--", linewidth=1, label="p = 0.05 threshold")
        ax.set_xlabel("sd ratio (max/min across dosage)")
        ax.set_ylabel("p (asymptotic)")
        ax.set_title("vQTL isolated power test — pure_variance variants")
        ax.legend(frameon=False, loc="upper right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = os.path.join(PLOTS_DIR, "vqtl_power_scatter.png")
        fig.savefig(p, dpi=150); plt.close(fig)
        paths["vqtl_power_scatter"] = p

    # ---- 4) Crosstalk / specificity summary ----
    ct_rows = []
    if not ge_crosstalk.empty:
        fp = int((~ge_crosstalk["recovered_ge"]).sum())
        ct_rows.append(("vQTL variants tested on GxE\n(expect: not significant)", fp, len(ge_crosstalk)))
    if not vqtl_crosstalk.empty:
        fp = int((~vqtl_crosstalk["recovered_vqtl"]).sum())
        ct_rows.append(("GxE variants tested on vQTL\n(expect: not significant)", fp, len(vqtl_crosstalk)))
    if ct_rows:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        labels = [r[0] for r in ct_rows]
        pct = [100 * r[1] / r[2] if r[2] else 0 for r in ct_rows]
        bars = ax.bar(labels, pct, color="#8a4fd6")
        for b, r in zip(bars, ct_rows):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                     f"{r[1]}/{r[2]}", ha="center", fontsize=9)
        ax.set_ylabel("spurious cross-mechanism detection rate (%)")
        ax.set_ylim(0, max(pct + [10]) * 1.3)
        ax.set_title("Cross-mechanism specificity check")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        p = os.path.join(PLOTS_DIR, "crosstalk_specificity.png")
        fig.savefig(p, dpi=150); plt.close(fig)
        paths["crosstalk_specificity"] = p

    # ---- 5) Overlap: power variants recovered by BOTH mechanisms ----
    both_defined = df[df["in_ge_pool"] & df["in_vqtl_pool"]].copy()
    if not both_defined.empty:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        cats = ["GxE only", "vQTL only", "Both", "Neither"]
        ge_ok = both_defined["recovered_ge"].fillna(False)
        vq_ok = both_defined["recovered_vqtl"].fillna(False)
        counts = [
            int((ge_ok & ~vq_ok).sum()),
            int((~ge_ok & vq_ok).sum()),
            int((ge_ok & vq_ok).sum()),
            int((~ge_ok & ~vq_ok).sum()),
        ]
        ax.bar(cats, counts, color=["#2a78d6", "#d68a2a", "#1D9E75", "#D85A30"])
        for i, c in enumerate(counts):
            ax.text(i, c + 0.1, str(c), ha="center", fontsize=9)
        ax.set_ylabel("number of variants")
        ax.set_title("Variants designed for BOTH mechanisms — detection overlap")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        p = os.path.join(PLOTS_DIR, "overlap_both_mechanisms.png")
        fig.savefig(p, dpi=150); plt.close(fig)
        paths["overlap_both_mechanisms"] = p

    return paths


def generate_word_report(df: pd.DataFrame, plot_paths: dict) -> str:
    ge_power = df[df["ge_role"] == "power"]
    ge_fpr = df[df["ge_role"] == "fpr_control"]
    vqtl_power = df[df["vqtl_role"] == "power"]
    ge_crosstalk = df[df["ge_role"] == "crosstalk_check"]
    vqtl_crosstalk = df[df["vqtl_role"] == "crosstalk_check"]

    doc = Document()
    doc.add_heading("Isolated causal variant test — combined results report", level=0)
    meta = doc.add_paragraph()
    meta.add_run(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}").italic = True
    doc.add_paragraph(
        "Every variant from either default pool was tested one at a time against an "
        "otherwise-null panel, on BOTH the G×E and the vQTL mechanism simultaneously "
        "(using a null/flat configuration on whichever mechanism it was not designed for). "
        "This gives, for every variant, both a power estimate on its designed mechanism "
        "and a specificity check on the other mechanism. Combined results below are "
        "assembled directly from the per-variant result files saved on disk."
    )

    doc.add_heading("Summary", level=1)
    if not ge_power.empty:
        n_rec, n_tot = int(ge_power["recovered_ge"].sum()), len(ge_power)
        doc.add_paragraph(
            f"GxE power test: {n_rec}/{n_tot} causal variants recovered "
            f"(overall power {100 * n_rec / n_tot:.1f}%)."
        )
    if not ge_fpr.empty:
        n_fp, n_tot = int((~ge_fpr["recovered_ge"]).sum()), len(ge_fpr)
        doc.add_paragraph(
            f"GxE false-positive controls (main effect only, beta_inter = 0): "
            f"{n_fp}/{n_tot} incorrectly flagged as significant "
            f"(false-positive rate {100 * n_fp / n_tot:.1f}%)."
        )
    if not vqtl_power.empty:
        n_rec, n_tot = int(vqtl_power["recovered_vqtl"].sum()), len(vqtl_power)
        doc.add_paragraph(
            f"vQTL power test (pure_variance): {n_rec}/{n_tot} causal variants recovered "
            f"(overall power {100 * n_rec / n_tot:.1f}%)."
        )
    if not ge_crosstalk.empty:
        n_fp, n_tot = int((~ge_crosstalk["recovered_ge"]).sum()), len(ge_crosstalk)
        doc.add_paragraph(
            f"Cross-mechanism check, vQTL-only variants tested on G×E: {n_fp}/{n_tot} "
            f"showed a spurious G×E signal ({100 * n_fp / n_tot:.1f}%)."
        )
    if not vqtl_crosstalk.empty:
        n_fp, n_tot = int((~vqtl_crosstalk["recovered_vqtl"]).sum()), len(vqtl_crosstalk)
        doc.add_paragraph(
            f"Cross-mechanism check, G×E-only variants tested on vQTL: {n_fp}/{n_tot} "
            f"showed a spurious vQTL signal ({100 * n_fp / n_tot:.1f}%)."
        )
    failed = df[df["status"] == "FAILED"]
    if not failed.empty:
        p = doc.add_paragraph()
        run = p.add_run(f"{len(failed)} variant run(s) raised an exception and could not be evaluated — see the detail table.")
        run.font.color.rgb = RGBColor(0xE2, 0x4B, 0x4A)

    if plot_paths:
        doc.add_heading("Plots", level=1)
        captions = {
            "ge_power_scatter": "GxE isolated power test — outcome by effect magnitude and sign.",
            "ge_power_by_bin": "GxE detection rate by effect-size bin.",
            "ge_fpr_controls": "Main-effect-only controls — false-positive check.",
            "vqtl_power_scatter": "vQTL isolated power test — pure_variance variants.",
            "crosstalk_specificity": "Cross-mechanism specificity check.",
            "overlap_both_mechanisms": "Detection overlap for variants designed on both mechanisms.",
        }
        for key in ["ge_power_scatter", "ge_power_by_bin", "ge_fpr_controls",
                    "vqtl_power_scatter", "crosstalk_specificity", "overlap_both_mechanisms"]:
            if key in plot_paths:
                doc.add_picture(plot_paths[key], width=Inches(6))
                cap = doc.add_paragraph(captions[key])
                cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in cap.runs:
                    run.italic = True
                    run.font.size = Pt(9)

    doc.add_heading("Detailed results", level=1)
    cols = ["variant", "in_ge_pool", "in_vqtl_pool", "ge_role", "beta_inter", "beta_main",
            "p_ge", "significant_ge", "recovered_ge",
            "vqtl_role", "sd_by_dosage", "p_vqtl", "significant_vqtl", "recovered_vqtl", "status"]
    cols = [c for c in cols if c in df.columns]
    table = doc.add_table(rows=1, cols=len(cols))
    table.style = "Light Grid Accent 1"
    for i, c in enumerate(cols):
        table.rows[0].cells[i].text = c
        table.rows[0].cells[i].paragraphs[0].runs[0].bold = True

    ordered = df.sort_values(["ge_role", "vqtl_role", "variant"])
    for _, row in ordered.iterrows():
        cells = table.add_row().cells
        for i, c in enumerate(cols):
            val = row[c]
            if c in ("p_ge", "p_vqtl"):
                val = _fmt_p(val)
            cells[i].text = "" if pd.isna(val) else str(val)
        for c in table.rows[-1].cells:
            for r in c.paragraphs[0].runs:
                r.font.size = Pt(8)

    out_path = os.path.join(ISOLATED_ROOT, "isolated_causal_test_report.docx")
    doc.save(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated test: every variant, both mechanisms")
    parser.add_argument("--workers", type=int, default=1,
                         help="Number of variants to test in parallel (separate processes). Default 1.")
    parser.add_argument("--force", action="store_true",
                         help="Recompute even variants that already have an isolated_summary.json with "
                              "status ok (default: skipped and cached result reused)")
    args = parser.parse_args()

    n_workers = max(1, args.workers)
    force = args.force
    os.makedirs(ISOLATED_ROOT, exist_ok=True)

    plan = build_variant_plan()
    print(f"{len(plan)} unique variants to test (union of both default pools), both mechanisms each.")

    t0 = time.time()

    if n_workers == 1:
        for entry in plan:
            run_isolated_variant(entry, n_workers=1, force=force)
    else:
        print(f"Running {len(plan)} isolated variants with {n_workers} parallel processes...")
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_worker, entry, n_workers, force): entry["variant"] for entry in plan}
            for fut in as_completed(futures):
                label = futures[fut]
                try:
                    res = fut.result()
                    status = res.get("status", "?")
                except Exception as exc:
                    status = "FAILED"
                    print(f"\n*** VARIANT '{label}' FAILED in worker process: {exc} ***")
                print(f"[{label}] done ({status}).")

    section("COMBINED RESULTS (assembled from disk)")
    # Always rebuild the combined table from the per-variant isolated_summary.json
    # files on disk, rather than from whatever this run's execution happened to
    # accumulate in memory. This is what makes the combined recap correct even
    # when some variants were skipped this run (cache hit) or computed in a
    # separate process/invocation.
    df = collect_results_from_disk(plan)
    if df.empty:
        print("No results found on disk -- nothing to report.")
        sys.exit(1)
    print(df.to_string(index=False))

    summary_csv = os.path.join(ISOLATED_ROOT, "isolated_power_curve.csv")
    df.to_csv(summary_csv, index=False)
    with open(os.path.join(ISOLATED_ROOT, "isolated_power_curve.json"), "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2, default=str)
    print(f"\n[export] {summary_csv}")

    section("REPORT — plots and Word")
    plot_paths = generate_plots(df)
    for name, p in plot_paths.items():
        print(f"[plot] {name}: {p}")
    report_path = generate_word_report(df, plot_paths)
    print(f"[export] {report_path}")

    print(f"\nCompleted in {time.time() - t0:.0f}s.")

    ok = df[df["status"] == "ok"]
    missed_power_ge = ok.loc[(ok["ge_role"] == "power") & (~ok["recovered_ge"]), "variant"].tolist()
    fp_ge = ok.loc[(ok["ge_role"] != "power") & (~ok["recovered_ge"]), "variant"].tolist()
    missed_power_vqtl = ok.loc[(ok["vqtl_role"] == "power") & (~ok["recovered_vqtl"]), "variant"].tolist()
    fp_vqtl = ok.loc[(ok["vqtl_role"] != "power") & (~ok["recovered_vqtl"]), "variant"].tolist()
    failed = df.loc[df["status"] == "FAILED", "variant"].tolist()

    if missed_power_ge:
        print(f"\n*** CAUSAL VARIANTS NOT RECOVERED (GxE power, isolated): {missed_power_ge} ***")
    if fp_ge:
        print(f"*** SPURIOUS GxE SIGNIFICANCE (fpr_control or crosstalk): {fp_ge} ***")
    if missed_power_vqtl:
        print(f"\n*** CAUSAL VARIANTS NOT RECOVERED (vQTL power, isolated): {missed_power_vqtl} ***")
    if fp_vqtl:
        print(f"*** SPURIOUS vQTL SIGNIFICANCE (crosstalk): {fp_vqtl} ***")
    if failed:
        print(f"*** FAILED VARIANTS (exception): {failed} ***")

    if missed_power_ge or fp_ge or missed_power_vqtl or fp_vqtl or failed:
        sys.exit(1)
    print("\n*** All isolated causal variants recovered correctly on both mechanisms. ***")


if __name__ == "__main__":
    main()