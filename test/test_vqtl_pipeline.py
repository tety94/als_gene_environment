"""
End-to-end vQTL pipeline library (Step 3: scan -> Step 4: filter ->
Step 5: interaction -> Step 6: rge_het -> Step 7: robustness+permutation).

Exposes a single public function, run_pipeline_for_method(), called by
run_scenarios.py (via run_vqtl_asymptotic) for the "full pipeline" part of
each scenario, and in general by any orchestrator that wants to run
Step 3->7 with a specific se_method on an already-prepared dataset.

PHILOSOPHY (unchanged): calls the REAL vqtl.core.* functions as they are
(run_vqtl_scan, filter_candidates, run_interaction_tests, run_rge_het,
run_robustness_and_permutation), no reimplementation of the statistics in
this module -- the only thing replaced is persistence: instead of a real
MySQL/MariaDB, fake_vqtl_repository.py is used (same interface as
vqtl/db/repository.py, in memory). This makes it possible to test the real
orchestration too (column renaming, fingerprint, resume, short-circuit) and
not just the pure statistical formulas -- and it is precisely there that
the real bugs fixed in vqtl were found before writing this test (see
CHANGELOG_VQTL_BUGFIX.md in the vqtl/ folder): cli.py/variant_subset,
core/data.py's wrong import of pca_utils + unpacking of
load_and_prepare_data + double merge of the PCAs, core/interaction.py +
rge_het.py + permutation.py not handling missing genotypes ("." in the VCF)
the way scan.py does.

THIS MODULE IS LIBRARY-ONLY: no module-level code touches the
env/filesystem, and there is no `main()`. work_dir and generation are
parameters of run_pipeline_for_method(), no longer module-level global
variables -- this makes it safe to call multiple times in the same process
(one scenario after another) without having to "monkey-patch" module
attributes before every call, and without needing a specific fake_data/
folder to exist next to THIS file at import time (the previous version
required both).

The single end-to-end entry point for the whole test battery remains
run_isolated_casual_test.py (see its docstring): this module, like
run_scenarios.py and fake_vqtl_repository.py, is meant to be imported from
there, not run standalone.

WHAT IS CHECKED (printed for human review + automated checks in
report_utils.run_checks -- these are statistical tests, a 100% boolean
pass/fail is not expected):
  1. Scan (Step 3): "pure vQTL" variants (variance-only effect, no G×E
     interaction) and causal G×E variants (which still induce
     heteroscedasticity via the exposure) should have low P/P_gc relative
     to the null pool.
  2. Filter (Step 4): lambda_GC should be reasonably close to 1. The
     selected candidates should include the causal variants and few/no
     false positives among the null ones.
  3. Interaction (Step 5): G×E variants should have low pval and a beta_I
     sign consistent with the injected effect; pure vQTL variants should
     NOT show a significant interaction (that is exactly the point: they
     are a pure variance signal, not an interaction).
  4. Permutation (Step 7): low empirical p-value for the G×E variants;
     Levene's test (heteroscedasticity by dosage) should be significant
     for the pure vQTL variants too, unlike the interaction test.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace as _dc_replace

import pandas as pd

import fake_vqtl_repository as fake_repo
from report_utils import export_csv, export_docx, run_checks
from vqtl.core.data import load_vqtl_dataset
from vqtl.core.filter_candidates import filter_candidates
from vqtl.core.interaction import run_interaction_tests
from vqtl.core.permutation import run_robustness_and_permutation
from vqtl.core.phenotype import prepare_phenotype
from vqtl.core.rge_het import run_rge_het
from vqtl.core.scan import run_vqtl_scan


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def run_pipeline_for_method(
    se_method: str,
    ge_cfg,
    vcfg_base,
    truth: pd.DataFrame,
    all_causal: set,
    n_null_truth: int,
    work_dir: str,
    generation: int,
    alpha: float = 0.05,
) -> dict:
    """Runs the full Step 3->7 pipeline with a specific se_method, saving
    everything under work_dir/vqtl_results/gen{generation}/{se_method}/ (so
    that asymptotic and bootstrap results do not overwrite each other and
    can be inspected separately). Returns the summary dict.
    """
    # Must be done BEFORE any call to vqtl.core.* below (see the
    # fake_vqtl_repository.py docstring on why it is safe to do this on
    # every call, even though the vqtl.core.* modules were already imported
    # above: their "from vqtl.db import repository as repo" is local to the
    # functions, resolved only when they actually run, not at import time).
    sys.modules["vqtl.db.repository"] = fake_repo

    vcfg = _dc_replace(vcfg_base, se_method=se_method)

    run_dir = os.path.join(work_dir, "vqtl_results", f"gen{generation}", se_method)
    tables_dir = os.path.join(run_dir, "tables")
    figures_dir = os.path.join(run_dir, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    section(f"[{se_method}] Step 1-2: load dataset + phenotype transform")
    ds = load_vqtl_dataset(ge_cfg, vcfg, generation=generation)
    ds.df = prepare_phenotype(ds.df, ge_cfg.target_col)
    print(f"Samples: {len(ds.df)} | variants: {len(ds.variant_cols)} | covariates: {ds.covariate_cols}")

    section(f"[{se_method}] Step 3: genome-wide vQTL scan (run_vqtl_scan)")
    fake_repo.reset_all()  # independent run: no short-circuit from a previous run's/method's fingerprint
    t0 = time.time()
    vqtl_df = run_vqtl_scan(ds, vcfg, ge_cfg.target_col, generation=generation, force=True)
    print(f"{len(vqtl_df)} variants scanned in {time.time() - t0:.1f}s")

    vqtl_df_display = vqtl_df.merge(truth[["variant", "effect_type"]], left_on="SNP", right_on="variant", how="left")
    print("\nRanking by P (causal variants should be at the top, not mixed in with the null ones):")
    top = vqtl_df_display.sort_values("P").head(15)[["SNP", "effect_type", "N", "MAF", "Z", "P"]]
    print(top.to_string(index=False))
    n_causal_in_top15 = sum(1 for s in top["SNP"] if s in all_causal)
    print(f"-> {n_causal_in_top15}/{len(all_causal)} causal variants among the top 15 by P.")
    export_csv(vqtl_df_display[["SNP", "effect_type", "N", "MAF", "Z", "P"]].sort_values("P"),
               tables_dir, "step3_scan_full")

    section(f"[{se_method}] Step 4: candidate filtering + lambda_GC (filter_candidates)")
    vqtl_df_annotated, candidates, lambda_gc = filter_candidates(vqtl_df, vcfg, figures_dir, generation=generation)
    print(f"lambda_GC = {lambda_gc:.3f} (null pool of {n_null_truth} variants)")
    candidates_display = candidates.merge(truth[["variant", "effect_type"]], left_on="SNP", right_on="variant", how="left")
    print(f"\n{len(candidates)} candidates selected (VQTL_FILTER_TOP_N={vcfg.filter_top_n}):")
    print(candidates_display[["SNP", "effect_type", "P", "P_gc"]].to_string(index=False))
    found_causal = set(candidates["SNP"]) & all_causal
    false_positives = set(candidates["SNP"]) - all_causal
    print(f"\n-> Causal variants recovered among the candidates: {len(found_causal)}/{len(all_causal)} {sorted(found_causal)}")
    print(f"-> Null variants that ended up among the candidates (false positives): "
          f"{len(false_positives)}/{n_null_truth} {sorted(false_positives)}")
    export_csv(candidates_display[["SNP", "effect_type", "P", "P_gc"]], tables_dir, "step4_candidates")

    summary = {
        "se_method": se_method,
        "lambda_gc": round(float(lambda_gc), 3),
        "n_causal_total": len(all_causal),
        "n_found_causal": len(found_causal),
        "found_causal": sorted(found_causal),
        "n_false_positives": len(false_positives),
        "n_null_truth": n_null_truth,
        "n_gxe_sig": 0, "n_gxe_total": 0, "n_pv_falsepos": 0, "n_pv_total": 0,
    }

    if candidates.empty:
        print(f"\n[{se_method}] No candidates selected: stopping here for this method (Step 5-7 skipped).")
        with open(os.path.join(tables_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return summary

    section(f"[{se_method}] Step 5: G×E interaction test (run_interaction_tests)")
    interaction_df = run_interaction_tests(ds, vcfg, candidates, ge_cfg.target_col, generation=generation)
    interaction_df_display = interaction_df.merge(
        truth[["variant", "effect_type", "true_beta_interaction"]], left_on="SNP", right_on="variant", how="left")
    print(interaction_df_display[
        ["SNP", "effect_type", "true_beta_interaction", "beta_I", "SE", "pval", "N"]
    ].to_string(index=False))
    export_csv(interaction_df_display[["SNP", "effect_type", "true_beta_interaction", "beta_I", "SE", "pval", "N"]],
               tables_dir, "step5_interaction")

    section(f"[{se_method}] Step 6: genotype-exposure relationship + heteroscedasticity (run_rge_het)")
    rge_df = run_rge_het(ds, vcfg, candidates, ge_cfg.target_col, generation=generation)
    rge_df_display = rge_df.merge(truth[["variant", "effect_type"]], left_on="SNP", right_on="variant", how="left")
    rge_cols = ["SNP", "effect_type", "rGE_pval", "rGE_flag", "het_BP_lm_pvalue", "heteroscedasticity_flag"]
    print(rge_df_display[rge_cols].to_string(index=False))
    export_csv(rge_df_display[rge_cols], tables_dir, "step6_rge_het")

    section(f"[{se_method}] Step 7: robustness + permutation (Freedman-Lane) + Levene's test")
    robustness_df, perm_df = run_robustness_and_permutation(
        ds, vcfg, interaction_df, ge_cfg.target_col, generation=generation)
    perm_df_display = perm_df.merge(truth[["variant", "effect_type"]], left_on="SNP", right_on="variant", how="left")
    perm_cols = ["SNP", "effect_type", "beta_I_observed", "empirical_pval", "asymptotic_pval",
                 "levene_stat_observed", "levene_pval"]
    print(perm_df_display[perm_cols].to_string(index=False))
    export_csv(perm_df_display[perm_cols], tables_dir, "step7_permutation")

    section(f"[{se_method}] Automated checks")
    suite = run_checks(
        lambda_gc=lambda_gc,
        all_causal=all_causal,
        found_causal=found_causal,
        candidates=candidates,
        interaction_df_display=interaction_df_display,
        perm_df_display=perm_df_display,
        alpha=alpha,
    )
    suite.print_report()

    gxe_rows = interaction_df_display[interaction_df_display["effect_type"] == "gxe_meanshift"]
    pv_rows = interaction_df_display[interaction_df_display["effect_type"] == "pure_variance"]
    summary.update({
        "n_gxe_sig": int((gxe_rows["pval"] < alpha).sum()) if not gxe_rows.empty else 0,
        "n_gxe_total": len(gxe_rows),
        "n_pv_falsepos": int((pv_rows["pval"] < alpha).sum()) if not pv_rows.empty else 0,
        "n_pv_total": len(pv_rows),
        "checks": suite.to_list(),
        "has_failures": suite.has_failures,
    })
    with open(os.path.join(tables_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[export] {os.path.join(tables_dir, 'summary.json')}")

    docx_path = os.path.join(run_dir, f"vqtl_report_gen{generation}_{se_method}.docx")
    export_docx(
        docx_path, generation, summary, suite,
        tables=[
            ("Step 3 — Genome-wide vQTL scan", "Full ranking by P. Highlighted rows = causal variants.",
             vqtl_df_display[["SNP", "effect_type", "N", "MAF", "Z", "P"]].sort_values("P")),
            ("Step 4 — Selected candidates", f"lambda_GC={lambda_gc:.3f}.",
             candidates_display[["SNP", "effect_type", "P", "P_gc"]]),
            ("Step 5 — G×E interaction test", "beta_I and pval of the interaction test.",
             interaction_df_display[["SNP", "effect_type", "true_beta_interaction", "beta_I", "SE", "pval", "N"]]),
            ("Step 6 — rGE and heteroscedasticity", "rGE_flag=True indicates possible confounding.",
             rge_df_display[rge_cols]),
            ("Step 7 — Robustness, permutation, and Levene's test", "empirical_pval and levene_pval.",
             perm_df_display[perm_cols]),
        ],
    )
    print(f"\n[{se_method}] Full output in: {run_dir}/")
    return summary
