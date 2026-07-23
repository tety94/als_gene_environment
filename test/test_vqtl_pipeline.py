"""
Test end-to-end della pipeline vqtl (Step 3: scan -> Step 4: filter ->
Step 5: interaction -> Step 6: rge_het -> Step 7: robustness+permutation)
sui dati sintetici di gen_fake_data_vqtl.py (da generare PRIMA di lanciare
questo script: `python gen_fake_data_vqtl.py`).

FILOSOFIA (stessa di run_pipeline_test.py): chiama le funzioni REALI di
vqtl.core.* cosi' come sono (run_vqtl_scan, filter_candidates,
run_interaction_tests, run_rge_het, run_robustness_and_permutation),
nessuna reimplementazione della statistica in questo script -- l'unica cosa
sostituita e' la persistenza: al posto di un vero MySQL/MariaDB si usa
fake_vqtl_repository.py (stessa interfaccia di vqtl/db/repository.py, in
memoria). Questo permette di testare anche l'orchestrazione vera (rename di
colonne, fingerprint, resume, short-circuit) e non solo le formule
statistiche pure -- ed e' proprio li' che sono stati trovati i bug reali
corretti in vqtl prima di scrivere questo test (vedi CHANGELOG_VQTL_BUGFIX.md
nella cartella vqtl/): cli.py/variant_subset, core/data.py import errato di
pca_utils + unpacking di load_and_prepare_data + doppio merge delle PCA,
core/interaction.py + rge_het.py + permutation.py che non gestivano i
genotipi mancanti ("." nel VCF) come fa scan.py.

Nessun MySQL/MariaDB richiesto. Richiede gene_environment importabile
(REPO_ROOT = root del repo dove vive gene_environment/, vedi sys.path sotto
-- va lanciato da dentro il repo vero, non da questa chat).

COSA VIENE VERIFICATO (stampato per revisione umana, non solo assert -- sono
test statistici, non ci si aspetta un pass/fail booleano al 100%):
  1. Scan (Step 3): le 2 varianti "vQTL pure" (solo effetto di varianza,
     nessuna interazione G×E) e le 5 varianti G×E causali (che producono
     comunque eteroschedasticita' via l'esposizione, vedi gen_fake_data_vqtl.py)
     devono avere P/P_gc bassi rispetto al pool nullo.
  2. Filtro (Step 4): lambda_GC deve essere ragionevolmente vicino a 1 (pool
     nullo di 300 varianti, non 60 -- se e' molto lontano da 1 con un pool
     cosi' grande, indica un problema nel modello/residualizzazione, non solo
     rumore campionario). I candidati selezionati devono includere le 7
     varianti causali (5 G×E + 2 vQTL pure) e pochi/nessun falso positivo fra
     le 300 nulle.
  3. Interazione (Step 5): le 5 varianti G×E devono avere pval basso e segno
     di beta_I coerente con l'effetto iniettato; le 2 vQTL pure NON devono
     avere un'interazione significativa (e' proprio il punto: sono un
     segnale di varianza puro, non un'interazione).
  4. Permutazione (Step 7): p-value empirico basso per le 5 G×E; il test di
     Levene (eteroschedasticita' per dosaggio) dovrebbe essere significativo
     anche per le 2 vQTL pure, a differenza del test di interazione.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import json
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)  # cartella che contiene sia vqtl/ sia gene_environment/
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCRIPT_DIR)

FAKE = os.path.join(SCRIPT_DIR, "fake_data")   # prima: "fake_data_vqtl"
if not os.path.isdir(FAKE):
    raise SystemExit(
        f"{FAKE} non esiste: lancia prima `python gen_fake_data_vqtl.py` in questa cartella."
    )

# ---- config ambiente, PRIMA di importare gene_environment/vqtl ----
os.environ.setdefault("DB_USER", "unused")
os.environ.setdefault("DB_PASSWORD", "unused")
os.environ.setdefault("DB_NAME", "unused")
os.environ.update({
    "USE_PCA_COVARIATES": "true",
    "PCA_N_COMPONENTS": "10",
    "PCA_COVARIATES_PATH_TEMPLATE": os.path.join(FAKE, "pca_covariates_gen{generation}.csv"),
    "GENERATION": "1",
    "TARGET_COL": "onset_age",
    "EXPOSURE": "exposure_env",
    "COVARIATES": "sex",
    "RAW_FILE": os.path.join(FAKE, "genetic.csv"),
    "ENV_FILE": os.path.join(FAKE, "env.csv"),
    "SEP": ",",
    "LOG_DIR": os.path.join(SCRIPT_DIR, "logs"),
    "VQTL_N_PERM": "500",
    "VQTL_CHUNK_SIZE": "20",
    "VQTL_N_JOBS": str(min(4, os.cpu_count() or 2)),
    # top_N invece della soglia P_gc<1e-5 di default: con un pool sintetico
    # (anche se ora piu' grande, 300 nulle) il calcolo di lambda_GC resta
    # meno stabile che su un vero scan genoma-wide con milioni di varianti,
    # quindi qui la selezione a soglia fissa sarebbe meno prevedibile per un
    # test automatico. lambda_GC viene comunque calcolato e stampato sotto
    # come diagnostica, a prescindere da come si selezionano i candidati.
    "VQTL_FILTER_TOP_N": "15",
})

import fake_vqtl_repository as fake_repo  # noqa: E402
sys.modules["vqtl.db.repository"] = fake_repo  # vedi docstring del modulo

from gene_environment.config import get_config  # noqa: E402
from gene_environment.logging_utils import configure_logging  # noqa: E402
from vqtl.config import get_vqtl_config  # noqa: E402
from vqtl.core.data import load_vqtl_dataset  # noqa: E402
from vqtl.core.phenotype import prepare_phenotype  # noqa: E402
from vqtl.core.scan import run_vqtl_scan  # noqa: E402
from vqtl.core.filter_candidates import filter_candidates  # noqa: E402
from vqtl.core.interaction import run_interaction_tests  # noqa: E402
from vqtl.core.rge_het import run_rge_het  # noqa: E402
from vqtl.core.permutation import run_robustness_and_permutation  # noqa: E402
from report_utils import export_csv, run_checks, export_docx

GENERATION = 1
ALPHA = 0.05


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def run_pipeline_for_method(se_method: str, ge_cfg, vcfg_base, truth, all_causal, n_null_truth) -> dict:
    """Esegue l'intera pipeline Step 3->7 con un se_method specifico,
    salvando tutto in una sottocartella dedicata (vqtl_results/gen{N}/{se_method}/)
    cosi' i risultati di asymptotic e bootstrap non si sovrascrivono a
    vicenda e sono ispezionabili separatamente. Ritorna il summary dict
    (stesso schema di prima) per il confronto finale in main()."""
    from dataclasses import replace as _dc_replace

    vcfg = _dc_replace(vcfg_base, se_method=se_method)

    run_dir = os.path.join(SCRIPT_DIR, "vqtl_results", f"gen{GENERATION}", se_method)
    tables_dir = os.path.join(run_dir, "tables")
    figures_dir = os.path.join(run_dir, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    section(f"[{se_method}] Step 1-2: caricamento dataset + trasformazione fenotipo")
    ds = load_vqtl_dataset(ge_cfg, vcfg, generation=GENERATION)
    ds.df = prepare_phenotype(ds.df, ge_cfg.target_col)
    print(f"Campioni: {len(ds.df)} | varianti: {len(ds.variant_cols)} | covariate: {ds.covariate_cols}")

    section(f"[{se_method}] Step 3: scan vQTL genoma-wide (run_vqtl_scan)")
    fake_repo.reset_all()  # run indipendente: niente short-circuit da fingerprint di un run precedente/altro metodo
    t0 = time.time()
    vqtl_df = run_vqtl_scan(ds, vcfg, ge_cfg.target_col, generation=GENERATION, force=True)
    print(f"{len(vqtl_df)} varianti scansionate in {time.time() - t0:.1f}s")

    vqtl_df_display = vqtl_df.merge(truth[["variant", "effect_type"]], left_on="SNP", right_on="variant", how="left")
    print("\nRanking per P (le 7 causali dovrebbero stare in cima, non in mezzo alle 300 nulle):")
    top = vqtl_df_display.sort_values("P").head(15)[["SNP", "effect_type", "N", "MAF", "Z", "P"]]
    print(top.to_string(index=False))
    n_causal_in_top15 = sum(1 for s in top["SNP"] if s in all_causal)
    print(f"-> {n_causal_in_top15}/{len(all_causal)} causali fra le prime 15 per P.")
    export_csv(vqtl_df_display[["SNP", "effect_type", "N", "MAF", "Z", "P"]].sort_values("P"), tables_dir,
               "step3_scan_full")

    section(f"[{se_method}] Step 4: filtro candidati + lambda_GC (filter_candidates)")
    vqtl_df_annotated, candidates, lambda_gc = filter_candidates(vqtl_df, vcfg, figures_dir, generation=GENERATION)
    print(f"lambda_GC = {lambda_gc:.3f} (pool nullo di {n_null_truth} varianti)")
    candidates_display = candidates.merge(truth[["variant", "effect_type"]], left_on="SNP", right_on="variant", how="left")
    print(f"\n{len(candidates)} candidati selezionati (VQTL_FILTER_TOP_N={vcfg.filter_top_n}):")
    print(candidates_display[["SNP", "effect_type", "P", "P_gc"]].to_string(index=False))
    found_causal = set(candidates["SNP"]) & all_causal
    false_positives = set(candidates["SNP"]) - all_causal
    print(f"\n-> Causali recuperate fra i candidati: {len(found_causal)}/{len(all_causal)} {sorted(found_causal)}")
    print(f"-> Varianti nulle finite fra i candidati (falsi positivi): {len(false_positives)}/{n_null_truth} {sorted(false_positives)}")
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
        print(f"\n[{se_method}] Nessun candidato selezionato: interrompo qui per questo metodo (Step 5-7 saltati).")
        with open(os.path.join(tables_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return summary

    section(f"[{se_method}] Step 5: test di interazione G×E (run_interaction_tests)")
    interaction_df = run_interaction_tests(ds, vcfg, candidates, ge_cfg.target_col, generation=GENERATION)
    interaction_df_display = interaction_df.merge(truth[["variant", "effect_type", "true_beta_interaction"]], left_on="SNP", right_on="variant", how="left")
    print(interaction_df_display[["SNP", "effect_type", "true_beta_interaction", "beta_I", "SE", "pval", "N"]].to_string(index=False))
    export_csv(interaction_df_display[["SNP", "effect_type", "true_beta_interaction", "beta_I", "SE", "pval", "N"]],
               tables_dir, "step5_interaction")

    section(f"[{se_method}] Step 6: relazione genotipo-esposizione + eteroschedasticita' (run_rge_het)")
    rge_df = run_rge_het(ds, vcfg, candidates, ge_cfg.target_col, generation=GENERATION)
    rge_df_display = rge_df.merge(truth[["variant", "effect_type"]], left_on="SNP", right_on="variant", how="left")
    cols = ["SNP", "effect_type", "rGE_pval", "rGE_flag", "het_BP_lm_pvalue", "heteroscedasticity_flag"]
    print(rge_df_display[cols].to_string(index=False))
    export_csv(
        rge_df_display[["SNP", "effect_type", "rGE_pval", "rGE_flag", "het_BP_lm_pvalue", "heteroscedasticity_flag"]],
        tables_dir, "step6_rge_het")

    section(f"[{se_method}] Step 7: robustezza + permutazione (Freedman-Lane) + test di Levene")
    robustness_df, perm_df = run_robustness_and_permutation(ds, vcfg, interaction_df, ge_cfg.target_col, generation=GENERATION)
    perm_df_display = perm_df.merge(truth[["variant", "effect_type"]], left_on="SNP", right_on="variant", how="left")
    cols = ["SNP", "effect_type", "beta_I_observed", "empirical_pval", "asymptotic_pval", "levene_stat_observed", "levene_pval"]
    print(perm_df_display[cols].to_string(index=False))
    export_csv(perm_df_display[["SNP", "effect_type", "beta_I_observed", "empirical_pval", "asymptotic_pval",
                                "levene_stat_observed", "levene_pval"]], tables_dir, "step7_permutation")

    section(f"[{se_method}] Controlli automatici")
    suite = run_checks(
        lambda_gc=lambda_gc,
        all_causal=all_causal,
        found_causal=found_causal,
        candidates=candidates,
        interaction_df_display=interaction_df_display,
        perm_df_display=perm_df_display,
        alpha=ALPHA,
    )
    suite.print_report()

    gxe_rows = interaction_df_display[interaction_df_display["effect_type"] == "gxe_meanshift"]
    pv_rows = interaction_df_display[interaction_df_display["effect_type"] == "pure_variance"]
    summary.update({
        "n_gxe_sig": int((gxe_rows["pval"] < ALPHA).sum()) if not gxe_rows.empty else 0,
        "n_gxe_total": len(gxe_rows),
        "n_pv_falsepos": int((pv_rows["pval"] < ALPHA).sum()) if not pv_rows.empty else 0,
        "n_pv_total": len(pv_rows),
        "checks": suite.to_list(),
        "has_failures": suite.has_failures,
    })
    with open(os.path.join(tables_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[export] {os.path.join(tables_dir, 'summary.json')}")

    docx_path = os.path.join(run_dir, f"vqtl_report_gen{GENERATION}_{se_method}.docx")
    export_docx(
        docx_path, GENERATION, summary, suite,
        tables=[
            ("Step 3 — Scan vQTL genoma-wide", "Ranking completo per P. Righe evidenziate = varianti causali.",
             vqtl_df_display[["SNP", "effect_type", "N", "MAF", "Z", "P"]].sort_values("P")),
            ("Step 4 — Candidati selezionati", f"lambda_GC={lambda_gc:.3f}.",
             candidates_display[["SNP", "effect_type", "P", "P_gc"]]),
            ("Step 5 — Test di interazione G×E", "beta_I e pval del test di interazione.",
             interaction_df_display[["SNP", "effect_type", "true_beta_interaction", "beta_I", "SE", "pval", "N"]]),
            ("Step 6 — rGE ed eteroschedasticita'", "rGE_flag=True indica possibile confondimento.",
             rge_df_display[["SNP", "effect_type", "rGE_pval", "rGE_flag", "het_BP_lm_pvalue", "heteroscedasticity_flag"]]),
            ("Step 7 — Robustezza e permutazione + Levene", "empirical_pval e levene_pval.",
             perm_df_display[["SNP", "effect_type", "beta_I_observed", "empirical_pval", "asymptotic_pval",
                              "levene_stat_observed", "levene_pval"]]),
        ],
    )
    print(f"\n[{se_method}] Output completo in: {run_dir}/")
    return summary


def main() -> None:
    ge_cfg = get_config()
    configure_logging(ge_cfg.log_dir)
    vcfg_base = get_vqtl_config()

    truth = pd.read_csv(os.path.join(FAKE, "ground_truth.csv"))
    causal_gxe = set(truth.loc[truth["effect_type"] == "gxe_meanshift", "variant"])
    causal_pure_var = set(truth.loc[truth["effect_type"] == "pure_variance", "variant"])
    all_causal = causal_gxe | causal_pure_var
    n_null_truth = int((truth["effect_type"] == "no_effect").sum())
    print(f"Ground truth: {len(causal_gxe)} G×E causali, {len(causal_pure_var)} vQTL pure, {n_null_truth} nulle")

    # ---- Intera pipeline (Step 3->7) girata SEPARATAMENTE per ciascun
    # se_method, output in sottocartelle distinte (vqtl_results/gen{N}/asymptotic/
    # e vqtl_results/gen{N}/bootstrap/) cosi' si vede cosa succede con
    # ciascun metodo end-to-end, non solo un confronto parziale sullo scan. ----
    summaries = {}
    for se_method in ["asymptotic", "bootstrap"]:
        section(f"########## RUN COMPLETO: se_method={se_method} ##########")
        summaries[se_method] = run_pipeline_for_method(se_method, ge_cfg, vcfg_base, truth, all_causal, n_null_truth)

    section("Confronto finale asymptotic vs bootstrap")
    cmp_rows = []
    for se_method, s in summaries.items():
        cmp_rows.append({
            "se_method": se_method,
            "lambda_gc": s.get("lambda_gc"),
            "n_found_causal": s.get("n_found_causal"),
            "n_causal_total": s.get("n_causal_total"),
            "n_false_positives": s.get("n_false_positives"),
            "n_gxe_sig": s.get("n_gxe_sig"),
            "n_gxe_total": s.get("n_gxe_total"),
            "n_pv_falsepos": s.get("n_pv_falsepos"),
            "has_failures": s.get("has_failures"),
        })
    cmp_df = pd.DataFrame(cmp_rows)
    print(cmp_df.to_string(index=False))

    combined_dir = os.path.join(SCRIPT_DIR, "vqtl_results", f"gen{GENERATION}")
    export_csv(cmp_df, combined_dir, "summary_comparison_asymptotic_vs_bootstrap")
    with open(os.path.join(combined_dir, "summary_comparison.json"), "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\n[export] {os.path.join(combined_dir, 'summary_comparison.json')}")
    print(f"Dettaglio per metodo in: {combined_dir}/asymptotic/ e {combined_dir}/bootstrap/")

    if any(s.get("has_failures") for s in summaries.values()):
        print("\n*** ALMENO UN METODO HA CONTROLLI CRITICI FALLITI (vedi sopra). ***")
        sys.exit(1)
    print("\n*** Entrambi i run completati (eventuali WARN sono attesi con dati sintetici / potenza limitata). ***")

if __name__ == "__main__":
    main()