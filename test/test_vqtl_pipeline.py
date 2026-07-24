"""
Libreria della pipeline vQTL end-to-end (Step 3: scan -> Step 4: filter ->
Step 5: interaction -> Step 6: rge_het -> Step 7: robustness+permutation).

Espone una sola funzione pubblica, run_pipeline_for_method(), richiamata da
run_scenarios.py (via run_vqtl_asymptotic) per la parte "pipeline completa"
di ogni scenario, e in generale da qualunque orchestratore voglia far girare
Step 3->7 con un se_method specifico su un dataset gia' pronto.

FILOSOFIA (invariata): chiama le funzioni REALI di vqtl.core.* cosi' come
sono (run_vqtl_scan, filter_candidates, run_interaction_tests, run_rge_het,
run_robustness_and_permutation), nessuna reimplementazione della statistica
in questo modulo -- l'unica cosa sostituita e' la persistenza: al posto di
un vero MySQL/MariaDB si usa fake_vqtl_repository.py (stessa interfaccia di
vqtl/db/repository.py, in memoria). Questo permette di testare anche
l'orchestrazione reale (rename di colonne, fingerprint, resume,
short-circuit) e non solo le formule statistiche pure -- ed e' proprio li'
che sono stati trovati i bug reali corretti in vqtl prima di scrivere
questo test (vedi CHANGELOG_VQTL_BUGFIX.md nella cartella vqtl/): cli.py/
variant_subset, core/data.py import errato di pca_utils + unpacking di
load_and_prepare_data + doppio merge delle PCA, core/interaction.py +
rge_het.py + permutation.py che non gestivano i genotipi mancanti ("." nel
VCF) come fa scan.py.

QUESTO MODULO E' SOLO UNA LIBRERIA: nessun codice a livello di modulo tocca
env/filesystem, e non c'e' un `main()`. work_dir e generation sono
parametri di run_pipeline_for_method(), non piu' variabili globali di
modulo -- questo la rende sicura da richiamare piu' volte nello stesso
processo (uno scenario dopo l'altro) senza dover "monkey-patchare" gli
attributi del modulo prima di ogni chiamata, e senza bisogno che una
cartella fake_data/ specifica esista accanto a QUESTO file al momento
dell'import (la versione precedente richiedeva entrambe le cose).

Un unico punto d'ingresso end-to-end per l'intera batteria di test resta
run_isolated_casual_test.py (vedi il suo docstring): questo modulo, come
run_scenarios.py e fake_vqtl_repository.py, e' pensato per essere importato
da li', non lanciato da solo.

COSA VIENE VERIFICATO (stampato per revisione umana + controlli automatici
in report_utils.run_checks -- sono test statistici, non ci si aspetta un
pass/fail booleano al 100%):
  1. Scan (Step 3): le varianti "vQTL pure" (solo effetto di varianza,
     nessuna interazione G×E) e le varianti G×E causali (che inducono
     comunque eteroschedasticita' via l'esposizione) devono avere P/P_gc
     bassi rispetto al pool nullo.
  2. Filtro (Step 4): lambda_GC deve essere ragionevolmente vicino a 1. I
     candidati selezionati devono includere le varianti causali e
     pochi/nessun falso positivo fra le nulle.
  3. Interazione (Step 5): le G×E devono avere pval basso e segno di beta_I
     coerente con l'effetto iniettato; le vQTL pure NON devono avere
     un'interazione significativa (e' proprio il punto: sono un segnale di
     varianza puro, non un'interazione).
  4. Permutazione (Step 7): p-value empirico basso per le G×E; il test di
     Levene (eteroschedasticita' per dosaggio) dovrebbe essere significativo
     anche per le vQTL pure, a differenza del test di interazione.
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
    """Esegue l'intera pipeline Step 3->7 con un se_method specifico,
    salvando tutto in work_dir/vqtl_results/gen{generation}/{se_method}/ (cosi'
    i risultati di asymptotic e bootstrap non si sovrascrivono a vicenda e
    sono ispezionabili separatamente). Ritorna il summary dict.
    """
    # Va fatto PRIMA di qualunque chiamata a vqtl.core.* qui sotto (vedi
    # docstring di fake_vqtl_repository.py sul perche' e' sicuro farlo ad
    # ogni chiamata, anche se i moduli vqtl.core.* sono gia' stati
    # importati sopra: il loro "from vqtl.db import repository as repo" e'
    # locale alle funzioni, risolto solo quando vengono effettivamente
    # eseguite, non al momento dell'import).
    sys.modules["vqtl.db.repository"] = fake_repo

    vcfg = _dc_replace(vcfg_base, se_method=se_method)

    run_dir = os.path.join(work_dir, "vqtl_results", f"gen{generation}", se_method)
    tables_dir = os.path.join(run_dir, "tables")
    figures_dir = os.path.join(run_dir, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    section(f"[{se_method}] Step 1-2: caricamento dataset + trasformazione fenotipo")
    ds = load_vqtl_dataset(ge_cfg, vcfg, generation=generation)
    ds.df = prepare_phenotype(ds.df, ge_cfg.target_col)
    print(f"Campioni: {len(ds.df)} | varianti: {len(ds.variant_cols)} | covariate: {ds.covariate_cols}")

    section(f"[{se_method}] Step 3: scan vQTL genoma-wide (run_vqtl_scan)")
    fake_repo.reset_all()  # run indipendente: niente short-circuit da fingerprint di un run precedente/altro metodo
    t0 = time.time()
    vqtl_df = run_vqtl_scan(ds, vcfg, ge_cfg.target_col, generation=generation, force=True)
    print(f"{len(vqtl_df)} varianti scansionate in {time.time() - t0:.1f}s")

    vqtl_df_display = vqtl_df.merge(truth[["variant", "effect_type"]], left_on="SNP", right_on="variant", how="left")
    print("\nRanking per P (le varianti causali dovrebbero stare in cima, non in mezzo alle nulle):")
    top = vqtl_df_display.sort_values("P").head(15)[["SNP", "effect_type", "N", "MAF", "Z", "P"]]
    print(top.to_string(index=False))
    n_causal_in_top15 = sum(1 for s in top["SNP"] if s in all_causal)
    print(f"-> {n_causal_in_top15}/{len(all_causal)} causali fra le prime 15 per P.")
    export_csv(vqtl_df_display[["SNP", "effect_type", "N", "MAF", "Z", "P"]].sort_values("P"),
               tables_dir, "step3_scan_full")

    section(f"[{se_method}] Step 4: filtro candidati + lambda_GC (filter_candidates)")
    vqtl_df_annotated, candidates, lambda_gc = filter_candidates(vqtl_df, vcfg, figures_dir, generation=generation)
    print(f"lambda_GC = {lambda_gc:.3f} (pool nullo di {n_null_truth} varianti)")
    candidates_display = candidates.merge(truth[["variant", "effect_type"]], left_on="SNP", right_on="variant", how="left")
    print(f"\n{len(candidates)} candidati selezionati (VQTL_FILTER_TOP_N={vcfg.filter_top_n}):")
    print(candidates_display[["SNP", "effect_type", "P", "P_gc"]].to_string(index=False))
    found_causal = set(candidates["SNP"]) & all_causal
    false_positives = set(candidates["SNP"]) - all_causal
    print(f"\n-> Causali recuperate fra i candidati: {len(found_causal)}/{len(all_causal)} {sorted(found_causal)}")
    print(f"-> Varianti nulle finite fra i candidati (falsi positivi): "
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
        print(f"\n[{se_method}] Nessun candidato selezionato: interrompo qui per questo metodo (Step 5-7 saltati).")
        with open(os.path.join(tables_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return summary

    section(f"[{se_method}] Step 5: test di interazione G×E (run_interaction_tests)")
    interaction_df = run_interaction_tests(ds, vcfg, candidates, ge_cfg.target_col, generation=generation)
    interaction_df_display = interaction_df.merge(
        truth[["variant", "effect_type", "true_beta_interaction"]], left_on="SNP", right_on="variant", how="left")
    print(interaction_df_display[
        ["SNP", "effect_type", "true_beta_interaction", "beta_I", "SE", "pval", "N"]
    ].to_string(index=False))
    export_csv(interaction_df_display[["SNP", "effect_type", "true_beta_interaction", "beta_I", "SE", "pval", "N"]],
               tables_dir, "step5_interaction")

    section(f"[{se_method}] Step 6: relazione genotipo-esposizione + eteroschedasticita' (run_rge_het)")
    rge_df = run_rge_het(ds, vcfg, candidates, ge_cfg.target_col, generation=generation)
    rge_df_display = rge_df.merge(truth[["variant", "effect_type"]], left_on="SNP", right_on="variant", how="left")
    rge_cols = ["SNP", "effect_type", "rGE_pval", "rGE_flag", "het_BP_lm_pvalue", "heteroscedasticity_flag"]
    print(rge_df_display[rge_cols].to_string(index=False))
    export_csv(rge_df_display[rge_cols], tables_dir, "step6_rge_het")

    section(f"[{se_method}] Step 7: robustezza + permutazione (Freedman-Lane) + test di Levene")
    robustness_df, perm_df = run_robustness_and_permutation(
        ds, vcfg, interaction_df, ge_cfg.target_col, generation=generation)
    perm_df_display = perm_df.merge(truth[["variant", "effect_type"]], left_on="SNP", right_on="variant", how="left")
    perm_cols = ["SNP", "effect_type", "beta_I_observed", "empirical_pval", "asymptotic_pval",
                 "levene_stat_observed", "levene_pval"]
    print(perm_df_display[perm_cols].to_string(index=False))
    export_csv(perm_df_display[perm_cols], tables_dir, "step7_permutation")

    section(f"[{se_method}] Controlli automatici")
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
            ("Step 3 — Scan vQTL genoma-wide", "Ranking completo per P. Righe evidenziate = varianti causali.",
             vqtl_df_display[["SNP", "effect_type", "N", "MAF", "Z", "P"]].sort_values("P")),
            ("Step 4 — Candidati selezionati", f"lambda_GC={lambda_gc:.3f}.",
             candidates_display[["SNP", "effect_type", "P", "P_gc"]]),
            ("Step 5 — Test di interazione G×E", "beta_I e pval del test di interazione.",
             interaction_df_display[["SNP", "effect_type", "true_beta_interaction", "beta_I", "SE", "pval", "N"]]),
            ("Step 6 — rGE ed eteroschedasticita'", "rGE_flag=True indica possibile confondimento.",
             rge_df_display[rge_cols]),
            ("Step 7 — Robustezza e permutazione + Levene", "empirical_pval e levene_pval.",
             perm_df_display[perm_cols]),
        ],
    )
    print(f"\n[{se_method}] Output completo in: {run_dir}/")
    return summary
