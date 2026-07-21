#!/usr/bin/env python3
"""
Entry point della pipeline vQTL, nello stesso stile di `gene_environment.cli`
(un sottocomando per step, piu' `run-all`). Sostituisce `run_pipeline.sh` +
gli 8 script `stepN_*.py` invocati singolarmente del progetto originale.

Esempi:
    python -m vqtl.cli run-all --generation 2   # prima volta: calcola tutto
    python -m vqtl.cli run-all --generation 1   # se gen1 ha gia' risultati
                                                 # significativi in cache,
                                                 # salta lo scan e legge da DB
    python -m vqtl.cli scan --generation 1
    python -m vqtl.cli filter --generation 1
    python -m vqtl.cli interaction --generation 1
    python -m vqtl.cli rge-het --generation 1
    python -m vqtl.cli permute --generation 1
    python -m vqtl.cli report --generation 1
    python -m vqtl.cli docx --generation 1

STATO: tutto (scan genoma-wide, candidati, interazione, rGE/eteroschedasticita',
robustezza, permutazioni) e' persistito a DB (tabelle vqtl_*, vedi
vqtl/db/schema.sql e vqtl/db/repository.py), non piu' in .tsv intermedi.
Solo i deliverable finali restano file: report.md, report.docx, figures/*.png.

SHORT-CIRCUIT: prima di calcolare qualunque cosa per una generazione,
`run-all` controlla se vqtl_scan_results_significant ha gia' righe per
quella generazione. Se si' (e non e' stato passato --force), lo scan
genoma-wide E il filtro vengono saltati del tutto -- i risultati si leggono
direttamente da DB -- e si passa subito allo Step 5 (interazione) sui
candidati gia' noti. Se e' la prima volta che una generazione viene
analizzata, o dopo --force, si calcola tutto da capo (come qualunque altro
step, resta comunque ripreso automaticamente se interrotto a meta', vedi
scan.py).

Il concetto di "coorte" (gen1 / gen2 / gen3 del progetto originale) coincide
qui con `cfg.generation` di gene_environment (variabile GENERATION nel
.env): --generation sovrascrive cfg.generation per il singolo comando senza
toccare il .env.
"""
from __future__ import annotations

import argparse
import os
import sys

from gene_environment.config import get_config
from gene_environment.logging_utils import configure_logging, get_logger

from vqtl.config import get_vqtl_config
from vqtl.core.data import VqtlDataset, get_or_build_dataset, select_variants_from_significant_results
from vqtl.core.docx_report import build_docx_report
from vqtl.core.filter_candidates import filter_candidates, regenerate_plots
from vqtl.core.interaction import run_interaction_tests
from vqtl.core.permutation import run_robustness_and_permutation
from vqtl.core.phenotype import prepare_phenotype
from vqtl.core.report import build_report
from vqtl.core.rge_het import run_rge_het
from vqtl.core.scan import run_vqtl_scan
from vqtl.db import repository as repo

log = get_logger(__name__)


def _load_prepared_dataset(generation: int | None, force: bool) -> tuple[VqtlDataset, str, int]:
    ge_cfg = get_config()
    vcfg = get_vqtl_config()
    dataset = get_or_build_dataset(ge_cfg, vcfg, generation=generation, force=force)
    gen = generation if generation is not None else ge_cfg.generation
    dataset.df = prepare_phenotype(dataset.df, ge_cfg.target_col)
    cohort_dir = vcfg.cohort_dir(gen)
    os.makedirs(cohort_dir, exist_ok=True)
    return dataset, cohort_dir, gen


def _resolve_variant_subset(args, dataset):
    if getattr(args, "significant_only", False):
        return select_variants_from_significant_results(dataset, exposure=args.exposure)
    return None

def cmd_scan(args):
    ge_cfg = get_config()
    vcfg = get_vqtl_config()
    dataset, _cohort_dir, gen = _load_prepared_dataset(args.generation, args.force)
    subset = _resolve_variant_subset(args, dataset)
    result = run_vqtl_scan(dataset, vcfg, ge_cfg.target_col, generation=gen, force=args.force, variant_subset=subset)
    return result


def cmd_filter(args):
    vcfg = get_vqtl_config()
    _dataset, cohort_dir, gen = _load_prepared_dataset(args.generation, force=False)
    vqtl_df = repo.get_scan_results(gen)
    _df_full, candidates, _lam = filter_candidates(vqtl_df, vcfg, os.path.join(cohort_dir, "figures"), generation=gen)
    return candidates


def cmd_interaction(args):
    ge_cfg = get_config()
    vcfg = get_vqtl_config()
    dataset, _cohort_dir, gen = _load_prepared_dataset(args.generation, force=False)
    candidates = repo.get_candidates(gen)
    return run_interaction_tests(dataset, vcfg, candidates, ge_cfg.target_col, generation=gen)


def cmd_rge_het(args):
    ge_cfg = get_config()
    vcfg = get_vqtl_config()
    dataset, _cohort_dir, gen = _load_prepared_dataset(args.generation, force=False)
    candidates = repo.get_candidates(gen)
    return run_rge_het(dataset, vcfg, candidates, ge_cfg.target_col, generation=gen)


def cmd_permute(args):
    ge_cfg = get_config()
    vcfg = get_vqtl_config()
    dataset, _cohort_dir, gen = _load_prepared_dataset(args.generation, force=False)
    interaction_df = repo.fetch_results("interaction", gen)
    if not interaction_df.empty:
        interaction_df = interaction_df.rename(columns={"beta_i": "beta_I", "n": "N", "maf": "MAF", "se": "SE"}).dropna(subset=["pval"])
    return run_robustness_and_permutation(dataset, vcfg, interaction_df, ge_cfg.target_col, generation=gen)


def _fetch_all_for_report(gen: int):
    vqtl_df = repo.get_scan_results(gen)
    candidates = repo.get_candidates(gen)
    interaction_df = repo.fetch_results("interaction", gen)
    if not interaction_df.empty:
        interaction_df = interaction_df.rename(columns={"beta_i": "beta_I", "n": "N", "maf": "MAF", "se": "SE"}).dropna(subset=["pval"])
    interaction_significant_df = repo.get_interaction_significant(gen)
    if not interaction_significant_df.empty:
        interaction_significant_df = interaction_significant_df.rename(columns={"beta_i": "beta_I", "n": "N", "maf": "MAF", "se": "SE"})
    rge_df = repo.fetch_results("rge_het", gen)
    if not rge_df.empty:
        rge_df = rge_df.rename(columns={
            "rge_beta_exposure_on_snp": "rGE_beta_exposure_on_snp", "rge_se": "rGE_SE", "rge_pval": "rGE_pval",
            "rge_flag": "rGE_flag", "het_bp_lm_stat": "het_BP_lm_stat", "het_bp_lm_pvalue": "het_BP_lm_pvalue",
            "het_bp_f_stat": "het_BP_f_stat", "het_bp_f_pvalue": "het_BP_f_pvalue",
        })
    perm_df = repo.fetch_results("permutation", gen)
    if not perm_df.empty:
        perm_df = perm_df.rename(columns={"beta_i_observed": "beta_I_observed"}).dropna(subset=["empirical_pval"])
    robustness_df = repo.fetch_results("robustness", gen)
    if not robustness_df.empty:
        robustness_df = robustness_df.rename(columns={"beta_i": "beta_I", "n": "N", "maf": "MAF", "se": "SE", "phenotype_variant": "variant"}).dropna(subset=["pval"])
    return vqtl_df, candidates, interaction_df, interaction_significant_df, rge_df, perm_df, robustness_df


def cmd_report(args) -> str:
    ge_cfg = get_config()
    vcfg = get_vqtl_config()
    dataset, cohort_dir, gen = _load_prepared_dataset(args.generation, force=False)
    vqtl_df, candidates, interaction_df, _interaction_sig_df, rge_df, perm_df, robustness_df = _fetch_all_for_report(gen)
    return build_report(
        dataset, vcfg, cohort_dir, gen,
        vqtl_df=vqtl_df, candidates=candidates, interaction_df=interaction_df,
        rge_df=rge_df, perm_df=perm_df, robustness_df=robustness_df,
        target_col=ge_cfg.target_col,
    )


def cmd_docx(args) -> str:
    ge_cfg = get_config()
    vcfg = get_vqtl_config()
    dataset, cohort_dir, gen = _load_prepared_dataset(args.generation, force=False)
    vqtl_df, candidates, interaction_df, interaction_significant_df, rge_df, perm_df, robustness_df = _fetch_all_for_report(gen)
    return build_docx_report(
        vcfg, cohort_dir, gen, n_samples=len(dataset.df),
        exposures=vcfg.exposures, target_col=ge_cfg.target_col,
        vqtl_df=vqtl_df, candidates=candidates, interaction_df=interaction_df,
        interaction_significant_df=interaction_significant_df,
        rge_df=rge_df, perm_df=perm_df, robustness_df=robustness_df,
    )


def cmd_run_all(args) -> None:
    ge_cfg = get_config()
    vcfg = get_vqtl_config()
    dataset, cohort_dir, gen = _load_prepared_dataset(args.generation, args.force)

    log.info("=== vQTL run-all: generazione %s ===", gen)

    n_sig = 0 if args.force else repo.count_significant_scan(gen)
    if n_sig > 0:
        log.info(
            "Generazione %s ha gia' %d risultati significativi salvati a DB: "
            "salto lo scan genoma-wide e il filtro, uso la cache (passa --force per rifare tutto da zero).",
            gen, n_sig,
        )
        vqtl_df = repo.get_scan_results(gen)
        candidates = repo.get_candidates(gen)
        regenerate_plots(vqtl_df, os.path.join(cohort_dir, "figures"))
    else:
        subset = _resolve_variant_subset(args, dataset)
        vqtl_df = run_vqtl_scan(dataset, vcfg, ge_cfg.target_col, generation=gen, force=args.force,
                                variant_subset=subset)
        vqtl_df, candidates, _lam = filter_candidates(vqtl_df, vcfg, os.path.join(cohort_dir, "figures"),
                                                      generation=gen)

    interaction_df = run_interaction_tests(dataset, vcfg, candidates, ge_cfg.target_col, generation=gen)
    interaction_significant_df = repo.get_interaction_significant(gen)
    if not interaction_significant_df.empty:
        interaction_significant_df = interaction_significant_df.rename(columns={"beta_i": "beta_I", "n": "N", "maf": "MAF", "se": "SE"})
    rge_df = run_rge_het(dataset, vcfg, candidates, ge_cfg.target_col, generation=gen)
    robustness_df, perm_df = run_robustness_and_permutation(dataset, vcfg, interaction_df, ge_cfg.target_col, generation=gen)

    report_path = build_report(
        dataset, vcfg, cohort_dir, gen,
        vqtl_df=vqtl_df, candidates=candidates, interaction_df=interaction_df,
        rge_df=rge_df, perm_df=perm_df, robustness_df=robustness_df,
        target_col=ge_cfg.target_col,
    )
    docx_path = build_docx_report(
        vcfg, cohort_dir, gen, n_samples=len(dataset.df),
        exposures=vcfg.exposures, target_col=ge_cfg.target_col,
        vqtl_df=vqtl_df, candidates=candidates, interaction_df=interaction_df,
        interaction_significant_df=interaction_significant_df,
        rge_df=rge_df, perm_df=perm_df, robustness_df=robustness_df,
    )

    log.info("=== vQTL run-all completato: generazione %s -> %s , %s ===", gen, report_path, docx_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pipeline vQTL / G x E")
    sub = parser.add_subparsers(dest="command", required=True)

    def _add_common(p):
        p.add_argument("--generation", type=int, default=None, help="Sovrascrive GENERATION del .env (coorte gen1/gen2/gen3) per questo comando")
        p.add_argument("--force", action="store_true", help="Ricalcola tutto da zero, ignorando cache dataset E cache DB (scan gia' 'done', significativi gia' noti)")
        p.add_argument("--significant-only", action="store_true",
                       help="Limita alle varianti gia' note come significative (get_significant_results)")
        p.add_argument("--exposure", type=str, default=None,
                       help="Filtra get_significant_results per esposizione (richiede --significant-only)")

    for name in ["scan", "filter", "interaction", "rge-het", "permute", "report", "docx", "run-all"]:
        p = sub.add_parser(name)
        _add_common(p)

    args = parser.parse_args(argv)

    cfg = get_config()
    configure_logging(cfg.log_dir)

    dispatch = {
        "scan": cmd_scan,
        "filter": cmd_filter,
        "interaction": cmd_interaction,
        "rge-het": cmd_rge_het,
        "permute": cmd_permute,
        "report": cmd_report,
        "docx": cmd_docx,
        "run-all": cmd_run_all,
    }
    dispatch[args.command](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
