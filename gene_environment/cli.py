#!/usr/bin/env python3
"""
Single entry point for the pipeline. Replaces "ordine_comandi.txt" (a text
file listing the scripts to run by hand, in the right order, hoping not to
get it wrong) with a CLI that explicitly lists the available steps and the
recommended order.

Examples:
    python -m gene_environment.cli filter-vcf
    python -m gene_environment.cli build-matrix
    python -m gene_environment.cli run-model
    python -m gene_environment.cli replicate-significant --source-generation 2 --target-generation 1
    python -m gene_environment.cli extract-significant
    python -m gene_environment.cli export-significant-csv
    python -m gene_environment.cli report-onset-age
    python -m gene_environment.cli assign-genes
    python -m gene_environment.cli annotate-genes
    python -m gene_environment.cli run-gxe-genetlib
    python -m gene_environment.cli generate-reports          # Table 1, Table 2, Table 2b, annotated tables
    python -m gene_environment.cli generate-reports --only table2 table2b
    python -m gene_environment.cli pipeline-order    # print the recommended order
"""
from __future__ import annotations

import argparse
import sys

from gene_environment.config import get_config
from gene_environment.logging_utils import configure_logging, get_logger

log = get_logger(__name__)

PIPELINE_ORDER = """
Recommended execution order:

  1. filter-vcf              Filter the raw VCFs (MAF, LD pruning) -> filtered VCFs per cohort
  2. build-matrix             Filtered VCFs -> genotype matrix parquet (per chromosome + whole genome)
  3. run-model                Builds the dataset (genetics+environment) and runs the
                               per-variant test (matching + permutations + onset_age
                               difference), saving everything to the DB as it goes
  4. extract-significant      Extracts, from the source VCFs, the genotype of only the
                               variants found significant, for all cohorts (1/2/3),
                               writing the combined CSV INCREMENTALLY
  5. export-significant-csv   (repeatable at any time) exports an up-to-date CSV
                               snapshot of the currently significant variants, into a
                               folder separate from step 4's
  6. report-onset-age         Boxplot + forest plot of the onset_age difference, from
                               data already in the DB (no recalculation)
  7. assign-genes             Assigns the Ensembl gene to the significant variants
  8. annotate-genes           Enriches genes with neuro annotations (CTD/GO)
  9. recalculate-final        High-precision recalculation (10000 perms) of the
                               significant variants in both cohorts, with all betas
                               (JSON) -- after assign-genes
 10. generate-reports         Generates the Word/CSV/figure reports (Table 1, Table 2,
                               Table 2b, annotated tables) -- run any time after the
                               relevant upstream steps have populated the DB/CSVs

Steps 4 and 5 can be repeated as many times as needed during the
"run-model" run (e.g. from cron), to always have an up-to-date snapshot
of the significant variants without waiting for the full run to finish.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gene-environment pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("filter-vcf", help="Filter the raw VCFs (MAF, LD pruning)")
    sub.add_parser("build-matrix", help="Filtered VCFs -> genotype matrix parquet")
    sub.add_parser("run-model", help="Dataset + per-variant test + onset_age, saved to DB")
    p_extract = sub.add_parser("extract-significant", help="Extracts the genotype of the significant variants")
    p_extract.add_argument("--force", action="store_true", help="Ignore the checkpoint and start over")
    p_export = sub.add_parser("export-significant-csv", help="Exports a CSV snapshot of the significant variants")
    p_export.add_argument("--alpha", type=float, default=None, help="FDR threshold (default: config.pvalue_threshold)")
    p_repl = sub.add_parser(
        "replicate-significant",
        help="Re-tests, on another generation, ONLY the significant variants from an already-completed generation",
    )
    p_repl.add_argument("--source-generation", type=int, required=True, help="Already-completed generation to take significant variants from")
    p_repl.add_argument("--target-generation", type=int, required=True, help="Generation to re-test those variants on")
    p_repl.add_argument("--exposure", type=str, default=None, help="Default: config.exposure")
    p_repl.add_argument("--alpha", type=float, default=None, help="FDR threshold (default: config.pvalue_threshold)")
    p_repl.add_argument("--test-label", type=str, default=None, help="Default: replication_of_gen{source}")
    p_repl.add_argument("--force-rebuild-dataset", action="store_true", help="Rebuilds the target generation's dataset even if already cached")
    sub.add_parser("report-onset-age", help="Boxplot + forest plot of onset_age")
    sub.add_parser("assign-genes", help="Assigns the Ensembl gene to the significant variants")
    sub.add_parser("annotate-genes", help="Neuro annotations on the genes")
    sub.add_parser("pipeline-order", help="Prints the recommended step order")
    sub.add_parser("run-gxe-genetlib", help="Chromosome-by-chromosome G x E analysis with GENetLib")

    p_final = sub.add_parser(
        "recalculate-final",
        help="High-precision recalculation (10000 perms by default) of the significant variants in both cohorts, with all betas saved to JSON",
    )
    p_final.add_argument("--exposure", type=str, default=None, help="Default: config.exposure")
    p_final.add_argument("--n-perm", type=int, default=10000, help="Number of permutations (default: 10000)")
    p_final.add_argument("--force-rebuild-dataset", action="store_true")

    p_reports = sub.add_parser(
        "generate-reports",
        help="Generates the Word/CSV/figure reports (Table 1, Table 2, Table 2b, annotated tables)",
    )
    p_reports.add_argument(
        "--only", nargs="+", default=None,
        choices=["table1", "table2", "table2b", "annotated-tables"],
        help="Generate only these report(s) instead of all of them",
    )

    sub.add_parser("build-cohort-mapping", help="Rebuilds the id -> generation mapping CSV by reading VCF headers (prerequisite for table1)")
    sub.add_parser("run-c9-check", help="Restricted genotype/environment/C9orf72-code merge diagnostic (see report/c9_check.py)")

    args = parser.parse_args(argv)

    cfg = get_config()
    configure_logging(cfg.log_dir)

    if args.command == "pipeline-order":
        print(PIPELINE_ORDER)
        return 0

    if args.command == "filter-vcf":
        from gene_environment.vcf_pipeline.filter_vcf import run_filter_vcf
        run_filter_vcf()

    elif args.command == "build-matrix":
        from gene_environment.vcf_pipeline.vcf_to_parquet import run_vcf_to_parquet_pipeline
        run_vcf_to_parquet_pipeline()

    elif args.command == "run-model":
        from gene_environment.analysis.orchestrator import run_main_pipeline
        run_main_pipeline()

    elif args.command == "extract-significant":
        from gene_environment.significant_variants.extract_matrix import run_extract_significant_matrices
        run_extract_significant_matrices(force=args.force)

    elif args.command == "export-significant-csv":
        from gene_environment.significant_variants.export_significant_csv import run_export
        run_export(alpha=args.alpha, from_export=True)

    elif args.command == "replicate-significant":
        from gene_environment.analysis.run_replication import run_replication_on_significant_variants
        run_replication_on_significant_variants(
            source_generation=args.source_generation,
            target_generation=args.target_generation,
            exposure=args.exposure,
            alpha=args.alpha,
            test_label=args.test_label,
            force_rebuild_dataset=args.force_rebuild_dataset,
        )

    elif args.command == "report-onset-age":
        from gene_environment.analysis.report_onset_age import run_report_onset_age
        run_report_onset_age()

    elif args.command == "assign-genes":
        from gene_environment.gene_annotation.annotate_genes import run_assign_genes
        run_assign_genes()

    elif args.command == "annotate-genes":
        from gene_environment.gene_annotation.annotate_genes import run_annotate_gene_neuro_info
        run_annotate_gene_neuro_info()

    elif args.command == "run-gxe-genetlib":
        from gene_environment.analysis.gxe_genetlib import run_gxe_genetlib_pipeline
        run_gxe_genetlib_pipeline()

    elif args.command == "recalculate-final":
        from gene_environment.analysis.run_final_recalc import run_final_recalculation
        run_final_recalculation(
            exposure=args.exposure,
            n_perm=args.n_perm,
            force_rebuild_dataset=args.force_rebuild_dataset,
        )

    elif args.command == "generate-reports":
        from gene_environment.report.runner import run_all_reports
        run_all_reports(only=args.only)

    elif args.command == "build-cohort-mapping":
        from gene_environment.report.build_cohort_mapping import run_build_cohort_mapping
        run_build_cohort_mapping()

    elif args.command == "run-c9-check":
        from gene_environment.report.c9_check import run_c9_check
        run_c9_check()

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
