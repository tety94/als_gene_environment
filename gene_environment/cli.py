#!/usr/bin/env python3
"""
Entry point unico della pipeline. Sostituisce "ordine_comandi.txt" (un file
di testo con l'elenco degli script da lanciare a mano, nell'ordine giusto,
sperando di non sbagliare) con un CLI che elenca esplicitamente gli step
disponibili e l'ordine consigliato.

Esempi:
    python -m gene_environment.cli filter-vcf
    python -m gene_environment.cli build-matrix
    python -m gene_environment.cli run-model
    python -m gene_environment.cli extract-significant
    python -m gene_environment.cli export-significant-csv
    python -m gene_environment.cli report-onset-age
    python -m gene_environment.cli assign-genes
    python -m gene_environment.cli annotate-genes
    python -m gene_environment.cli pipeline-order    # stampa l'ordine consigliato
"""
from __future__ import annotations

import argparse
import sys

from gene_environment.config import get_config
from gene_environment.logging_utils import configure_logging, get_logger

log = get_logger(__name__)

PIPELINE_ORDER = """
Ordine consigliato di esecuzione:

  1. filter-vcf              Filtra i VCF grezzi (MAF, LD pruning) -> VCF filtrati per coorte
  2. build-matrix             VCF filtrati -> matrice genotipica parquet (per cromosoma + genoma intero)
  3. run-model                Costruisce il dataset (genetica+ambiente) e lancia il test
                               per-variante (matching + permutazioni + differenza onset_age),
                               salvando tutto a DB man mano
  4. extract-significant      Estrae dai VCF sorgente il genotipo delle sole varianti
                               risultate significative, per tutte le coorti (1/2/3),
                               scrivendo il CSV combinato IN MODO INCREMENTALE
  5. export-significant-csv   (ripetibile in ogni momento) esporta uno snapshot CSV
                               aggiornato delle varianti significative correnti, in una
                               cartella separata da quella dello step 4
  6. report-onset-age         Boxplot + forest plot sulla differenza onset_age, dai dati
                               già in DB (nessun ricalcolo)
  7. assign-genes             Assegna il gene Ensembl alle varianti significative
  8. annotate-genes           Arricchisce i geni con annotazioni neuro (CTD/GO)

Gli step 4 e 5 possono essere ripetuti quante volte serve durante il run di
"run-model" (es. da cron), per avere sempre uno snapshot aggiornato delle
varianti significative senza dover aspettare la fine del run completo.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pipeline gene-environment")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("filter-vcf", help="Filtra i VCF grezzi (MAF, LD pruning)")
    sub.add_parser("build-matrix", help="VCF filtrati -> matrice genotipica parquet")
    sub.add_parser("run-model", help="Dataset + test per-variante + onset_age, salvati a DB")
    p_extract = sub.add_parser("extract-significant", help="Estrae il genotipo delle varianti significative")
    p_extract.add_argument("--force", action="store_true", help="Ignora il checkpoint e riparte da zero")
    p_export = sub.add_parser("export-significant-csv", help="Esporta uno snapshot CSV delle varianti significative")
    p_export.add_argument("--alpha", type=float, default=None, help="Soglia FDR (default: config.pvalue_threshold)")
    sub.add_parser("report-onset-age", help="Boxplot + forest plot onset_age")
    sub.add_parser("assign-genes", help="Assegna il gene Ensembl alle varianti significative")
    sub.add_parser("annotate-genes", help="Annotazioni neuro sui geni")
    sub.add_parser("pipeline-order", help="Stampa l'ordine consigliato degli step")

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
        run_export(alpha=args.alpha)

    elif args.command == "report-onset-age":
        from gene_environment.analysis.report_onset_age import run_report_onset_age
        run_report_onset_age()

    elif args.command == "assign-genes":
        from gene_environment.gene_annotation.annotate_genes import run_assign_genes
        run_assign_genes()

    elif args.command == "annotate-genes":
        from gene_environment.gene_annotation.annotate_genes import run_annotate_gene_neuro_info
        run_annotate_gene_neuro_info()

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
