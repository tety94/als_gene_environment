#!/usr/bin/env python3
"""Gene annotation of significant variants: assigns the Ensembl gene to each variant, then enriches genes with neuro annotations (CTD/GO)."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed

from gene_environment.apis.ctd_api import CTDAPI
from gene_environment.config import get_config
from gene_environment.db.connection import get_connection
from gene_environment.db.repository import (
    get_empty_variants_gene,
    update_variant_gene,
    get_genes_to_annotate,
)
from gene_environment.logging_utils import configure_logging, get_logger

log = get_logger(__name__)


def _get_ensembl_api():
    try:
        from gene_environment.apis.ensembl_api import EnsemblAPI
        return EnsemblAPI
    except ImportError as e:
        raise ImportError(
            "Module 'apis.ensembl_api.EnsemblAPI' not found."
        ) from e


def _get_gene_annotator():
    try:
        from gene_environment.apis.gene_annotator import GeneAnnotator
        return GeneAnnotator
    except ImportError as e:
        raise ImportError(
            "Module 'gene_environment.apis.gene_annotator' not found."
        ) from e


def run_assign_genes(iterations: int | None = None) -> None:
    """Assign the Ensembl gene to every significant variant that doesn't
    have a gene assigned yet."""
    cfg = get_config()
    configure_logging(cfg.log_dir)
    EnsemblAPI = _get_ensembl_api()

    variants = get_empty_variants_gene()
    log.info("Variants without an assigned gene: %d", len(variants))

    COMMIT_EVERY = 20  # number of variants between commits

    ok, failed = 0, 0
    with get_connection() as conn:
        for i, (_, variant) in enumerate(variants.iterrows(), start=1):
            chrom, pos = variant["chromosome"], variant["position"]
            try:
                gene_id, gene_name = EnsemblAPI.fetch_gene(chrom, pos)
            except Exception:
                log.exception("Ensembl error for variant %s (chr%s:%s)", variant["variant"], chrom, pos)
                failed += 1
                continue

            if gene_id:
                update_variant_gene(conn, variant["variant"], gene_id, gene_name)
                log.info("%s -> gene %s (%s)", variant["variant"], gene_id, gene_name)
                ok += 1
            else:
                update_variant_gene(conn, variant["variant"], "NO-GENE", "NO-GENE")
                log.info("No gene found for %s", variant["variant"])
                ok += 1

            if i % COMMIT_EVERY == 0:
                conn.commit()
                log.info("Intermediate commit after %d variants processed", i)

        conn.commit()  # final flush for the last partial batch (< 20 variants)

    log.info("Gene assignment complete: %d ok, %d failed", ok, failed)

def _annotate_one_gene(gene: str) -> tuple[str, bool, str | None]:
    GeneAnnotator = _get_gene_annotator()
    try:
        GeneAnnotator.annotate(gene)
        return gene, True, None
    except Exception as e:
        return gene, False, str(e)


def run_annotate_gene_neuro_info() -> None:
    cfg = get_config()
    configure_logging(cfg.log_dir)

    genes = get_genes_to_annotate()
    log.info("Genes to annotate: %d", len(genes))
    if not genes:
        return

    # The external APIs (PanelApp, Open Targets) have their own rate limits,
    # independent of how many cores the machine has: the generic
    # max_workers is too aggressive here. A low, fixed value is safer.
    annotation_workers = min(cfg.max_workers, 3)

    failed = []
    with ProcessPoolExecutor(max_workers=annotation_workers) as ex:
        futures = {ex.submit(_annotate_one_gene, g): g for g in genes}
        for fut in as_completed(futures):
            gene, ok, err = fut.result()
            if not ok:
                failed.append((gene, err))
                log.error("Error on gene %s: %s", gene, err)

    log.info("Annotation complete: %d ok, %d failed", len(genes) - len(failed), len(failed))

if __name__ == "__main__":
    run_assign_genes()
    run_annotate_gene_neuro_info()
