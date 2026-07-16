#!/usr/bin/env python3
"""
Annotazione genica delle varianti significative. Unisce due script originali
che facevano cose consecutive e correlate (e condividevano lo stesso
pattern "ProcessPoolExecutor + chiamata API esterna"):
  - process_variants.py     -> assegna il gene (Ensembl) a ogni variante
  - main_gene_analysis.py   -> annota i geni trovati con info neuro (CTD/GO)

Fix:
  - `apis.ensembl_api.EnsemblAPI` e `pipeline.gene_annotator.GeneAnnotator`
    sono moduli esterni al set di file fornito: qui l'import è protetto con
    un messaggio d'errore chiaro invece di un ImportError criptico, così chi
    esegue lo script sa subito cosa manca.
  - Logging invece di print(); un'eccezione su UNA variante/gene non ferma
    più silenziosamente l'intero batch senza traccia (era già gestito con
    try/except in main_gene_analysis.py, qui uniformato e loggato anche in
    process_variants, che nell'originale non aveva alcuna gestione errori:
    un fallimento della chiamata Ensembl per una variante avrebbe fermato
    l'intero script, perdendo il lavoro fatto su tutte le altre).
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed

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
            "Modulo 'apis.ensembl_api.EnsemblAPI' non trovato. È una dipendenza esterna "
            "al pacchetto refattorizzato: assicurati che sia installato/nel PYTHONPATH "
            "prima di lanciare annotate_genes."
        ) from e


def _get_gene_annotator():
    try:
        from pipeline.gene_annotator import GeneAnnotator
        return GeneAnnotator
    except ImportError as e:
        raise ImportError(
            "Modulo 'pipeline.gene_annotator.GeneAnnotator' non trovato. È una dipendenza "
            "esterna al pacchetto refattorizzato: assicurati che sia installato/nel "
            "PYTHONPATH prima di lanciare annotate_genes."
        ) from e


def run_assign_genes(iterations: int | None = None) -> None:
    """ex process_variants.py: assegna il gene Ensembl a ogni variante
    significativa senza gene ancora assegnato."""
    cfg = get_config()
    configure_logging(cfg.log_dir)
    EnsemblAPI = _get_ensembl_api()

    variants = get_empty_variants_gene()
    log.info("Varianti senza gene assegnato: %d", len(variants))

    COMMIT_EVERY = 20  # numero di varianti tra un commit e l'altro

    ok, failed = 0, 0
    with get_connection() as conn:
        for i, (_, variant) in enumerate(variants.iterrows(), start=1):
            chrom, pos = variant["chromosome"], variant["position"]
            try:
                gene_id, gene_name = EnsemblAPI.fetch_gene(chrom, pos)
            except Exception:
                log.exception("Errore Ensembl per variante %s (chr%s:%s)", variant["variant"], chrom, pos)
                failed += 1
                continue

            if gene_id:
                update_variant_gene(conn, variant["variant"], gene_id, gene_name)
                log.info("%s -> gene %s (%s)", variant["variant"], gene_id, gene_name)
                ok += 1
            else:
                update_variant_gene(conn, variant["variant"], "NO-GENE", "NO-GENE")
                log.info("Nessun gene trovato per %s", variant["variant"])
                ok += 1

            if i % COMMIT_EVERY == 0:
                conn.commit()
                log.info("Commit intermedio dopo %d varianti processate", i)

        conn.commit()  # flush finale per l'ultimo blocco parziale (< 20 varianti)

    log.info("Assegnazione geni completata: %d ok, %d falliti", ok, failed)

def _annotate_one_gene(gene: str) -> tuple[str, bool, str | None]:
    GeneAnnotator = _get_gene_annotator()
    try:
        GeneAnnotator.annotate(gene)
        return gene, True, None
    except Exception as e:
        return gene, False, str(e)


def run_annotate_gene_neuro_info() -> None:
    """ex main_gene_analysis.py: arricchisce i geni trovati con annotazioni
    neuro (espressione cerebrale, processi GO, malattie CTD, ecc.)."""
    cfg = get_config()
    configure_logging(cfg.log_dir)

    genes = get_genes_to_annotate()
    log.info("Geni da annotare: %d", len(genes))
    if not genes:
        return

    failed = []
    with ProcessPoolExecutor(max_workers=cfg.max_workers) as ex:
        futures = {ex.submit(_annotate_one_gene, g): g for g in genes}
        for fut in as_completed(futures):
            gene, ok, err = fut.result()
            if not ok:
                failed.append((gene, err))
                log.error("Errore sul gene %s: %s", gene, err)

    log.info("Annotazione completata: %d ok, %d falliti", len(genes) - len(failed), len(failed))


if __name__ == "__main__":
    run_assign_genes()
    run_annotate_gene_neuro_info()
