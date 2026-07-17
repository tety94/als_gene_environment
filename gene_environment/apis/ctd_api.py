"""
Client per file .tsv scaricati manualmente dal CTD Batch Query
(https://ctdbase.org/tools/batchQuery.go).

Uso: l'utente esegue la query a mano nel browser (report "Chemical-Gene
Interactions" e/o "Gene-Disease Associations"), scarica il TSV, e lo passa
a questo modulo per costruire un indice gene -> interazioni.

Formato Chemical-Gene Interactions (esempio osservato):
  # Input  ChemicalName  ChemicalID  CasRN  GeneSymbol  GeneID  Organism
    OrganismID  Interaction  InteractionActions  PubMedIDs

Formato Gene-Disease Associations: colonne diverse (da confermare al primo
file reale — il parser rileva il tipo guardando l'header).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from gene_environment.logging_utils import get_logger

log = get_logger(__name__)

PESTICIDE_KEYWORDS = (
    "pesticide", "herbicide", "insecticide", "fungicide", "rodenticide",
    "chlorpyrifos", "glyphosate", "paraquat", "atrazine", "malathion",
    "permethrin", "carbaryl", "diazinon",
)


@dataclass
class ChemGeneInteraction:
    gene_symbol: str
    chemical_name: str
    chemical_id: str
    cas_rn: str
    organism: str
    interaction: str
    interaction_actions: str
    pubmed_ids: List[str] = field(default_factory=list)

    @property
    def is_pesticide_by_keyword(self) -> bool:
        return any(kw in self.chemical_name.lower() for kw in PESTICIDE_KEYWORDS)

@dataclass
class GeneDiseaseAssociation:
    gene_symbol: str
    disease_name: str
    disease_id: str
    disease_categories: str
    direct_evidence: str              # non vuoto = associazione curata diretta
    inference_chemical_name: str      # valorizzato = associazione inferita TRAMITE questo chimico
    inference_score: Optional[float]
    omim_ids: str
    pubmed_ids: List[str] = field(default_factory=list)

    @property
    def is_direct(self) -> bool:
        return bool(self.direct_evidence)

    @property
    def is_inferred_via_chemical(self) -> bool:
        return bool(self.inference_chemical_name) and not self.is_direct

    @property
    def is_pesticide_mediated(self) -> bool:
        return self.is_inferred_via_chemical and any(
            kw in self.inference_chemical_name.lower() for kw in PESTICIDE_KEYWORDS
        )


class CTDAPI:
    # path di default, sovrascrivibili da config se preferisci
    CHEM_GENE_TSV_PATH = "/srv/python-projects/gene_environment_v2/data/ctd_chem_gene_export.tsv"
    DISEASE_TSV_PATH = "/srv/python-projects/gene_environment_v2/data/ctd_gene_disease_export.tsv"

    _chem_index_cache: Dict[str, List["ChemGeneInteraction"]] | None = None
    _disease_index_cache: Dict[str, List["GeneDiseaseAssociation"]] | None = None

    @staticmethod
    def build_disease_index(path: str, keyword_filter: Optional[str] = None
                             ) -> Dict[str, List[GeneDiseaseAssociation]]:
        """Formato reale: Input, DiseaseName, DiseaseID, GeneSymbol, GeneID,
        DiseaseCategories, DirectEvidence, InferenceChemicalName,
        InferenceScore, OmimIDs, PubMedIDs

        keyword_filter: se valorizzato, tiene solo le righe il cui
        DiseaseName contiene questa stringa (case-insensitive)."""
        rows = CTDAPI._read_tsv_rows(path)
        index: Dict[str, List[GeneDiseaseAssociation]] = {}
        skipped_no_gene = 0

        for row in rows:
            gene = (row.get("GeneSymbol") or "").strip().upper()
            disease = (row.get("DiseaseName") or "").strip()
            if not gene or not disease:
                skipped_no_gene += 1
                continue
            if keyword_filter and keyword_filter.lower() not in disease.lower():
                continue

            pubmed_raw = (row.get("PubMedIDs") or "").strip()
            pubmed_ids = [p for p in pubmed_raw.split("|") if p]

            score_raw = (row.get("InferenceScore") or "").strip()
            try:
                score = float(score_raw) if score_raw else None
            except ValueError:
                score = None

            index.setdefault(gene, []).append(GeneDiseaseAssociation(
                gene_symbol=gene,
                disease_name=disease,
                disease_id=(row.get("DiseaseID") or "").strip(),
                disease_categories=(row.get("DiseaseCategories") or "").strip(),
                direct_evidence=(row.get("DirectEvidence") or "").strip(),
                inference_chemical_name=(row.get("InferenceChemicalName") or "").strip(),
                inference_score=score,
                omim_ids=(row.get("OmimIDs") or "").strip(),
                pubmed_ids=pubmed_ids,
            ))

        log.info(
            "CTD disease index: %d geni distinti, %d righe senza gene risolto (filtro keyword='%s')",
            len(index), skipped_no_gene, keyword_filter,
        )
        return index



    @classmethod
    def get_chem_index(cls) -> Dict[str, List["ChemGeneInteraction"]]:
        """Lazy-load, cache a livello di processo. In un ProcessPoolExecutor
        ogni worker ha la propria cache: il file viene letto una volta per
        processo, non una volta per gene."""
        if cls._chem_index_cache is None:
            cls._chem_index_cache = cls.build_chem_gene_index(cls.CHEM_GENE_TSV_PATH)
        return cls._chem_index_cache

    @classmethod
    def get_disease_index(cls) -> Dict[str, List["GeneDiseaseAssociation"]]:
        if cls._disease_index_cache is None:
            cls._disease_index_cache = cls.build_disease_index(cls.DISEASE_TSV_PATH, keyword_filter=None)
        return cls._disease_index_cache

    @staticmethod
    def _read_tsv_rows(path: str) -> List[dict]:
        """Legge un TSV di CTD Batch Query: trova l'header (riga che inizia
        con '#' contenente i nomi colonna, tipicamente l'ultima riga di
        commento prima dei dati) e ritorna una lista di dict per riga."""
        header = None
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                if line.startswith("#"):
                    header = [h.strip() for h in line.lstrip("#").strip().split("\t")]
                    continue
                if header is None:
                    continue
                values = line.split("\t")
                if len(values) < len(header):
                    values += [""] * (len(header) - len(values))
                elif len(values) > len(header):
                    values = values[:len(header)]
                rows.append(dict(zip(header, values)))
        log.info("CTD batch file %s: %d righe dati lette (header=%s)", path, len(rows), header)
        return rows

    @staticmethod
    def build_chem_gene_index(path: str, organism_filter: Optional[str] = "Homo sapiens"
                               ) -> Dict[str, List[ChemGeneInteraction]]:
        """Costruisce indice gene_symbol -> lista interazioni chimiche, dal
        file 'Chemical-Gene Interactions' scaricato dal Batch Query.

        organism_filter: se valorizzato, tiene solo le righe con quell'
        organismo esatto (default: solo Homo sapiens, dato che i risultati
        CTD includono anche modelli murini/ratto che potrebbero non
        interessarti). Passa None per tenere tutte le specie."""
        rows = CTDAPI._read_tsv_rows(path)
        index: Dict[str, List[ChemGeneInteraction]] = {}
        skipped_organism = 0

        for row in rows:
            gene = (row.get("GeneSymbol") or "").strip().upper()
            if not gene:
                continue

            organism = (row.get("Organism") or "").strip()
            if organism_filter and organism != organism_filter:
                skipped_organism += 1
                continue

            pubmed_raw = (row.get("PubMedIDs") or "").strip()
            pubmed_ids = [p for p in pubmed_raw.split("|") if p]

            index.setdefault(gene, []).append(ChemGeneInteraction(
                gene_symbol=gene,
                chemical_name=(row.get("ChemicalName") or "").strip(),
                chemical_id=(row.get("ChemicalID") or "").strip(),
                cas_rn=(row.get("CasRN") or "").strip(),
                organism=organism,
                interaction=(row.get("Interaction") or "").strip(),
                interaction_actions=(row.get("InteractionActions") or "").strip(),
                pubmed_ids=pubmed_ids,
            ))

        log.info(
            "CTD chem-gene index: %d geni distinti, %d righe scartate per filtro organismo (%s)",
            len(index), skipped_organism, organism_filter,
        )
        return index

    # --------------------------------------------------------------
    # Query per singolo gene (lookup in memoria)
    # --------------------------------------------------------------

    @staticmethod
    def query_gene(gene_symbol: str,
                    chem_index: Dict[str, List[ChemGeneInteraction]],
                    disease_index: Optional[Dict[str, List[GeneDiseaseAssociation]]] = None
                    ) -> dict:
        symbol = (gene_symbol or "").strip().upper()

        chem_interactions = chem_index.get(symbol, [])
        pesticide_interactions = [ci for ci in chem_interactions if ci.is_pesticide_by_keyword]

        diseases = disease_index.get(symbol, []) if disease_index else []

        return {
            "chemicals": [ci.chemical_name for ci in chem_interactions],
            "chemical_interactions": chem_interactions,
            "pesticide_interactions": pesticide_interactions,
            "diseases": diseases,
        }