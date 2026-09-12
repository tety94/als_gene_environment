"""Client for .tsv files manually downloaded from the CTD Batch Query
(https://ctdbase.org/tools/batchQuery.go).

Usage: the user runs the query by hand in the browser (the "Chemical-Gene
Interactions" and/or "Gene-Disease Associations" report), downloads the
TSV, and passes it to this module to build a gene -> interactions index.

Chemical-Gene Interactions format (observed example):
  # Input  ChemicalName  ChemicalID  CasRN  GeneSymbol  GeneID  Organism
    OrganismID  Interaction  InteractionActions  PubMedIDs

Gene-Disease Associations format: different columns (the parser detects
the type by looking at the header).
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

    "ddt", "dichlorodiphenyltrichloroethane", "rotenone", "fipronil",
    "2,4-d", "dieldrin", "endosulfan", "deltamethrin", "imidacloprid",
    "endrin", "lindane", "aldrin", "chlordane", "heptachlor",
    "organophosphate", "organochlorine", "carbamate",
    "diquat", "metolachlor", "pyrethroid", "pyrethrin",
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
    direct_evidence: str              # non-empty = direct curated association
    inference_chemical_name: str      # set = association inferred THROUGH this chemical
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
    # default paths, can be overridden via config if preferred
    CHEM_GENE_TSV_PATH = "/srv/python-projects/gene_environment_v2/data/ctd_chem_gene_export.tsv"
    DISEASE_TSV_PATH = "/srv/python-projects/gene_environment_v2/data/ctd_gene_disease_export.tsv"

    _chem_index_cache: Dict[str, List["ChemGeneInteraction"]] | None = None
    _disease_index_cache: Dict[str, List["GeneDiseaseAssociation"]] | None = None

    @staticmethod
    def build_disease_index(path: str, keyword_filter: Optional[str] = None
                             ) -> Dict[str, List[GeneDiseaseAssociation]]:
        """Real format: Input, DiseaseName, DiseaseID, GeneSymbol, GeneID,
        DiseaseCategories, DirectEvidence, InferenceChemicalName,
        InferenceScore, OmimIDs, PubMedIDs

        keyword_filter: if set, keeps only rows whose DiseaseName contains
        this string (case-insensitive)."""
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
            "CTD disease index: %d distinct genes, %d rows with no resolved gene (keyword filter='%s')",
            len(index), skipped_no_gene, keyword_filter,
        )
        return index



    @classmethod
    def get_chem_index(cls) -> Dict[str, List["ChemGeneInteraction"]]:
        """Lazy-load, process-level cache. In a ProcessPoolExecutor each
        worker has its own cache: the file is read once per process, not
        once per gene."""
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
        """Read a CTD Batch Query TSV: finds the header (the line starting
        with '#' containing the column names, typically the last comment
        line before the data) and returns a list of dicts, one per row."""
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
        log.info("CTD batch file %s: %d data rows read (header=%s)", path, len(rows), header)
        return rows

    @staticmethod
    def build_chem_gene_index(path: str, organism_filter: Optional[str] = "Homo sapiens"
                               ) -> Dict[str, List[ChemGeneInteraction]]:
        """Build a gene_symbol -> chemical interactions list index from the
        'Chemical-Gene Interactions' file downloaded from the Batch Query.

        organism_filter: if set, keeps only rows matching that exact
        organism (default: Homo sapiens only, since CTD results also
        include mouse/rat models). Pass None to keep all species."""
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
            "CTD chem-gene index: %d distinct genes, %d rows dropped by organism filter (%s)",
            len(index), skipped_organism, organism_filter,
        )
        return index

    # --------------------------------------------------------------
    # Single-gene query (in-memory lookup)
    # --------------------------------------------------------------

    @staticmethod
    @staticmethod
    def query_gene(gene_symbol: str,
                   chem_index: Dict[str, List[ChemGeneAssociation]],
                   disease_index: Optional[Dict[str, List[GeneDiseaseAssociation]]] = None
                   ) -> dict:
        symbol = (gene_symbol or "").strip().upper()

        if not symbol or symbol.startswith("ENSG"):
            log.warning("CTD: missing or unresolved gene symbol ('%s'), skipping.", gene_symbol)
            return {"chemicals": [], "chemical_interactions": [], "pesticide_interactions": [], "diseases": []}

        chem_interactions = chem_index.get(symbol, [])
        pesticide_interactions = [ci for ci in chem_interactions if ci.is_pesticide_by_keyword]
        diseases = disease_index.get(symbol, []) if disease_index else []

        log.info(
            "CTD: gene=%s -> %d total chemical interactions (%d pesticides), %d associated diseases",
            symbol, len(chem_interactions), len(pesticide_interactions), len(diseases),
        )

        if pesticide_interactions:
            log.info(
                "CTD pesticide MATCH: gene=%s pesticides=%s",
                symbol, [ci.chemical_name for ci in pesticide_interactions],
            )

        return {
            "chemicals": [ci.chemical_name for ci in chem_interactions],
            "chemical_interactions": chem_interactions,
            "pesticide_interactions": pesticide_interactions,
            "diseases": diseases,
        }