# gene_environment/apis/open_targets_api.py
"""Client for the Open Targets Platform (https://platform.opentargets.org).
Public GraphQL API, no key required.

Used to fetch the gene-disease association score (aggregated from
literature, GWAS, animal models, etc.) for ALS, given an Ensembl gene ID.
"""
from __future__ import annotations

import requests

from gene_environment.logging_utils import get_logger

log = get_logger(__name__)

GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"

ALS_KEYWORDS = ("amyotrophic lateral sclerosis", "motor neuron", "motor neurone")

_QUERY = """
query AssociatedDiseases($ensemblId: String!) {
    target(ensemblId: $ensemblId) {
        id
        approvedSymbol
        associatedDiseases(page: {index: 0, size: 2000}) {
            rows {
                disease {
                    id
                    name
                }
                score
            }
        }
    }
}
"""


class OpenTargetsAPI:

    @staticmethod
    def get_als_association(ensembl_gene_id: str, timeout: int = 15) -> dict:
        resp = requests.post(
            GRAPHQL_URL,
            json={"query": _QUERY, "variables": {"ensemblId": ensembl_gene_id}},
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()

        if "errors" in payload:
            log.error("Open Targets API error for %s: %s", ensembl_gene_id, payload["errors"])
            return {"associated": False, "score": None, "disease_name": None, "disease_id": None}

        target = (payload.get("data") or {}).get("target")
        if not target:
            log.info("Open Targets: %s not found (target=None, gene not in Open Targets)", ensembl_gene_id)
            return {"associated": False, "score": None, "disease_name": None, "disease_id": None}

        rows = ((target.get("associatedDiseases") or {}).get("rows")) or []
        log.info(
            "Open Targets: %s (%s) found, %d total associated diseases",
            ensembl_gene_id, target.get("approvedSymbol"), len(rows),
        )

        als_rows = [
            r for r in rows
            if any(kw in (r["disease"]["name"] or "").lower() for kw in ALS_KEYWORDS)
        ]

        if not als_rows:
            log.info(
                "Open Targets: %s has %d associated diseases, none ALS-related",
                ensembl_gene_id, len(rows),
            )
            return {"associated": False, "score": None, "disease_name": None, "disease_id": None}

        best = max(als_rows, key=lambda r: r["score"])
        log.info(
            "Open Targets MATCH: %s -> '%s' score=%.3f",
            ensembl_gene_id, best["disease"]["name"], best["score"],
        )
        return {
            "associated": True,
            "score": best["score"],
            "disease_name": best["disease"]["name"],
            "disease_id": best["disease"]["id"],
        }