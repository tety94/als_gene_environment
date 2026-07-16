# gene_environment/apis/open_targets_api.py
"""
Client per Open Targets Platform (https://platform.opentargets.org).
API GraphQL pubblica, nessuna key richiesta.

Usata per ottenere lo score di associazione gene-malattia (aggregato da
letteratura, GWAS, modelli animali, ecc.) per SLA, dato un Ensembl gene ID.
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
        associatedDiseases(page: {index: 0, size: 200}) {
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
        """Ritorna l'evidenza di associazione gene-SLA da Open Targets.

        Output:
            {
                "associated": bool,
                "score": float | None,     # 0.0-1.0, quanto più alto quanto più forte l'evidenza aggregata
                "disease_name": str | None,
                "disease_id": str | None,
            }
        """
        resp = requests.post(
            GRAPHQL_URL,
            json={"query": _QUERY, "variables": {"ensemblId": ensembl_gene_id}},
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()

        if "errors" in payload:
            log.error("Open Targets API error per %s: %s", ensembl_gene_id, payload["errors"])
            return {"associated": False, "score": None, "disease_name": None, "disease_id": None}

        target = (payload.get("data") or {}).get("target")
        if not target:
            return {"associated": False, "score": None, "disease_name": None, "disease_id": None}

        rows = ((target.get("associatedDiseases") or {}).get("rows")) or []
        als_rows = [
            r for r in rows
            if any(kw in (r["disease"]["name"] or "").lower() for kw in ALS_KEYWORDS)
        ]

        if not als_rows:
            return {"associated": False, "score": None, "disease_name": None, "disease_id": None}

        best = max(als_rows, key=lambda r: r["score"])
        return {
            "associated": True,
            "score": best["score"],
            "disease_name": best["disease"]["name"],
            "disease_id": best["disease"]["id"],
        }