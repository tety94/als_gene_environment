# gene_environment/apis/panelapp_api.py
"""
Client per Genomics England PanelApp (https://panelapp.genomicsengland.co.uk).
API REST pubblica, solo lettura. Nessuna API key richiesta.

Usata per verificare se un gene è incluso in pannelli diagnostici relativi
a SLA / malattia del motoneurone, con relativo livello di confidenza
(green = evidenza solida, amber = moderata, red = limitata/da escludere).
"""
from __future__ import annotations

import requests

from gene_environment.logging_utils import get_logger

log = get_logger(__name__)

BASE_URL = "https://panelapp.genomicsengland.co.uk/api/v1"

# Filtro per nome, non per ID pannello: più robusto a variazioni di ID/versione.
ALS_KEYWORDS = ("amyotrophic lateral sclerosis", "motor neuron", "motor neurone", "mnd")


class PanelAppAPI:

    @staticmethod
    def get_als_status(gene_symbol: str, timeout: int = 15) -> dict:
        """Ritorna lo stato del gene rispetto ai pannelli SLA/MND in PanelApp.

        Output:
            {
                "found_in_als_panel": bool,
                "confidence_level": str | None,   # "3"=green,"2"=amber,"1"=red (vedi nota sotto)
                "panel_name": str | None,
                "matches": list[dict],            # tutte le entry gene trovate, per debug
            }

        NOTA: PanelApp esprime confidence_level come stringa numerica ("0","1","2","3"),
        dove 3 = green (diagnostic-grade), 2 = amber, 1 = red, 0 = non valutato/rimosso.
        Verifica questa mappatura contro la risposta reale prima di usarla in produzione:
        i valori esatti possono variare a seconda della versione dell'API.
        """
        resp = requests.get(
            f"{BASE_URL}/genes/",
            params={"entity_name": gene_symbol},
            timeout=timeout,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])

        als_matches = []
        for entry in results:
            panel_name = (entry.get("panel") or {}).get("name", "") or ""
            relevant_disorders = " ".join(entry.get("relevant_disorders") or [])
            haystack = f"{panel_name} {relevant_disorders}".lower()
            if any(kw in haystack for kw in ALS_KEYWORDS):
                als_matches.append(entry)

        if not als_matches:
            return {
                "found_in_als_panel": False,
                "confidence_level": None,
                "panel_name": None,
                "matches": [],
            }

        # Se il gene compare in più pannelli SLA, prendi la confidenza più alta.
        best = max(als_matches, key=lambda e: e.get("confidence_level", "0"))
        return {
            "found_in_als_panel": True,
            "confidence_level": best.get("confidence_level"),
            "panel_name": (best.get("panel") or {}).get("name"),
            "matches": als_matches,
        }