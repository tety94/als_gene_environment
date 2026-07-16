# gene_environment/apis/panelapp_api.py
from __future__ import annotations

import time
import requests

from gene_environment.logging_utils import get_logger

log = get_logger(__name__)

BASE_URL = "https://panelapp.genomicsengland.co.uk/api/v1"
ALS_KEYWORDS = ("amyotrophic lateral sclerosis", "motor neuron", "motor neurone", "mnd")


def _get_with_retry(url: str, params: dict, timeout: int, max_retries: int = 5) -> requests.Response:
    """GET con retry/backoff su 429 (rate limit) e 5xx (errori transitori
    del server). Rispetta l'header Retry-After se presente, altrimenti
    usa un backoff esponenziale con jitter."""
    for attempt in range(max_retries):
        resp = requests.get(url, params=params, timeout=timeout)

        if resp.status_code == 429 or resp.status_code >= 500:
            retry_after = resp.headers.get("Retry-After")
            if retry_after is not None:
                wait = float(retry_after)
            else:
                wait = (2 ** attempt) + (0.1 * attempt)  # backoff esponenziale + piccolo jitter
            log.warning(
                "PanelApp %s (tentativo %d/%d), attendo %.1fs: %s",
                resp.status_code, attempt + 1, max_retries, wait, url,
            )
            time.sleep(wait)
            continue

        resp.raise_for_status()
        return resp

    resp.raise_for_status()  # ultimo tentativo: se ancora in errore, solleva
    return resp


class PanelAppAPI:

    @staticmethod
    class PanelAppAPI:

        @staticmethod
        def get_als_status(gene_symbol: str, timeout: int = 15) -> dict:
            if not gene_symbol or gene_symbol.startswith("ENSG"):
                log.warning(
                    "PanelApp: simbolo gene mancante o non risolto ('%s'), skip query.",
                    gene_symbol,
                )
                return {"found_in_als_panel": False, "confidence_level": None,
                        "panel_name": None, "matches": [], "skipped_no_symbol": True}

            resp = _get_with_retry(f"{BASE_URL}/genes/", params={"entity_name": gene_symbol}, timeout=timeout)
            payload = resp.json()

            log.info(
                "PanelApp query gene_symbol=%s -> count=%s risultati totali",
                gene_symbol, payload.get("count"),
            )

            results = payload.get("results", [])
            als_matches = []
            for entry in results:
                panel_name = (entry.get("panel") or {}).get("name", "") or ""
                relevant_disorders = " ".join(entry.get("relevant_disorders") or [])
                haystack = f"{panel_name} {relevant_disorders}".lower()
                if any(kw in haystack for kw in ALS_KEYWORDS):
                    als_matches.append(entry)
                    log.info(
                        "PanelApp MATCH gene=%s panel='%s' confidence=%s",
                        gene_symbol, panel_name, entry.get("confidence_level"),
                    )

            if not als_matches:
                log.info(
                    "PanelApp: gene=%s trovato in %d pannelli totali, nessuno relativo a SLA/MND",
                    gene_symbol, len(results),
                )
                return {"found_in_als_panel": False, "confidence_level": None,
                        "panel_name": None, "matches": [], "skipped_no_symbol": False}

            best = max(als_matches, key=lambda e: e.get("confidence_level", "0"))
            return {
                "found_in_als_panel": True,
                "confidence_level": best.get("confidence_level"),
                "panel_name": (best.get("panel") or {}).get("name"),
                "matches": als_matches,
                "skipped_no_symbol": False,
            }