"""Human Protein Atlas (HPA) client: fetches the full per-gene JSON and extracts single-cell expression info."""
import requests

class HPAAPI:
    """Queries the full HPA JSON for a gene (there's no dedicated single_cell
    endpoint). Uses the URL:
        https://www.proteinatlas.org/<ENSG>.json
    """

    BASE_URL = "https://www.proteinatlas.org"

    @staticmethod
    def fetch_hpa_json(ensg: str) -> dict:
        """Download the full JSON for the gene."""
        url = f"{HPAAPI.BASE_URL}/{ensg}.json"
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            return {}

    @staticmethod
    def get_single_cell_info(ensg: str) -> dict:
        """Extract single-cell data (if present) from the HPA JSON.
        HPA has no dedicated single_cell REST endpoint, but the full JSON
        can contain sections like 'rna_single_cell_type' or similar."""
        j = HPAAPI.fetch_hpa_json(ensg)
        result = {
            "neurons": False,
            "glia": False,
            "cell_types": []
        }

        # The HPA JSON can have various sections; check fields related to "rna_single_cell"
        sc_nCPM = j.get("RNA single cell type specific nCPM", {}) or {}

        for cell_type in sc_nCPM.keys():
            ct_lower = cell_type.lower()
            result["cell_types"].append(cell_type)
            if "neuron" in ct_lower:
                result["neurons"] = True
            if any(x in ct_lower for x in ["astro", "oligo", "microglia", "glia"]):
                result["glia"] = True

        # remove duplicates
        result["cell_types"] = list(set(result["cell_types"]))
        return result
