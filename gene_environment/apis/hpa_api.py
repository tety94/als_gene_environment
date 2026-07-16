import requests

class HPAAPI:
    """
    Interroga il JSON completo di HPA per un gene (non un endpoint single_cell dedicato).
    Usa l'URL:
        https://www.proteinatlas.org/<ENSG>.json
    """

    BASE_URL = "https://www.proteinatlas.org"

    @staticmethod
    def fetch_hpa_json(ensg: str) -> dict:
        """
        Scarica il JSON completo per il gene.
        """
        url = f"{HPAAPI.BASE_URL}/{ensg}.json"
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            return {}

    @staticmethod
    def get_single_cell_info(ensg: str) -> dict:
        """
        Estrae i dati single-cell (se presenti) dal JSON HPA.
        HPA non ha un endpoint REST single_cell dedicato ma il JSON
        completo può contenere sezioni come 'rna_single_cell_type' o simili.
        """
        j = HPAAPI.fetch_hpa_json(ensg)
        result = {
            "neurons": False,
            "glia": False,
            "cell_types": []
        }

        # Il JSON di HPA può avere varie sezioni
        # controlla campi legati a "rna_single_cell"
        sc_nCPM = j.get("RNA single cell type specific nCPM", {}) or {}

        for cell_type in sc_nCPM.keys():
            ct_lower = cell_type.lower()
            result["cell_types"].append(cell_type)
            if "neuron" in ct_lower:
                result["neurons"] = True
            if any(x in ct_lower for x in ["astro", "oligo", "microglia", "glia"]):
                result["glia"] = True

        # rimuovi duplicati
        result["cell_types"] = list(set(result["cell_types"]))
        return result