import requests

# GENE ONTOLOGY API
class GOAPI:
    GO_REST = "https://www.ebi.ac.uk/QuickGO/services"

    @staticmethod
    def get_go_terms(uniprot: str) -> dict:
        """
        Recupera termini GO associati a un geneProductId (UniProt).
        Filtra termini contenenti parole chiave 'neuro' e 'toxic'.
        """

        url = f"{GOAPI.GO_REST}/annotation/search"
        params = {
            "geneProductId": uniprot,  # esempio: "UniProtKB:P04637"
            "limit": 200
        }

        try:
            r = requests.get(url, params=params)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"Errore API: {e}")
            return {"neuro": [], "toxic": []}

        neuro = set()
        toxic = set()

        for res in r.json().get("results", []):
            go_term = res.get("goName") or GOAPI.lookup_go_name(res.get("goId")) or res.get("goId")
            if not go_term:
                continue

            term_lower = go_term.lower()
            if any(x in term_lower for x in ["neuron", "synapse", "axon", "dendrit", "glial"]):
                neuro.add(go_term)
            if any(x in term_lower for x in ["toxic", "oxidative", "stress"]):
                toxic.add(go_term)

        return {
            "neuro": list(neuro),
            "toxic": list(toxic)
        }

    @staticmethod
    def lookup_go_name(go_id: str) -> str:
        """Recupera il nome leggibile di un GO term dato il suo ID."""
        if not go_id:
            return None
        url = f"{GOAPI.GO_REST}/ontology/go/terms/{go_id}"
        try:
            r = requests.get(url)
            r.raise_for_status()
            results = r.json().get("results")
            if results and "name" in results[0]:
                return results[0]["name"]
        except requests.RequestException:
            return None
        return None
