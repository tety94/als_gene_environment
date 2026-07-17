from gene_environment.apis.ctd_api import CTDAPI
from gene_environment.apis.ensembl_api import EnsemblAPI
from gene_environment.apis.gtex_api import GTExAPI
from gene_environment.apis.hpa_api import HPAAPI
from gene_environment.apis.neuro_score import NeuroScore
from gene_environment.apis.open_targets_api import OpenTargetsAPI
from gene_environment.apis.panelapp_api import PanelAppAPI
from gene_environment.db.repository import upsert_gene_neuro_annotation


class GeneAnnotator:

    @staticmethod
    def annotate(ensg: str):
        info = EnsemblAPI.get_gene_info(ensg)
        gene_symbol = info.get("gene_symbol")  # puo' essere None: gestito a valle da PanelApp/CTD
        gene_type = info.get("gene_type")

        gtex = GTExAPI.get_brain_expression(ensg)
        hpa = HPAAPI.get_single_cell_info(ensg)

        # GO disattivato: niente chiamata esterna, campo a NULL per velocizzare
        go_neuro_processes = None
        go_toxic_response = None

        panelapp = PanelAppAPI.get_als_status(gene_symbol)
        opentargets = OpenTargetsAPI.get_als_association(ensg)

        # CTD: indici caricati una volta per processo worker (lazy cache),
        # query per simbolo (non ENSG, l'indice CTD e' keyed su GeneSymbol)
        ctd_data = CTDAPI.query_gene(
            gene_symbol,
            CTDAPI.get_chem_index(),
            CTDAPI.get_disease_index(),
        )

        pesticide_names = [
            ci.chemical_name for ci in ctd_data["chemical_interactions"]
            if ci.is_pesticide_by_keyword
        ]

        neuro_diseases_all = [
            d for d in ctd_data["diseases"]
            if "amyotrophic" in d.disease_name.lower() or "motor neuron" in d.disease_name.lower()
        ]
        neuro_diseases_direct = [d for d in neuro_diseases_all if d.is_direct]
        neuro_diseases_pesticide_mediated = [d for d in neuro_diseases_all if d.is_pesticide_mediated]

        neuro_disease_names = [d.disease_name for d in neuro_diseases_all]

        data = {
            "gene_id": ensg,
            "gene_symbol": gene_symbol,
            "gene_type": gene_type,
            "expressed_brain": gtex["expressed_brain"],
            "brain_tissues": ",".join(gtex["tissues"]),
            "expressed_neurons": hpa["neurons"],
            "expressed_glia": hpa["glia"],
            "cell_types": ",".join(hpa["cell_types"]),
            "go_neuro_processes": go_neuro_processes,
            "go_toxic_response": go_toxic_response,
            "ctd_chemicals": ",".join(pesticide_names) if pesticide_names else None,
            "ctd_neuro_diseases": ",".join(neuro_disease_names) if neuro_disease_names else None,
            "ctd_neuro_disease_direct": bool(neuro_diseases_direct),
            "ctd_neuro_disease_pesticide_mediated": bool(neuro_diseases_pesticide_mediated),
            "als_panelapp_confidence": panelapp["confidence_level"],
            "als_opentargets_score": opentargets["score"],
        }

        data["neuro_plausibility_score"] = NeuroScore.compute(data)
        upsert_gene_neuro_annotation(data)