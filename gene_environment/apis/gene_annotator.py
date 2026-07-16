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

        gtex = GTExAPI.get_brain_expression(ensg)
        hpa = HPAAPI.get_single_cell_info(ensg)

        # GO e CTD disattivati: niente chiamate esterne, campi a NULL per velocizzare
        go_neuro_processes = None
        go_toxic_response = None
        ctd_chemicals = None
        ctd_neuro_diseases = None

        panelapp = PanelAppAPI.get_als_status(info["gene_symbol"])
        opentargets = OpenTargetsAPI.get_als_association(ensg)

        data = {
            "gene_id": ensg,
            "gene_symbol": info["gene_symbol"],
            "gene_type": info["gene_type"],
            "expressed_brain": gtex["expressed_brain"],
            "brain_tissues": ",".join(gtex["tissues"]),
            "expressed_neurons": hpa["neurons"],
            "expressed_glia": hpa["glia"],
            "cell_types": ",".join(hpa["cell_types"]),
            "go_neuro_processes": go_neuro_processes,
            "go_toxic_response": go_toxic_response,
            "ctd_chemicals": ctd_chemicals,
            "ctd_neuro_diseases": ctd_neuro_diseases,
            "als_panelapp_confidence" : panelapp["confidence_level"],
            "als_opentargets_score": opentargets["score"],
        }

        data["neuro_plausibility_score"] = NeuroScore.compute(data)
        upsert_gene_neuro_annotation(data)