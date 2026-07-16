from apis.ensembl_api import EnsemblAPI
from apis.gtex_api import GTExAPI
from apis.hpa_api import HPAAPI
from apis.ctd_api import CTDAPI
from scoring.neuro_score import NeuroScore
from db import upsert_gene_neuro_annotation
from apis.go_api import GOAPI

class GeneAnnotator:


    @staticmethod
    def annotate(ensg: str):
        info = EnsemblAPI.get_gene_info(ensg)

        gtex = GTExAPI.get_brain_expression(ensg)
        uniprots = EnsemblAPI.ensg_to_uniprot(ensg)

        hpa = HPAAPI.get_single_cell_info(ensg)

        go_terms = {"neuro": [], "toxic": []}

        for up in uniprots:
            uniprot_full = f"UniProtKB:{up}"
            data = GOAPI.get_go_terms(uniprot_full)
            go_terms["neuro"].extend(data.get("neuro", []))
            go_terms["toxic"].extend(data.get("toxic", []))

        # rimuovi duplicati
        go_terms["neuro"] = list(set(go_terms["neuro"]))
        go_terms["toxic"] = list(set(go_terms["toxic"]))

        ctd = CTDAPI.query_gene(info["gene_symbol"])

        data = {
            "gene_id": ensg,
            "gene_symbol": info["gene_symbol"],
            "gene_type": info["gene_type"],
            "expressed_brain": gtex["expressed_brain"],
            "brain_tissues": ",".join(gtex["tissues"]),
            "expressed_neurons": hpa["neurons"],
            "expressed_glia": hpa["glia"],
            "cell_types": ",".join(hpa["cell_types"]),
            "go_neuro_processes": ",".join(go_terms["neuro"]),
            "go_toxic_response": ",".join(go_terms["toxic"]),
            "ctd_chemicals": ",".join(ctd["chemicals"]),
            "ctd_neuro_diseases": ",".join(ctd["neuro_diseases"])
        }

        data["neuro_plausibility_score"] = NeuroScore.compute(data)
        upsert_gene_neuro_annotation(data)
