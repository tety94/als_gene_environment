import requests
from numpy import generic


class GTExAPI:
    GTEX_REST = "https://gtexportal.org/api/v2"

    BRAIN_TISSUES = [
        "Brain_Amygdala",
        "Brain_Anterior_cingulate_cortex_BA24",
        "Brain_Caudate_basal_ganglia",
        "Brain_Cerebellar_Hemisphere",
        "Brain_Cerebellum",
        "Brain_Cortex",
        "Brain_Frontal_Cortex_BA9",
        "Brain_Hippocampus",
        "Brain_Hypothalamus",
        "Brain_Nucleus_accumbens_basal_ganglia",
        "Brain_Putamen_basal_ganglia",
        "Brain_Spinal_cord_cervical_c-1",
        "Brain_Substantia_nigra",
    ]

    # Versioni GENCODE effettive e dataset GTEx compatibile
    GENCODE_TO_DATASET = {
        # GTEx v8 → GENCODE v19 e versioni precedenti usate in v8
        "v8": "gtex_v8",
        "v9": "gtex_v8",
        "v10": "gtex_v8",
        "v11": "gtex_v8",
        "v12": "gtex_v8",
        "v13": "gtex_v8",
        "v14": "gtex_v8",
        "v15": "gtex_v8",
        "v16": "gtex_v8",
        "v17": "gtex_v8",
        "v18": "gtex_v8",
        "v19": "gtex_v8",

        # GTEx v10 → GENCODE v26 e versioni precedenti compatibili
        "v20": "gtex_v10",
        "v21": "gtex_v10",
        "v22": "gtex_v10",
        "v23": "gtex_v10",
        "v24": "gtex_v10",
        "v25": "gtex_v10",
        "v26": "gtex_v10",

        # snRNA-seq pilot → GENCODE v39
        "v39": "gtex_snrnaseq_pilot",
    }

    @staticmethod
    def parse_versioned_ensg(ensg: str):
        if '.' in ensg:
            parts = ensg.split('.')
            return f"v{parts[1]}"
        return None

    def extract_version(gencode_id: str) -> str:
        """
        Prende un GENCODE ID versionato e restituisce solo la versione come stringa.
        Esempio:
            ENSG00000141027.20 -> '20'
        """
        if "." in gencode_id:
            return 'v' + str(gencode_id.split(".")[-1])
        return None  # se non c'è versione


    @staticmethod
    def get_versioned_gencode(gene_symbol: str):
        """
        Restituisce (gencodeId compatibile, versione GENCODE, dataset GTEx)
        """
        url = f"{GTExAPI.GTEX_REST}/reference/gene"
        params = {"geneId": gene_symbol}
        r = requests.get(url, params=params)
        if not r.ok:
            return None, None, None

        data = r.json().get("data", [])
        if not data:
            return None, None, None

        # Ordina dalla versione più alta alla più bassa
        sorted_genes = sorted(
            data,
            key=lambda x: int(x["gencodeId"].split('.')[-1]) if x.get("gencodeId") else 0,
            reverse=True
        )

        # Trova prima versione compatibile con dataset GTEx
        for g in sorted_genes:
            # version = g.get("gencodeVersion")
            genocode_id = g.get("gencodeId")
            version = GTExAPI.parse_versioned_ensg(genocode_id)
            dataset = GTExAPI.GENCODE_TO_DATASET.get(version)
            if dataset:
                return genocode_id, version, dataset

        return None, None, None

    @staticmethod
    def get_brain_expression(gene_symbol: str):
        gencode_id, gencode_version, dataset = GTExAPI.get_versioned_gencode(gene_symbol)
        if not gencode_id or not dataset:
            return {"expressed_brain": False, "tissues": [], "gencode_version": None, "dataset": None}

        url = f"{GTExAPI.GTEX_REST}/expression/geneExpression"
        params = {"gencodeId": gencode_id, "datasetId": dataset}
        for t in GTExAPI.BRAIN_TISSUES:
            params.setdefault("tissueSiteDetailId", []).append(t)

        r = requests.get(url, params=params)

        if not r.ok:
            return {"expressed_brain": False, "tissues": [], "gencode_version": gencode_version, "dataset": dataset}

        data = r.json().get("data", [])
        brain_tissues = [
            d["tissueSiteDetailId"]
            for d in data
            if d.get("data") and d.get("tissueSiteDetailId") in GTExAPI.BRAIN_TISSUES
        ]

        return {
            "expressed_brain": len(brain_tissues) > 0,
            "tissues": brain_tissues,
            "gencode_version": gencode_version,
            "dataset": dataset,
        }
