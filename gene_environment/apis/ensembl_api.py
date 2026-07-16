import requests
import requests

class EnsemblAPI:

    HEADERS = {"Content-Type": "application/json"}
    ENSEMBL_REST = "https://rest.ensembl.org"

    @staticmethod
    def get_gene_info(ensg_id: str) -> dict:
        url = f"{EnsemblAPI.ENSEMBL_REST}/lookup/id/{ensg_id}"
        r = requests.get(url, headers=EnsemblAPI.HEADERS)
        if not r.ok:
            return {}

        j = r.json()
        return {
            "gene_symbol": j.get("display_name"),
            "gene_type": j.get("biotype"),
            "chromosome": j.get("seq_region_name")
        }


    @staticmethod
    def fetch_gene(chrom, pos):
        """
        Fetches gene information based on chromosome and position.
        """
        url = f"{EnsemblAPI.ENSEMBL_REST}/overlap/region/human/{chrom}:{pos}-{pos}?feature=gene"
        try:
            r = requests.get(url, headers=EnsemblAPI.HEADERS)
            r.raise_for_status()
            data = r.json()
            if data:
                gene = data[0]
                gene_id = gene['id']
                gene_name = gene.get('external_name', gene_id)
                return gene_id, gene_name
        except requests.exceptions.RequestException as e:
            print(f"Error fetching gene at {chrom}:{pos} -> {str(e)}")
        return None, None

    @staticmethod
    def get_variant_coordinates(variant):
        """
        Get coordinates (chrom, start, end) for a given variant.
        """
        url = f"{EnsemblAPI.ENSEMBL_REST}/lookup/symbol/homo_sapiens/{variant}?content-type=application/json"
        try:
            r = requests.get(url, headers=EnsemblAPI.HEADERS)
            r.raise_for_status()
            d = r.json()
            return d["seq_region_name"], d["start"], d["end"]
        except requests.exceptions.RequestException as e:
            print(f"Error fetching variant coordinates for {variant} -> {str(e)}")
        return None, None, None

    @staticmethod
    def ensg_to_uniprot(ensg):
        url = f"https://rest.ensembl.org/xrefs/id/{ensg}?content-type=application/json"
        r = requests.get(url, headers={"Content-Type": "application/json"})
        r.raise_for_status()
        xrefs = r.json()

        #  può averne più di uno perchè può codificare più proteine
        uniprots = [x["primary_id"] for x in xrefs if x["dbname"].lower().startswith("uniprot")]

        return uniprots