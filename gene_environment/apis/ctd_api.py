import gzip
from typing import Dict, List, Set


class CTDAPI:
    GENE_DISEASE_FILE = "CTD_curated_genes_diseases.tsv.gz"
    CHEM_GENE_FILE = "CTD_chem_gene_ixns.tsv.gz"

    @staticmethod
    def _get_header_and_rows(f):
        """
        Legge il file tsv.gz e ritorna header e righe.
        Gestisce righe con colonne mancanti o extra.
        """
        header = None
        rows = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                # Considera la prima riga commentata come header
                header = [h.strip() for h in line.lstrip("#").split("\t")]
                break
        if header is None:
            raise ValueError("Header non trovato")

        # Leggi tutte le altre righe
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row_values = line.split("\t")
            # Se mancano colonne, aggiungi stringhe vuote
            if len(row_values) < len(header):
                row_values += [""] * (len(header) - len(row_values))
            # Se ci sono colonne extra, tronca
            elif len(row_values) > len(header):
                row_values = row_values[:len(header)]
            rows.append(row_values)
        return header, rows

    @staticmethod
    def query_gene(gene_symbol: str) -> Dict[str, List[str]]:
        chemicals: Set[str] = set()
        neuro_diseases: Set[str] = set()

        # Interazioni chemical-gene
        try:
            with gzip.open(CTDAPI.CHEM_GENE_FILE, mode="rt", encoding="utf-8") as f:
                header, rows = CTDAPI._get_header_and_rows(f)
                for r in rows:

                    if len(r) != len(header):
                        continue  # salta righe incomplete
                    row = dict(zip(header, r))
                    gene = row.get("GeneSymbol")
                    if not gene or not isinstance(gene, str) or gene.strip() == "":
                        continue  # salta righe senza GeneSymbol valido
                    if gene.upper() == gene_symbol.upper():
                        chem = row.get("ChemicalName")
                        if chem:
                            chemicals.add(chem.strip())
        except FileNotFoundError:
            print(f"Errore: file {CTDAPI.CHEM_GENE_FILE} non trovato.")
        except Exception as e:
            print(gene_symbol)
            print(f"Errore durante la lettura di {CTDAPI.CHEM_GENE_FILE}: {e}")

        # Malattie neurologiche curate
        try:
            with gzip.open(CTDAPI.GENE_DISEASE_FILE, mode="rt", encoding="utf-8") as f:
                header, rows = CTDAPI._get_header_and_rows(f)
                for r in rows:
                    if len(r) != len(header):
                        print("Riga problematica in GENE_DISEASE_FILE (saltata):", r)
                        print(header)
                        print(r)
                        exit()
                        continue  # salta righe incomplete
                    row = dict(zip(header, r))
                    gene = row.get("GeneSymbol")
                    if not gene or not isinstance(gene, str) or gene.strip() == "":
                        continue  # salta righe senza GeneSymbol valido
                    disease = row.get("DiseaseName")
                    if gene.upper() == gene_symbol.upper() and disease and "neuro" in disease.lower():
                        neuro_diseases.add(disease.strip())
        except FileNotFoundError:
            print(f"Errore: file {CTDAPI.GENE_DISEASE_FILE} non trovato.")
        except Exception as e:
            print(gene_symbol)
            print(f"Errore durante la lettura di {CTDAPI.GENE_DISEASE_FILE} per il gene {gene_symbol}: {e}")

        return {
            "chemicals": list(chemicals),
            "neuro_diseases": list(neuro_diseases)
        }
