"""
Caricamento delle componenti principali (PCA) calcolate dalla pipeline QC
(00_run_plink_qc.sh -> extract_pca_covariates.py), da usare come covariate
di correzione per struttura di popolazione nel modello OLS.

IMPORTANTE: la PCA e' stata calcolata SEPARATAMENTE per generazione (gen1 =
discovery, gen2 = replicazione): non esiste una PCA "congiunta" da usare
qui. Ogni generazione ha il proprio file pca_covariates.csv, prodotto da
extract_pca_covariates.py sulla PCA calcolata SOLO su quella coorte. Per
questo il path del file dipende da cfg.generation (vedi
load_pca_covariates).

NOTA: cfg.generation e' un int (1, 2, 3 -- vedi config.py), non la stringa
"gen1"/"gen2": il prefisso "gen" e' gia' hardcoded nel template di default
(PCA_COVARIATES_PATH_TEMPLATE), non nel valore sostituito.
"""
from __future__ import annotations

import pandas as pd

from gene_environment.logging_utils import get_logger

log = get_logger(__name__)

# Colonna usata come chiave di merge tra il dataframe principale e le PCA.
# Deve corrispondere al nome della colonna sample ID nel dataframe
# principale prodotto da build_dataset.py. Gli ID sono nel formato
# "NOME_NOME" (FamilyID_IndividualID duplicato) sia nel dataframe
# principale sia in pca_covariates.csv (estratto SENZA --strip-doubled-id,
# per scelta esplicita), quindi il merge funziona senza trasformazioni.
PCA_ID_COLUMN = "IID"


def load_pca_covariates(path_template: str, generation: int, n_components: int) -> pd.DataFrame:
    """Carica pca_covariates.csv per la generazione data e restituisce un
    dataframe con colonne [PCA_ID_COLUMN, PC1 .. PC<n_components>].

    path_template puo' contenere il placeholder {generation}, che viene
    sostituito con cfg.generation COSI' COM'E' (un int: 1, 2, 3 -- il
    prefisso "gen" va scritto nel template stesso), es. default:
      "/mnt/cresla_prod/genome_datasets/qc_output_gen{generation}/pca_covariates.csv"
    con cfg.generation=1 diventa ".../qc_output_gen1/pca_covariates.csv".
    """
    path = path_template.format(generation=generation)
    df = pd.read_csv(path)

    if PCA_ID_COLUMN not in df.columns:
        raise ValueError(
            f"Colonna '{PCA_ID_COLUMN}' non trovata in {path}. "
            f"Colonne disponibili: {list(df.columns)}"
        )

    pc_cols = [f"PC{i}" for i in range(1, n_components + 1)]
    missing = [c for c in pc_cols if c not in df.columns]
    if missing:
        available = sorted(
            (c for c in df.columns if c.startswith("PC")),
            key=lambda c: int(c[2:]) if c[2:].isdigit() else 0,
        )
        raise ValueError(
            f"Richieste {n_components} PC ma mancano {missing} in {path}. "
            f"PC disponibili: {available}"
        )

    log.info("PCA: caricate %d PC per %d campioni da %s (generazione=%s)",
              n_components, len(df), path, generation)
    return df[[PCA_ID_COLUMN] + pc_cols]


def merge_pca_covariates(df: pd.DataFrame, pca_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Merge left di df con le PCA sulla colonna PCA_ID_COLUMN.

    Logga quanti campioni restano SENZA PCA dopo il merge (mancata
    corrispondenza di ID): quei campioni verranno comunque scartati piu'
    avanti dal dropna() sulle colonne PC in process_single_variant, ma e'
    importante accorgersene qui e non in silenzio -- puo' segnalare un
    problema di formato ID (es. se pca_covariates.csv fosse stato
    rigenerato per errore con --strip-doubled-id mentre il dataframe
    principale usa ancora gli ID doppi, o viceversa).
    """
    pc_cols = [c for c in pca_df.columns if c != PCA_ID_COLUMN]

    if PCA_ID_COLUMN not in df.columns:
        raise ValueError(
            f"Colonna '{PCA_ID_COLUMN}' non trovata nel dataframe principale: "
            f"impossibile fare il merge con le PCA. Colonne disponibili: {list(df.columns)}"
        )

    merged = df.merge(pca_df, on=PCA_ID_COLUMN, how="left", validate="many_to_one")

    n_missing = int(merged[pc_cols[0]].isna().sum())
    if n_missing > 0:
        pct = 100 * n_missing / len(merged)
        log.warning(
            "PCA: %d/%d campioni (%.1f%%) senza corrispondenza dopo il merge "
            "(ID non trovato nel file PCA). Verranno esclusi dal modello per "
            "ogni variante che li coinvolge (dropna sulle colonne PC in "
            "process_single_variant). Se la percentuale e' alta, controlla "
            "il formato degli ID in entrambi i file.",
            n_missing, len(merged), pct,
        )
    else:
        log.info("PCA: merge completato, tutti i %d campioni hanno le PC.", len(merged))

    return merged, pc_cols
