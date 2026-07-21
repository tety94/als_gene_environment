"""
Costruzione del dataset "pronto per vQTL": dosaggio genotipico + fenotipo +
covariate + PCA, in un unico DataFrame.

PRIMA (vqtl_pipeline originale): Step 0 (bcftools norm + manifest VCF per
cromosoma) + meta' dello Step 1 (join fenotipo<->campioni genotipati,
lettura VCF per-cromosoma con cyvcf2) + Step 2 (merge con le PCA "fatte in
casa"). Ogni step successivo (5, 6, 7, 8) doveva poi RILEGGERE i VCF
originali con cyvcf2 per estrarre il dosaggio delle sole SNP candidate,
duplicando la stessa logica di parsing quattro volte (vedi
`extract_snp_dosage` in step5/6/7/8 del progetto originale).

ORA: gene_environment ha gia' prodotto (comandi `filter-vcf` + `build-matrix`
della sua CLI) un'unica matrice genotipica genoma-wide gia' filtrata
MAF/LD-pruned (RAW_FILE, parquet) e sa gia' come unirla al file ambientale
per id-campione, ripulire gli id, e selezionare la generazione/coorte
corretta (`gene_environment.vcf_pipeline.build_dataset.load_and_prepare_data`).
Qui ci limitiamo a:
  1. riusare quella funzione cosi' com'e' (nessuna riscrittura della logica
     di join/id-cleaning/filtro-generazione);
  2. aggiungere le colonne di esposizione extra richieste da vqtl (nel
     progetto originale erano testate piu' esposizioni separatamente, mentre
     gene_environment ne gestisce una sola per run -- vedi VqtlConfig.exposures);
  3. fare il merge con le PCA reali calcolate dalla pipeline QC
     (quality_control/00_run_plink_qc.sh -> extract_pca_covariates.py),
     al posto della PCA "fatta in casa" (thinning random + sklearn PCA su un
     sottoinsieme minuscolo di SNP) che il progetto vqtl calcolava da solo
     nel suo Step 1 -- QC e PCA vere sono gia' un output della pipeline QC
     ufficiale, non ha senso ricalcolarne una versione piu' debole qui.

Il dosaggio genotipico risultante NON contiene piu' -1/NaN "grezzi" da VCF:
e' gia' passato per la stessa strategia di gestione dei missing usata da
tutta la pipeline gene_environment (MISSING_GENOTYPE_STRATEGY, vedi
vcf_pipeline/vcf_to_parquet.py); qui ci si limita a un controllo difensivo
(dosaggio NaN o fuori range trattato come mancante) nel caso in cui questo
comportamento cambi in futuro.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from gene_environment.analysis.pca_utils import load_pca_covariates
from gene_environment.config import Config
from gene_environment.logging_utils import get_logger
from gene_environment.utils.id_utils import clean_sample_id, parse_variant_label
from gene_environment.vcf_pipeline.build_dataset import load_and_prepare_data

from vqtl.config import VqtlConfig

log = get_logger(__name__)


@dataclass
class VqtlDataset:
    df: pd.DataFrame
    # colonne "safe" (variant_0, variant_1, ...) dei genotipi, valori
    # dosaggio 0/1/2 (eventuale mancante -> NaN, gia' gestito a monte)
    variant_cols: list[str]
    # variant_safe -> label reale "CHROM_POS_REF_ALT"
    mapping: dict[str, str]
    # esposizione richiesta (raw) -> colonna standardizzata usata nei modelli
    exposure_std_cols: dict[str, str]
    # covariate di correzione: cfg.covariates (dummy-encoded se categoriali) + PC1..PCn
    covariate_cols: list[str]
    # covariate RICHIESTE (vcfg.covariates, prima del dummy-encoding): usate
    # solo per capire se la cache va invalidata quando VQTL_COVARIATES cambia
    # tra un run e l'altro (vedi get_or_build_dataset)
    requested_covariates: list[str] = field(default_factory=list)


def _generation_config(ge_cfg: Config, generation: int | None) -> Config:
    """Ritorna una copia di ge_cfg con .generation sovrascritto, se richiesto
    (es. da `--generation` in cli.py), senza mutare il singleton globale."""
    if generation is None or generation == ge_cfg.generation:
        return ge_cfg
    return replace(ge_cfg, generation=generation)


def _dummy_encode(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Dummy-encoding (drop_first) delle covariate non numeriche (es. 'sex',
    'onset_site' se salvate come stringhe/categorie nel file ambientale).
    Le covariate gia' numeriche non vengono toccate. Necessario perche' i
    modelli statsmodels usati da vqtl (residualizzazione, interazione, rGE,
    permutazioni) costruiscono la design matrix direttamente da array numpy
    (niente parsing a formula/patsy come in gene_environment.analysis.modeling),
    quindi le covariate devono arrivare gia' numeriche."""
    out_cols = []
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            log.warning("Covariata '%s' non trovata nel dataset, ignorata.", c)
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out_cols.append(c)
            continue
        dummies = pd.get_dummies(df[c], prefix=c, drop_first=True, dtype=float)
        df = pd.concat([df, dummies], axis=1)
        out_cols.extend(dummies.columns.tolist())
        log.info("Covariata '%s' non numerica: dummy-encoded in %s", c, dummies.columns.tolist())
    return df, out_cols


def load_vqtl_dataset(
    ge_cfg: Config, vcfg: VqtlConfig, generation: int | None = None
) -> VqtlDataset:
    ge_cfg = _generation_config(ge_cfg, generation)

    log.info(
        "Carico dataset vQTL (generazione=%s) da RAW_FILE=%s + ENV_FILE=%s",
        ge_cfg.generation, ge_cfg.raw_file, ge_cfg.env_file,
    )
    df, variant_cols, mapping, _ecols_default, _variant_cols_real = load_and_prepare_data(ge_cfg)
    log.info("Dataset caricato: %d campioni, %d varianti", len(df), len(variant_cols))

    # ---- Esposizioni: gene_environment standardizza solo cfg.exposure di
    # default; qui aggiungiamo (se non gia' presente) una colonna
    # standardizzata per OGNI esposizione richiesta da vqtl. ----
    exposure_std_cols: dict[str, str] = {}
    for exp in vcfg.exposures:
        if exp not in df.columns:
            raise ValueError(
                f"Esposizione '{exp}' (VQTL_EXPOSURES) non trovata nel file ambientale "
                f"{ge_cfg.env_file}. Colonne disponibili: {list(df.columns)}"
            )
        df[exp] = pd.to_numeric(df[exp], errors="coerce")
        std_col = f"{exp}_std" if ge_cfg.standardize else exp
        if std_col not in df.columns:
            if ge_cfg.standardize:
                df[std_col] = StandardScaler().fit_transform(df[[exp]])
            else:
                std_col = exp
        exposure_std_cols[exp] = std_col
    log.info("Esposizioni vqtl: %s", exposure_std_cols)

    # ---- Covariate base (sex, onset_site, diagnostic_delay, ...) ----
    # NOTA: qui si usa vcfg.covariates (override specifico di vqtl, default
    # = ge_cfg.covariates se VQTL_COVARIATES non e' impostato -- vedi
    # vqtl/config.py), non ge_cfg.covariates direttamente: permette di
    # escludere una covariata solo da vqtl senza toccare COVARIATES nel .env
    # condiviso con il resto della pipeline.
    df, covariate_cols = _dummy_encode(df, list(vcfg.covariates))

    # ---- PCA reali (quality_control), non quelle "fatte in casa" ----
    if ge_cfg.use_pca_covariates:
        pca_df = load_pca_covariates(
            ge_cfg.pca_covariates_path_template, ge_cfg.generation, ge_cfg.pca_n_components,
        )
        # NOTA IMPORTANTE su formato ID: pca_covariates.csv contiene la
        # colonna "IID" nel formato "grezzo" prodotto da plink (osservato
        # nella forma doppia "NOME_NOME", vedi il commento su PCA_ID_COLUMN
        # in pca_utils.py), mentre il dataframe principale qui ha gia' la
        # colonna "id" ripulita/deduplicata da `clean_sample_id`
        # (load_and_prepare_data la applica all'indice del file genetico).
        # Un merge diretto "IID"->"id" senza normalizzare non farebbe
        # match quasi nessun campione ("NOME_NOME" != "NOME"). Percio' qui
        # applichiamo la STESSA `clean_sample_id` gia' usata dal resto della
        # pipeline anche alla colonna IID delle PCA, invece di introdurre
        # una logica di normalizzazione diversa/nuova.
        pca_df = pca_df.rename(columns={"IID": "id"})
        pca_df["id"] = pca_df["id"].astype(str).map(clean_sample_id)
        pc_cols = [c for c in pca_df.columns if c != "id"]
        n_before = len(df)
        df = df.merge(pca_df, on="id", how="left", validate="many_to_one")
        n_missing_pca = int(df[pc_cols[0]].isna().sum()) if pc_cols else 0
        if n_missing_pca:
            log.warning(
                "%d/%d campioni senza PCA dopo il merge (esclusi a valle dai dropna per-modello).",
                n_missing_pca, n_before,
            )
        covariate_cols = covariate_cols + pc_cols
        log.info("PCA attive come covariate di correzione: %s", pc_cols)
    else:
        log.info("PCA disattivate (USE_PCA_COVARIATES=false).")

    return VqtlDataset(
        df=df,
        variant_cols=variant_cols,
        mapping=mapping,
        exposure_std_cols=exposure_std_cols,
        covariate_cols=covariate_cols,
        requested_covariates=list(vcfg.covariates),
    )


def variant_chrom_pos(variant_label: str) -> tuple[str | None, int | None]:
    """CHROM/POS a partire dal label reale ('CHROM_POS_REF_ALT'), riusando
    il parser condiviso di gene_environment invece di un regex locale."""
    chrom, pos, _mutation = parse_variant_label(variant_label)
    return chrom, pos


def get_or_build_dataset(
    ge_cfg: Config, vcfg: VqtlConfig, generation: int | None = None, force: bool = False,
) -> VqtlDataset:
    """Come `load_vqtl_dataset`, ma con una cache su disco (pickle) dentro
    la cartella della coorte -- stesso principio del TEMP_DF_PATH usato da
    gene_environment.analysis.orchestrator, per non rifare il join
    genetica+ambiente+PCA (costoso, legge l'intero RAW_FILE parquet) ad ogni
    singolo sotto-comando della CLI vqtl (scan/filter/interaction/...).
    Passa force=True per ricostruire da zero (es. dopo aver rilanciato
    filter-vcf/build-matrix a monte)."""
    ge_cfg_eff = _generation_config(ge_cfg, generation)
    cache_path = os.path.join(vcfg.cohort_dir(ge_cfg_eff.generation), "vqtl_dataset.pkl")

    if not force and os.path.exists(cache_path):
        log.info("Carico dataset vQTL dalla cache: %s", cache_path)
        with open(cache_path, "rb") as f:
            dataset = pickle.load(f)
        # getattr per compatibilita' con cache scritte da una versione
        # precedente di questo modulo, prima dell'aggiunta di requested_covariates
        cached_covariates = getattr(dataset, "requested_covariates", None)
        exposures_changed = set(dataset.exposure_std_cols.keys()) != set(vcfg.exposures)
        covariates_changed = cached_covariates is not None and set(cached_covariates) != set(vcfg.covariates)
        if exposures_changed or covariates_changed:
            log.warning(
                "Config vqtl cambiata rispetto alla cache (esposizioni: cache=%s richieste=%s; "
                "covariate: cache=%s richieste=%s): ricostruisco.",
                list(dataset.exposure_std_cols.keys()), vcfg.exposures, cached_covariates, vcfg.covariates,
            )
        else:
            return dataset

    dataset = load_vqtl_dataset(ge_cfg, vcfg, generation=generation)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(dataset, f)
    log.info("Dataset vQTL salvato in cache: %s", cache_path)
    return dataset


def dosage_matrix(dataset: VqtlDataset, safe_cols: list[str]) -> np.ndarray:
    """Estrae il sotto-blocco di dosaggio (n_samples x len(safe_cols)) come
    array numpy, con eventuale valore mancante/non valido -> NaN (rete di
    sicurezza; il dosaggio prodotto da gene_environment e' gia' 0/1/2)."""
    sub = dataset.df[safe_cols].apply(pd.to_numeric, errors="coerce")
    arr = sub.to_numpy(dtype=float, copy=True)
    arr[(arr < 0) | (arr > 2)] = np.nan
    return arr

def select_variants_from_significant_results(
    dataset: VqtlDataset, exposure: str | None = None,
) -> list[str]:
    """Restringe variant_cols alle sole varianti gia' note come significative
    (gene_environment.db.repository.get_significant_results), invece che a
    un range genomico. Nessuna modifica al dataset in memoria/cache."""
    from gene_environment.db.repository import get_significant_results

    sig_df = get_significant_results(exposure=exposure)
    if sig_df.empty:
        log.warning("get_significant_results(exposure=%s): nessuna riga trovata, subset vuoto.", exposure)
        return []

    log.debug("Esempio label da get_significant_results: %s", sig_df["variant"].head(5).tolist())
    log.debug("Esempio label da dataset.mapping: %s", list(dataset.mapping.values())[:5])

    inv_mapping = {v: k for k, v in dataset.mapping.items()}
    wanted = sig_df["variant"].unique().tolist()
    safe_cols = [inv_mapping[v] for v in wanted if v in inv_mapping]

    missing = [v for v in wanted if v not in inv_mapping]
    if missing:
        log.warning(
            "%d/%d varianti significative non trovate nel dataset corrente (label non matchata): %s%s",
            len(missing), len(wanted), missing[:5], "..." if len(missing) > 5 else "",
        )
    log.info("Subset da get_significant_results: %d/%d varianti mappate su dataset.variant_cols.",
              len(safe_cols), len(wanted))
    return safe_cols