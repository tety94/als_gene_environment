"""
Configurazione della pipeline vQTL.

PRIMA: `vqtl_pipeline` era un progetto completamente slegato da
`gene_environment_v2`, con un proprio `config/config.yaml` (path VCF, path
ENV_FILE, soglie MAF/LD, credenziali-equivalenti come le directory dei dati)
duplicati a mano rispetto al `.env` di gene_environment -- due fonti di
verità per le stesse informazioni (es. `env_file` in config.yaml conteneva
letteralmente lo stesso path di ENV_FILE nel .env), col rischio concreto che
vengano aggiornate una sì e una no.

ORA: vqtl non ha un proprio file di configurazione. Usa la STESSA istanza di
`gene_environment.config.Config` (stesso `.env`, stesso singleton
`get_config()`) per tutto ciò che già esiste lì (path dei file, soglie MAF/LD
già applicate a monte dal filtraggio VCF, colonne target/covariate/esposizione,
generazione/coorte, PCA, permutazioni, parallelismo, cartelle di log/output).
Questo modulo aggiunge SOLO i parametri che sono realmente specifici del
metodo vQTL (QUAIL-style scan, filtro genomic-control, permutazioni
Freedman-Lane, ecc.) e che non hanno un equivalente in gene_environment,
letti dalle stesse variabili d'ambiente (nuove chiavi, prefisso VQTL_, da
aggiungere al `.env` esistente -- vedi `.env.vqtl.example` in questa cartella).

Cosa NON serve più duplicare qui (era in config.yaml originale) perché già
in gene_environment.config.Config:
  - cohorts.gen1/gen2.vcf_dir           -> non serve: vqtl legge il dosage
                                            genotipico da cfg.raw_file (la
                                            matrice genoma-wide già filtrata
                                            MAF/LD e prodotta da
                                            `gene_environment.cli filter-vcf`
                                            + `build-matrix`), non più dai
                                            VCF grezzi per cromosoma.
  - env_file, id_col, phenotype_col      -> cfg.env_file, "id", cfg.target_col
  - covariates                           -> cfg.covariates
  - sample_qc (missingness/relatedness/PCA) -> QC e PCA già fatte a monte da
                                            quality_control/00_run_plink_qc.sh
                                            + extract_pca_covariates.py (vedi
                                            core/data.py); lo step Python
                                            "fatto in casa" del progetto
                                            originale (proxy IBS, PCA su
                                            thinning casuale) viene eliminato.
  - vqtl.min_maf / min_call_rate          -> il filtro MAF/LD è già applicato
                                            dal filtraggio VCF a monte
                                            (MAF_THRESHOLD/LD_* in gene_environment);
                                            qui restano solo come RETE DI
                                            SICUREZZA opzionale (default 0,
                                            cioè disattivi) nel caso in cui
                                            RAW_FILE non sia già stato filtrato.
  - results_dir per cohort/gen1/gen2      -> cfg.generation (1/2/3, stesso
                                            meccanismo con cui gene_environment
                                            distingue le coorti) seleziona la
                                            coorte; l'output finisce sotto
                                            VQTL_RESULTS_DIR/gen<N>/.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from gene_environment.config import (
    Config,
    _env,
    _env_bool,
    _env_float,
    _env_int,
    _env_list,
    get_config,
)


def _env_float_list(name: str, default: str) -> list[float]:
    return [float(v) for v in _env_list(name, default)]


def _env_optional_int(name: str) -> int | None:
    val = _env(name, "")
    return int(val) if val not in (None, "") else None


@dataclass(frozen=True)
class VqtlConfig:
    # Config di gene_environment, riusata cosi' com'e' (paths, target/covariate/
    # esposizione, generazione/coorte, PCA, parallelismo, permutazioni, log).
    ge: Config = field(default_factory=get_config)

    # ---- Covariate di correzione ----
    # Se VQTL_COVARIATES non e' impostato, il default e' ge.covariates (le
    # stesse usate dal resto della pipeline, da COVARIATES nel .env). Imposta
    # VQTL_COVARIATES esplicitamente per usare un elenco diverso SOLO per
    # vqtl, senza toccare COVARIATES (che resta condivisa con gene_environment
    # e con il suo test di interazione via matching). Esempio: per escludere
    # onset_site solo da vqtl mantenendola nel resto della pipeline, imposta
    # VQTL_COVARIATES=sex,diagnostic_delay nel .env.
    covariates: list[str] = field(default_factory=lambda: _env_list("VQTL_COVARIATES", ""))

    # ---- Esposizioni testate (Step 5/6/7) ----
    # Nel progetto vqtl originale erano testate 3 esposizioni separatamente
    # (seminativi_1500, risaie_1500, vigneti_1500), mentre gene_environment
    # ne testa una sola per run (EXPOSURE). Qui, se VQTL_EXPOSURES non e'
    # impostato, il default e' [ge.exposure] (comportamento minimo,
    # coerente col resto della pipeline); imposta VQTL_EXPOSURES=exp1,exp2,exp3
    # nel .env per testarne piu' di una in un solo run vqtl, come nella
    # pipeline originale.
    exposures: list[str] = field(default_factory=lambda: _env_list("VQTL_EXPOSURES", ""))

    # ---- Step 3: QUAIL-style vQTL scan ----
    taus: list[float] = field(
        default_factory=lambda: _env_float_list(
            "VQTL_TAUS", "0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45"
        )
    )
    se_method: str = field(default_factory=lambda: _env("VQTL_SE_METHOD", "asymptotic"))
    bootstrap_k: int = field(default_factory=lambda: _env_int("VQTL_BOOTSTRAP_K", 200))
    chunk_size: int = field(default_factory=lambda: _env_int("VQTL_CHUNK_SIZE", 2000))
    # Default: cfg.max_workers (stesso parallelismo del resto della pipeline,
    # niente terzo parametro da tenere sincronizzato a mano). -1 = tutti i core.
    n_jobs: int = field(default_factory=lambda: _env_int("VQTL_N_JOBS", 0))
    # Rete di sicurezza opzionale: MAF/call-rate sono gia' filtrati a monte
    # (vedi docstring del modulo), di default qui NON si rifiltra nulla.
    min_maf: float = field(default_factory=lambda: _env_float("VQTL_MIN_MAF", 0.0))
    min_call_rate: float = field(default_factory=lambda: _env_float("VQTL_MIN_CALL_RATE", 0.0))

    # ---- Step 4: filtro candidati genomic-control-corrected ----
    # Di default filtra su P_gc (corretta per inflazione genomica), non sulla
    # P asintotica grezza -- vedi audit descritto nel README originale: sui
    # dati simulati la P grezza era marcatamente anti-conservativa per un
    # predittore di dosaggio discreto (0/1/2). Imposta VQTL_FILTER_P_COLUMN=P
    # per tornare al comportamento vecchio (sconsigliato).
    filter_p_column: str = field(default_factory=lambda: _env("VQTL_FILTER_P_COLUMN", "P_gc"))
    filter_p_threshold: float = field(default_factory=lambda: _env_float("VQTL_FILTER_P_THRESHOLD", 1e-5))
    filter_top_n: int | None = field(default_factory=lambda: _env_optional_int("VQTL_FILTER_TOP_N"))

    # ---- Step 5/6: test di interazione SNP x esposizione, rGE, eteroschedasticita' ----
    # HC3 (OLS) / HC1 (logit): standard error robuste all'eteroschedasticita',
    # attive di default (vedi audit nel README originale, punto 2).
    robust_se: bool = field(default_factory=lambda: _env_bool("VQTL_ROBUST_SE", True))
    rge_het_alpha: float = field(default_factory=lambda: _env_float("VQTL_RGE_HET_ALPHA", 0.05))
    # Soglia nominale (non corretta per test multipli) usata per popolare
    # vqtl_interaction_results_significant -- vedi db/schema.sql.
    interaction_sig_threshold: float = field(default_factory=lambda: _env_float("VQTL_INTERACTION_SIG_THRESHOLD", 0.05))

    # ---- Step 7: permutazioni Freedman-Lane sui top loci ----
    # Default: riusa N_PERM_HIGH di gene_environment se VQTL_N_PERM non e'
    # impostato, cosi' non serve un terzo numero di permutazioni da scegliere.
    n_perm: int = field(default_factory=lambda: _env_int("VQTL_N_PERM", 0))
    perm_top_n_loci: int = field(default_factory=lambda: _env_int("VQTL_PERM_TOP_N_LOCI", 10))

    # ---- Output ----
    results_dir: str = field(default_factory=lambda: _env("VQTL_RESULTS_DIR", "./vqtl_results"))

    # ---- Step 9: export .docx (Results + Supplementary Material per il paper) ----
    # Quante righe dello scan genoma-wide mostrare nella Tabella 1 (Results, corpo
    # del paper) vs. nella Supplementary Table S1 (elenco esteso).
    docx_top_n_scan: int = field(default_factory=lambda: _env_int("VQTL_DOCX_TOP_N_SCAN", 20))
    # Cap sulle righe della Supplementary Table S1 (scan completo): una tabella Word
    # con decine/centinaia di migliaia di righe e' inutilizzabile in un manoscritto;
    # oltre questo numero si tronca con una nota che rimanda al file .tsv completo.
    docx_supp_max_rows: int = field(default_factory=lambda: _env_int("VQTL_DOCX_SUPP_MAX_ROWS", 200))

    def __post_init__(self):
        if not self.covariates:
            object.__setattr__(self, "covariates", list(self.ge.covariates))
        if not self.exposures:
            object.__setattr__(self, "exposures", [self.ge.exposure])
        if self.n_jobs == 0:
            object.__setattr__(self, "n_jobs", self.ge.max_workers)
        if self.n_perm == 0:
            object.__setattr__(self, "n_perm", self.ge.n_perm_high)

    def cohort_dir(self, generation: int | None = None) -> str:
        """Cartella di output per la generazione/coorte corrente (o quella
        passata esplicitamente, per sovrascrivere cfg.ge.generation senza
        mutare la config globale -- vedi cli.py --generation)."""
        gen = generation if generation is not None else self.ge.generation
        return f"{self.results_dir}/gen{gen}"


_vqtl_config_instance: VqtlConfig | None = None


def get_vqtl_config() -> VqtlConfig:
    global _vqtl_config_instance
    if _vqtl_config_instance is None:
        _vqtl_config_instance = VqtlConfig()
    return _vqtl_config_instance
