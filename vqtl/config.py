"""
Configuration for the vQTL pipeline.

vQTL has no configuration file of its own. It reuses the same instance of
`gene_environment.config.Config` (same `.env`, same `get_config()`
singleton) for everything that already lives there: file paths, MAF/LD
thresholds already applied upstream by VCF filtering, target/covariate/
exposure columns, generation/cohort, PCA, permutations, parallelism, and
log/output directories.

This module only adds the parameters that are genuinely specific to the
vQTL method (QUAIL-style scan, genomic-control filtering, Freedman-Lane
permutations, etc.) and that have no equivalent in gene_environment. They
are read from dedicated environment variables (VQTL_-prefixed keys, to be
added to the existing `.env` -- see `.env.vqtl.example` in this folder).

What is already covered by `gene_environment.config.Config` and therefore
does NOT need its own entry here:
  - VCF/cohort directories        -> not needed: vqtl reads genotype dosage
                                      from cfg.raw_file (the genome-wide
                                      matrix, already MAF/LD-filtered,
                                      produced by `gene_environment.cli
                                      filter-vcf` + `build-matrix`), not
                                      from raw per-chromosome VCFs.
  - env_file, id_col, phenotype_col  -> cfg.env_file, "id", cfg.target_col
  - covariates                       -> cfg.covariates
  - sample QC (missingness/relatedness/PCA) -> QC and PCA are computed
                                      upstream by
                                      quality_control/00_run_plink_qc.sh +
                                      extract_pca_covariates.py (see
                                      core/data.py).
  - vqtl.min_maf / min_call_rate     -> the MAF/LD filter is already applied
                                      by the upstream VCF filtering step
                                      (MAF_THRESHOLD/LD_* in
                                      gene_environment); here they remain
                                      only as an optional SAFETY NET
                                      (default 0, i.e. disabled) in case
                                      RAW_FILE has not already been
                                      filtered.
  - results_dir per cohort/gen1/gen2 -> cfg.generation (1/2/3, the same
                                      mechanism gene_environment uses to
                                      distinguish cohorts) selects the
                                      cohort; output is written under
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
    # gene_environment config, reused as-is (paths, target/covariate/
    # exposure, generation/cohort, PCA, parallelism, permutations, log).
    ge: Config = field(default_factory=get_config)

    # ---- Adjustment covariates ----
    # If VQTL_COVARIATES is not set, the default is ge.covariates (the same
    # ones used by the rest of the pipeline, from COVARIATES in .env). Set
    # VQTL_COVARIATES explicitly to use a different list ONLY for vqtl,
    # without touching COVARIATES (which stays shared with gene_environment
    # and its matching-based interaction test). Example: to exclude
    # onset_site from vqtl only, while keeping it in the rest of the
    # pipeline, set VQTL_COVARIATES=sex,diagnostic_delay in .env.
    covariates: list[str] = field(default_factory=lambda: _env_list("VQTL_COVARIATES", ""))

    # ---- Tested exposures (Step 5/6/7) ----
    # gene_environment tests a single exposure per run (EXPOSURE). Here, if
    # VQTL_EXPOSURES is not set, the default is [ge.exposure] (minimal
    # behavior, consistent with the rest of the pipeline); set
    # VQTL_EXPOSURES=exp1,exp2,exp3 in .env to test more than one exposure
    # in a single vqtl run.
    exposures: list[str] = field(default_factory=lambda: _env_list("VQTL_EXPOSURES", ""))

    # ---- Step 3: QUAIL-style vQTL scan ----
    taus: list[float] = field(
        default_factory=lambda: _env_float_list(
            "VQTL_TAUS", "0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45"
        )
    )
    se_method: str = field(default_factory=lambda: _env("VQTL_SE_METHOD", "bootstrap"))
    bootstrap_k: int = field(default_factory=lambda: _env_int("VQTL_BOOTSTRAP_K", 200))
    asymptotic_bootstrap_k: int = field(default_factory=lambda: _env_int("VQTL_ASYMPTOTIC_BOOTSTRAP_K", 50))
    chunk_size: int = field(default_factory=lambda: _env_int("VQTL_CHUNK_SIZE", 2000))
    # Default: cfg.max_workers (same parallelism as the rest of the
    # pipeline, no separate parameter to keep in sync by hand). -1 = all cores.
    n_jobs: int = field(default_factory=lambda: _env_int("VQTL_N_JOBS", 0))
    # Optional safety net: MAF/call-rate are already filtered upstream (see
    # module docstring); by default nothing is re-filtered here.
    min_maf: float = field(default_factory=lambda: _env_float("VQTL_MIN_MAF", 0.0))
    min_call_rate: float = field(default_factory=lambda: _env_float("VQTL_MIN_CALL_RATE", 0.0))

    # ---- Step 4: genomic-control-corrected candidate filter ----
    # By default filters on P_gc (corrected for genomic inflation), not on
    # the raw asymptotic P: on a discrete dosage predictor (0/1/2) the raw P
    # is markedly anti-conservative. Set VQTL_FILTER_P_COLUMN=P to fall back
    # to filtering on the raw P value.
    filter_p_column: str = field(default_factory=lambda: _env("VQTL_FILTER_P_COLUMN", "P_gc"))
    filter_p_threshold: float = field(default_factory=lambda: _env_float("VQTL_FILTER_P_THRESHOLD", 1e-5))
    filter_top_n: int | None = field(default_factory=lambda: _env_optional_int("VQTL_FILTER_TOP_N"))

    # ---- Step 5/6: SNP x exposure interaction test, rGE, heteroscedasticity ----
    # HC3 (OLS) / HC1 (logit): heteroscedasticity-robust standard errors,
    # enabled by default.
    robust_se: bool = field(default_factory=lambda: _env_bool("VQTL_ROBUST_SE", True))
    rge_het_alpha: float = field(default_factory=lambda: _env_float("VQTL_RGE_HET_ALPHA", 0.05))
    # Nominal threshold (not corrected for multiple testing) used to populate
    # vqtl_interaction_results_significant -- see db/schema.sql.
    interaction_sig_threshold: float = field(default_factory=lambda: _env_float("VQTL_INTERACTION_SIG_THRESHOLD", 0.05))

    # ---- Step 7: Freedman-Lane permutations on top loci ----
    # Default: reuses gene_environment's N_PERM_HIGH if VQTL_N_PERM is not
    # set, so there is no separate permutation count to choose.
    n_perm: int = field(default_factory=lambda: _env_int("VQTL_N_PERM", 0))
    perm_top_n_loci: int = field(default_factory=lambda: _env_int("VQTL_PERM_TOP_N_LOCI", 10))

    # ---- Output ----
    results_dir: str = field(default_factory=lambda: _env("VQTL_RESULTS_DIR", "./vqtl_results"))

    # ---- Step 9: .docx export (Results + Supplementary Material for the paper) ----
    # How many rows of the genome-wide scan to show in Table 1 (Results, main
    # body of the paper) vs. in Supplementary Table S1 (extended list).
    docx_top_n_scan: int = field(default_factory=lambda: _env_int("VQTL_DOCX_TOP_N_SCAN", 20))
    # Cap on the rows of Supplementary Table S1 (full scan): a Word table
    # with tens/hundreds of thousands of rows is unusable in a manuscript;
    # beyond this number the table is truncated with a note pointing to the
    # full .tsv file.
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
        """Output folder for the current generation/cohort (or the one
        passed explicitly, to override cfg.ge.generation without mutating
        the global config -- see cli.py --generation)."""
        gen = generation if generation is not None else self.ge.generation
        return f"{self.results_dir}/gen{gen}"


_vqtl_config_instance: VqtlConfig | None = None


def get_vqtl_config() -> VqtlConfig:
    global _vqtl_config_instance
    if _vqtl_config_instance is None:
        _vqtl_config_instance = VqtlConfig()
    return _vqtl_config_instance
