"""
Configurazione centralizzata della pipeline.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv  # type: ignore

    # Cerca un .env nella cwd o nella root del progetto; non fallisce se manca.
    _here = Path(__file__).resolve().parent.parent
    for candidate in (Path.cwd() / ".env", _here / ".env"):
        if candidate.exists():
            load_dotenv(candidate)
            break
except ImportError:
    # python-dotenv è opzionale: se non installato, ci si affida alle env vars
    # già presenti nell'ambiente (va benissimo in produzione/CI).
    pass


class ConfigError(RuntimeError):
    pass


def _env(name: str, default: str | None = None, required: bool = False) -> str | None:
    val = os.environ.get(name, default)
    if required and (val is None or val == ""):
        raise ConfigError(
            f"Variabile d'ambiente obbligatoria mancante: {name}. "
            f"Copia .env.example in .env e compilala, oppure esportala nell'ambiente."
        )
    return val


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    return int(val) if val not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    return float(val) if val not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_list(name: str, default: str = "") -> list[str]:
    val = os.environ.get(name, default)
    return [v.strip() for v in val.split(",") if v.strip()]


@dataclass(frozen=True)
class DBConfig:
    user: str = field(default_factory=lambda: _env("DB_USER", required=True))
    password: str = field(default_factory=lambda: _env("DB_PASSWORD", required=True))
    name: str = field(default_factory=lambda: _env("DB_NAME", required=True))
    host: str = field(default_factory=lambda: _env("DB_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("DB_PORT", 3306))
    pool_size: int = field(default_factory=lambda: _env_int("DB_POOL_SIZE", 8))


@dataclass(frozen=True)
class Config:
    # ---- FILE PATH ----
    raw_file: str = field(default_factory=lambda: _env("RAW_FILE", ""))
    env_file: str = field(default_factory=lambda: _env("ENV_FILE", ""))
    temp_df_path: str = field(default_factory=lambda: _env("TEMP_DF_PATH", "temp_df.pkl"))

    generation: int = field(default_factory=lambda: _env_int("GENERATION", 1))
    test_label: str = field(default_factory=lambda: _env("TEST", "0_1"))

    # ---- DATA SETTINGS ----
    sep: str = field(default_factory=lambda: _env("SEP", ","))
    decimal: str = field(default_factory=lambda: _env("DECIMAL", "."))
    target_col: str = field(default_factory=lambda: _env("TARGET_COL", "onset_age"))
    exposure: str = field(default_factory=lambda: _env("EXPOSURE", ""))
    covariates: list[str] = field(default_factory=lambda: _env_list("COVARIATES", "sex"))
    sample_id_col: str = field(default_factory=lambda: _env("SAMPLE_ID_COL", "id"))

    # ---- MATCHING ----
    match_k: int = field(default_factory=lambda: _env_int("MATCH_K", 3))
    min_treated: int = field(default_factory=lambda: _env_int("MIN_TREATED", 5))
    min_sample_size: int = field(default_factory=lambda: _env_int("MIN_SAMPLE_SIZE", 10))
    max_smd: float = field(default_factory=lambda: _env_float("MAX_SMD", 0.25))

    # ---- PERMUTATION ----
    n_perm: int = field(default_factory=lambda: _env_int("N_PERM", 500))
    n_perm_high: int = field(default_factory=lambda: _env_int("N_PERM_HIGH", 10000))
    random_state: int = field(default_factory=lambda: _env_int("RANDOM_STATE", 42))
    min_obs_coef: float = field(default_factory=lambda: _env_float("MIN_OBS_COEF", 2))
    pvalue_threshold: float = field(default_factory=lambda: _env_float("PVALUE_THRESHOLD", 0.05))
    # ottimizzazione: interrompe le permutazioni "light" in anticipo se il
    # risultato parziale è già chiaramente non significativo (vedi modeling.py)
    adaptive_perm_check_every: int = field(default_factory=lambda: _env_int("ADAPTIVE_PERM_CHECK_EVERY", 100))
    adaptive_perm_futility_p: float = field(default_factory=lambda: _env_float("ADAPTIVE_PERM_FUTILITY_P", 0.5))

    # ---- SCALING / PARALLEL ----
    standardize: bool = field(default_factory=lambda: _env_bool("STANDARDIZE", True))
    max_workers: int = field(default_factory=lambda: _env_int("MAX_WORKERS", os.cpu_count() or 4))

    # ---- ONSET AGE ANALYSIS ----
    use_mann_whitney: bool = field(default_factory=lambda: _env_bool("USE_MANN_WHITNEY", True))
    onset_alpha: float = field(default_factory=lambda: _env_float("ONSET_ALPHA", 0.05))
    onset_min_group_size: int = field(default_factory=lambda: _env_int("ONSET_MIN_GROUP_SIZE", 5))
    onset_low_power_threshold: int = field(default_factory=lambda: _env_int("ONSET_LOW_POWER_THRESHOLD", 10))
    n_boot: int = field(default_factory=lambda: _env_int("N_BOOT", 2000))

    # ---- GENE REDUCTION / VCF ----
    vcf_folders: list[str] = field(default_factory=lambda: _env_list("VCF_FOLDERS", ""))
    null_percentage: float = field(default_factory=lambda: _env_float("NULL_PERCENTAGE", 0.30))
    output_folder: str = field(default_factory=lambda: _env("OUTPUT_FOLDER", ""))
    maf_threshold: float = field(default_factory=lambda: _env_float("MAF_THRESHOLD", 0.001))
    ld_window_size: int = field(default_factory=lambda: _env_int("LD_WINDOW_SIZE", 50))
    ld_step: int = field(default_factory=lambda: _env_int("LD_STEP", 5))
    ld_r2_threshold: float = field(default_factory=lambda: _env_float("LD_R2_THRESHOLD", 0.8))

    # ---- VCF sorgenti per generazione (extract_matrix) ----
    vcf_dir_gen1: str = field(default_factory=lambda: _env("VCF_DIR_GEN1", ""))
    vcf_dir_gen2: str = field(default_factory=lambda: _env("VCF_DIR_GEN2", ""))
    vcf_dir_gen3: str = field(default_factory=lambda: _env("VCF_DIR_GEN3", ""))

    # ---- OUTPUT ----
    significant_matrix_dir: str = field(default_factory=lambda: _env("SIGNIFICANT_MATRIX_DIR", "./output/significant_variant_matrices"))
    significant_export_dir: str = field(default_factory=lambda: _env("SIGNIFICANT_EXPORT_DIR", "./output/significant_export"))
    onset_age_out_dir: str = field(default_factory=lambda: _env("ONSET_AGE_OUT_DIR", "./output/onset_age_analysis"))
    log_dir: str = field(default_factory=lambda: _env("LOG_DIR", "./logs"))

    db: DBConfig = field(default_factory=DBConfig)


_config_instance: Config | None = None


def get_config() -> Config:
    """Singleton lazy: la config viene letta/validata una sola volta al primo utilizzo."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance
