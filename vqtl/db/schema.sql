-- ============================================================
-- Schema for the DB-backed persistence of the vqtl pipeline.
-- Run this ONCE against the same MySQL/MariaDB database already used by
-- gene_environment (same DB_NAME from .env, same connection pool managed
-- by gene_environment/db/connection.py -- vqtl does not open its own DB).
--
--   mysql -u <DB_USER> -p <DB_NAME> < vqtl/db/schema.sql
--
-- Why these tables instead of variant_results (already existing): the data
-- model is different. variant_results is one row per (variant, exposure,
-- generation, test) with a SINGLE interaction result per row. vqtl instead
-- produces results across several, progressively more selective stages
-- (genome-wide scan -> candidates -> interaction -> rGE/heteroscedasticity
-- -> permutation -> robustness), each with its own key and its own
-- columns: using variant_results for everything would have required either
-- a single table with dozens of almost-always-NULL columns, or overloading
-- the semantics of its existing columns. Separate tables, same pattern
-- (status pending/in_progress/done/failed, insert placeholders, then
-- bulk-update), same connection pool.
-- ============================================================

-- ---- Step 3+4: genome-wide scan + filter/candidates (same row; the
-- filter step only performs an UPDATE on is_candidate/p_gc/fdr_gc) ----
CREATE TABLE IF NOT EXISTS vqtl_scan_results (
    generation      TINYINT UNSIGNED NOT NULL,
    variant         VARCHAR(191) NOT NULL,
    chromosome      VARCHAR(16),
    position        INT UNSIGNED,
    status          ENUM('pending','in_progress','done','failed') NOT NULL DEFAULT 'pending',
    n               INT UNSIGNED,
    maf             DOUBLE,
    beta_qi         DOUBLE,
    se              DOUBLE,
    z               DOUBLE,
    p               DOUBLE,
    p_gc            DOUBLE,
    fdr_gc          DOUBLE,
    is_candidate    TINYINT(1) NOT NULL DEFAULT 0,
    error_message   TEXT,
    updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (generation, variant),
    INDEX idx_vqtl_scan_status (generation, status),
    INDEX idx_vqtl_scan_candidate (generation, is_candidate),
    INDEX idx_vqtl_scan_p (generation, p)
) ENGINE=InnoDB;

-- Significant-results mirror of vqtl_scan_results, holding ONLY the
-- candidate variants (is_candidate=1), resynchronized (DELETE + INSERT)
-- every time the Step 4 filter runs for that generation. It serves two
-- purposes:
--   1) it is the direct source for Table 1 (Results) of report.docx, while
--      vqtl_scan_results remains the source for Supplementary Table S1;
--   2) it is the "short-circuit" signal: if this table already has rows
--      for a generation, a new run for that generation SKIPS the
--      genome-wide scan (the expensive computation) and the filter step
--      entirely, and reads directly from here + from vqtl_scan_results --
--      see vqtl/cli.py and the count_significant_scan/sync_scan_significant
--      functions in vqtl/db/repository.py.
CREATE TABLE IF NOT EXISTS vqtl_scan_results_significant (
    generation      TINYINT UNSIGNED NOT NULL,
    variant         VARCHAR(191) NOT NULL,
    chromosome      VARCHAR(16),
    position        INT UNSIGNED,
    n               INT UNSIGNED,
    maf             DOUBLE,
    beta_qi         DOUBLE,
    se              DOUBLE,
    z               DOUBLE,
    p               DOUBLE,
    p_gc            DOUBLE,
    fdr_gc          DOUBLE,
    updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (generation, variant)
) ENGINE=InnoDB;

-- Fingerprint of the current statistical configuration for each generation
-- (taus, se_method, ...): if it differs from what is saved, the repository
-- clears vqtl_scan_results for that generation and reinserts the
-- placeholders, instead of accidentally reusing 'done' rows computed with
-- different parameters.
CREATE TABLE IF NOT EXISTS vqtl_scan_runs (
    generation   TINYINT UNSIGNED NOT NULL PRIMARY KEY,
    fingerprint  JSON NOT NULL,
    updated_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- ---- Step 5: SNP x exposure interaction test (candidates only) ----
CREATE TABLE IF NOT EXISTS vqtl_interaction_results (
    generation      TINYINT UNSIGNED NOT NULL,
    variant         VARCHAR(191) NOT NULL,
    exposure        VARCHAR(191) NOT NULL,
    chromosome      VARCHAR(16),
    position        INT UNSIGNED,
    status          ENUM('pending','in_progress','done','failed') NOT NULL DEFAULT 'pending',
    beta_i          DOUBLE,
    se              DOUBLE,
    pval            DOUBLE,
    n               INT UNSIGNED,
    maf             DOUBLE,
    error_message   TEXT,
    updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (generation, variant, exposure),
    INDEX idx_vqtl_interaction_status (generation, status)
) ENGINE=InnoDB;

-- Significant-results mirror of vqtl_interaction_results for Step 5: only
-- the SNP x exposure pairs with a nominal pval < VQTL_INTERACTION_SIG_THRESHOLD
-- (default 0.05 -- see vqtl/config.py), resynchronized every time Step 5
-- runs for that generation. Direct source for Table 2 (Results) of
-- report.docx (while vqtl_interaction_results remains the source for
-- Supplementary Table S2). Unlike the scan/scan_significant pair, this is
-- NOT used as a short-circuit to skip computation: Step 5 only runs on the
-- candidates (already a small set) and already skips individual
-- status='done' pairs on its own (see get_done_keys in repository.py), so
-- no additional whole-generation short-circuit is needed here.
CREATE TABLE IF NOT EXISTS vqtl_interaction_results_significant (
    generation   TINYINT UNSIGNED NOT NULL,
    variant      VARCHAR(191) NOT NULL,
    exposure     VARCHAR(191) NOT NULL,
    chromosome   VARCHAR(16),
    position     INT UNSIGNED,
    beta_i       DOUBLE,
    se           DOUBLE,
    pval         DOUBLE,
    n            INT UNSIGNED,
    maf          DOUBLE,
    updated_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (generation, variant, exposure)
) ENGINE=InnoDB;

-- ---- Step 6: rGE + heteroscedasticity (candidates only) ----
CREATE TABLE IF NOT EXISTS vqtl_rge_het_results (
    generation                  TINYINT UNSIGNED NOT NULL,
    variant                     VARCHAR(191) NOT NULL,
    exposure                    VARCHAR(191) NOT NULL,
    chromosome                  VARCHAR(16),
    position                    INT UNSIGNED,
    status                      ENUM('pending','in_progress','done','failed') NOT NULL DEFAULT 'pending',
    rge_beta_exposure_on_snp    DOUBLE,
    rge_se                      DOUBLE,
    rge_pval                    DOUBLE,
    rge_flag                    TINYINT(1),
    het_bp_lm_stat              DOUBLE,
    het_bp_lm_pvalue            DOUBLE,
    het_bp_f_stat               DOUBLE,
    het_bp_f_pvalue             DOUBLE,
    heteroscedasticity_flag     TINYINT(1),
    error_message               TEXT,
    updated_at                  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (generation, variant, exposure),
    INDEX idx_vqtl_rgehet_status (generation, status)
) ENGINE=InnoDB;

-- ---- Step 7a: robustness to phenotype transformations (top loci only) ----
CREATE TABLE IF NOT EXISTS vqtl_robustness_results (
    generation          TINYINT UNSIGNED NOT NULL,
    variant              VARCHAR(191) NOT NULL,
    exposure             VARCHAR(191) NOT NULL,
    phenotype_variant    VARCHAR(32) NOT NULL,  -- 'original' | 'log_transform' | 'rank_inverse_normal' | 'outliers_removed'
    chromosome           VARCHAR(16),
    position              INT UNSIGNED,
    status                ENUM('pending','in_progress','done','failed') NOT NULL DEFAULT 'pending',
    beta_i                DOUBLE,
    se                    DOUBLE,
    pval                  DOUBLE,
    n                     INT UNSIGNED,
    maf                   DOUBLE,
    error_message         TEXT,
    updated_at            TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (generation, variant, exposure, phenotype_variant)
) ENGINE=InnoDB;

-- ---- Step 7b: Freedman-Lane permutations (top loci only) ----
-- Besides the Freedman-Lane permutation on the interaction (beta_i_observed/
-- empirical_pval), the SAME loop (same iteration, not a separate step)
-- also computes a permutation-based Levene test on the variance of the
-- residualized phenotype by genotype group: observed Levene statistic on
-- the real genotype groups -> permutation of the genotype LABELS (not of
-- the residuals, unlike the Freedman-Lane test above) -> statistic
-- recomputed on the permuted groups -> empirical null distribution ->
-- empirical p-value. This confirms (or does not confirm), on the same
-- data and with minimal distributional assumptions, the variance effect
-- detected by the Step 3 QUAIL scan for that locus, without the asymptotic
-- assumptions of quantile regression. The value is the same across all
-- rows for the same variant (it does not depend on exposure; the Levene
-- test is univariate on genotype).
CREATE TABLE IF NOT EXISTS vqtl_permutation_results (
    generation           TINYINT UNSIGNED NOT NULL,
    variant              VARCHAR(191) NOT NULL,
    exposure             VARCHAR(191) NOT NULL,
    chromosome           VARCHAR(16),
    position             INT UNSIGNED,
    status               ENUM('pending','in_progress','done','failed') NOT NULL DEFAULT 'pending',
    beta_i_observed      DOUBLE,
    n_perm_valid         INT UNSIGNED,
    empirical_pval       DOUBLE,
    asymptotic_pval      DOUBLE,
    levene_stat_observed DOUBLE,
    levene_pval          DOUBLE,
    levene_n_perm_valid  INT UNSIGNED,
    error_message        TEXT,
    updated_at            TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (generation, variant, exposure)
) ENGINE=InnoDB;
