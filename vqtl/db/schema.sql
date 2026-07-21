-- ============================================================
-- Schema per la persistenza a DB della pipeline vqtl.
-- Da eseguire UNA VOLTA sullo stesso database MySQL/MariaDB gia' usato da
-- gene_environment (stesso DB_NAME del .env, stesso pool di connessioni
-- gestito da gene_environment/db/connection.py -- vqtl non apre un DB suo).
--
--   mysql -u <DB_USER> -p <DB_NAME> < vqtl/db/schema.sql
--
-- NOTA MIGRAZIONE: se hai gia' eseguito una versione precedente di questo
-- schema (prima dell'aggiunta del test di Levene permutazionale in
-- vqtl_permutation_results), il CREATE TABLE IF NOT EXISTS qui sotto NON
-- aggiunge le nuove colonne a una tabella gia' esistente. In quel caso:
--   ALTER TABLE vqtl_permutation_results
--     ADD COLUMN levene_stat_observed DOUBLE,
--     ADD COLUMN levene_pval DOUBLE,
--     ADD COLUMN levene_n_perm_valid INT UNSIGNED;
-- (oppure, se i dati gia' calcolati non servono piu', DROP TABLE
-- vqtl_permutation_results; e rilancia questo file cosi' com'e'.)
--
-- Perche' queste tabelle e non variant_results (gia' esistente): il modello
-- dati e' diverso. variant_results e' una riga per (variant, exposure,
-- generation, test) con UN solo risultato di interazione per riga.
-- vqtl produce invece risultati a piu' stadi via via piu' selettivi (scan
-- genoma-wide -> candidati -> interazione -> rGE/eteroschedasticita' ->
-- permutazione -> robustezza), ciascuno con la propria chiave e le proprie
-- colonne: usare variant_results per tutto avrebbe richiesto o una singola
-- tabella con decine di colonne quasi sempre NULL, o sovraccaricare la
-- semantica delle colonne esistenti. Tabelle separate, stesso pattern
-- (status pending/in_progress/done/failed, insert dei placeholder, poi
-- update in bulk), stesso connection pool.
-- ============================================================

-- ---- Step 3+4: scan genoma-wide + filtro/candidati (stessa riga, il
-- filtro fa solo un UPDATE su is_candidate/p_gc/fdr_gc) ----
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

-- "Tabella dove salva solamente le cose significative": mirror di
-- vqtl_scan_results, ma SOLO le varianti candidate (is_candidate=1),
-- risincronizzata (DELETE + INSERT) ogni volta che gira lo Step 4 filter
-- per quella generazione. Ha due scopi:
--   1) e' la fonte diretta per Table 1 (Results) del report.docx, mentre
--      vqtl_scan_results resta la fonte per Supplementary Table S1;
--   2) e' il segnale di "short-circuit": se per una generazione questa
--      tabella ha gia' righe, un nuovo run per quella generazione SALTA
--      del tutto lo scan genoma-wide (il calcolo costoso) e il filtro,
--      e legge direttamente da qui + da vqtl_scan_results -- vedi
--      vqtl/cli.py e la funzione count_significant_scan/sync_scan_significant
--      in vqtl/db/repository.py.
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

-- Fingerprint della configurazione statistica corrente per ciascuna
-- generazione (taus, se_method, ...): se cambia rispetto a quanto salvato,
-- il repository ripulisce vqtl_scan_results per quella generazione e
-- reinserisce i placeholder, invece di riusare per sbaglio righe 'done'
-- calcolate con parametri diversi.
CREATE TABLE IF NOT EXISTS vqtl_scan_runs (
    generation   TINYINT UNSIGNED NOT NULL PRIMARY KEY,
    fingerprint  JSON NOT NULL,
    updated_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- ---- Step 5: test di interazione SNP x esposizione (solo candidati) ----
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

-- "Tabella dove salva solamente le cose significative" per lo Step 5:
-- mirror di vqtl_interaction_results, solo le coppie SNP x esposizione con
-- pval nominale < VQTL_INTERACTION_SIG_THRESHOLD (default 0.05 -- vedi
-- vqtl/config.py), risincronizzata ogni volta che gira lo Step 5 per quella
-- generazione. Fonte diretta per Table 2 (Results) del report.docx (mentre
-- vqtl_interaction_results resta la fonte per Supplementary Table S2). A
-- differenza della coppia scan/scan_significant, qui NON e' usata come
-- short-circuit per saltare il calcolo: lo Step 5 gira solo sui candidati
-- (gia' un insieme piccolo) e salta gia' da solo le coppie con status='done'
-- individualmente (vedi get_done_keys in repository.py), quindi non serve
-- un meccanismo di short-circuit aggiuntivo a livello di intera generazione.
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

-- ---- Step 6: rGE + eteroschedasticita' (solo candidati) ----
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

-- ---- Step 7a: robustezza a trasformazioni del fenotipo (solo top loci) ----
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

-- ---- Step 7b: permutazioni Freedman-Lane (solo top loci) ----
-- Oltre alla permutazione Freedman-Lane sull'interazione (beta_i_observed/
-- empirical_pval), nello STESSO loop (stessa iterazione, non uno step a
-- parte) si calcola anche un test di Levene permutazionale sulla varianza
-- del fenotipo residualizzato per gruppo genotipico: statistica di Levene
-- osservata sui gruppi genotipo reali -> permutazione delle ETICHETTE di
-- genotipo (non dei residui, a differenza del Freedman-Lane sopra) ->
-- statistica ricalcolata sui gruppi permutati -> distribuzione nulla
-- empirica -> p-value empirica. Conferma (o smentisce) in modo
-- assumption-light, sugli stessi dati, l'effetto di varianza rilevato dallo
-- scan QUAIL del Step 3 per il locus, senza le assunzioni asintotiche della
-- quantile regression. Il valore e' lo stesso per tutte le righe dello
-- stesso variant (non dipende dall'esposizione, la Levene e' univariata sul
-- genotipo).
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
    updated_at           TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (generation, variant, exposure)
) ENGINE=InnoDB;
