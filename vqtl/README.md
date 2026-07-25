# vQTL / Gene-by-Environment Interaction Pipeline

`vqtl` is a module inside the same repository as `gene_environment` and
`quality_control`, not a standalone project: it shares the same `.env`,
the same `Config`, the same logging, and the same id/PCA/statistics
helpers as the rest of the pipeline.

```
gene_environment_v2/
  gene_environment/     # main pipeline
  quality_control/      # QC + PCA (plink2)
  vqtl/                 # this package
    config.py
    cli.py
    core/
      data.py             Steps 0-2 (in part): genetics + environment + PCA join
      phenotype.py        Step 2: phenotype transformations (z/log/rint)
      scan.py              Step 3: QUAIL-style vQTL scan
      filter_candidates.py Step 4: Manhattan/QQ plots, P_gc, candidate filter
      interaction.py        Step 5: SNP x exposure interaction test
      rge_het.py             Step 6: rGE + heteroscedasticity
      permutation.py          Step 7: robustness + Freedman-Lane permutations
      report.py                 Step 8: report.md + figures
      docx_report.py             Step 9: report.docx (Results + Supplementary
                                  Material, paper-ready)
    db/
      schema.sql          CREATE TABLE statements (run once, see below)
      repository.py        All DB queries (placeholders/updates/fetches)
    .env.vqtl.example    variables to add to the existing .env
  .env
```

## Architecture

**Genotype dosage comes from `RAW_FILE`, not from VCF files.** vqtl does
not parse VCFs itself: it reads the genome-wide genotype matrix already
produced by `gene_environment.cli filter-vcf` + `build-matrix` (already
MAF/LD-pruned). `vqtl/core/data.py` loads it by reusing
`gene_environment.vcf_pipeline.build_dataset.load_and_prepare_data`, the
same function the rest of the pipeline uses -- same id join, same id
cleaning (`clean_sample_id`), same generation/cohort filter. As a
consequence, Steps 5-8 read candidate-SNP dosage directly from a column of
the DataFrame already in memory, without touching VCFs at any point.

**QC and PCA come from `quality_control/`.** The `quality_control/`
pipeline in this repository computes QC and PCA with `plink2`
(`00_run_plink_qc.sh` -> `extract_pca_covariates.py`); its output
(`pca_covariates.csv`, one per generation) is the same
`PCA_COVARIATES_PATH_TEMPLATE` that `gene_environment` itself uses as an
adjustment covariate. `vqtl/core/data.py` reuses those same PCA via
`gene_environment.utils.pca_utils.load_pca_covariates`.

**A single configuration.** `vqtl/config.py` has no configuration file of
its own: it reads the same environment variables from the existing `.env`
through `gene_environment.config.get_config()`, and only adds the
parameters that are genuinely specific to the vQTL method (`VQTL_`-
prefixed, see `.env.vqtl.example`).

**Cohorts** (gen1/gen2/gen3) map onto the same `GENERATION` variable used
by `gene_environment` (or `--generation` on the command line, which
overrides `GENERATION` for that command only, without touching the `.env`
file).

## Statistical method

The statistical core of vqtl is genuinely specific to the method and has
no equivalent in `gene_environment` (which runs a matching + permutation
interaction test, a complementary but different approach):

1. Genome-wide QUAIL-style vQTL scan (`core/scan.py`): OLS residualization
   + paired-tau quantile regression, with the SE estimated via bootstrap
   (asymptotic-style mini-bootstrap or a fuller bootstrap, see the
   module's docstring for the accuracy/cost trade-off between the two).
2. Candidate filter, by default on `P_gc` (genomic-control-corrected),
   not on the raw asymptotic p-value (`core/filter_candidates.py`).
3. Interaction and rGE tests with heteroscedasticity-robust standard
   errors (HC3/HC1) by default (`core/interaction.py`, `core/rge_het.py`).
4. Freedman-Lane permutation on the top loci, rather than a naive
   permutation of the raw phenotype (`core/permutation.py`).

## Usage

```bash
# from the repo root (where both gene_environment/ and vqtl/ live)
cd /srv/python-projects/gene_environment_v2

# 1. add the VQTL_* variables to the existing .env
cat vqtl/.env.vqtl.example >> .env
# ... then edit the values as needed (VQTL_EXPOSURES, etc.)

# 2. create the vqtl_* tables on the same DB as gene_environment (ONCE)
mysql -u <DB_USER> -p <DB_NAME> < vqtl/db/schema.sql

# 3. run the pipeline
python3 -m vqtl.cli run-all --generation 2   # first run: computes everything
python3 -m vqtl.cli run-all --generation 1   # if gen1 already has significant
                                              # results cached, skips the scan
                                              # and reads from the DB

# individual steps (for debugging), same cohort:
python3 -m vqtl.cli scan --generation 1
python3 -m vqtl.cli filter --generation 1
python3 -m vqtl.cli interaction --generation 1
python3 -m vqtl.cli rge-het --generation 1
python3 -m vqtl.cli permute --generation 1
python3 -m vqtl.cli report --generation 1
python3 -m vqtl.cli docx --generation 1
```

**All intermediate results live in the DB, not in `.tsv` files** (see the
dedicated section below). Only the final deliverables remain files, under
`VQTL_RESULTS_DIR/gen<N>/`: `report.md`, `report.docx` (Results +
Supplementary Material for the paper -- see the dedicated section below),
and `figures/*.png`.

The merged dataset (genetics + environment + PCA -- the most expensive
step, since it reads the entire `RAW_FILE`) is cached at
`VQTL_RESULTS_DIR/gen<N>/vqtl_dataset.pkl` (the same principle as
`gene_environment.analysis.orchestrator`'s `TEMP_DF_PATH`), so running the
steps one at a time from the command line does not rebuild it every time.
Pass `--force` to rebuild it (e.g. after rerunning
`filter-vcf`/`build-matrix` upstream, or after recomputing the PCA).

## Word export for the paper (Step 9)

`report.docx` (generated automatically by `run-all`, or on its own with
`python3 -m vqtl.cli docx --generation <N>`) is meant to be pasted or
attached directly into a manuscript: its content -- titles, captions,
column headers, notes -- is entirely in English. It contains:

- **Results**: Table 1 (top loci from the genome-wide scan), Table 2
  (ONLY the nominally significant interaction tests, `P <
  VQTL_INTERACTION_SIG_THRESHOLD`, default 0.05 -- sourced from
  `vqtl_interaction_results_significant`), Table 3 (permutation-based
  validation of the top loci: interaction + variance by genotype, see
  below), Figures 1-3 (Manhattan, QQ, forest plot).
- **Supplementary Material**: Table S1 (full genome-wide scan, truncated
  to `VQTL_DOCX_SUPP_MAX_ROWS` rows with a note pointing to the
  `vqtl_scan_results` DB table for the complete list), Table S2 (ALL
  interaction tests on the candidates, including the non-significant
  pairs excluded from Table 2), Table S3 (rGE/heteroscedasticity screen),
  Table S4 (robustness to phenotype transformations), Supplementary
  Figures (per-locus boxplot + scatter).

If Table 2 has no rows (no SNP x exposure pair reaches the nominal
threshold), the document says so explicitly ("No rows to display.")
instead of leaving an unexplained empty table or, worse, silently showing
every test as if it were a significant result.

**Table 3 also includes a permutation-based Levene test for variance by
genotype**, not only the Freedman-Lane permutation on the interaction: for
each top locus, the Levene (Brown-Forsythe, robust to non-normal
phenotypes) statistic is computed on the real genotype groups (dosage
0/1/2) of the covariate-residualized phenotype; the genotype LABELS are
then permuted (not the residuals, unlike the interaction permutation) and
the statistic is recomputed at each permutation, building an empirical
null distribution -- no asymptotic assumptions, the p-value comes entirely
from the data itself. This provides an assumption-light confirmation (or
otherwise) of the variance effect that the Step 3 scan (QUAIL, quantile
regression, with its own asymptotic assumptions) detected for that locus.
It runs in the SAME permutation loop already used for the interaction test
(same joblib/n_splits infrastructure), not as a separate pass; it is
computed once per SNP (independent of exposure) and reused for subsequent
rows of the same SNP if it appears with more than one exposure among the
top loci.

Tables use a "three-line" style (border above, border below the header,
border at the bottom, no internal gridlines), Times New Roman, in a
landscape document with reduced margins -- needed because the tables have
8-11 columns, and in portrait orientation with standard margins Word
breaks mid-word on tokens with no spaces (variant ids, exposure names).
`VQTL_DOCX_TOP_N_SCAN` controls how many scan rows go into Table 1 (main
body of the paper, default 20); `VQTL_DOCX_SUPP_MAX_ROWS` controls the
truncation of Supplementary Table S1 (default 200 -- beyond that, a Word
table becomes unusable and the reader is pointed to the `vqtl_scan_results`
DB table instead).

## Candidate consistency across filter runs

If you rerun Step 4 (`filter`, or `run-all` without the short-circuit)
with a different threshold or `VQTL_FILTER_TOP_N` than a previous run, the
candidate list can change. Variants that DROP OUT of the new list are
automatically cleared from all downstream tables (`vqtl_interaction_
results`, `vqtl_rge_het_results`, `vqtl_robustness_results`, `vqtl_
permutation_results`) for that generation -- otherwise they would remain
as orphaned rows from a previous filter, which `fetch_results()` would
still include in the results with no way of knowing that variant is no
longer a current candidate. This is logged explicitly when it happens.

## DB persistence: full vs. significant, resumability, per-generation short-circuit

All steps (3-7) write to MySQL/MariaDB tables (`vqtl_*`, the same
database and the same connection pool as `gene_environment` --
`gene_environment.db.connection`, "PID-aware": safe even if worker
processes were to open their own connections in the future), rather than
to intermediate `.tsv` files. The full schema is in `vqtl/db/schema.sql`
(run once, see "Usage" above).

**Pattern shared by all tables** (the same as `gene_environment`'s
`variant_results`): for every unit of work (a variant, or a
variant+exposure pair), a `status='pending'` placeholder is inserted
before anything is computed; each completed chunk/row is updated
(`status='done'`, or `'failed'` with `error_message` on an exception)
IMMEDIATELY, not at the end of the step. On restart, units already marked
`'done'` are not repeated -- if the process is interrupted (Ctrl+C, OOM, a
cluster job being killed), it resumes from where it left off, not from
scratch. If the statistical parameters change (`VQTL_TAUS`,
`VQTL_SE_METHOD`, etc.) or the variant set in `RAW_FILE` changes, the
"fingerprint" saved in `vqtl_scan_runs` no longer matches and the rows for
that generation are cleared and recomputed automatically (no manual
intervention needed).

**"Full" vs. "significant only"**: `vqtl_scan_results` (Step 3+4) and
`vqtl_interaction_results` (Step 5) each have a twin table --
`vqtl_scan_results_significant` and `vqtl_interaction_results_significant`
-- holding ONLY the candidate subset (`is_candidate=1`, from the Step 4
filter) and the nominally significant subset (`pval <
VQTL_INTERACTION_SIG_THRESHOLD`, default 0.05), respectively. They are
resynchronized (DELETE + INSERT) every time the corresponding step runs,
so they are always an exact mirror, never stale. They serve two purposes:
1. they are the direct source for Table 1/Table 2 (Results) of
   `report.docx`, while the "full" tables remain the source for
   Supplementary Tables S1/S2;
2. **`vqtl_scan_results_significant` is also the short-circuit signal**
   for the whole generation (see below).

rGE/heteroscedasticity (Step 6), robustness and permutations (Step 7)
remain "full only" (no `_significant` table): they already run only on
the small set of candidates/top loci, and already skip pairs that are
already `'done'` on their own -- an extra `_significant` table would not
have added a real short-circuit, only another table to keep in sync.

**Per-generation short-circuit**: before loading any data, every `run-all`
checks `vqtl_scan_results_significant` for that generation. If it already
has rows (and `--force` was not passed), the genome-wide scan AND the
filter step are skipped entirely -- no quantile regression, no new
placeholder queries -- and Steps 5-7 start directly from the already-known
candidates read from the DB (logged explicitly). The Manhattan/QQ figures
are still regenerated (cheap) from the cached data, as a safety measure.
If a generation is being analyzed for the first time (no
`vqtl_scan_results_significant` rows for that number yet), everything is
computed from scratch, exactly as for a generation never seen before --
`--force` always forces a full recomputation even if the cache already
exists.

## Step 3 (genome-wide scan): cost

Step 3 is the most expensive part of the pipeline: for every variant it
fits up to `2 x len(VQTL_TAUS)` quantile regressions (default 9 taus -> 18
fits/variant with `VQTL_SE_METHOD=asymptotic`; beta and SE both come from
those same fits, not from two separate rounds). With
`VQTL_SE_METHOD=bootstrap` (the default) the cost rises to `18 x
VQTL_BOOTSTRAP_K` fits per variant (default 200x -> ~3600 fits/variant):
this is intentionally NOT meant to be run genome-wide, only on a handful
of already-selected loci -- see the accuracy/cost note in
`core/scan.py`'s module docstring for why the (cheaper but less accurate)
`asymptotic` method exists as an alternative. On a real genome-wide scan
(tens/hundreds of thousands of variants after the MAF/LD filter), total
runtime therefore depends mostly on: the number of variants in `RAW_FILE`,
`VQTL_TAUS` (fewer taus = linearly faster), `VQTL_SE_METHOD`, and
`MAX_WORKERS`/`VQTL_N_JOBS` (parallelized by chunk, one chunk =
`VQTL_CHUNK_SIZE` variants; each chunk's results reach the main process
and are written to the DB as soon as they are ready, not all at once at
the end).

## Excluding a covariate from vqtl only (e.g. onset_site)

The default covariates are the same as `gene_environment`'s (`COVARIATES`
in `.env`). To use a different subset ONLY in vqtl, without touching
`COVARIATES` (which stays shared with the rest of the pipeline, including
`gene_environment`'s matching-based interaction test), set
`VQTL_COVARIATES` in `.env`:

```
VQTL_COVARIATES=sex,diagnostic_delay
```

If `VQTL_COVARIATES` is not set, the default behavior is unchanged (uses
`COVARIATES`). To drop `onset_site` everywhere, including
`gene_environment`'s own analyses, edit the shared `COVARIATES` directly
instead (no need for `VQTL_COVARIATES` in that case).

## Checklist before a real run

- **`DB_USER` / `DB_PASSWORD` / `DB_NAME` (/ `DB_HOST`, default
  `127.0.0.1`) in `.env`**: vqtl actively uses the database (all
  intermediate results live there, see above), so these variables are not
  merely a formal requirement of `gene_environment.config.get_config()`
  (`DBConfig` requires them regardless, even for commands that never touch
  the DB) -- vqtl genuinely needs them to run. Before the first run:
  `mysql -u <DB_USER> -p <DB_NAME> < vqtl/db/schema.sql`.
- **Dependencies**: see `requirements-vqtl.txt`. Includes
  `mysql-connector-python` (most likely already installed, since it is
  already a dependency of `gene_environment.db.connection`).
- **`VQTL_EXPOSURES`**: if not set, only `EXPOSURE` (the single exposure
  used by the rest of the pipeline) is tested. Set this explicitly (e.g.
  `VQTL_EXPOSURES=exposure_a,exposure_b,exposure_c`) to test more than one
  exposure in the same run. If this variable changes between runs, the
  dataset cache (see above) is invalidated and rebuilt automatically.
- **PCA**: make sure `quality_control/00_run_plink_qc.sh` +
  `extract_pca_covariates.py` have already been run for the generation you
  want to analyze (the same requirement as `gene_environment`, not
  specific to vqtl). IDs in `pca_covariates.csv` are normalized with the
  same `clean_sample_id` used by the rest of the pipeline before the merge
  (necessary: the PCA file has ids in plink's raw "doubled" format, the
  main dataframe already has them cleaned).
- **MAF/LD**: the filter is already applied upstream by
  `gene_environment.cli filter-vcf`; `VQTL_MIN_MAF`/`VQTL_MIN_CALL_RATE`
  here default to disabled (0.0) so as not to filter twice -- they act
  purely as an optional safety net, not as the primary filter.
- **`VQTL_N_PERM`**: if left at 0, reuses `gene_environment`'s
  `N_PERM_HIGH`. For a full vqtl run with permutations intended for actual
  inference on the top loci, verify that value is adequate for your use
  case.
