# Genomic QC Pipeline — README

This describes the full order of execution, from raw VCFs to the final
Word supplementary report. Every script is resumable / non-destructive:
re-running a step after a crash skips work that is already done (see each
script's own header for details), and none of the Python reporting
scripts recompute anything — they only read files produced by earlier
steps.

## IMPORTANT: run the whole sequence ONCE PER GENERATION, not once for
## the pooled cohort

This study uses gen1 as the **discovery/validation cohort** and gen2 as
the **replication cohort** (gen3 is excluded entirely, not used anywhere
below). Discovery and replication must be analyzed independently — in
particular the PCA used to correct for population structure in the G×E
model must be computed **separately per cohort**: a PCA fit on the pooled
gen1+gen2 data would leak information between the two cohorts and defeats
the purpose of having a separate replication set.

Concretely, this means running the **entire sequence below twice**, with
two separate output directories — once with only `gen1` as input to step
1, once with only `gen2`. You get two independent
`Supplementary_QC_Report.docx` (one per cohort), two independent
`pca_covariates.csv`, etc. Nothing is shared between the two runs.

All commands below assume you are on the server that has `plink2` and
`bcftools` in `PATH` (the `geneenv` conda environment).

```bash
# Discovery / validation cohort
export GEN1_DIR=/mnt/cresla_prod/genome_datasets/gen1
export OUT_DIR_GEN1=/mnt/cresla_prod/genome_datasets/qc_output_gen1

# Replication cohort
export GEN2_DIR=/mnt/cresla_prod/genome_datasets/gen2
export OUT_DIR_GEN2=/mnt/cresla_prod/genome_datasets/qc_output_gen2
```

Run everything below with `$OUT_DIR=$OUT_DIR_GEN1` and only `$GEN1_DIR` as
input to step 1, then again with `$OUT_DIR=$OUT_DIR_GEN2` and only
`$GEN2_DIR`. The "Full command sequence" section at the bottom has both
runs spelled out end to end.

---

## 1. Main QC pipeline (bash, on the server)

Filters, LD-prunes, computes kinship and PCA for a SINGLE cohort (pass
only that cohort's VCF directory — do not pass gen1 and gen2 together).

```bash
./00_run_plink_qc.sh --use-filtered --jobs 16 \
    "$GEN1_DIR" \
    "$OUT_DIR_GEN1"
```

Produces (among others): `king.kin0`, `pca.eigenvec`, `pca.eigenval`,
`merged_all/geno/qc/pruned.*`, `missingness.vmiss`, `missingness.smiss`.

---

## 2. Extra QC checks (bash, on the server — run after step 1)

Sex check, heterozygosity check, MAF spectrum, and a software/version log.
Does **not** touch `00_run_plink_qc.sh` or its output; it only adds new
files to the same `$OUT_DIR`.

```bash
./01_run_extra_qc_checks.sh "$OUT_DIR_GEN1"
```

Produces: `sex_check.sexcheck`, `heterozygosity.het`, `maf.afreq`,
`run_metadata.txt`.

**Requires** `merged_qc.*` and `merged_pruned.*` from step 1 to still be
on disk.

---

## 3. Python diagnostics and QC reports

These can run on any machine with Python 3 + pandas/numpy/matplotlib/scipy
(they do not need plink2/bcftools — they only read plink2's output
files). Order between 3a–3d does not matter; 3e needs all of them to have
already run.

### 3a. Sample/variant attrition table

```bash
python3 qc_attrition_summary.py \
    --qc-dir "$OUT_DIR_GEN1" \
    --out "$OUT_DIR_GEN1/qc_attrition.csv"
```

### 3b. Kinship + PCA/batch-effect report

```bash
python3 qc_report.py \
    --kin "$OUT_DIR_GEN1/king.kin0" \
    --eigenvec "$OUT_DIR_GEN1/pca.eigenvec" \
    --eigenval "$OUT_DIR_GEN1/pca.eigenval" \
    --vcf-dirs "$GEN1_DIR" \
    --use-filtered \
    --out-dir "$OUT_DIR_GEN1/qc_report"
```

`--vcf-dirs` here takes ONLY this cohort's directory (not gen1+gen2+gen3
together as in earlier drafts of this pipeline) — the batch-effect check
this script does is meaningless within a single cohort/single VCF source
and will just show one color; the point of running it per-cohort is the
kinship report (looking for unexpected relatedness/duplicates WITHIN this
cohort) and a plain PCA scatter to eyeball outliers. Omit
`--vcf-dirs`/`--use-filtered` entirely if you don't need the batch
coloring at all.

### 3c. Relatedness / PC-vs-exposure / lambda GC interpretation

```bash
python3 interpret_plink_output.py \
    --kin0 "$OUT_DIR_GEN1/king.kin0" \
    --eigenvec "$OUT_DIR_GEN1/pca.eigenvec" \
    --metadata sample_metadata_gen1.csv \
    --exposure-col exposure_agri_score \
    --pvalues gwas_results_gen1.csv --pvalue-col p \
    --out-dir "$OUT_DIR_GEN1/diagnostics_output"
```

`--pvalues` is optional (only needed for the lambda GC / QQ-plot section).
`--metadata`/`--pvalues` must be the metadata/results for THIS cohort
only.

### 3d. Supplementary plots (missingness, sex-check, heterozygosity, MAF)

Requires step 2 to have already produced `sex_check.sexcheck`,
`heterozygosity.het`, `maf.afreq`.

```bash
python3 qc_supplementary_plots.py \
    --qc-dir "$OUT_DIR_GEN1" \
    --out-dir "$OUT_DIR_GEN1/supplementary_plots"
```

### 3e. PCA covariates for the G×E model (independent of the report — run whenever needed)

This is the file the gene-environment pipeline (`pca_utils.py` /
`modeling.py`) actually consumes — one per cohort, matched by
`cfg.generation`.

```bash
python3 extract_pca_covariates.py \
    --eigenvec "$OUT_DIR_GEN1/pca.eigenvec" \
    --n-pcs 10 \
    --strip-doubled-id \
    --out "$OUT_DIR_GEN1/pca_covariates.csv"
```

Use `--strip-doubled-id` consistently with whatever the gene-environment
dataframe's IID format actually is (doubled `NOME_NOME` vs plain `NOME`)
— see that script's `--help` for what it does and doesn't touch.

---

## 4. Assemble the Word supplementary report

Run after 3a, 3b, 3c, and 3d have all produced their output (missing
inputs are skipped with a note in the document rather than causing a
crash, so you can also run this earlier to see what's still missing).

```bash
python3 build_supplementary_report.py \
    --qc-dir "$OUT_DIR_GEN1" \
    --kinship-report-dir "$OUT_DIR_GEN1/qc_report" \
    --diagnostics-dir "$OUT_DIR_GEN1/diagnostics_output" \
    --attrition-csv "$OUT_DIR_GEN1/qc_attrition.csv" \
    --supp-plots-dir "$OUT_DIR_GEN1/supplementary_plots" \
    --out "$OUT_DIR_GEN1/Supplementary_QC_Report_gen1.docx"
```

Produces `Supplementary_QC_Report_gen1.docx`: all tables and figures in
English, numbered (Table S1…, Figure S1…), ready to paste into or attach
to the paper's Supplementary Materials. Review the auto-generated
verdicts (LMM needed? PCs in the main model?) before submission — they
are a starting point, not a final decision. Name the file per cohort
(`_gen1`/`_gen2`) so the two don't overwrite each other.

---

## Full command sequence, copy-paste order (BOTH cohorts)

```bash
# ============ DISCOVERY / VALIDATION (gen1) ============
export METADATA=/srv/python-projects/gene_environment_v2/data/componenti_ambientali_full.csv

# ============ DISCOVERY / VALIDATION (gen1) ============
export OUT_DIR=/mnt/cresla_prod/genome_datasets/qc_output_gen1

bash ./00_run_plink_qc.sh --use-filtered --jobs 16 \
    /mnt/cresla_prod/genome_datasets/gen1 \
    "$OUT_DIR"
bash ./01_run_extra_qc_checks.sh "$OUT_DIR"

python3 qc_attrition_summary.py --qc-dir "$OUT_DIR" --out "$OUT_DIR/qc_attrition.csv"
python3 qc_report.py --kin "$OUT_DIR/king.kin0" --eigenvec "$OUT_DIR/pca.eigenvec" \
    --eigenval "$OUT_DIR/pca.eigenval" \
    --vcf-dirs /mnt/cresla_prod/genome_datasets/gen1 --use-filtered \
    --out-dir "$OUT_DIR/qc_report"

# uno per ciascuna esposizione che usi nel modello
for EXPOSURE in seminativi_500 vigneti_500 risaie_500 seminativi_1000 vigneti_1000 risaie_1000 seminativi_1500 vigneti_1500 risaie_1500; do
    python3 interpret_plink_output.py --kin0 "$OUT_DIR/king.kin0" --eigenvec "$OUT_DIR/pca.eigenvec" \
        --metadata "$METADATA" --exposure-col "$EXPOSURE" \
        --out-dir "$OUT_DIR/diagnostics_output_${EXPOSURE}"
done

python3 qc_supplementary_plots.py --qc-dir "$OUT_DIR" --out-dir "$OUT_DIR/supplementary_plots"
python3 extract_pca_covariates.py --eigenvec "$OUT_DIR/pca.eigenvec" --n-pcs 10 \
    --strip-doubled-id --out "$OUT_DIR/pca_covariates.csv"

python3 build_supplementary_report.py --qc-dir "$OUT_DIR" \
    --kinship-report-dir "$OUT_DIR/qc_report" \
    --diagnostics-dir "$OUT_DIR/diagnostics_output_seminativi_1500" \
    --attrition-csv "$OUT_DIR/qc_attrition.csv" \
    --supp-plots-dir "$OUT_DIR/supplementary_plots" \
    --out "$OUT_DIR/Supplementary_QC_Report_gen1.docx"

# ============ REPLICATION (gen2) ============
export OUT_DIR=/mnt/cresla_prod/genome_datasets/qc_output_gen2

bash ./00_run_plink_qc.sh --use-filtered --jobs 16 \
    /mnt/cresla_prod/genome_datasets/gen2 \
    "$OUT_DIR"
bash ./01_run_extra_qc_checks.sh "$OUT_DIR"

python3 qc_attrition_summary.py --qc-dir "$OUT_DIR" --out "$OUT_DIR/qc_attrition.csv"
python3 qc_report.py --kin "$OUT_DIR/king.kin0" --eigenvec "$OUT_DIR/pca.eigenvec" \
    --eigenval "$OUT_DIR/pca.eigenval" \
    --vcf-dirs /mnt/cresla_prod/genome_datasets/gen2 --use-filtered \
    --out-dir "$OUT_DIR/qc_report"

for EXPOSURE in seminativi_500 vigneti_500 risaie_500 seminativi_1000 vigneti_1000 risaie_1000 seminativi_1500 vigneti_1500 risaie_1500; do
    python3 interpret_plink_output.py --kin0 "$OUT_DIR/king.kin0" --eigenvec "$OUT_DIR/pca.eigenvec" \
        --metadata "$METADATA" --exposure-col "$EXPOSURE" \
        --out-dir "$OUT_DIR/diagnostics_output_${EXPOSURE}"
done

python3 qc_supplementary_plots.py --qc-dir "$OUT_DIR" --out-dir "$OUT_DIR/supplementary_plots"
python3 extract_pca_covariates.py --eigenvec "$OUT_DIR/pca.eigenvec" --n-pcs 10 \
    --strip-doubled-id --out "$OUT_DIR/pca_covariates.csv"

python3 build_supplementary_report.py --qc-dir "$OUT_DIR" \
    --kinship-report-dir "$OUT_DIR/qc_report" \
    --diagnostics-dir "$OUT_DIR/diagnostics_output_seminativi_1500" \
    --attrition-csv "$OUT_DIR/qc_attrition.csv" \
    --supp-plots-dir "$OUT_DIR/supplementary_plots" \
    --out "$OUT_DIR/Supplementary_QC_Report_gen2.docx"

```

---

## File map (what each script reads and writes)

| Script | Reads | Writes |
|---|---|---|
| `00_run_plink_qc.sh` | raw VCFs (ONE cohort's directory) | `merged_*.{pgen,pvar,psam}`, `king.kin0`, `pca.eigenvec/.eigenval`, `missingness.*` |
| `01_run_extra_qc_checks.sh` | `merged_qc.*`, `merged_pruned.*` | `sex_check.sexcheck`, `heterozygosity.het`, `maf.afreq`, `run_metadata.txt` |
| `qc_attrition_summary.py` | `merged_all/geno/qc/pruned.*`, `pruned.prune.in` | `qc_attrition.csv`, `qc_attrition.png` |
| `qc_report.py` | `king.kin0`, `pca.eigenvec/.eigenval`, (optionally this cohort's raw VCFs) | `kinship_*.csv/png`, `pca_batch_eta2.csv`, `pca_scatter_by_batch.png`, `pca_scree_plot.png` |
| `interpret_plink_output.py` | `king.kin0`, `pca.eigenvec`, this cohort's metadata, (optionally p-values) | `diagnostics_report.txt`, `pc_exposure_correlation.csv`, `pi_hat_distribution.png`, `pca_vs_exposure.png`, `qq_plot.png` |
| `qc_supplementary_plots.py` | `missingness.*`, `sex_check.sexcheck`, `heterozygosity.het`, `maf.afreq` | `missingness_distributions.png`, `sex_check_distribution.png`, `heterozygosity_distribution.png`, `maf_spectrum.png`, flagged/outlier sample CSVs |
| `extract_pca_covariates.py` | `pca.eigenvec` | `pca_covariates.csv` (for the G×E model, not part of the QC report; use `--strip-doubled-id` consistently with the G×E dataframe's ID format) |
| `build_supplementary_report.py` | all of the above CSVs/PNGs (for THIS cohort) | `Supplementary_QC_Report_<cohort>.docx` |

## Notes

- Everything above is run **once per cohort** (gen1, gen2), each with its
  own `$OUT_DIR`. Nothing is pooled or shared between the two runs — this
  is required by the discovery/replication design (see top of this file).
  gen3 is not used anywhere in this pipeline.
- If you re-run any step with different data (new samples, changed
  thresholds), re-run every downstream step in order — nothing is
  automatically invalidated/re-triggered across scripts.
- `qc_attrition_summary.py` and `01_run_extra_qc_checks.sh` need the
  intermediate `merged_*` files from step 1 to still exist on disk. If
  you've deleted them to save space, those two steps will skip with a
  warning instead of failing.
- `build_supplementary_report.py` never recomputes anything; it only
  formats what's already there. Missing inputs show up as a highlighted
  note in the document instead of crashing the build, so you can run it
  at any point to see what's left to generate.
