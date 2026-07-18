# Genomic QC Pipeline — README

This describes the full order of execution, from raw VCFs to the final
Word supplementary report. Every script is resumable / non-destructive:
re-running a step after a crash skips work that is already done (see each
script's own header for details), and none of the Python reporting
scripts recompute anything — they only read files produced by earlier
steps.

All commands below assume you are on the server that has `plink2` and
`bcftools` in `PATH` (the `geneenv` conda environment), and that
`$OUT_DIR` is the same folder passed to every step.

```bash
export OUT_DIR=/mnt/cresla_prod/genome_datasets/qc_output
```

---

## 1. Main QC pipeline (bash, on the server)

Merges batches, filters, LD-prunes, computes kinship and PCA.

```bash
./00_run_plink_qc.sh --use-filtered --jobs 16 \
    /mnt/cresla_prod/genome_datasets/gen1 \
    /mnt/cresla_prod/genome_datasets/gen2 \
    "$OUT_DIR"
```

Produces (among others): `king.kin0`, `pca.eigenvec`, `pca.eigenval`,
`merged_all/geno/qc/pruned.*`, `missingness.vmiss`, `missingness.smiss`.

---

## 2. Extra QC checks (bash, on the server — run after step 1)

Sex check, heterozygosity check, MAF spectrum, and a software/version log.
Does **not** touch `00_run_plink_qc.sh` or its output; it only adds new
files to the same `$OUT_DIR`.

```bash
./01_run_extra_qc_checks.sh "$OUT_DIR"
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
    --qc-dir "$OUT_DIR" \
    --out "$OUT_DIR/qc_attrition.csv"
```

### 3b. Kinship + PCA/batch-effect report

```bash
python3 qc_report.py \
    --kin "$OUT_DIR/king.kin0" \
    --eigenvec "$OUT_DIR/pca.eigenvec" \
    --eigenval "$OUT_DIR/pca.eigenval" \
    --vcf-dirs /mnt/cresla_prod/genome_datasets/gen1 \
               /mnt/cresla_prod/genome_datasets/gen2 \
               /mnt/cresla_prod/genome_datasets/gen3 \
    --use-filtered \
    --out-dir "$OUT_DIR/qc_report"
```

Omit `--vcf-dirs`/`--use-filtered` if the original VCF directories are no
longer accessible from where you're running this — you'll get kinship
diagnostics and a plain (uncolored) PCA scatter instead of the
batch-colored one.

### 3c. Relatedness / PC-vs-exposure / lambda GC interpretation

```bash
python3 interpret_plink_output.py \
    --kin0 "$OUT_DIR/king.kin0" \
    --eigenvec "$OUT_DIR/pca.eigenvec" \
    --metadata sample_metadata.csv \
    --exposure-col exposure_agri_score \
    --pvalues gwas_results.csv --pvalue-col p \
    --out-dir "$OUT_DIR/diagnostics_output"
```

`--pvalues` is optional (only needed for the lambda GC / QQ-plot section).

### 3d. Supplementary plots (missingness, sex-check, heterozygosity, MAF)

Requires step 2 to have already produced `sex_check.sexcheck`,
`heterozygosity.het`, `maf.afreq`.

```bash
python3 qc_supplementary_plots.py \
    --qc-dir "$OUT_DIR" \
    --out-dir "$OUT_DIR/supplementary_plots"
```

### 3e. PCA covariates for the G×E model (independent of the report — run whenever needed)

```bash
python3 extract_pca_covariates.py \
    --eigenvec "$OUT_DIR/pca.eigenvec" \
    --n-pcs 10 \
    --out "$OUT_DIR/pca_covariates.csv"
```

---

## 4. Assemble the Word supplementary report

Run after 3a, 3b, 3c, and 3d have all produced their output (missing
inputs are skipped with a note in the document rather than causing a
crash, so you can also run this earlier to see what's still missing).

```bash
python3 build_supplementary_report.py \
    --qc-dir "$OUT_DIR" \
    --kinship-report-dir "$OUT_DIR/qc_report" \
    --diagnostics-dir "$OUT_DIR/diagnostics_output" \
    --attrition-csv "$OUT_DIR/qc_attrition.csv" \
    --supp-plots-dir "$OUT_DIR/supplementary_plots" \
    --out "$OUT_DIR/Supplementary_QC_Report.docx"
```

Produces `Supplementary_QC_Report.docx`: all tables and figures in
English, numbered (Table S1…, Figure S1…), ready to paste into or attach
to the paper's Supplementary Materials. Review the auto-generated
verdicts (LMM needed? PCs in the main model?) before submission — they
are a starting point, not a final decision.

---

## Full command sequence, copy-paste order

```bash
export OUT_DIR=/mnt/cresla_prod/genome_datasets/qc_output

./00_run_plink_qc.sh --use-filtered --jobs 16 gen1_dir gen2_dir gen3_dir "$OUT_DIR"
./01_run_extra_qc_checks.sh "$OUT_DIR"

python3 qc_attrition_summary.py --qc-dir "$OUT_DIR" --out "$OUT_DIR/qc_attrition.csv"
python3 qc_report.py --kin "$OUT_DIR/king.kin0" --eigenvec "$OUT_DIR/pca.eigenvec" \
    --eigenval "$OUT_DIR/pca.eigenval" --out-dir "$OUT_DIR/qc_report"
python3 interpret_plink_output.py --kin0 "$OUT_DIR/king.kin0" --eigenvec "$OUT_DIR/pca.eigenvec" \
    --metadata sample_metadata.csv --exposure-col exposure_agri_score \
    --out-dir "$OUT_DIR/diagnostics_output"
python3 qc_supplementary_plots.py --qc-dir "$OUT_DIR" --out-dir "$OUT_DIR/supplementary_plots"
python3 extract_pca_covariates.py --eigenvec "$OUT_DIR/pca.eigenvec" --n-pcs 10 \
    --out "$OUT_DIR/pca_covariates.csv"

python3 build_supplementary_report.py --qc-dir "$OUT_DIR" \
    --kinship-report-dir "$OUT_DIR/qc_report" \
    --diagnostics-dir "$OUT_DIR/diagnostics_output" \
    --attrition-csv "$OUT_DIR/qc_attrition.csv" \
    --supp-plots-dir "$OUT_DIR/supplementary_plots" \
    --out "$OUT_DIR/Supplementary_QC_Report.docx"
```

---

## File map (what each script reads and writes)

| Script | Reads | Writes |
|---|---|---|
| `00_run_plink_qc.sh` | raw VCFs | `merged_*.{pgen,pvar,psam}`, `king.kin0`, `pca.eigenvec/.eigenval`, `missingness.*` |
| `01_run_extra_qc_checks.sh` | `merged_qc.*`, `merged_pruned.*` | `sex_check.sexcheck`, `heterozygosity.het`, `maf.afreq`, `run_metadata.txt` |
| `qc_attrition_summary.py` | `merged_all/geno/qc/pruned.*`, `pruned.prune.in` | `qc_attrition.csv`, `qc_attrition.png` |
| `qc_report.py` | `king.kin0`, `pca.eigenvec/.eigenval`, (optionally raw VCFs) | `kinship_*.csv/png`, `pca_batch_eta2.csv`, `pca_scatter_by_batch.png`, `pca_scree_plot.png` |
| `interpret_plink_output.py` | `king.kin0`, `pca.eigenvec`, metadata, (optionally p-values) | `diagnostics_report.txt`, `pc_exposure_correlation.csv`, `pi_hat_distribution.png`, `pca_vs_exposure.png`, `qq_plot.png` |
| `qc_supplementary_plots.py` | `missingness.*`, `sex_check.sexcheck`, `heterozygosity.het`, `maf.afreq` | `missingness_distributions.png`, `sex_check_distribution.png`, `heterozygosity_distribution.png`, `maf_spectrum.png`, flagged/outlier sample CSVs |
| `extract_pca_covariates.py` | `pca.eigenvec` | `pca_covariates.csv` (for the G×E model, not part of the QC report) |
| `build_supplementary_report.py` | all of the above CSVs/PNGs | `Supplementary_QC_Report.docx` |

## Notes

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
