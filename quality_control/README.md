# Genomic QC Pipeline — README

End-to-end genotype QC, from raw per-chromosome VCFs to a Word
supplementary report, for as many cohorts as your study has. The whole
thing runs with **one command**:

```bash
python3 run_pipeline.py --config pipeline_config.yaml
```

This document explains what that command does, how to configure it, and
how to run individual steps by hand (useful for debugging, or for
re-running just one thing without waiting for everything else).

---

## 0. Setup (once)

You need `plink2` and `bcftools` in `PATH` (normally already true inside
the conda environment used for the rest of the genetics pipeline — check
with `which plink2 bcftools`), and Python 3 with `pandas`, `numpy`,
`matplotlib`, `scipy`, `python-docx`, and `pyyaml`
(`pip install pyyaml --break-system-packages` if missing).

Copy the example config and fill in your real paths:

```bash
cp pipeline_config.example.yaml pipeline_config.yaml
```

Open `pipeline_config.yaml` and set, at minimum: each cohort's
`vcf_dirs` and `out_dir`, `metadata_csv`, and the `exposures` list. Every
field is documented inline in the file. In particular:

- **One cohort = one entry under `cohorts:`.** `vcf_dirs` is a *list*
  because a single cohort can have been genotyped in more than one raw
  VCF batch (see the note at the top of `00_run_plink_qc.sh`); almost
  always it's a list of one directory.
- **PCA must be computed separately per cohort.** If your study has a
  discovery and a replication cohort (as this one does, `gen1` /
  `gen2`), never merge them into one `cohorts:` entry — a PCA fit on
  pooled data leaks information between discovery and replication and
  defeats the point of having an independent replication set. Keep them
  as separate entries; the orchestrator already runs each one through
  its own, independent pipeline end to end.

---

## 1. Running everything: `run_pipeline.py`

```bash
python3 run_pipeline.py --config pipeline_config.yaml
```

For every cohort listed in the config, in order, this runs:

| # | Step | Script | What it does |
|---|------|--------|---------------|
| 1 | `qc` | `00_run_plink_qc.sh` | merge batches, filter, LD-prune, kinship (KING), PCA |
| 2 | `extra` | `01_run_extra_qc_checks.sh` | sex check, heterozygosity, MAF spectrum, version metadata |
| 3 | `attrition` | `qc_attrition_summary.py` | sample/variant counts at each QC stage |
| 4 | `kinship` | `qc_report.py` | kinship distribution + PCA/batch-effect plots |
| 5 | `diagnostics` | `interpret_plink_output.py` | once per exposure in the config: relatedness verdict, PCs-vs-exposure, lambda GC |
| 6 | `plots` | `qc_supplementary_plots.py` | missingness/sex-check/heterozygosity/MAF figures |
| 7 | `covariates` | `extract_pca_covariates.py` | PCs → CSV, ready to merge into the G×E model |
| 8 | `docx` | `build_supplementary_report.py` | assembles everything above into one Word report |

Steps 1–2 need the server (plink2/bcftools + access to the raw VCF
storage); steps 3–8 are pure Python and read only the files steps 1–2
already wrote to `out_dir`.

At the end, each cohort has its own `out_dir` containing everything,
including `Supplementary_QC_Report_<cohort>.docx`. A full run log is
also written to `logs/run_pipeline_<timestamp>.log`.

### Useful flags

```bash
# See every command that would run, without running anything:
python3 run_pipeline.py --config pipeline_config.yaml --dry-run

# Only one cohort:
python3 run_pipeline.py --config pipeline_config.yaml --cohorts gen1

# Only re-run the Python reporting steps (e.g. you tweaked a threshold
# and don't want to wait hours for steps 1-2 to redo the genomic work):
python3 run_pipeline.py --config pipeline_config.yaml \
    --only attrition,kinship,diagnostics,plots,covariates,docx

# Redo absolutely everything from scratch, ignoring existing output:
python3 run_pipeline.py --config pipeline_config.yaml --force

# If gen1 fails, still attempt gen2 instead of stopping:
python3 run_pipeline.py --config pipeline_config.yaml --keep-going
```

`--only` and `--skip` are mutually exclusive; both take a comma-separated
subset of: `qc,extra,attrition,kinship,diagnostics,plots,covariates,docx`.

### Resume behavior

Steps 1–2 (`00_run_plink_qc.sh`, `01_run_extra_qc_checks.sh`) check,
before doing any work, whether their own final output already exists,
and skip it if so — this is the expensive, multi-hour part, and it
resumes automatically after a crash or an interrupted run. Steps 3–8 are
fast (seconds) and always recompute, so re-running the whole pipeline
after steps 1–2 have completed just cheaply refreshes the tables,
figures, and Word report. Use `--force` to ignore all of this and redo
every step, including 1–2, from scratch.

The "output already exists" check for steps 1–2 looks at whether the
expected output file is present, not whether its content is complete —
if a step was killed mid-write, use `--force` or delete the suspect file
by hand before re-running.

---

## 2. Running individual steps by hand

Every script under here also works stand-alone with its own `--help`;
this is what `run_pipeline.py` is calling under the hood, so it's useful
for debugging a single step, or for a one-off analysis outside the
config. Example, for one cohort's `out_dir` after step 1–2 have already
run:

```bash
python3 qc_attrition_summary.py --qc-dir "$OUT_DIR" --out "$OUT_DIR/qc_attrition.csv"

python3 qc_report.py \
    --kin "$OUT_DIR/king.kin0" --eigenvec "$OUT_DIR/pca.eigenvec" --eigenval "$OUT_DIR/pca.eigenval" \
    --vcf-dirs "$VCF_DIR" --use-filtered --out-dir "$OUT_DIR/qc_report"

python3 interpret_plink_output.py \
    --kin0 "$OUT_DIR/king.kin0" --eigenvec "$OUT_DIR/pca.eigenvec" \
    --metadata metadata.csv --exposure-col my_exposure --strip-doubled-id \
    --out-dir "$OUT_DIR/diagnostics_output_my_exposure"

python3 qc_supplementary_plots.py --qc-dir "$OUT_DIR" --out-dir "$OUT_DIR/supplementary_plots"

python3 extract_pca_covariates.py \
    --eigenvec "$OUT_DIR/pca.eigenvec" --n-pcs 10 --strip-doubled-id \
    --out "$OUT_DIR/pca_covariates.csv"

python3 build_supplementary_report.py \
    --qc-dir "$OUT_DIR" \
    --diagnostics-dir "$OUT_DIR/diagnostics_output_my_exposure" \
    --cohort-label "gen1 (discovery)" \
    --out "$OUT_DIR/Supplementary_QC_Report.docx"
```

---

## 3. Shared module: `plink_io.py`

Every Python script above imports its plink2-table reading and
doubled-ID stripping logic from `plink_io.py` instead of re-implementing
it. Two things live there worth knowing about if you're extending the
pipeline:

- **`strip_doubled_id(s)`** reduces an IID of the form `NAME_NAME` (the
  two halves identical, typical when a VCF had `FamilyID == IndividualID`
  and got a doubled ID) down to `NAME`. It leaves anything else alone.
  This is **opt-in everywhere** (`--strip-doubled-id`) — it is never
  applied silently. Both `interpret_plink_output.py` and
  `extract_pca_covariates.py` take this flag; use the *same* setting for
  both, for the same cohort, since they need to agree on what an IID
  looks like when merging against your metadata.
- **`read_plink_table` / `load_eigenvec` / `load_kinship`** centralize
  the "strip the leading `#` plink2 puts on header columns" logic and a
  couple of column-name normalizations (e.g. `ID1`/`ID2` → `IID1`/`IID2`
  in `.kin0`). If a future plink2 version changes a column name again,
  fix it once here.

---

## 4. How `build_supplementary_report.py` gets its numbers

`interpret_plink_output.py` writes two files per run:
`diagnostics_report.txt` (human-readable log) and
`diagnostics_summary.json` (the same key numbers, structured).
`build_supplementary_report.py` reads **only the JSON**. This means a
future wording change to a log line in `interpret_plink_output.py`
cannot silently drop a number from the Word report — if the JSON key
isn't there, the corresponding section prints an explicit
"[Not available: ...]" note instead of a wrong or missing paragraph.

`primary_exposure_for_report` in the config picks which
`diagnostics_output_<exposure>/` folder feeds that report (relatedness,
PC-exposure correlation, lambda GC sections) — all `exposures` are still
run and get their own diagnostics folder, this setting just decides
which one's numbers go in the Word document.