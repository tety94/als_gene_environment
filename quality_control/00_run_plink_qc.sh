#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# 00_run_plink_qc.sh
# ============================================================================
# QC pipeline on already-existing VCFs for ONE cohort: filters to common
# biallelic SNPs, does LD pruning, computes relatedness (KING-robust, via
# plink2) and PCA.
#
# RUN THIS ON THE SERVER that has plink2 and bcftools: this is genomic
# compute, it does not run in a sandbox without access to the raw VCF
# storage.
#
# REQUIREMENTS: plink2 and bcftools in PATH (normally already present in
# the conda environment used for the rest of the genetics pipeline --
# check with: which plink2 bcftools
#
# WHAT "BATCH" MEANS HERE (do not confuse with "cohort"): a single study
# cohort can itself have been genotyped in more than one raw-VCF batch
# (e.g. two sequencing waves for the same cohort, each delivered as its
# own directory of per-chromosome VCFs). This script accepts one or more
# input directories and merges them into a single cohort dataset. If your
# cohort was genotyped in a single batch, just pass one directory -- most
# of the time that is exactly what happens, and each run of this script
# corresponds to exactly one cohort. This script has no built-in notion
# of "cohort" beyond "whatever you pass it in one invocation, with one
# --out_dir": how many cohorts your study has, and which directories
# belong to which cohort, is decided by the pipeline config / orchestrator
# that calls this script (see run_pipeline.py and the root README), not
# by this script itself.
#
# NOTE ON STRUCTURE (IMPORTANT, learned after a real production error):
#   plink2 --pmerge-list handles ONE kind of merge well at a time: either
#   merging samples (same variants, different samples) or concatenating
#   (same samples, different variants). Here we need BOTH at once (N
#   batches = different samples, 22 chromosomes = different variants),
#   and that combination triggers "Error: Non-concatenating
#   --pmerge[-list] is under development." -- a known plink2 limitation,
#   not a bug in your data.
#
#   Because of that, ALL merging (across batches and across chromosomes)
#   is done in bcftools, which natively handles both cases, and we
#   convert to pgen ONCE at the end, on the already-complete genome-wide
#   VCF:
#     Step 1: filter (if needed) + index, per batch x chromosome
#     Step 2: bcftools merge across batches, per chromosome (parallel)
#     Step 3: bcftools concat of the 22 chromosomes -> one genome-wide VCF
#     Step 4: ONE plink2 --vcf ... --make-pgen conversion
#     Step 5: diagnostics + missingness filter (--geno/--mind) -> merged_qc
#     Step 6: LD pruning
#     Step 7: relatedness (KING)
#     Step 8: PCA
#
# WHAT CHANGED vs. the first version of this script:
#   - Step 1 and Step 2 are parallelized with xargs -P (up to 16 workers).
#   - All output (stdout+stderr) goes both to screen and to
#     $OUT_DIR/logs/pipeline.log (via tee).
#   - Every Step 1 and Step 2 job writes its own dedicated log file in
#     $OUT_DIR/logs/, so a failure inside the parallel run is easy to
#     trace back to a specific batch/chromosome.
#
# RESUME (continue where you left off): EVERY step checks, before running,
# whether its own final output already exists, and if so SKIPS it
# (logging "[skip, already present]") instead of redoing it. This applies
# both to the individual parallel jobs of Step 1/2 (an already-completed
# batch/chromosome is not redone) and to the sequential Steps 3-7. If you
# re-run this script after an interruption (crash, server reboot, Ctrl-C),
# it automatically resumes from the first missing output. Use --force to
# ignore all of this and redo everything from scratch.
#
# NOTE: the "existence" check is based on the final output file (plus its
# companion/index file, e.g. .vcf.gz + .tbi, or .pgen + .pvar + .psam)
# being present, NOT on a content-integrity check. If a step is killed
# HALFWAY through writing (e.g. kill -9 in the middle of a bcftools
# merge), the file may exist but be truncated/corrupt without resume
# noticing. In that case use --force, or manually delete the suspect
# output before re-running.
#
# USAGE:
#   ./00_run_plink_qc.sh [--use-filtered] [--jobs N] [--force] <vcf_dir_1> [<vcf_dir_2> ...] <out_dir>
#
# --use-filtered: if present, the script looks for *_filtered.vcf.gz
#   inside a vcf_filtered/ subfolder of each input directory and SKIPS
#   the bcftools view -m2 -M2 --min-af step, assuming that filter was
#   already applied upstream. Only use this after verifying the upstream
#   filter already includes biallelic-only + a reasonable MAF threshold.
# --jobs N: number of parallel workers for Step 1 and Step 2 (default 16).
# --force: ignore all existing output and redo the entire pipeline from
#   scratch, overwriting.
#
# FILTER PARAMETERS (environment variables, not flags -- defaults match
# what the rest of the genetics pipeline already uses elsewhere, for
# consistency):
#   MAF_THRESHOLD   (default 0.01) MAF threshold, used both in the
#                   bcftools view filter of Step 1 (only if NOT
#                   --use-filtered) and in the pruning/extraction of
#                   Step 6.
#   LD_WINDOW_SIZE  (default 50)   window size for --indep-pairwise.
#   LD_STEP         (default 5)    step size for --indep-pairwise.
#   LD_R2_THRESHOLD (default 0.5)  r2 threshold for --indep-pairwise.
#   GENO_THRESH     (default 0.05) per-variant missingness threshold (Step 5).
#   MIND_THRESH     (default 0.05) per-sample missingness threshold (Step 5).
#   EXCLUDE_ID_PREFIXES (default empty) sample-ID prefixes to EXCLUDE
#                   entirely from the QC pipeline (kinship, PCA, etc.),
#                   comma-separated, e.g. "ACH" or "ACH,XYZ". Same name
#                   and purpose as the EXCLUDE_ID_PREFIXES variable used
#                   upstream of this cohort's genotype data: if those
#                   samples are excluded before reaching the downstream
#                   gene-environment model (because absent from the
#                   exposure file, or for any other reason), they must be
#                   excluded here too -- otherwise kinship, missingness
#                   and PCA end up computed on a broader cohort than the
#                   one actually analyzed, with numbers (sample size,
#                   percentages) that will not match what ends up in the
#                   paper. Applied per batch, based on the sample IDs of
#                   that batch's chr1 VCF.
# Example of overriding them:
#   MAF_THRESHOLD=0.01 LD_R2_THRESHOLD=0.5 ./00_run_plink_qc.sh --use-filtered ...
#   EXCLUDE_ID_PREFIXES=ACH ./00_run_plink_qc.sh --use-filtered ...
#
# Example, single-batch cohort, already-filtered VCFs, 16 workers:
#   ./00_run_plink_qc.sh --use-filtered --jobs 16 \
#       /mnt/genome_datasets/cohortA \
#       /mnt/genome_datasets/qc_output_cohortA
#
# IMPORTANT NOTE: if the same patient appears in more than one batch,
# plink2/the merge will misbehave (duplicate IDs, or false "twins" in the
# kinship report if the IDs were made unique downstream). Step 0 below
# automatically checks for sample-ID overlap across the batches passed in
# this invocation.
#
# NOTE ON THE BATCH MERGE: bcftools merge takes the union of sites across
# batches; a sample that has no genotype at a site present only in
# another batch is filled in as missing (./.) rather than excluded. With
# different MAF thresholds or slightly different calling pipelines
# between batches this can introduce non-trivial missingness at some
# sites, which artificially inflates kinship estimates. This is why
# Step 5 applies an explicit --geno/--mind filter (with a diagnostic
# report) AFTER the merge and BEFORE pruning/kinship/PCA -- see the
# comments in Step 5 further below.
#
# NOTE ON PARALLELISM: watch I/O. On shared network storage, 16 parallel
# workers can saturate the disk before the CPU. If the server slows down,
# lower --jobs (e.g. 8).
# ============================================================================

USE_FILTERED=0
JOBS=16
FORCE=0
POSITIONAL=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --use-filtered)
            USE_FILTERED=1
            shift
            ;;
        --jobs)
            JOBS="$2"
            shift 2
            ;;
        --force)
            FORCE=1
            shift
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done
set -- "${POSITIONAL[@]}"

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 [--use-filtered] [--jobs N] [--force] <vcf_dir_1> [<vcf_dir_2> ...] <out_dir>"
    exit 1
fi

ARGS=("$@")
OUT_DIR="${ARGS[-1]}"
VCF_DIRS=("${ARGS[@]:0:${#ARGS[@]}-1}")

mkdir -p "$OUT_DIR" "$OUT_DIR/logs" "$OUT_DIR/filtered" "$OUT_DIR/merged"
cd "$OUT_DIR"

# From here on, all stdout+stderr goes both to screen and to the main log.
LOGFILE="$OUT_DIR/logs/pipeline.log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "==> Starting pipeline: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Mode: $([ $USE_FILTERED -eq 1 ] && echo 'USING ALREADY-FILTERED VCFs (skip bcftools view)' || echo 'filtering MAF/biallelic from scratch')"
echo "Parallel workers: $JOBS"
echo "Resume: $([ $FORCE -eq 1 ] && echo 'DISABLED (--force: redoing everything from scratch)' || echo 'active (skipping steps whose output already exists)')"
echo "Main log: $LOGFILE"

# MAF/LD-pruning filter parameters, overridable via environment variables.
# Defaults match the values already used elsewhere in the study's
# genetics pipeline, for consistency:
#   MAF_THRESHOLD=0.01 LD_WINDOW_SIZE=50 LD_STEP=5 LD_R2_THRESHOLD=0.5
MAF_THRESHOLD="${MAF_THRESHOLD:-0.01}"
LD_WINDOW_SIZE="${LD_WINDOW_SIZE:-50}"
LD_STEP="${LD_STEP:-5}"
LD_R2_THRESHOLD="${LD_R2_THRESHOLD:-0.5}"
EXCLUDE_ID_PREFIXES="${EXCLUDE_ID_PREFIXES:-}"
export MAF_THRESHOLD LD_WINDOW_SIZE LD_STEP LD_R2_THRESHOLD EXCLUDE_ID_PREFIXES
echo "MAF filter: $MAF_THRESHOLD | LD pruning: window $LD_WINDOW_SIZE, step $LD_STEP, r2 < $LD_R2_THRESHOLD"
if [ -n "$EXCLUDE_ID_PREFIXES" ]; then
    echo "Sample exclusion by ID prefix: $EXCLUDE_ID_PREFIXES"
else
    echo "Sample exclusion by ID prefix: none (EXCLUDE_ID_PREFIXES not set)"
fi

echo "==> Checking required tools"
command -v plink2 >/dev/null 2>&1 || { echo "ERROR: plink2 not found in PATH."; exit 1; }
command -v bcftools >/dev/null 2>&1 || { echo "ERROR: bcftools not found in PATH."; exit 1; }
command -v xargs >/dev/null 2>&1 || { echo "ERROR: xargs not found in PATH."; exit 1; }

echo "==> Step 0: checking sample-ID overlap across batches (${VCF_DIRS[*]})"
: > "$OUT_DIR/all_sample_ids.txt"
for d in "${VCF_DIRS[@]}"; do
    batch=$(basename "$d")
    if [ "$USE_FILTERED" -eq 1 ]; then
        search_dir="$d/vcf_filtered"
        first_vcf=$(ls "$search_dir"/*chr1_filtered.vcf.gz 2>/dev/null | head -n1)
    else
        search_dir="$d"
        first_vcf=$(ls "$d"/*chr1.vcf.gz 2>/dev/null | head -n1)
    fi
    if [ -z "$first_vcf" ]; then
        echo "  WARNING: no chr1 file found in $search_dir, skipping it for the ID check."
        continue
    fi
    n_samples=$(bcftools query -l "$first_vcf" | wc -l)
    n_dup=$(bcftools query -l "$first_vcf" | sort | uniq -d | wc -l)

    # Exclusion file for THIS batch: one ID per line, used both here (for
    # the count/overlap check) and in Step 1 (process_one) to physically
    # exclude these samples from the VCFs before the merge.
    exclude_file="$OUT_DIR/exclude_samples_${batch}.txt"
    : > "$exclude_file"
    if [ -n "$EXCLUDE_ID_PREFIXES" ]; then
        pattern=""
        IFS=',' read -ra PREFIXES <<< "$EXCLUDE_ID_PREFIXES"
        for p in "${PREFIXES[@]}"; do
            pattern="${pattern}^${p}|"
        done
        pattern="${pattern%|}"
        bcftools query -l "$first_vcf" | grep -E "$pattern" > "$exclude_file" || true
    fi
    n_excluded=$(wc -l < "$exclude_file")
    n_kept=$((n_samples - n_excluded))

    if [ "$n_excluded" -gt 0 ]; then
        echo "  $d -> $n_samples samples (chr1), $n_dup internal duplicate IDs, $n_excluded excluded by prefix ($EXCLUDE_ID_PREFIXES) -> $n_kept in the QC cohort"
    else
        echo "  $d -> $n_samples samples (chr1), $n_dup internal duplicate IDs"
    fi

    bcftools query -l "$first_vcf" | grep -vxFf "$exclude_file" >> "$OUT_DIR/all_sample_ids.txt" || true
done
n_total=$(wc -l < "$OUT_DIR/all_sample_ids.txt")
n_unique=$(sort -u "$OUT_DIR/all_sample_ids.txt" | wc -l)
n_overlap=$((n_total - n_unique))
echo "  Total IDs in the QC cohort, post-exclusion (sum over batches): $n_total | unique IDs: $n_unique | overlap: $n_overlap"
if [ "$n_overlap" -gt 0 ]; then
    echo "  >>> WARNING: $n_overlap sample IDs repeat across batches."
    echo "  >>> If these are the same patient genotyped more than once, deduplicate"
    echo "  >>> BEFORE relatedness/kinship, or you will see false 'twins'."
fi

echo ""
if [ "$USE_FILTERED" -eq 1 ]; then
    echo "==> Step 1: using already-filtered VCFs (skip bcftools view), indexing for the merge"
else
    echo "==> Step 1: filtering to common biallelic SNPs (MAF >= 0.05) per chromosome, per batch, indexing"
fi
echo "    (parallelized over $JOBS workers; per-job logs in $OUT_DIR/logs/step1_*.log)"

# ---------------------------------------------------------------------------
# Step 1 worker: prepares the merge-ready VCF for one batch/chromosome
# (filtering if necessary) and indexes it (.tbi). No longer converts to
# pgen here: that conversion happens once at the end of the pipeline, on
# the already-merged genome-wide VCF. Prints the path of the ready VCF to
# stdout, prefixed by the chromosome number, so the caller can group by
# chromosome in Step 2.
# ---------------------------------------------------------------------------
process_one() {
    local d="$1" chr="$2" use_filtered="$3" out_dir="$4" force="$5"
    local batch vcf_in out_prefix log_file merge_vcf exclude_file exclude_args
    batch=$(basename "$d")
    out_prefix="$out_dir/filtered/${batch}_chr${chr}"
    merge_vcf="${out_prefix}.merge_input.vcf.gz"
    log_file="$out_dir/logs/step1_${batch}_chr${chr}.log"
    exclude_file="$out_dir/exclude_samples_${batch}.txt"

    if [ "$force" -ne 1 ] && [ -f "$merge_vcf" ] && [ -f "${merge_vcf}.tbi" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$batch] chr${chr}: [skip, already present]" >> "$log_file"
        echo "${chr} ${merge_vcf}"
        return 0
    fi

    {
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$batch] chr${chr}: starting"
        if [ "$use_filtered" -eq 1 ]; then
            vcf_in=$(ls "$d/vcf_filtered"/*chr${chr}_filtered.vcf.gz 2>/dev/null | head -n1)
        else
            vcf_in=$(ls "$d"/*chr${chr}.vcf.gz 2>/dev/null | head -n1)
        fi

        if [ -z "$vcf_in" ]; then
            echo "[skip] chr${chr} not found for $batch"
            exit 0
        fi

        # Exclusion by sample-ID prefix (EXCLUDE_ID_PREFIXES, see Step 0):
        # if the file exists and is NOT empty, pass -S ^file to bcftools
        # in BOTH modes (filtered/unfiltered) -- otherwise these samples
        # would remain in the VCF passed to the merge despite being
        # excluded downstream of this QC pipeline, leaving the QC cohort
        # (kinship/PCA/missingness) larger than the one actually analyzed
        # by the model.
        exclude_args=()
        if [ -s "$exclude_file" ]; then
            exclude_args=(-S "^${exclude_file}")
        fi

        if [ "$use_filtered" -eq 1 ]; then
            if [ "${#exclude_args[@]}" -eq 0 ]; then
                echo "  already filtered: creating symlink (no copy, no extra filtering)"
                ln -sf "$(readlink -f "$vcf_in")" "$merge_vcf"
            else
                echo "  already filtered, but excluding $(wc -l < "$exclude_file") samples by ID prefix (bcftools view -S)"
                bcftools view "${exclude_args[@]}" "$vcf_in" -Oz -o "$merge_vcf"
            fi
        else
            if [ "${#exclude_args[@]}" -eq 0 ]; then
                echo "  filtering (biallelic, MAF >= $MAF_THRESHOLD)"
            else
                echo "  filtering (biallelic, MAF >= $MAF_THRESHOLD), excluding $(wc -l < "$exclude_file") samples by ID prefix"
            fi
            bcftools view -m2 -M2 -v snps --min-af "${MAF_THRESHOLD}:minor" "${exclude_args[@]}" "$vcf_in" -Oz -o "$merge_vcf"
        fi

        echo "  indexing"
        bcftools index -t -f "$merge_vcf"

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$batch] chr${chr}: done"
        echo "MERGE_VCF_OK:$merge_vcf"
    } > "$log_file" 2>&1

    if [ -f "${merge_vcf}.tbi" ]; then
        # format: <chr> <path>   -- the caller groups by chr
        echo "${chr} ${merge_vcf}"
    fi
}
export -f process_one

JOBLIST1="$OUT_DIR/logs/step1_joblist.txt"
: > "$JOBLIST1"
for d in "${VCF_DIRS[@]}"; do
    for chr in $(seq 1 22); do
        echo "$d|$chr" >> "$JOBLIST1"
    done
done
echo "  Total Step 1 jobs: $(wc -l < "$JOBLIST1") (batches x 22 chromosomes)"

STEP1_OUT="$OUT_DIR/logs/step1_output.txt"
cat "$JOBLIST1" | xargs -P "$JOBS" -I{} bash -c '
    IFS="|" read -r d chr <<< "{}"
    process_one "$d" "$chr" "'"$USE_FILTERED"'" "'"$OUT_DIR"'" "'"$FORCE"'"
' > "$STEP1_OUT"

n_ok=$(grep -c . "$STEP1_OUT" || true)
echo "  Jobs completed with a merge-ready VCF: $n_ok"
echo "  If this is lower than expected (batches x 22), check the logs in"
echo "  $OUT_DIR/logs/step1_*.log for missing chromosomes/batches (likely [skip])."

echo ""
echo "==> Step 2: merging batches, per chromosome (bcftools merge, parallel over $JOBS workers)"
echo "    (per-chromosome logs in $OUT_DIR/logs/step2_chr*.log)"

merge_chr() {
    local chr="$1" out_dir="$2" force="$3"
    shift 3
    local vcfs=("$@")
    local out_vcf="$out_dir/merged/merged_chr${chr}.vcf.gz"
    local log_file="$out_dir/logs/step2_chr${chr}.log"

    if [ "$force" -ne 1 ] && [ -f "$out_vcf" ] && [ -f "${out_vcf}.tbi" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] chr${chr}: [skip, already present]" >> "$log_file"
        echo "$out_vcf"
        return 0
    fi

    {
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] chr${chr}: merging ${#vcfs[@]} batch(es)"
        if [ "${#vcfs[@]}" -eq 1 ]; then
            echo "  only one batch present for this chromosome, copying (no merge needed)"
            ln -sf "$(readlink -f "${vcfs[0]}")" "$out_vcf"
        else
            bcftools merge -m none -Oz -o "$out_vcf" "${vcfs[@]}"
        fi
        bcftools index -t -f "$out_vcf"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] chr${chr}: done"
        echo "MERGE_CHR_OK:$out_vcf"
    } > "$log_file" 2>&1

    if [ -f "${out_vcf}.tbi" ]; then
        echo "$out_vcf"
    fi
}
export -f merge_chr

# Group Step 1's VCFs by chromosome (also handles a batch missing some
# chromosome).
JOBLIST2="$OUT_DIR/logs/step2_joblist.txt"
: > "$JOBLIST2"
for chr in $(seq 1 22); do
    files=$(awk -v c="$chr" '$1 == c { $1=""; sub(/^ /,""); print }' "$STEP1_OUT" | tr '\n' ' ' | sed 's/ *$//')
    if [ -n "$files" ]; then
        echo "${chr}|${files}" >> "$JOBLIST2"
    else
        echo "  WARNING: no VCF available for chr${chr}, skipping this chromosome entirely."
    fi
done

MERGED_CHR_LIST="$OUT_DIR/logs/step2_output.txt"
cat "$JOBLIST2" | xargs -P "$JOBS" -I{} bash -c '
    IFS="|" read -r chr files <<< "{}"
    merge_chr "$chr" "'"$OUT_DIR"'" "'"$FORCE"'" $files
' > "$MERGED_CHR_LIST"

n_chr_ok=$(grep -c . "$MERGED_CHR_LIST" || true)
echo "  Chromosomes merged successfully: $n_chr_ok / 22"

echo ""
echo "==> Step 3: concatenating the 22 chromosomes into one genome-wide VCF"
if [ "$FORCE" -ne 1 ] && [ -f "$OUT_DIR/merged_all.vcf.gz" ] && [ -f "$OUT_DIR/merged_all.vcf.gz.tbi" ]; then
    echo "  [skip, already present] $OUT_DIR/merged_all.vcf.gz"
else
    CONCAT_LIST="$OUT_DIR/logs/concat_list.txt"
    : > "$CONCAT_LIST"
    for chr in $(seq 1 22); do
        f="$OUT_DIR/merged/merged_chr${chr}.vcf.gz"
        if [ -f "$f" ]; then
            echo "$f" >> "$CONCAT_LIST"
        fi
    done
    bcftools concat -f "$CONCAT_LIST" -Oz -o "$OUT_DIR/merged_all.vcf.gz"
    bcftools index -t -f "$OUT_DIR/merged_all.vcf.gz"
    echo "  Genome-wide VCF: $OUT_DIR/merged_all.vcf.gz"
fi

echo ""
echo "==> Step 4: converting to pgen (once, on the already-complete VCF)"
echo "    + assigning unique variant IDs (chr:pos:ref:alt) and dropping exact"
echo "    duplicates: needed because --indep-pairwise requires unique IDs, and"
echo "    raw VCFs often have '.' as the ID for (almost) every variant."
if [ "$FORCE" -ne 1 ] && [ -f "$OUT_DIR/merged_all.pgen" ] && [ -f "$OUT_DIR/merged_all.pvar" ] && [ -f "$OUT_DIR/merged_all.psam" ]; then
    echo "  [skip, already present] $OUT_DIR/merged_all.pgen"
else
    plink2 --vcf "$OUT_DIR/merged_all.vcf.gz" \
           --double-id \
           --set-all-var-ids '@:#:$r:$a' \
           --new-id-max-allele-len 200 truncate \
           --rm-dup force-first \
           --make-pgen \
           --out "$OUT_DIR/merged_all"
fi

echo ""
echo "==> Step 5: missingness diagnostics + quality filter (--geno then --mind)"
echo "    bcftools merge takes the union of sites across batches: a variant"
echo "    present in only 1 of N batches is genotyped only in that batch's"
echo "    samples and missing in the rest, despite having a perfectly normal"
echo "    apparent MAF -- --maf alone does NOT catch this. This artificially"
echo "    inflates estimated kinship, especially between samples from"
echo "    different batches."
echo ""
echo "    IMPORTANT: --geno and --mind must be applied in TWO SEQUENTIAL"
echo "    plink2 calls, not one. If run together, plink2 computes per-sample"
echo "    missingness (--mind) on the dataset BEFORE it has been cleaned of"
echo "    batch-unbalanced variants: an entire batch can end up with"
echo "    artificially high aggregate missingness (because it 'misses' every"
echo "    variant called only in the other batch(es)) and get almost"
echo "    entirely dropped BEFORE --geno has removed those variants. So here"
echo "    --geno is applied first (cleans the variants), then, ON THE"
echo "    ALREADY-CLEANED DATASET, --mind (now per-sample missingness"
echo "    reflects real coverage, not the batch artifact)."
echo ""
echo "    Default threshold 0.05 for both. Change via GENO_THRESH/MIND_THRESH"
echo "    environment variables before running the script, e.g.:"
echo "      GENO_THRESH=0.02 MIND_THRESH=0.05 ./00_run_plink_qc.sh ..."
GENO_THRESH="${GENO_THRESH:-0.05}"
MIND_THRESH="${MIND_THRESH:-0.05}"
echo "    Thresholds in use: --geno $GENO_THRESH  --mind $MIND_THRESH"

if [ "$FORCE" -ne 1 ] && [ -f "$OUT_DIR/missingness.vmiss" ] && [ -f "$OUT_DIR/missingness.smiss" ]; then
    echo "  [skip, already present] $OUT_DIR/missingness.vmiss / .smiss"
else
    plink2 --pfile "$OUT_DIR/merged_all" \
           --missing \
           --out "$OUT_DIR/missingness"
fi
echo "  Diagnostic report (BEFORE filtering): $OUT_DIR/missingness.vmiss (per variant),"
echo "  $OUT_DIR/missingness.smiss (per sample). If you want to pick the threshold by eye"
echo "  instead of using the 0.05 default, look at the F_MISS distribution in these"
echo "  files (e.g. is it bimodal? a block of variants around missing ~1/n_batches is"
echo "  the signature of a site present in only some batches)."

echo ""
echo "  -- Step 5a: removing batch-unbalanced variants (--geno $GENO_THRESH) --"
if [ "$FORCE" -ne 1 ] && [ -f "$OUT_DIR/merged_geno.pgen" ] && [ -f "$OUT_DIR/merged_geno.pvar" ] && [ -f "$OUT_DIR/merged_geno.psam" ]; then
    echo "  [skip, already present] $OUT_DIR/merged_geno.pgen"
else
    plink2 --pfile "$OUT_DIR/merged_all" \
           --geno "$GENO_THRESH" \
           --make-pgen \
           --out "$OUT_DIR/merged_geno"
fi

echo ""
echo "  -- Step 5b: removing samples with genuinely poor coverage (--mind $MIND_THRESH), on the already-cleaned dataset --"
if [ "$FORCE" -ne 1 ] && [ -f "$OUT_DIR/merged_qc.pgen" ] && [ -f "$OUT_DIR/merged_qc.pvar" ] && [ -f "$OUT_DIR/merged_qc.psam" ]; then
    echo "  [skip, already present] $OUT_DIR/merged_qc.pgen"
else
    plink2 --pfile "$OUT_DIR/merged_geno" \
           --mind "$MIND_THRESH" \
           --make-pgen \
           --out "$OUT_DIR/merged_qc"
fi

n_samples_before=$(($(wc -l < "$OUT_DIR/merged_all.psam") - 1))
n_samples_after=$(($(wc -l < "$OUT_DIR/merged_qc.psam") - 1))
n_samples_dropped=$((n_samples_before - n_samples_after))
echo "  Samples before --mind filter: $n_samples_before | after: $n_samples_after | dropped: $n_samples_dropped"
if [ "$n_samples_dropped" -gt $((n_samples_before / 10)) ]; then
    echo "  >>> WARNING: more than 10% of samples were dropped for missingness."
    echo "  >>> Check $OUT_DIR/merged_qc.log to see whether they are concentrated"
    echo "  >>> in one specific batch (possible real coverage problem in that"
    echo "  >>> batch, not just a merge artifact) before proceeding."
fi
echo "  Filtered dataset: $OUT_DIR/merged_qc (used from here on for pruning/kinship/PCA)"


echo ""
echo "==> Step 6: LD pruning (window $LD_WINDOW_SIZE, step $LD_STEP, r2 < $LD_R2_THRESHOLD)"
echo "    + safety MAF >= $MAF_THRESHOLD filter here, independently of whatever"
echo "    filter was applied upstream in the _filtered VCFs (PCA/kinship behave"
echo "    poorly with rare variants, so we reapply it regardless)."
if [ "$FORCE" -ne 1 ] && [ -f "$OUT_DIR/pruned.prune.in" ]; then
    echo "  [skip, already present] $OUT_DIR/pruned.prune.in"
else
    plink2 --pfile "$OUT_DIR/merged_qc" \
           --maf "$MAF_THRESHOLD" \
           --indep-pairwise "$LD_WINDOW_SIZE" "$LD_STEP" "$LD_R2_THRESHOLD" \
           --out "$OUT_DIR/pruned"
fi

if [ "$FORCE" -ne 1 ] && [ -f "$OUT_DIR/merged_pruned.pgen" ] && [ -f "$OUT_DIR/merged_pruned.pvar" ] && [ -f "$OUT_DIR/merged_pruned.psam" ]; then
    echo "  [skip, already present] $OUT_DIR/merged_pruned.pgen"
else
    plink2 --pfile "$OUT_DIR/merged_qc" \
           --maf "$MAF_THRESHOLD" \
           --extract "$OUT_DIR/pruned.prune.in" \
           --make-pgen \
           --out "$OUT_DIR/merged_pruned"
fi

n_pruned=$(wc -l < "$OUT_DIR/pruned.prune.in")
echo "  Independent SNPs after pruning: $n_pruned"

echo ""
echo "==> Step 7: relatedness (KING-robust kinship, via plink2)"
if [ "$FORCE" -ne 1 ] && [ -f "$OUT_DIR/king.kin0" ]; then
    echo "  [skip, already present] $OUT_DIR/king.kin0"
else
    plink2 --pfile "$OUT_DIR/merged_pruned" \
           --make-king-table \
           --out "$OUT_DIR/king"
fi
echo "  Output: $OUT_DIR/king.kin0 (columns: #FID1 ID1 FID2 ID2 NSNP HETHET IBS0 KINSHIP)"

echo ""
echo "==> Step 8: PCA (10 components)"
if [ "$FORCE" -ne 1 ] && [ -f "$OUT_DIR/pca.eigenvec" ]; then
    echo "  [skip, already present] $OUT_DIR/pca.eigenvec"
else
    plink2 --pfile "$OUT_DIR/merged_pruned" \
           --pca 10 \
           --out "$OUT_DIR/pca"
fi
echo "  Output: $OUT_DIR/pca.eigenvec, $OUT_DIR/pca.eigenval"

echo ""
echo "==> DONE: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Key files for the downstream Python scripts:"
echo "    - $OUT_DIR/king.kin0"
echo "    - $OUT_DIR/pca.eigenvec"
echo ""
echo "Full log for this run: $LOGFILE"
echo "Per-job Step 1 logs: $OUT_DIR/logs/step1_<batch>_chr<N>.log"
echo "Per-job Step 2 logs: $OUT_DIR/logs/step2_chr<N>.log"
echo ""
echo "Next step: 01_run_extra_qc_checks.sh on this same out_dir."
