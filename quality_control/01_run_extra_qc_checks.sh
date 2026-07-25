#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# 01_run_extra_qc_checks.sh
# ============================================================================
# Extra QC checks that 00_run_plink_qc.sh does not do, but that are
# typically requested in a genomics paper. Runs ON THE SAME SERVER, AFTER
# 00_run_plink_qc.sh: it reuses the intermediate .pgen files that script
# leaves on disk (merged_qc, merged_pruned) -- it does not recompute them
# and does not touch 00_run_plink_qc.sh in any way.
#
# WHAT IT DOES (in $OUT_DIR, the same --out-dir passed to 00_run_plink_qc.sh):
#   A) --check-sex   on merged_qc     -> sex_check.sexcheck
#      (uses merged_qc, not pruned: --check-sex wants the full X
#      chromosome, not the LD-pruned subset which may have dropped many
#      chrX SNPs)
#   B) --het         on merged_pruned -> heterozygosity.het
#      (here the LD-pruned set is used on purpose: heterozygosity should
#      be estimated on independent SNPs, otherwise LD distorts it)
#   C) --freq        on merged_pruned -> maf.afreq
#   D) plink2/bcftools versions + exact command -> run_metadata.txt
#
# REQUIREMENTS: plink2 and bcftools in PATH, and the output of
# 00_run_plink_qc.sh still present (merged_qc.pgen/.pvar/.psam and
# merged_pruned.pgen/.pvar/.psam).
#
# RESUME: as in the main script, skips any step whose output already
# exists. Use --force to redo everything.
#
# USAGE:
#   ./01_run_extra_qc_checks.sh [--force] <out_dir>
#
# Example:
#   ./01_run_extra_qc_checks.sh /mnt/genome_datasets/qc_output_cohortA
# ============================================================================

FORCE=0
POSITIONAL=()
while [ "$#" -gt 0 ]; do
    case "$1" in
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

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [--force] <out_dir>"
    echo "  <out_dir> must be the same --out-dir already used for 00_run_plink_qc.sh"
    exit 1
fi

OUT_DIR="$1"

if [ ! -f "$OUT_DIR/merged_qc.pgen" ] || [ ! -f "$OUT_DIR/merged_pruned.pgen" ]; then
    echo "ERROR: cannot find $OUT_DIR/merged_qc.pgen and/or $OUT_DIR/merged_pruned.pgen."
    echo "Run 00_run_plink_qc.sh first (or check that you passed the same out_dir)."
    exit 1
fi

mkdir -p "$OUT_DIR/logs"
LOGFILE="$OUT_DIR/logs/extra_qc_checks.log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "==> Starting extra QC checks: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Out dir: $OUT_DIR"
echo "Resume: $([ $FORCE -eq 1 ] && echo 'DISABLED (--force)' || echo 'active')"

echo ""
echo "==> A) Sex check (plink2 --check-sex, on merged_qc)"
if [ "$FORCE" -ne 1 ] && [ -f "$OUT_DIR/sex_check.sexcheck" ]; then
    echo "  [skip, already present] $OUT_DIR/sex_check.sexcheck"
else
    plink2 --pfile "$OUT_DIR/merged_qc" \
           --check-sex \
           --out "$OUT_DIR/sex_check" \
        || echo "  WARNING: --check-sex failed (likely too few chrX SNPs in the dataset). Check $OUT_DIR/sex_check.log."
fi
if [ -f "$OUT_DIR/sex_check.sexcheck" ]; then
    n_mismatch=$(awk 'NR>1 && $4=="PROBLEM"' "$OUT_DIR/sex_check.sexcheck" | wc -l)
    echo "  Samples with genetic sex != declared sex (PROBLEM): $n_mismatch"
    echo "  Output: $OUT_DIR/sex_check.sexcheck"
fi

echo ""
echo "==> B) Heterozygosity check (plink2 --het, on merged_pruned)"
if [ "$FORCE" -ne 1 ] && [ -f "$OUT_DIR/heterozygosity.het" ]; then
    echo "  [skip, already present] $OUT_DIR/heterozygosity.het"
else
    plink2 --pfile "$OUT_DIR/merged_pruned" \
           --het \
           --out "$OUT_DIR/heterozygosity"
fi
echo "  Output: $OUT_DIR/heterozygosity.het (F-statistic per sample)"

echo ""
echo "==> C) Allele frequencies / MAF spectrum (plink2 --freq, on merged_pruned)"
if [ "$FORCE" -ne 1 ] && [ -f "$OUT_DIR/maf.afreq" ]; then
    echo "  [skip, already present] $OUT_DIR/maf.afreq"
else
    plink2 --pfile "$OUT_DIR/merged_pruned" \
           --freq \
           --out "$OUT_DIR/maf"
fi
echo "  Output: $OUT_DIR/maf.afreq"

echo ""
echo "==> D) Reproducibility metadata (software versions + command)"
META_FILE="$OUT_DIR/run_metadata.txt"
{
    echo "Extra checks run date: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Host: $(hostname)"
    echo ""
    echo "plink2 --version:"
    plink2 --version 2>&1 | head -n 1
    echo ""
    echo "bcftools --version:"
    bcftools --version 2>&1 | head -n 1
    echo ""
    echo "01_run_extra_qc_checks.sh command:"
    echo "  $0 $* (force=$FORCE)"
    if [ -f "$OUT_DIR/logs/pipeline.log" ]; then
        echo ""
        echo "Startup line of the main pipeline (00_run_plink_qc.sh), from $OUT_DIR/logs/pipeline.log:"
        head -n 5 "$OUT_DIR/logs/pipeline.log"
    fi
} > "$META_FILE"
echo "  Saved to: $META_FILE"

echo ""
echo "==> DONE: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Files produced in this step: sex_check.sexcheck, heterozygosity.het, maf.afreq, run_metadata.txt"
echo "Next step: qc_supplementary_plots.py and qc_attrition_summary.py for the final tables/figures."
