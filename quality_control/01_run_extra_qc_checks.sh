#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# 01_run_extra_qc_checks.sh
# ============================================================================
# Check QC aggiuntivi che 00_run_plink_qc.sh non fa, ma che tipicamente
# vengono chiesti in un paper genomico. Gira SUL TUO SERVER, DOPO
# 00_run_plink_qc.sh: riusa i file .pgen intermedi che quello script
# lascia sul disco (merged_qc, merged_pruned) -- non li ricalcola e non
# tocca 00_run_plink_qc.sh in alcun modo.
#
# COSA FA (in $OUT_DIR, stesso --out-dir passato a 00_run_plink_qc.sh):
#   A) --check-sex   su merged_qc     -> sex_check.sexcheck
#      (usa merged_qc, non pruned: --check-sex vuole il cromosoma X intero,
#      non il subset LD-pruned che puo' aver scartato molti SNP di chrX)
#   B) --het         su merged_pruned -> heterozygosity.het
#      (qui invece si usa il set LD-pruned apposta: l'eterozigosita' va
#      stimata su SNP indipendenti, altrimenti l'LD la distorce)
#   C) --freq        su merged_pruned -> maf.afreq
#   D) versioni plink2/bcftools + comando esatto -> run_metadata.txt
#
# REQUISITI: plink2 e bcftools nel PATH, e l'output di 00_run_plink_qc.sh
# ancora presente (merged_qc.pgen/.pvar/.psam e merged_pruned.pgen/.pvar/.psam).
#
# RESUME: come nello script principale, salta gli step il cui output
# esiste gia'. Usa --force per rifare tutto.
#
# USO:
#   ./01_run_extra_qc_checks.sh [--force] <out_dir>
#
# Esempio:
#   ./01_run_extra_qc_checks.sh /mnt/cresla_prod/genome_datasets/qc_output
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
    echo "Uso: $0 [--force] <out_dir>"
    echo "  <out_dir> deve essere lo stesso --out-dir gia' usato per 00_run_plink_qc.sh"
    exit 1
fi

OUT_DIR="$1"

if [ ! -f "$OUT_DIR/merged_qc.pgen" ] || [ ! -f "$OUT_DIR/merged_pruned.pgen" ]; then
    echo "ERRORE: non trovo $OUT_DIR/merged_qc.pgen e/o $OUT_DIR/merged_pruned.pgen."
    echo "Lancia prima 00_run_plink_qc.sh (o verifica di aver passato lo stesso out_dir)."
    exit 1
fi

mkdir -p "$OUT_DIR/logs"
LOGFILE="$OUT_DIR/logs/extra_qc_checks.log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "==> Avvio extra QC checks: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Out dir: $OUT_DIR"
echo "Resume: $([ $FORCE -eq 1 ] && echo 'DISATTIVATO (--force)' || echo 'attivo')"

echo ""
echo "==> A) Sex check (plink2 --check-sex, su merged_qc)"
if [ "$FORCE" -ne 1 ] && [ -f "$OUT_DIR/sex_check.sexcheck" ]; then
    echo "  [skip, gia' presente] $OUT_DIR/sex_check.sexcheck"
else
    plink2 --pfile "$OUT_DIR/merged_qc" \
           --check-sex \
           --out "$OUT_DIR/sex_check" \
        || echo "  ATTENZIONE: --check-sex ha fallito (probabile assenza di sufficienti SNP su chrX nel dataset). Controlla $OUT_DIR/sex_check.log."
fi
if [ -f "$OUT_DIR/sex_check.sexcheck" ]; then
    n_mismatch=$(awk 'NR>1 && $4=="PROBLEM"' "$OUT_DIR/sex_check.sexcheck" | wc -l)
    echo "  Campioni con sesso genetico != sesso dichiarato (PROBLEM): $n_mismatch"
    echo "  Output: $OUT_DIR/sex_check.sexcheck"
fi

echo ""
echo "==> B) Heterozygosity check (plink2 --het, su merged_pruned)"
if [ "$FORCE" -ne 1 ] && [ -f "$OUT_DIR/heterozygosity.het" ]; then
    echo "  [skip, gia' presente] $OUT_DIR/heterozygosity.het"
else
    plink2 --pfile "$OUT_DIR/merged_pruned" \
           --het \
           --out "$OUT_DIR/heterozygosity"
fi
echo "  Output: $OUT_DIR/heterozygosity.het (F-stat per campione)"

echo ""
echo "==> C) Frequenze alleliche / MAF spectrum (plink2 --freq, su merged_pruned)"
if [ "$FORCE" -ne 1 ] && [ -f "$OUT_DIR/maf.afreq" ]; then
    echo "  [skip, gia' presente] $OUT_DIR/maf.afreq"
else
    plink2 --pfile "$OUT_DIR/merged_pruned" \
           --freq \
           --out "$OUT_DIR/maf"
fi
echo "  Output: $OUT_DIR/maf.afreq"

echo ""
echo "==> D) Metadata di riproducibilita' (versioni software + comando)"
META_FILE="$OUT_DIR/run_metadata.txt"
{
    echo "Data esecuzione extra checks: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Host: $(hostname)"
    echo ""
    echo "plink2 --version:"
    plink2 --version 2>&1 | head -n 1
    echo ""
    echo "bcftools --version:"
    bcftools --version 2>&1 | head -n 1
    echo ""
    echo "Comando 01_run_extra_qc_checks.sh:"
    echo "  $0 $* (force=$FORCE)"
    if [ -f "$OUT_DIR/logs/pipeline.log" ]; then
        echo ""
        echo "Riga di avvio della pipeline principale (00_run_plink_qc.sh), da $OUT_DIR/logs/pipeline.log:"
        head -n 5 "$OUT_DIR/logs/pipeline.log"
    fi
} > "$META_FILE"
echo "  Salvato in: $META_FILE"

echo ""
echo "==> FATTO: $(date '+%Y-%m-%d %H:%M:%S')"
echo "File prodotti in questo step: sex_check.sexcheck, heterozygosity.het, maf.afreq, run_metadata.txt"
echo "Prossimo step: lancia qc_supplementary_plots.py e qc_attrition_summary.py per i grafici/tabelle finali."
