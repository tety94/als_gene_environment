#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# 00_run_plink_qc.sh
# ============================================================================
# Pipeline QC su VCF gia' esistenti: filtra SNP bialleliche comuni, fa LD
# pruning, calcola relatedness (KING-robust, via plink2) e PCA.
#
# DA ESEGUIRE SUL TUO SERVER (hereditary), non funziona nel mio sandbox:
# io non ho accesso ai tuoi file /mnt/cresla_prod/...
#
# REQUISITI: plink2 e bcftools nel PATH (probabilmente gia' presenti
# nell'ambiente conda "geneenv" che usi per gene_environment_v2 -- verifica
# con: which plink2 bcftools
#
# NOVITA' rispetto alla versione precedente:
#   - Step 1 (filtro + conversione pgen per batch/cromosoma) e' ora
#     parallelizzato su fino a 16 worker con xargs -P, invece di essere
#     sequenziale. Step 2-5 restano sequenziali perche' operano sul merge
#     completo e non sono parallelizzabili in modo banale.
#   - Tutto l'output (stdout+stderr) viene sia mostrato a schermo sia
#     salvato su $OUT_DIR/logs/pipeline.log (via tee).
#   - Ogni job di Step 1 scrive anche il proprio log dedicato in
#     $OUT_DIR/logs/step1_<batch>_chr<N>.log, cosi' se qualcosa fallisce in
#     parallelo sai subito quale batch/cromosoma e' stato.
#
# USO:
#   ./00_run_plink_qc.sh [--use-filtered] [--jobs N] <dir_vcf_1> [<dir_vcf_2> ...] <out_dir>
#
# --use-filtered: vedi sotto (invariato rispetto a prima).
# --jobs N: numero di worker paralleli per lo Step 1 (default 16).
#
# Esempio con i tuoi 3 batch, VCF gia' filtrati, 16 worker:
#   ./00_run_plink_qc.sh --use-filtered --jobs 16 \
#       /mnt/cresla_prod/genome_datasets/gen1 \
#       /mnt/cresla_prod/genome_datasets/gen2 \
#       /mnt/cresla_prod/genome_datasets/gen3 \
#       /mnt/cresla_prod/genome_datasets/qc_output
#
# NOTA IMPORTANTE: se lo stesso paziente compare in piu' di un batch
# (gen1/gen2/gen3), plink2 ti dara' un ERRORE di sample ID duplicati in
# fase di merge, oppure -- se gli ID sono stati resi unici a valle -- il
# controllo di relatedness qui sotto lo trovera' come kinship ~0.5 (falso
# "gemello monozigote"). In entrambi i casi, PRIMA di procedere controlla
# se gli ID campione si sovrappongono tra batch (fatto automaticamente
# nello Step 0 qui sotto).
#
# NOTA SUL PARALLELISMO: attenzione a I/O e RAM. Ogni worker di Step 1
# lancia bcftools view + plink2 su un VCF per cromosoma: con 16 worker
# in parallelo il carico su disco (specialmente se /mnt/cresla_prod/ e'
# uno storage di rete condiviso) puo' diventare il collo di bottiglia
# reale, non la CPU. Se noti I/O wait altissimo o il server rallenta per
# altri utenti, abbassa --jobs (es. 8).
# ============================================================================

USE_FILTERED=0
JOBS=16
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
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done
set -- "${POSITIONAL[@]}"

if [ "$#" -lt 2 ]; then
    echo "Uso: $0 [--use-filtered] [--jobs N] <dir_vcf_1> [<dir_vcf_2> ...] <out_dir>"
    exit 1
fi

ARGS=("$@")
OUT_DIR="${ARGS[-1]}"
VCF_DIRS=("${ARGS[@]:0:${#ARGS[@]}-1}")

mkdir -p "$OUT_DIR" "$OUT_DIR/logs"
cd "$OUT_DIR"

# Da qui in poi tutto stdout+stderr va sia a schermo sia sul log principale.
LOGFILE="$OUT_DIR/logs/pipeline.log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "==> Avvio pipeline: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Modalita': $([ $USE_FILTERED -eq 1 ] && echo 'USO VCF GIA FILTRATI (skip bcftools view)' || echo 'filtro MAF/bialleliche da zero')"
echo "Worker paralleli per Step 1: $JOBS"
echo "Log principale: $LOGFILE"

echo "==> Verifica strumenti disponibili"
command -v plink2 >/dev/null 2>&1 || { echo "ERRORE: plink2 non trovato nel PATH."; exit 1; }
command -v bcftools >/dev/null 2>&1 || { echo "ERRORE: bcftools non trovato nel PATH."; exit 1; }
command -v xargs >/dev/null 2>&1 || { echo "ERRORE: xargs non trovato nel PATH."; exit 1; }

echo "==> Step 0: controllo sovrapposizione sample ID tra i batch (${VCF_DIRS[*]})"
: > "$OUT_DIR/all_sample_ids.txt"
for d in "${VCF_DIRS[@]}"; do
    if [ "$USE_FILTERED" -eq 1 ]; then
        search_dir="$d/vcf_filtered"
        first_vcf=$(ls "$search_dir"/*chr1_filtered.vcf.gz 2>/dev/null | head -n1)
    else
        search_dir="$d"
        first_vcf=$(ls "$d"/*chr1.vcf.gz 2>/dev/null | head -n1)
    fi
    if [ -z "$first_vcf" ]; then
        echo "  ATTENZIONE: nessun file chr1 trovato in $search_dir, salto per il controllo ID."
        continue
    fi
    n_samples=$(bcftools query -l "$first_vcf" | wc -l)
    n_dup=$(bcftools query -l "$first_vcf" | sort | uniq -d | wc -l)
    echo "  $d -> $n_samples campioni (chr1), $n_dup ID duplicati interni"
    bcftools query -l "$first_vcf" >> "$OUT_DIR/all_sample_ids.txt"
done
n_total=$(wc -l < "$OUT_DIR/all_sample_ids.txt")
n_unique=$(sort -u "$OUT_DIR/all_sample_ids.txt" | wc -l)
n_overlap=$((n_total - n_unique))
echo "  Totale ID (somma batch): $n_total | ID unici: $n_unique | overlap: $n_overlap"
if [ "$n_overlap" -gt 0 ]; then
    echo "  >>> ATTENZIONE: $n_overlap sample ID si ripetono tra i batch."
    echo "  >>> Se sono lo stesso paziente genotipizzato piu' volte, deduplica"
    echo "  >>> PRIMA della relatedness, altrimenti troverai falsi 'gemelli'."
fi

echo ""
if [ "$USE_FILTERED" -eq 1 ]; then
    echo "==> Step 1: uso i VCF gia' filtrati (skip bcftools view), converto direttamente in pgen"
else
    echo "==> Step 1: filtro SNP bialleliche comuni (MAF >= 0.05) per cromosoma, per batch"
fi
echo "    (parallelizzato su $JOBS worker; log per singolo job in $OUT_DIR/logs/step1_*.log)"

mkdir -p "$OUT_DIR/filtered"
PGEN_LIST="$OUT_DIR/pgen_list.txt"
: > "$PGEN_LIST"

# ---------------------------------------------------------------------------
# Funzione worker per un singolo batch/cromosoma. Deve essere una funzione
# esportata (export -f) perche' xargs -P la lancia in sotto-shell separate.
# Ogni chiamata scrive il proprio log dedicato, e in caso di successo
# stampa il prefisso pgen su stdout (che noi raccogliamo per il pgen_list).
# ---------------------------------------------------------------------------
process_one() {
    local d="$1" chr="$2" use_filtered="$3" out_dir="$4"
    local batch vcf_in out_prefix log_file
    batch=$(basename "$d")
    out_prefix="$out_dir/filtered/${batch}_chr${chr}"
    log_file="$out_dir/logs/step1_${batch}_chr${chr}.log"

    {
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$batch] chr${chr}: avvio"
        if [ "$use_filtered" -eq 1 ]; then
            vcf_in=$(ls "$d/vcf_filtered"/*chr${chr}_filtered.vcf.gz 2>/dev/null | head -n1)
        else
            vcf_in=$(ls "$d"/*chr${chr}.vcf.gz 2>/dev/null | head -n1)
        fi

        if [ -z "$vcf_in" ]; then
            echo "[skip] chr${chr} non trovato per $batch"
            exit 0
        fi

        if [ "$use_filtered" -eq 1 ]; then
            echo "  converto direttamente in pgen (nessun filtro aggiuntivo qui)"
            plink2 --vcf "$vcf_in" \
                   --double-id \
                   --make-pgen \
                   --out "$out_prefix" \
                   --silent
        else
            echo "  filtro + converto in pgen"
            bcftools view -m2 -M2 -v snps --min-af 0.05:minor "$vcf_in" -Oz -o "${out_prefix}.filt.vcf.gz"
            plink2 --vcf "${out_prefix}.filt.vcf.gz" \
                   --double-id \
                   --make-pgen \
                   --out "$out_prefix" \
                   --silent
        fi
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$batch] chr${chr}: completato"
        # Riga marcatore: la raccogliamo dal chiamante per costruire pgen_list.txt
        echo "PGEN_OUT:$out_prefix"
    } > "$log_file" 2>&1

    # Se il job e' andato a buon fine e ha prodotto un pgen, rilancia la riga
    # marcatore anche su stdout del worker (fuori dal blocco loggato sopra),
    # cosi' il processo padre puo' raccoglierla.
    if [ -f "${out_prefix}.pgen" ]; then
        echo "$out_prefix"
    fi
}
export -f process_one

# Genera la lista di job (batch x cromosoma 1..22) e lancia con xargs -P.
JOBLIST="$OUT_DIR/logs/step1_joblist.txt"
: > "$JOBLIST"
for d in "${VCF_DIRS[@]}"; do
    for chr in $(seq 1 22); do
        echo "$d|$chr" >> "$JOBLIST"
    done
done

echo "  Totale job Step 1: $(wc -l < "$JOBLIST") (batch x 22 cromosomi)"

cat "$JOBLIST" | xargs -P "$JOBS" -I{} bash -c '
    IFS="|" read -r d chr <<< "{}"
    process_one "$d" "$chr" "'"$USE_FILTERED"'" "'"$OUT_DIR"'"
' > "$PGEN_LIST.raw"

# Filtra eventuali righe vuote/rumorose e deduplica mantenendo l'ordine.
grep -v '^\s*$' "$PGEN_LIST.raw" | sort -u > "$PGEN_LIST"
rm -f "$PGEN_LIST.raw"

n_ok=$(wc -l < "$PGEN_LIST")
echo "  Job completati con pgen prodotto: $n_ok"
echo "  Se il numero e' inferiore a quanto atteso, controlla i log in"
echo "  $OUT_DIR/logs/step1_*.log per i cromosomi/batch mancanti (probabile [skip])."

echo ""
echo "==> Step 2: merge di tutti i file pgen (tutti i batch, tutti i cromosomi)"
plink2 --pmerge-list "$PGEN_LIST" pfile \
       --make-pgen \
       --out "$OUT_DIR/merged_all"

echo ""
echo "==> Step 3: LD pruning (finestra 50, step 5, r2 < 0.2 -- standard)"
echo "    + filtro MAF >= 0.05 di sicurezza qui, indipendentemente dal filtro"
echo "    applicato a monte nei VCF _filtered (PCA/kinship funzionano male"
echo "    con varianti rare, quindi lo riapplichiamo comunque)."
plink2 --pfile "$OUT_DIR/merged_all" \
       --maf 0.05 \
       --indep-pairwise 50 5 0.2 \
       --out "$OUT_DIR/pruned"

plink2 --pfile "$OUT_DIR/merged_all" \
       --maf 0.05 \
       --extract "$OUT_DIR/pruned.prune.in" \
       --make-pgen \
       --out "$OUT_DIR/merged_pruned"

n_pruned=$(wc -l < "$OUT_DIR/pruned.prune.in")
echo "  SNP indipendenti dopo pruning: $n_pruned"

echo ""
echo "==> Step 4: relatedness (KING-robust kinship, via plink2)"
plink2 --pfile "$OUT_DIR/merged_pruned" \
       --make-king-table \
       --out "$OUT_DIR/king"
echo "  Output: $OUT_DIR/king.kin0 (colonne: #FID1 ID1 FID2 ID2 NSNP HETHET IBS0 KINSHIP)"

echo ""
echo "==> Step 5: PCA (10 componenti)"
plink2 --pfile "$OUT_DIR/merged_pruned" \
       --pca 10 \
       --out "$OUT_DIR/pca"
echo "  Output: $OUT_DIR/pca.eigenvec, $OUT_DIR/pca.eigenval"

echo ""
echo "==> FATTO: $(date '+%Y-%m-%d %H:%M:%S')"
echo "File chiave per lo script di interpretazione python:"
echo "    - $OUT_DIR/king.kin0"
echo "    - $OUT_DIR/pca.eigenvec"
echo ""
echo "Log completo di questa run: $LOGFILE"
echo "Log per-job dello Step 1: $OUT_DIR/logs/step1_<batch>_chr<N>.log"
echo ""
echo "Prossimo step: lancia interpret_plink_output.py passando questi due file"
echo "piu' i tuoi metadati di esposizione."