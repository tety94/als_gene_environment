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
# USO:
#   ./00_run_plink_qc.sh [--use-filtered] <dir_vcf_1> [<dir_vcf_2> <dir_vcf_3> ...] <out_dir>
#
# --use-filtered: se presente, lo script cerca *_filtered.vcf.gz dentro una
#   sottocartella vcf_filtered/ di ciascuna directory data in input (es.
#   gen1/vcf_filtered/gen1_onlycases_vcf_chr1_filtered.vcf.gz) e SALTA lo
#   step di bcftools view -m2 -M2 --min-af, assumendo che il filtro sia
#   gia' stato applicato a monte da gene_environment_v2. Usalo SOLO dopo
#   aver verificato (vedi commento sotto) che il filtro a monte includa
#   gia' bialleliche + una soglia MAF ragionevole; altrimenti lascia lo
#   script senza questo flag cosi' il filtro MAF/bialleliche viene comunque
#   applicato qui.
#
# Esempio con i tuoi 3 batch, usando i VCF gia' filtrati:
#   ./00_run_plink_qc.sh --use-filtered \
#       /mnt/cresla_prod/genome_datasets/gen1 \
#       /mnt/cresla_prod/genome_datasets/gen2 \
#       /mnt/cresla_prod/genome_datasets/gen3 \
#       /mnt/cresla_prod/genome_datasets/qc_output
#
# Esempio senza (rifa' il filtro MAF/bialleliche da zero sui VCF grezzi):
#   ./00_run_plink_qc.sh \
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
# se gli ID campione si sovrappongono tra batch:
#   for f in gen1 gen2 gen3; do
#       bcftools query -l $DIR/$f/${f}_onlycases_vcf_chr1.vcf.gz
#   done
# e confronta le liste (es. con `comm` o in python/pandas) per capire se
# hai overlap prima di lanciare la pipeline.
# ============================================================================

USE_FILTERED=0
if [ "${1:-}" == "--use-filtered" ]; then
    USE_FILTERED=1
    shift
fi

if [ "$#" -lt 2 ]; then
    echo "Uso: $0 [--use-filtered] <dir_vcf_1> [<dir_vcf_2> ...] <out_dir>"
    exit 1
fi

ARGS=("$@")
OUT_DIR="${ARGS[-1]}"
VCF_DIRS=("${ARGS[@]:0:${#ARGS[@]}-1}")

echo "Modalita': $([ $USE_FILTERED -eq 1 ] && echo 'USO VCF GIA FILTRATI (skip bcftools view)' || echo 'filtro MAF/bialleliche da zero')"

mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

echo "==> Verifica strumenti disponibili"
command -v plink2 >/dev/null 2>&1 || { echo "ERRORE: plink2 non trovato nel PATH."; exit 1; }
command -v bcftools >/dev/null 2>&1 || { echo "ERRORE: bcftools non trovato nel PATH."; exit 1; }

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
mkdir -p "$OUT_DIR/filtered"
PGEN_LIST="$OUT_DIR/pgen_list.txt"
: > "$PGEN_LIST"

for d in "${VCF_DIRS[@]}"; do
    batch=$(basename "$d")
    for chr in $(seq 1 22); do
        if [ "$USE_FILTERED" -eq 1 ]; then
            vcf_in=$(ls "$d/vcf_filtered"/*chr${chr}_filtered.vcf.gz 2>/dev/null | head -n1)
        else
            vcf_in=$(ls "$d"/*chr${chr}.vcf.gz 2>/dev/null | head -n1)
        fi
        if [ -z "$vcf_in" ]; then
            echo "  [skip] chr${chr} non trovato per $batch"
            continue
        fi
        out_prefix="$OUT_DIR/filtered/${batch}_chr${chr}"
        if [ "$USE_FILTERED" -eq 1 ]; then
            echo "  [$batch] chr${chr}: converto direttamente in pgen (nessun filtro aggiuntivo qui)"
            plink2 --vcf "$vcf_in" \
                   --double-id \
                   --make-pgen \
                   --out "$out_prefix" \
                   --silent
        else
            echo "  [$batch] chr${chr}: filtro + converto in pgen"
            bcftools view -m2 -M2 -v snps --min-af 0.05:minor "$vcf_in" -Oz -o "${out_prefix}.filt.vcf.gz"
            plink2 --vcf "${out_prefix}.filt.vcf.gz" \
                   --double-id \
                   --make-pgen \
                   --out "$out_prefix" \
                   --silent
        fi
        echo "$out_prefix" >> "$PGEN_LIST"
    done
done

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
echo "==> FATTO. File chiave per lo script di interpretazione python:"
echo "    - $OUT_DIR/king.kin0"
echo "    - $OUT_DIR/pca.eigenvec"
echo ""
echo "Prossimo step: lancia interpret_plink_output.py passando questi due file"
echo "piu' i tuoi metadati di esposizione."