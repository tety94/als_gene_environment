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
# NOTA SULLA STRUTTURA (IMPORTANTE, letta dopo un errore reale in produzione):
#   plink2 --pmerge-list gestisce bene UN SOLO tipo di merge alla volta:
#   o fondere campioni (stessi varianti, sample diversi) o concatenare
#   (stessi campioni, varianti diverse). Qui abbiamo ENTRAMBE le cose
#   insieme (3 batch = sample diversi, 22 cromosomi = varianti diverse),
#   e quella combinazione da' "Error: Non-concatenating --pmerge[-list] is
#   under development." -- una limitazione nota di plink2, non un bug
#   nei tuoi dati.
#
#   Per questo tutto il merge (sia tra batch che tra cromosomi) viene fatto
#   in bcftools, che gestisce nativamente entrambi i casi, e si converte
#   in pgen UNA SOLA VOLTA alla fine, sul VCF genome-wide gia' completo:
#     Step 1: filtro (se serve) + indicizzazione per batch x cromosoma
#     Step 2: bcftools merge tra i batch, per ciascun cromosoma (parallelo)
#     Step 3: bcftools concat dei 22 cromosomi -> un VCF genome-wide
#     Step 4: UNA conversione plink2 --vcf ... --make-pgen
#     Step 5: diagnostica + filtro missingness (--geno/--mind) -> merged_qc
#     Step 6: LD pruning
#     Step 7: relatedness (KING)
#     Step 8: PCA
#
# NOVITA' rispetto alla prima versione:
#   - Step 1 e Step 2 sono parallelizzati con xargs -P (fino a 16 worker).
#   - Tutto l'output (stdout+stderr) va sia a schermo sia su
#     $OUT_DIR/logs/pipeline.log (via tee).
#   - Ogni job di Step 1 e Step 2 scrive il proprio log dedicato in
#     $OUT_DIR/logs/, cosi' un fallimento in parallelo e' facile da tracciare.
#
# RESUME (riprendi da dove eri rimasto): OGNI step, prima di eseguire,
# controlla se il proprio output finale esiste gia' e in caso affermativo
# lo SALTA (loggando "[skip, gia' presente]"), invece di rifarlo. Vale sia
# per i singoli job paralleli di Step 1/2 (un batch/cromosoma o un
# cromosoma gia' completato non viene rifatto) sia per gli step
# sequenziali 3-7. Se rilanci lo script dopo un'interruzione (crash,
# server riavviato, Ctrl-C), riparte automaticamente dal primo output
# mancante. Usa --force per ignorare tutto questo e rifare comunque tutto
# da zero.
#
# NOTA: il controllo di "esistenza" si basa sulla presenza del file di
# output finale + relativo indice/companion file (es. .vcf.gz + .tbi, o
# .pgen + .pvar + .psam), NON su un controllo di integrita' del contenuto.
# Se uno step si interrompe A META' scrittura (es. kill -9 nel mezzo di un
# bcftools merge), il file puo' esistere ma essere troncato/corrotto senza
# che il resume se ne accorga. In quel caso usa --force, o cancella a mano
# l'output sospetto prima di rilanciare.
#
# USO:
#   ./00_run_plink_qc.sh [--use-filtered] [--jobs N] [--force] <dir_vcf_1> [<dir_vcf_2> ...] <out_dir>
#
# --use-filtered: se presente, lo script cerca *_filtered.vcf.gz dentro una
#   sottocartella vcf_filtered/ di ciascuna directory data in input e SALTA
#   lo step di bcftools view -m2 -M2 --min-af, assumendo che il filtro sia
#   gia' stato applicato a monte da gene_environment_v2. Usalo SOLO dopo
#   aver verificato che il filtro a monte includa gia' bialleliche + una
#   soglia MAF ragionevole.
# --jobs N: numero di worker paralleli per Step 1 e Step 2 (default 16).
# --force: ignora tutti gli output gia' presenti e rifa' l'intera pipeline
#   da zero, sovrascrivendo.
#
# PARAMETRI DI FILTRO (variabili d'ambiente, non flag -- default = quelli
# gia' usati nel resto della pipeline gene_environment_v2):
#   MAF_THRESHOLD   (default 0.01) soglia MAF, usata sia nel filtro
#                   bcftools view di Step 1 (solo se NON --use-filtered)
#                   sia nel pruning/estrazione di Step 6.
#   LD_WINDOW_SIZE  (default 50)   dimensione finestra per --indep-pairwise.
#   LD_STEP         (default 5)    step per --indep-pairwise.
#   LD_R2_THRESHOLD (default 0.5)  soglia r2 per --indep-pairwise.
#   GENO_THRESH     (default 0.05) soglia missingness per variante (Step 5).
#   MIND_THRESH     (default 0.05) soglia missingness per campione (Step 5).
# Esempio per cambiarli:
#   MAF_THRESHOLD=0.01 LD_R2_THRESHOLD=0.5 ./00_run_plink_qc.sh --use-filtered ...
#
# Esempio con i tuoi 3 batch, VCF gia' filtrati, 16 worker:
#   ./00_run_plink_qc.sh --use-filtered --jobs 16 \
#       /mnt/cresla_prod/genome_datasets/gen1 \
#       /mnt/cresla_prod/genome_datasets/gen2 \
#       /mnt/cresla_prod/genome_datasets/gen3 \
#       /mnt/cresla_prod/genome_datasets/qc_output
#
# NOTA IMPORTANTE: se lo stesso paziente compare in piu' di un batch,
# plink2/il merge dara' problemi (ID duplicati, o falsi "gemelli" nel
# kinship se gli ID sono stati resi unici a valle). Lo Step 0 controlla
# automaticamente la sovrapposizione tra gli ID campione tra i batch.
#
# NOTA SUL MERGE TRA BATCH: bcftools merge fa l'unione dei siti tra i
# batch; un campione che non ha genotipo in un sito presente solo in un
# altro batch viene riempito come mancante (./.) invece di essere escluso.
# Con soglie MAF diverse o pipeline di chiamata leggermente diverse tra
# batch questo puo' introdurre missingness non banale in alcuni siti, che
# gonfia artificialmente le stime di kinship. Per questo lo Step 5 applica
# un filtro esplicito --geno/--mind (con report diagnostico) DOPO il merge
# e PRIMA di pruning/kinship/PCA -- vedi commenti nello Step 5 piu' sotto.
#
# NOTA SUL PARALLELISMO: attenzione a I/O. Con storage di rete condiviso
# (es. /mnt/cresla_prod/), 16 worker in parallelo possono saturare il
# disco prima della CPU. Se il server rallenta, abbassa --jobs (es. 8).
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
    echo "Uso: $0 [--use-filtered] [--jobs N] [--force] <dir_vcf_1> [<dir_vcf_2> ...] <out_dir>"
    exit 1
fi

ARGS=("$@")
OUT_DIR="${ARGS[-1]}"
VCF_DIRS=("${ARGS[@]:0:${#ARGS[@]}-1}")

mkdir -p "$OUT_DIR" "$OUT_DIR/logs" "$OUT_DIR/filtered" "$OUT_DIR/merged"
cd "$OUT_DIR"

# Da qui in poi tutto stdout+stderr va sia a schermo sia sul log principale.
LOGFILE="$OUT_DIR/logs/pipeline.log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "==> Avvio pipeline: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Modalita': $([ $USE_FILTERED -eq 1 ] && echo 'USO VCF GIA FILTRATI (skip bcftools view)' || echo 'filtro MAF/bialleliche da zero')"
echo "Worker paralleli: $JOBS"
echo "Resume: $([ $FORCE -eq 1 ] && echo 'DISATTIVATO (--force: rifaccio tutto da zero)' || echo 'attivo (salto gli step il cui output esiste gia'"'"')')"
echo "Log principale: $LOGFILE"

# Parametri di filtro MAF/LD pruning, sovrascrivibili via variabili
# d'ambiente. Default = gli stessi valori gia' usati nel resto della tua
# pipeline (gene_environment_v2), per coerenza tra i due:
#   MAF_THRESHOLD=0.01 LD_WINDOW_SIZE=50 LD_STEP=5 LD_R2_THRESHOLD=0.5
MAF_THRESHOLD="${MAF_THRESHOLD:-0.01}"
LD_WINDOW_SIZE="${LD_WINDOW_SIZE:-50}"
LD_STEP="${LD_STEP:-5}"
LD_R2_THRESHOLD="${LD_R2_THRESHOLD:-0.5}"
export MAF_THRESHOLD LD_WINDOW_SIZE LD_STEP LD_R2_THRESHOLD
echo "Filtro MAF: $MAF_THRESHOLD | LD pruning: finestra $LD_WINDOW_SIZE, step $LD_STEP, r2 < $LD_R2_THRESHOLD"

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
    echo "==> Step 1: uso i VCF gia' filtrati (skip bcftools view), indicizzo per il merge"
else
    echo "==> Step 1: filtro SNP bialleliche comuni (MAF >= 0.05) per cromosoma, per batch, indicizzo"
fi
echo "    (parallelizzato su $JOBS worker; log per singolo job in $OUT_DIR/logs/step1_*.log)"

# ---------------------------------------------------------------------------
# Step 1 worker: prepara il VCF pronto per il merge per un batch/cromosoma
# (filtrando se necessario) e lo indicizza (.tbi). NON converte piu' in pgen
# qui: la conversione avviene una sola volta a fine pipeline, sul VCF
# genome-wide gia' unito. Stampa su stdout il percorso del VCF pronto,
# preceduto dal numero di cromosoma, cosi' il chiamante puo' raggruppare
# per cromosoma nello Step 2.
# ---------------------------------------------------------------------------
process_one() {
    local d="$1" chr="$2" use_filtered="$3" out_dir="$4" force="$5"
    local batch vcf_in out_prefix log_file merge_vcf
    batch=$(basename "$d")
    out_prefix="$out_dir/filtered/${batch}_chr${chr}"
    merge_vcf="${out_prefix}.merge_input.vcf.gz"
    log_file="$out_dir/logs/step1_${batch}_chr${chr}.log"

    if [ "$force" -ne 1 ] && [ -f "$merge_vcf" ] && [ -f "${merge_vcf}.tbi" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$batch] chr${chr}: [skip, gia' presente]" >> "$log_file"
        echo "${chr} ${merge_vcf}"
        return 0
    fi

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
            echo "  gia' filtrato: creo symlink (nessuna copia, nessun filtro aggiuntivo)"
            ln -sf "$(readlink -f "$vcf_in")" "$merge_vcf"
        else
            echo "  filtro (bialleliche, MAF >= $MAF_THRESHOLD)"
            bcftools view -m2 -M2 -v snps --min-af "${MAF_THRESHOLD}:minor" "$vcf_in" -Oz -o "$merge_vcf"
        fi

        echo "  indicizzo"
        bcftools index -t -f "$merge_vcf"

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$batch] chr${chr}: completato"
        echo "MERGE_VCF_OK:$merge_vcf"
    } > "$log_file" 2>&1

    if [ -f "${merge_vcf}.tbi" ]; then
        # formato: <chr> <path>   -- il chiamante raggruppa per chr
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
echo "  Totale job Step 1: $(wc -l < "$JOBLIST1") (batch x 22 cromosomi)"

STEP1_OUT="$OUT_DIR/logs/step1_output.txt"
cat "$JOBLIST1" | xargs -P "$JOBS" -I{} bash -c '
    IFS="|" read -r d chr <<< "{}"
    process_one "$d" "$chr" "'"$USE_FILTERED"'" "'"$OUT_DIR"'" "'"$FORCE"'"
' > "$STEP1_OUT"

n_ok=$(grep -c . "$STEP1_OUT" || true)
echo "  Job completati con VCF pronto per il merge: $n_ok"
echo "  Se il numero e' inferiore a quanto atteso (batch x 22), controlla i log"
echo "  in $OUT_DIR/logs/step1_*.log per i cromosomi/batch mancanti (probabile [skip])."

echo ""
echo "==> Step 2: merge dei batch, per cromosoma (bcftools merge, parallelo su $JOBS worker)"
echo "    (log per singolo cromosoma in $OUT_DIR/logs/step2_chr*.log)"

merge_chr() {
    local chr="$1" out_dir="$2" force="$3"
    shift 3
    local vcfs=("$@")
    local out_vcf="$out_dir/merged/merged_chr${chr}.vcf.gz"
    local log_file="$out_dir/logs/step2_chr${chr}.log"

    if [ "$force" -ne 1 ] && [ -f "$out_vcf" ] && [ -f "${out_vcf}.tbi" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] chr${chr}: [skip, gia' presente]" >> "$log_file"
        echo "$out_vcf"
        return 0
    fi

    {
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] chr${chr}: merge di ${#vcfs[@]} batch"
        if [ "${#vcfs[@]}" -eq 1 ]; then
            echo "  un solo batch presente per questo cromosoma, copio (nessun merge necessario)"
            ln -sf "$(readlink -f "${vcfs[0]}")" "$out_vcf"
        else
            bcftools merge -m none -Oz -o "$out_vcf" "${vcfs[@]}"
        fi
        bcftools index -t -f "$out_vcf"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] chr${chr}: completato"
        echo "MERGE_CHR_OK:$out_vcf"
    } > "$log_file" 2>&1

    if [ -f "${out_vcf}.tbi" ]; then
        echo "$out_vcf"
    fi
}
export -f merge_chr

# Raggruppa i VCF di Step 1 per cromosoma (gestisce anche il caso in cui un
# batch manchi di qualche cromosoma).
JOBLIST2="$OUT_DIR/logs/step2_joblist.txt"
: > "$JOBLIST2"
for chr in $(seq 1 22); do
    files=$(awk -v c="$chr" '$1 == c { $1=""; sub(/^ /,""); print }' "$STEP1_OUT" | tr '\n' ' ' | sed 's/ *$//')
    if [ -n "$files" ]; then
        echo "${chr}|${files}" >> "$JOBLIST2"
    else
        echo "  ATTENZIONE: nessun VCF disponibile per chr${chr}, salto interamente questo cromosoma."
    fi
done

MERGED_CHR_LIST="$OUT_DIR/logs/step2_output.txt"
cat "$JOBLIST2" | xargs -P "$JOBS" -I{} bash -c '
    IFS="|" read -r chr files <<< "{}"
    merge_chr "$chr" "'"$OUT_DIR"'" "'"$FORCE"'" $files
' > "$MERGED_CHR_LIST"

n_chr_ok=$(grep -c . "$MERGED_CHR_LIST" || true)
echo "  Cromosomi mergiati con successo: $n_chr_ok / 22"

echo ""
echo "==> Step 3: concatenazione dei 22 cromosomi in un unico VCF genome-wide"
if [ "$FORCE" -ne 1 ] && [ -f "$OUT_DIR/merged_all.vcf.gz" ] && [ -f "$OUT_DIR/merged_all.vcf.gz.tbi" ]; then
    echo "  [skip, gia' presente] $OUT_DIR/merged_all.vcf.gz"
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
    echo "  VCF genome-wide: $OUT_DIR/merged_all.vcf.gz"
fi

echo ""
echo "==> Step 4: conversione in pgen (una sola volta, sul VCF gia' completo)"
echo "    + assegno ID univoci alle varianti (chr:pos:ref:alt) e scarto duplicati"
echo "    esatti: necessario perche' --indep-pairwise richiede ID univoci, e i"
echo "    VCF grezzi hanno spesso '.' come ID per (quasi) tutte le varianti."
if [ "$FORCE" -ne 1 ] && [ -f "$OUT_DIR/merged_all.pgen" ] && [ -f "$OUT_DIR/merged_all.pvar" ] && [ -f "$OUT_DIR/merged_all.psam" ]; then
    echo "  [skip, gia' presente] $OUT_DIR/merged_all.pgen"
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
echo "==> Step 5: diagnostica missingness + filtro qualita' (--geno poi --mind)"
echo "    bcftools merge fa l'unione dei siti tra batch: una variante presente"
echo "    solo in 1-2 batch su 3 (o su 2, se ne usi meno) risulta genotipata"
echo "    solo in quei campioni e mancante (missing) negli altri, pur avendo"
echo "    una MAF apparente perfettamente normale -- --maf da solo NON la"
echo "    intercetta. Questo gonfia artificialmente il kinship stimato,"
echo "    specialmente tra campioni di batch diversi."
echo ""
echo "    IMPORTANTE: --geno e --mind vanno applicati in DUE chiamate plink2"
echo "    SEQUENZIALI, non in una sola. Se li lanci insieme, plink2 calcola la"
echo "    missingness per campione (--mind) sul dataset ANCORA NON ripulito"
echo "    dalle varianti sbilanciate tra batch: un intero batch puo' risultare"
echo "    con missingness aggregata elevata (perche' 'manca' su tutte le"
echo "    varianti chiamate solo nell'altro batch) e finire scartato quasi"
echo "    per intero PRIMA che --geno abbia rimosso quelle varianti. Per"
echo "    questo qui si applica prima --geno (pulisce le varianti), poi,"
echo "    SUL DATASET GIA' RIPULITO, --mind (ora la missingness per campione"
echo "    riflette la copertura reale, non l'artefatto batch)."
echo ""
echo "    Soglia default 0.05 per entrambi. Cambiabile con GENO_THRESH/MIND_THRESH"
echo "    come variabili d'ambiente prima di lanciare lo script, es.:"
echo "      GENO_THRESH=0.02 MIND_THRESH=0.05 ./00_run_plink_qc.sh ..."
GENO_THRESH="${GENO_THRESH:-0.05}"
MIND_THRESH="${MIND_THRESH:-0.05}"
echo "    Soglie in uso: --geno $GENO_THRESH  --mind $MIND_THRESH"

if [ "$FORCE" -ne 1 ] && [ -f "$OUT_DIR/missingness.vmiss" ] && [ -f "$OUT_DIR/missingness.smiss" ]; then
    echo "  [skip, gia' presente] $OUT_DIR/missingness.vmiss / .smiss"
else
    plink2 --pfile "$OUT_DIR/merged_all" \
           --missing \
           --out "$OUT_DIR/missingness"
fi
echo "  Report diagnostico (PRIMA del filtro): $OUT_DIR/missingness.vmiss (per variante),"
echo "  $OUT_DIR/missingness.smiss (per campione). Se vuoi scegliere la soglia a"
echo "  occhio invece di usare il default 0.05, guarda la distribuzione di"
echo "  F_MISS in questi file (es. e' bimodale? un blocco di varianti intorno"
echo "  a missing ~1/3 o ~1/2 e' la firma di un sito presente solo in alcuni batch)."

echo ""
echo "  -- Step 5a: rimuovo le varianti sbilanciate (--geno $GENO_THRESH) --"
if [ "$FORCE" -ne 1 ] && [ -f "$OUT_DIR/merged_geno.pgen" ] && [ -f "$OUT_DIR/merged_geno.pvar" ] && [ -f "$OUT_DIR/merged_geno.psam" ]; then
    echo "  [skip, gia' presente] $OUT_DIR/merged_geno.pgen"
else
    plink2 --pfile "$OUT_DIR/merged_all" \
           --geno "$GENO_THRESH" \
           --make-pgen \
           --out "$OUT_DIR/merged_geno"
fi

echo ""
echo "  -- Step 5b: rimuovo i campioni con copertura reale scarsa (--mind $MIND_THRESH), sul dataset gia' ripulito --"
if [ "$FORCE" -ne 1 ] && [ -f "$OUT_DIR/merged_qc.pgen" ] && [ -f "$OUT_DIR/merged_qc.pvar" ] && [ -f "$OUT_DIR/merged_qc.psam" ]; then
    echo "  [skip, gia' presente] $OUT_DIR/merged_qc.pgen"
else
    plink2 --pfile "$OUT_DIR/merged_geno" \
           --mind "$MIND_THRESH" \
           --make-pgen \
           --out "$OUT_DIR/merged_qc"
fi

n_samples_before=$(($(wc -l < "$OUT_DIR/merged_all.psam") - 1))
n_samples_after=$(($(wc -l < "$OUT_DIR/merged_qc.psam") - 1))
n_samples_dropped=$((n_samples_before - n_samples_after))
echo "  Campioni prima del filtro --mind: $n_samples_before | dopo: $n_samples_after | scartati: $n_samples_dropped"
if [ "$n_samples_dropped" -gt $((n_samples_before / 10)) ]; then
    echo "  >>> ATTENZIONE: scartato piu' del 10% dei campioni per missingness."
    echo "  >>> Controlla $OUT_DIR/merged_qc.log per capire se sono concentrati"
    echo "  >>> in un batch specifico (possibile problema di copertura reale di"
    echo "  >>> quel batch, non solo artefatto da merge) prima di proseguire."
fi
echo "  Dataset filtrato: $OUT_DIR/merged_qc (usato da qui in poi per pruning/kinship/PCA)"


echo ""
echo "==> Step 6: LD pruning (finestra $LD_WINDOW_SIZE, step $LD_STEP, r2 < $LD_R2_THRESHOLD)"
echo "    + filtro MAF >= $MAF_THRESHOLD di sicurezza qui, indipendentemente dal filtro"
echo "    applicato a monte nei VCF _filtered (PCA/kinship funzionano male"
echo "    con varianti rare, quindi lo riapplichiamo comunque)."
if [ "$FORCE" -ne 1 ] && [ -f "$OUT_DIR/pruned.prune.in" ]; then
    echo "  [skip, gia' presente] $OUT_DIR/pruned.prune.in"
else
    plink2 --pfile "$OUT_DIR/merged_qc" \
           --maf "$MAF_THRESHOLD" \
           --indep-pairwise "$LD_WINDOW_SIZE" "$LD_STEP" "$LD_R2_THRESHOLD" \
           --out "$OUT_DIR/pruned"
fi

if [ "$FORCE" -ne 1 ] && [ -f "$OUT_DIR/merged_pruned.pgen" ] && [ -f "$OUT_DIR/merged_pruned.pvar" ] && [ -f "$OUT_DIR/merged_pruned.psam" ]; then
    echo "  [skip, gia' presente] $OUT_DIR/merged_pruned.pgen"
else
    plink2 --pfile "$OUT_DIR/merged_qc" \
           --maf "$MAF_THRESHOLD" \
           --extract "$OUT_DIR/pruned.prune.in" \
           --make-pgen \
           --out "$OUT_DIR/merged_pruned"
fi

n_pruned=$(wc -l < "$OUT_DIR/pruned.prune.in")
echo "  SNP indipendenti dopo pruning: $n_pruned"

echo ""
echo "==> Step 7: relatedness (KING-robust kinship, via plink2)"
if [ "$FORCE" -ne 1 ] && [ -f "$OUT_DIR/king.kin0" ]; then
    echo "  [skip, gia' presente] $OUT_DIR/king.kin0"
else
    plink2 --pfile "$OUT_DIR/merged_pruned" \
           --make-king-table \
           --out "$OUT_DIR/king"
fi
echo "  Output: $OUT_DIR/king.kin0 (colonne: #FID1 ID1 FID2 ID2 NSNP HETHET IBS0 KINSHIP)"

echo ""
echo "==> Step 8: PCA (10 componenti)"
if [ "$FORCE" -ne 1 ] && [ -f "$OUT_DIR/pca.eigenvec" ]; then
    echo "  [skip, gia' presente] $OUT_DIR/pca.eigenvec"
else
    plink2 --pfile "$OUT_DIR/merged_pruned" \
           --pca 10 \
           --out "$OUT_DIR/pca"
fi
echo "  Output: $OUT_DIR/pca.eigenvec, $OUT_DIR/pca.eigenval"

echo ""
echo "==> FATTO: $(date '+%Y-%m-%d %H:%M:%S')"
echo "File chiave per lo script di interpretazione python:"
echo "    - $OUT_DIR/king.kin0"
echo "    - $OUT_DIR/pca.eigenvec"
echo ""
echo "Log completo di questa run: $LOGFILE"
echo "Log per-job Step 1: $OUT_DIR/logs/step1_<batch>_chr<N>.log"
echo "Log per-job Step 2: $OUT_DIR/logs/step2_chr<N>.log"
echo ""
echo "Prossimo step: lancia interpret_plink_output.py passando questi due file"
echo "piu' i tuoi metadati di esposizione."