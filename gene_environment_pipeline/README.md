# Gene-Environment Pipeline — refactor

Refactor completo della pipeline di analisi gene x ambiente. Sostituisce i
19 script originali (eseguiti a mano, nell'ordine descritto in
`ordine_comandi.txt`) con un pacchetto Python organizzato, un CLI unico, e
tutti i fix elencati sotto.

## ⚠️ Prima cosa da fare: ruota la password del DB

La configurazione condivisa conteneva `DB_PASSWORD` in chiaro. Quella
password va considerata compromessa (è finita in una chat) — **vai a
ruotarla sul server MySQL prima di mettere in produzione questo codice**.
Nel refactor le credenziali non sono più nel codice: vanno in un file
`.env` locale (mai committato) o in variabili d'ambiente di sistema. Vedi
`.env.example`.

## Struttura

```
gene_environment/
  config.py                 configurazione centralizzata (env vars)
  logging_utils.py          logging centralizzato (console + file, rotazione)
  cli.py                    entry point unico: python -m gene_environment.cli <comando>
  db/
    connection.py           connection pool MySQL + context manager
    repository.py           tutte le query (ex db.py), incluse le colonne onset_age
  vcf_pipeline/
    filter_vcf.py            ex gene_reduction.py — ora PARALLELIZZATO per file
    vcf_to_parquet.py        ex vcf_to_csv+create_chr_csv+create_full_csv+csv_to_parquet
                              — merge fra cromosomi ora basato su ID, non su ordine riga
    build_dataset.py         ex data_loader.py — bug duplicazione id risolto
  analysis/
    matching.py              ex matching.py (fix minori, log)
    onset_age_stats.py       statistiche onset_age, riusabili e vettorizzate
    modeling.py              ex modeling.py — fix seed instabile, early-stop adattivo,
                              onset_age calcolato e salvato subito
    orchestrator.py          ex main.py — bulk insert, log, path configurabili
    report_onset_age.py      ex analyze_variant_onset_age.py — legge da DB, non ricalcola
  significant_variants/
    extract_matrix.py        ex extract_significant_variant_matrices.py — scrittura
                              incrementale + checkpoint di ripresa + parallelo per cromosoma
    export_significant_csv.py  NUOVO script standalone, esportabile in ogni momento
  gene_annotation/
    annotate_genes.py        ex process_variants.py + main_gene_analysis.py, uniti
  utils/
    id_utils.py               clean_sample_id/parse_variant_label centralizzati
    stats_utils.py            add_fdr/volcano_plot (fix log10(0))
schema_migration.sql          ALTER TABLE per le nuove colonne onset_age
.env.example                  template configurazione (senza segreti)
```

## Uso

```bash
pip install -r requirements.txt
cp .env.example .env   # e compila i valori veri
python -m gene_environment.cli pipeline-order   # stampa l'ordine consigliato
```

Ordine consigliato degli step (uguale nello spirito a `ordine_comandi.txt`,
ma ognuno è un comando esplicito invece di uno script da ricordare a mano):

```bash
python -m gene_environment.cli filter-vcf
python -m gene_environment.cli build-matrix
python -m gene_environment.cli run-model
python -m gene_environment.cli extract-significant
python -m gene_environment.cli export-significant-csv
python -m gene_environment.cli report-onset-age
python -m gene_environment.cli assign-genes
python -m gene_environment.cli annotate-genes
```

`export-significant-csv` è pensato per essere rilanciato quante volte serve
(anche da cron) mentre `run-model` è ancora in corso: produce sempre uno
snapshot CSV aggiornato delle varianti significative, con le statistiche
onset_age già incluse, in una cartella separata (`SIGNIFICANT_EXPORT_DIR`).

## Problemi critici risolti

1. **Credenziali in chiaro nel codice** → spostate in variabili d'ambiente
   (`config.py` + `.env.example`), con validazione esplicita all'avvio se
   mancano.
2. **Seed non riproducibile** (`modeling.py`): `hash(variant_col)` su
   stringa non è stabile fra esecuzioni diverse di Python (hash
   randomization). Sostituito con un hash MD5 deterministico → risultati
   delle permutazioni ora riproducibili al 100% a parità di RANDOM_STATE.
3. **Merge fra cromosomi fragile e basato sull'ordine delle righe**
   (`create_full_csv.py`): sostituito con un merge basato su ID (pandas),
   robusto indipendentemente dall'ordine dei campioni nei singoli file.
4. **Duplicazione dell'id paziente come workaround**
   (`df_env['id']+'_'+df_env['id']` in `data_loader.py`, poi "ripulito" di
   nuovo in `analyze_variant_onset_age.py`): unificata in una sola funzione
   `clean_sample_id`, usata ovunque serva.
5. **Statistiche onset_age calcolate a posteriori in uno script separato**:
   ora calcolate DENTRO `modeling.py`, sullo stesso dataset usato per il
   modello, e salvate a DB nella stessa riga/stessa transazione (vedi
   `schema_migration.sql`).
6. **Nessuna scrittura incrementale** nell'estrazione delle varianti
   significative: `extract_matrix.py` ora scrive/appende il CSV dopo ogni
   generazione completata, con un checkpoint JSON per riprendere un run
   interrotto senza ripartire da zero.
7. **Nessun export separato e ripetibile delle varianti significative**:
   aggiunto `export_significant_csv.py`, eseguibile in qualunque momento,
   che scrive in una cartella diversa da quella dell'estrazione genotipi.
8. **`gene_reduction.py` sequenziale** nonostante `MAX_WORKERS` fosse già
   configurato: ora parallelizzato per file VCF.
9. **`volcano_plot`**: `-log10(0)` produceva `+inf` quando un p-value
   empirico era 0 (capita con permutation test). Ora i p-value sono
   clippati a un minimo positivo prima del log.
10. **Bootstrap onset_age vettorizzato**: il loop Python a 2000 iterazioni
    per variante x coorte è ora un resample matriciale numpy (10-50x più
    veloce), impattante perché ripetuto per ogni variante significativa.
11. **Early-stop adattivo nelle permutazioni** (`modeling.py`): se dopo un
    primo blocco di permutazioni il p-value parziale è già chiaramente non
    significativo, ci si ferma prima di sprecare il resto delle N_PERM
    (l'operazione più costosa di tutta la pipeline).
12. **Connection pooling MySQL**: prima ogni funzione in `db.py` apriva una
    connessione nuova; ora un pool condiviso (`db/connection.py`) e
    `executemany` per gli insert/update in batch.
13. **Logging uniforme** (console + file con rotazione) al posto di `print()`
    sparsi, con PID nei worker paralleli per distinguere i log.
14. **`vcf_to_csv.py`**: costruzione del genotipo vettorizzata con numpy
    invece di liste Python annidate, e scrittura diretta in Parquet invece
    di CSV intermedi giganti.
15. **Trattamento del genotipo mancante come "0/wild-type"**: comportamento
    mantenuto per compatibilità con le analisi precedenti, ma reso
    ESPLICITO e configurabile (`missing_strategy` in `vcf_to_parquet.py`)
    invece di una riga di codice silenziosa — è una scelta di modellazione,
    non un dettaglio tecnico, e vale la pena discuterla con chi ha definito
    l'analisi originale.
16. **`get_variants_after_gen1.py`**: superato da `extract_matrix.py`, che
    usa query indicizzate bcftools invece di scansionare i VCF riga per
    riga; non incluso nel refactor (tenuto solo come riferimento storico).

## Cosa NON è stato (e non poteva essere) testato end-to-end

Questo refactor è stato scritto e verificato per correttezza sintattica
(`python -m py_compile` su tutti i moduli) ma **non** è stato eseguito
contro il database, i VCF o i CSV reali (non disponibili in questo
ambiente). Prima di usarlo in produzione:
- esegui `schema_migration.sql` su un DB di **staging** e verifica che
  corrisponda esattamente allo schema reale (i nomi/tipi delle colonne
  esistenti sono stati dedotti dalle query in `db.py`, non da uno schema
  esplicito);
- lancia `run-model` su un piccolo sottoinsieme di varianti/pazienti e
  confronta i risultati con l'output della pipeline originale;
- rivedi la scelta "missing genotype -> 0" (punto 15) con chi ha il
  contesto biologico/clinico del progetto.
