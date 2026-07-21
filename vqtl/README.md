# vQTL / G x E -- riscritto sopra `gene_environment_v2`

Questo package sostituisce il vecchio progetto `vqtl_pipeline` (config.yaml
proprio, VCF letti da soli con `cyvcf2`, QC/PCA "fatte in casa", niente
legame col resto della pipeline). Non e' piu' un progetto a se stante: e'
un terzo modulo dentro lo stesso repo di `gene_environment` e
`quality_control`, e usa la stessa `.env`, la stessa `Config`, lo stesso
logging, gli stessi helper id/PCA/statistiche.

```
gene_environment_v2/
  gene_environment/     # pipeline principale (invariata)
  quality_control/      # QC + PCA "vere" (plink2, invariata)
  vqtl/                 # <- questo package
    config.py
    cli.py
    core/
      data.py            Step 0/1/2 (in parte): join genetica+ambiente+PCA
      phenotype.py        Step 2: trasformazioni fenotipo (z/log/rint)
      scan.py              Step 3: scan vQTL QUAIL-style
      filter_candidates.py Step 4: manhattan/qq, P_gc, filtro candidati
      interaction.py        Step 5: test di interazione SNP x esposizione
      rge_het.py             Step 6: rGE + eteroschedasticita'
      permutation.py          Step 7: robustezza + permutazioni Freedman-Lane
      report.py                 Step 8: report.md + figure
      docx_report.py             Step 9: report.docx (Results + Supplementary
                                  Material pronti per il paper, in inglese)
    db/
      schema.sql          CREATE TABLE (da eseguire una volta, vedi sotto)
      repository.py        Tutte le query (placeholder/update/fetch a DB)
    .env.vqtl.example    variabili da aggiungere al .env esistente
  .env
```

## Cosa e' cambiato rispetto al vecchio `vqtl_pipeline`, e perche'

**Niente piu' lettura diretta dei VCF.** Il vecchio Step 0 (bcftools norm +
manifest per cromosoma) e la meta' del vecchio Step 1 (lettura VCF con
`cyvcf2` per costruire la matrice di dosaggio) sparisce del tutto: la
matrice genotipica genoma-wide, gia' filtrata MAF/LD-pruned, e' esattamente
`RAW_FILE` (il parquet prodotto da `gene_environment.cli filter-vcf` +
`build-matrix`). `vqtl/core/data.py` la carica riusando
`gene_environment.vcf_pipeline.build_dataset.load_and_prepare_data`, la
stessa funzione che usa il resto della pipeline: stesso join per id, stessa
pulizia degli id (`clean_sample_id`), stesso filtro di generazione/coorte.
Come conseguenza, anche gli Step 5/6/7/8 del vecchio progetto -- che
dovevano ciascuno *rileggere* i VCF con `extract_snp_dosage` per estrarre
solo le SNP candidate -- ora leggono semplicemente una colonna del
DataFrame gia' in memoria. E' la semplificazione piu' grande di questa
riscrittura.

**Niente piu' QC/PCA "fatte in casa".** Il vecchio Step 1 calcolava da solo,
in Python, un proxy IBS per la relatedness e una PCA su un campione casuale
ridotto di SNP (documentato nel vecchio README come "simplification, non
production-grade"). La pipeline `quality_control/` di questo repo fa gia'
QC e PCA vere con `plink2` (`00_run_plink_qc.sh` -> `extract_pca_covariates.py`),
il cui output (`pca_covariates.csv`, uno per generazione) e' gia' il
`PCA_COVARIATES_PATH_TEMPLATE` che `gene_environment` stesso usa come
covariata di correzione. `vqtl/core/data.py` carica le stesse PCA reali con
`gene_environment.analysis.pca_utils.load_pca_covariates` invece di
ricalcolarne una versione piu' debole.

**Una sola config, non due.** Il vecchio `config/config.yaml` duplicava a
mano `env_file`, covariate, soglie MAF/LD gia' presenti nel `.env` di
gene_environment (stesso path scritto due volte in due formati diversi).
`vqtl/config.py` NON ha un proprio file: legge le stesse variabili
d'ambiente del `.env` esistente tramite `gene_environment.config.get_config()`
e aggiunge solo le variabili genuinamente specifiche del metodo vQTL
(prefisso `VQTL_`, vedi `.env.vqtl.example`).

**La nozione di "coorte" (gen1/gen2/gen3) e' la stessa di gene_environment**:
`GENERATION` nel `.env` (o `--generation` da riga di comando, che sovrascrive
`GENERATION` solo per quel comando, senza toccare il file). Non serve piu'
un blocco `cohorts:` separato con path VCF propri.

## Cosa NON e' cambiato (per scelta esplicita)

La logica statistica del metodo vQTL vero e proprio -- quella che il README
del progetto originale descrive come corretta dopo un audit -- e' rimasta
**identica**, perche' e' l'unica parte genuinamente specifica di vqtl e non
ha alcun equivalente in gene_environment (che fa un test di interazione con
matching+permutazione, un metodo complementare ma diverso):

1. Scan vQTL genome-wide stile QUAIL (`core/scan.py`): residualizzazione
   OLS + quantile regression a coppie di tau, SE asintotica o bootstrap.
2. Filtro candidati di default su `P_gc` (corretta per inflazione genomica),
   non sulla P asintotica grezza (`core/filter_candidates.py`).
3. Test di interazione e rGE con SE robuste HC3/HC1 di default
   (`core/interaction.py`, `core/rge_het.py`).
4. Permutazione Freedman-Lane sui top loci, non permutazione ingenua del
   fenotipo grezzo (`core/permutation.py`).

## Uso

```bash
# dalla root del repo (dove vivono sia gene_environment/ sia vqtl/)
cd /srv/python-projects/gene_environment_v2

# 1. aggiungi le variabili VQTL_* al .env esistente
cat vqtl/.env.vqtl.example >> .env
# ... poi modifica i valori secondo necessita' (VQTL_EXPOSURES ecc.)

# 2. crea le tabelle vqtl_* sullo stesso DB di gene_environment (UNA VOLTA SOLA)
mysql -u <DB_USER> -p <DB_NAME> < vqtl/db/schema.sql

# 3. lancia la pipeline
python3 -m vqtl.cli run-all --generation 2   # prima volta: calcola tutto
python3 -m vqtl.cli run-all --generation 1   # se gen1 ha gia' risultati
                                              # significativi salvati, salta
                                              # lo scan e legge dalla cache

# singoli step (debug), stessa coorte:
python3 -m vqtl.cli scan --generation 1
python3 -m vqtl.cli filter --generation 1
python3 -m vqtl.cli interaction --generation 1
python3 -m vqtl.cli rge-het --generation 1
python3 -m vqtl.cli permute --generation 1
python3 -m vqtl.cli report --generation 1
python3 -m vqtl.cli docx --generation 1
```

**Tutti i risultati intermedi sono a DB, non piu' in `.tsv`** (vedi sezione
dedicata sotto). Solo i deliverable finali restano file, dentro
`VQTL_RESULTS_DIR/gen<N>/`: `report.md` (interno, in italiano), `report.docx`
(Results + Supplementary Material per il paper, in inglese -- vedi sezione
dedicata sotto), `figures/*.png`.

Il dataset unito (genetica+ambiente+PCA, il passaggio piu' costoso, legge
l'intero `RAW_FILE`) viene messo in cache in
`VQTL_RESULTS_DIR/gen<N>/vqtl_dataset.pkl` (stesso principio del
`TEMP_DF_PATH` usato da `gene_environment.analysis.orchestrator`), cosi'
lanciare gli step uno alla volta da riga di comando non lo ricostruisce ogni
volta. Passa `--force` per ricostruirlo (es. dopo aver rilanciato
`filter-vcf`/`build-matrix` a monte, o dopo un ricalcolo della PCA).

## Export Word per il paper (Step 9)

`report.docx` (generato automaticamente da `run-all`, o a se' con
`python3 -m vqtl.cli docx --generation <N>`) e' pensato per essere copiato
o allegato direttamente a un manoscritto: a differenza di `report.md`
(interno, in italiano), il suo CONTENUTO -- titoli, didascalie, intestazioni
di colonna, note -- e' interamente in **inglese**. Contiene:

- **Results**: Table 1 (top loci dello scan genoma-wide), Table 2 (SOLO i
  test di interazione nominale-significativi, `P < VQTL_INTERACTION_SIG_
  THRESHOLD`, default 0.05 -- fonte: `vqtl_interaction_results_significant`),
  Table 3 (validazione via permutazione dei top loci: interazione +
  varianza per genotipo, vedi sotto), Figure 1-3 (Manhattan, QQ, forest plot).
- **Supplementary Material**: Table S1 (scan genoma-wide completo, troncato
  a `VQTL_DOCX_SUPP_MAX_ROWS` righe con nota che rimanda alla tabella DB
  `vqtl_scan_results` per l'elenco integrale), Table S2 (TUTTI i test di
  interazione sui candidati, comprese le coppie non significative escluse
  da Table 2), Table S3 (screening rGE/eteroschedasticita'), Table S4
  (robustezza a trasformazioni del fenotipo), Supplementary Figures
  (boxplot + scatter per-locus).

Se Table 2 non ha righe (nessuna coppia SNP x esposizione raggiunge la
soglia nominale), il documento lo dice esplicitamente ("No rows to
display.") invece di lasciare una tabella vuota senza spiegazione o, peggio,
mostrare per sbaglio tutti i test come se fossero risultati significativi.

**Table 3 include anche un test di Levene permutazionale per la varianza
per genotipo**, non solo la permutazione Freedman-Lane sull'interazione:
per ogni top locus si calcola la statistica di Levene (Brown-Forsythe,
robusta a fenotipi non normali) sui gruppi genotipici reali (dosaggio
0/1/2) del fenotipo residualizzato sulle covariate, poi si permutano le
ETICHETTE di genotipo (non i residui, a differenza della permutazione
sull'interazione) e si ricalcola la statistica ad ogni permutazione,
costruendo una distribuzione nulla empirica -- niente assunzioni
asintotiche, la p-value viene interamente dai dati stessi. Serve a
confermare (o smentire), in modo assumption-light, l'effetto di varianza
che lo scan Step 3 (QUAIL, quantile regression, con le sue proprie
assunzioni asintotiche) ha rilevato per quel locus. Gira nello STESSO ciclo
di permutazione gia' usato per l'interazione (stessa infrastruttura
joblib/n_splits), non e' un passaggio separato; e' calcolato una volta per
SNP (non dipende dall'esposizione) e riusato per le righe successive dello
stesso SNP se compare con piu' esposizioni tra i top loci.

Tabelle in stile "three-line" (bordo sopra, bordo sotto l'header, bordo in
fondo, nessuna griglia interna), Times New Roman, documento in formato
landscape con margini ridotti -- necessario perche' le tabelle hanno
8-11 colonne e in portrait/margini standard Word va a capo a meta' parola
sui token senza spazi (id varianti, nomi esposizione). `VQTL_DOCX_TOP_N_SCAN`
controlla quante righe dello scan finiscono in Table 1 (corpo del paper,
default 20); `VQTL_DOCX_SUPP_MAX_ROWS` controlla il troncamento di
Supplementary Table S1 (default 200 -- oltre e' un file Word inutilizzabile,
si rimanda alla tabella DB `vqtl_scan_results`).

## Coerenza dei candidati tra un filtro e l'altro

Se rilanci lo Step 4 (`filter`, o `run-all` senza short-circuit) con una
soglia/`VQTL_FILTER_TOP_N` diversa da un run precedente, l'elenco dei
candidati puo' cambiare. Le varianti che ESCONO dal nuovo elenco vengono
ripulite automaticamente da tutte le tabelle a valle (`vqtl_interaction_
results`, `vqtl_rge_het_results`, `vqtl_robustness_results`, `vqtl_
permutation_results`) per quella generazione -- altrimenti resterebbero
righe orfane di un filtro precedente, che `fetch_results()` includerebbe
comunque nei risultati non avendo modo di sapere che quella variante non e'
piu' un candidato corrente. Loggato esplicitamente quando succede.

## Persistenza a DB: tutto/significativi, ripresa, short-circuit per generazione

Tutti gli step (3-7) scrivono su tabelle MySQL/MariaDB (`vqtl_*`, stesso DB
e stesso connection pool di `gene_environment` -- `gene_environment.db.
connection`, "PID-aware": sicuro anche se in futuro si aprissero connessioni
da processi worker), non piu' su `.tsv` intermedi. Schema completo in
`vqtl/db/schema.sql` (da eseguire una volta, vedi "Uso" sopra).

**Pattern comune a tutte le tabelle** (uguale a `variant_results` di
gene_environment): per ogni unita' di lavoro (variante, o coppia
variante+esposizione) viene inserito un placeholder `status='pending'`
prima di calcolare qualunque cosa; ogni chunk/riga completata viene
aggiornata (`status='done'`, o `'failed'` con `error_message` se e' andata
in eccezione) SUBITO, non alla fine dello step. Al riavvio dello stesso
comando, le unita' gia' `'done'` non vengono ripetute -- se il processo
viene interrotto (Ctrl+C, OOM, job del cluster ucciso), si riparte da dove
si era, non da zero. Se cambiano i parametri statistici (`VQTL_TAUS`,
`VQTL_SE_METHOD`, ecc.) o il set di varianti in `RAW_FILE`, la "fingerprint"
salvata in `vqtl_scan_runs` non corrisponde piu' e le righe di quella
generazione vengono ripulite e ricalcolate da zero automaticamente (non
serve fare nulla a mano).

**"Tutto" vs "solo i significativi"**: `vqtl_scan_results` (Step 3+4) e
`vqtl_interaction_results` (Step 5) hanno una tabella gemella --
`vqtl_scan_results_significant` e `vqtl_interaction_results_significant` --
che contiene SOLO il sottoinsieme rispettivamente candidato
(`is_candidate=1`, filtro Step 4) e nominale-significativo
(`pval < VQTL_INTERACTION_SIG_THRESHOLD`, default 0.05). Sono risincronizzate
(DELETE + INSERT) ogni volta che gira lo step corrispondente, quindi sono
sempre uno specchio esatto, mai stantie. Servono a due cose:
1. sono la fonte diretta di Table 1/Table 2 (Results) del `report.docx`,
   mentre le tabelle "tutto" restano la fonte delle Supplementary Table S1/S2;
2. **`vqtl_scan_results_significant` e' anche il segnale di short-circuit**
   per l'intera generazione (vedi sotto).

rGE/eteroschedasticita' (Step 6), robustezza e permutazioni (Step 7) restano
solo "tutto" (niente tabella `_significant`): girano gia' solo sui pochi
candidati/top-loci (un insieme gia' piccolo), e saltano gia' da soli le
coppie gia' `'done'` -- una tabella `_significant` in piu' non avrebbe
aggiunto un vero short-circuit, solo una tabella da mantenere sincronizzata.

**Short-circuit per generazione** (questo e' il comportamento chiesto: "gen2
fa tutto, gen1 se ha gia' i significativi non deve ricalcolare niente"): a
ogni `run-all`, prima di caricare qualunque dato, si controlla
`vqtl_scan_results_significant` per quella generazione. Se ha gia' righe (e
non e' stato passato `--force`), lo scan genoma-wide E il filtro vengono
saltati del tutto -- niente quantile regression, niente nuove query di
placeholder -- e Step 5-7 partono direttamente dai candidati gia' noti letti
da DB (loggato esplicitamente: *"Generazione N ha gia' M risultati
significativi salvati a DB: salto lo scan genoma-wide e il filtro"*). Le
figure Manhattan/QQ vengono comunque rigenerate (economico) dai dati gia' in
cache, per sicurezza. Se e' la prima volta che una generazione viene
analizzata (nessun `vqtl_scan_results_significant` per quel numero), si
calcola tutto da capo, esattamente come per una generazione mai vista prima
-- `--force` forza comunque un ricalcolo completo anche se la cache c'e' gia'.

Verificato con un run reale su MariaDB in questo lavoro di sviluppo: prima
esecuzione di gen2 (calcolo completo), prima esecuzione di gen1 (calcolo
completo), seconda esecuzione di gen1 (short-circuit: 0 combinazioni
ricalcolate in tutti gli step, report e docx rigenerati dalla sola cache).

## Step 3 (scan genoma-wide): quanto costa

Lo Step 3 e' il piu' pesante della pipeline: per ogni variante fitta fino a
`2 x len(VQTL_TAUS)` quantile regression (default 9 tau -> 18 fit/variante
con `VQTL_SE_METHOD=asymptotic`, il default -- beta e SE si ricavano dagli
stessi fit, non da due round separati). Con `VQTL_SE_METHOD=bootstrap` il
costo sale a `18 x VQTL_BOOTSTRAP_K` fit per variante (default 200x -> ~3600
fit/variante): **non usarlo genoma-wide**, ha senso solo su una manciata di
loci gia' selezionati. Su uno scan genoma-wide reale (decine/centinaia di
migliaia di varianti dopo il filtro MAF/LD) il tempo totale dipende quindi
soprattutto da: numero di varianti in `RAW_FILE`, `VQTL_TAUS` (meno tau =
piu' veloce, linearmente), `VQTL_SE_METHOD`, e `MAX_WORKERS`/`VQTL_N_JOBS`
(parallelizzato per chunk, un chunk = `VQTL_CHUNK_SIZE` varianti, i risultati
di ogni chunk arrivano al processo principale e vengono scritti a DB man
mano che sono pronti, non tutti insieme alla fine).

## Escludere una covariata solo da vqtl (es. onset_site)

Le covariate di default sono le stesse di `gene_environment`
(`COVARIATES` nel `.env`). Per usarne un sottoinsieme diverso SOLO in vqtl,
senza toccare `COVARIATES` (che resta condivisa col resto della pipeline,
incluso il test di interazione via matching di gene_environment), imposta
`VQTL_COVARIATES` nel `.env`:

```
VQTL_COVARIATES=sex,diagnostic_delay
```

Se `VQTL_COVARIATES` non e' impostato, il comportamento resta quello di
sempre (usa `COVARIATES`). Se invece vuoi togliere `onset_site` ovunque,
comprese le analisi di gene_environment, modifica direttamente `COVARIATES`
nel `.env` condiviso (non serve `VQTL_COVARIATES` in quel caso).

## Cosa verificare prima di un run reale

- **`DB_USER` / `DB_PASSWORD` / `DB_NAME` (/ `DB_HOST`, default `127.0.0.1`) nel `.env`**:
  a differenza di prima, ora vqtl usa DAVVERO il database (tutti i risultati
  intermedi ci vivono, vedi sopra), quindi queste variabili non sono piu'
  solo un requisito formale di `gene_environment.config.get_config()`
  (`DBConfig` le richiede comunque, sempre, anche per comandi che non
  toccano il DB) ma servono a vqtl per funzionare. Prima del primo run:
  `mysql -u <DB_USER> -p <DB_NAME> < vqtl/db/schema.sql`.
- **Dipendenze**: vedi `requirements-vqtl.txt`. Aggiunta `mysql-connector-python`
  (probabilmente gia' installata, essendo gia' una dipendenza di
  `gene_environment.db.connection`). Rispetto al vecchio `requirements.txt`
  di `vqtl_pipeline` non servono piu' `cyvcf2`, `pysnptools`, `dask`,
  `pyyaml` (rimossi perche' non piu' usati, vedi sopra).
- **`VQTL_EXPOSURES`**: se non lo imposti, viene testata solo `EXPOSURE`
  (singola, come nel resto della pipeline). Il progetto originale ne testava
  3 (`seminativi_1500, risaie_1500, vigneti_1500`) -- imposta esplicitamente
  la lista se vuoi lo stesso comportamento. Se cambi questa variabile tra un
  run e l'altro, la cache del dataset (vedi sopra) viene invalidata e
  ricostruita automaticamente.
- **PCA**: assicurati che `quality_control/00_run_plink_qc.sh` +
  `extract_pca_covariates.py` siano gia' stati eseguiti per la generazione
  che vuoi analizzare (stesso requisito di `gene_environment`, non
  specifico di vqtl). Gli ID in `pca_covariates.csv` vengono normalizzati
  con la stessa `clean_sample_id` usata per il resto della pipeline prima
  del merge (necessario: il file PCA ha gli ID nel formato "raddoppiato"
  grezzo di plink, il dataframe principale li ha gia' ripuliti).
- **MAF/LD**: il filtro e' gia' applicato a monte da
  `gene_environment.cli filter-vcf`; `VQTL_MIN_MAF`/`VQTL_MIN_CALL_RATE`
  qui sono di default disattivi (0.0) per non rifiltrare due volte -- non
  sono piu' equivalenti ai `vqtl.min_maf`/`min_call_rate` del vecchio
  `config.yaml`, che invece erano l'UNICO filtro applicato.
- **`VQTL_N_PERM`**: se lasciato a 0 riusa `N_PERM_HIGH` di
  gene_environment; per un run vqtl completo con permutazioni "serie" sui
  top loci, verifica che quel valore sia adeguato (nel progetto originale il
  default era 1000, comparabile).
