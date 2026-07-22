# vqtl -- bug trovati e corretti scrivendo test/test_vqtl_pipeline.py

Trovati eseguendo davvero la pipeline (Step 1-7) su dati sintetici con
gene_environment reale, non solo per lettura del codice. Ordine: quello in
cui sono emersi.

## 1. `cli.py`: `TypeError` su `scan` e `run-all` (CORRETTO)

`cmd_scan`/`cmd_run_all` chiamavano `run_vqtl_scan(..., variant_subset=subset)`
ma la funzione non accettava quel parametro -- crash immediato su entrambi i
comandi principali, indipendentemente da `--significant-only`.

Fix: `run_vqtl_scan()` in `core/scan.py` ora accetta `variant_subset` e lo
usa per restringere le colonne scansionate (comportamento invariato se
`None`).

## 2. `core/data.py`: import errato di `load_pca_covariates` (CORRETTO)

Importava da `gene_environment.analysis.pca_utils` (modulo inesistente); il
modulo reale, secondo `gene_environment/vcf_pipeline/build_dataset.py` e i
commenti in `gene_environment/config.py`, e' `gene_environment.utils.pca_utils`.
`ModuleNotFoundError` al primo import di `vqtl.core.data`.

## 3. `core/data.py`: `load_and_prepare_data()` sballava lo unpacking (CORRETTO)

`load_and_prepare_data` ritorna 6 valori (`df, variant_cols_safe, mapping,
Ecols, variant_cols, covariate_cols`), `load_vqtl_dataset` ne spacchettava
solo 5. `ValueError: too many values to unpack` ad ogni chiamata.

## 4. `core/data.py`: doppio merge delle PCA (CORRETTO)

`gene_environment.vcf_pipeline.build_dataset.load_and_prepare_data` unisce
GIA' le PCA al dataframe restituito quando `USE_PCA_COVARIATES=true` (dentro
`_build_narrow_covariates`). `vqtl/core/data.py` non lo sapeva e le
ri-caricava/ri-univa una seconda volta: pandas rinominava le colonne
sovrapposte in `PC1_x`/`PC1_y` invece di `PC1`, e tutto il codice a valle che
si aspettava `df["PC1"]` falliva con `KeyError`. Bug bloccante ogni volta
che le PCA sono attive (il caso normale/di default).

Fix: rimosso il secondo merge, si riusano le colonne PC gia' presenti nel
dataframe.

## 5. `interaction.py` / `rge_het.py` / `permutation.py`: genotipi mancanti (CORRETTO)

`core/scan.py` gestisce correttamente i genotipi mancanti (codificati come
`"."` nei file prodotti dalla pipeline VCF reale) tramite `dosage_matrix()`
(`pd.to_numeric(..., errors="coerce")`). Gli altri tre step leggevano invece
il dosaggio con `dataset.df[safe_col].to_numpy(dtype=float)` diretto, che
esplode (`ValueError: could not convert string to float: '.'`) appena c'e'
un genotipo mancante -- cioe' sempre, sui dati reali. Corretto in tutti e 4 i
punti (`interaction.py` x1, `rge_het.py` x1, `permutation.py` x2) per usare
`dosage_matrix()` come `scan.py`.

## 6. `core/scan.py`: SE "asymptotic" del Step 3 NON calibrata (NON CORRETTO -- vedi sotto)

**Non un typo, una questione metodologica**: lasciato inalterato in attesa
di una decisione.

`_beta_qi_and_asymptotic_se()` calcola `beta_QI` come media di K differenze
di regressione quantile (una per ogni `tau` in `vcfg.taus`, K=9 di default),
e stima la sua SE come `mean(SE_per_tau) / sqrt(K)` -- formula corretta SOLO
se le K differenze fossero fra loro indipendenti. Non lo sono: sono tutte
stime di regressione quantile sullo STESSO campione a livelli di quantile
adiacenti, quindi correlate positivamente. La formula sottostima
sistematicamente la SE reale.

Verificato con una simulazione pulita (dosaggio indipendente dal residuo per
costruzione, quindi sotto la vera H0): su 2000 repliche, gli Z attesi N(0,1)
hanno invece sd=1.72 e lambda_GC=2.9 (atteso 1 in entrambi i casi). Lo stesso
pattern si vede nella pipeline reale su dati sintetici (test_vqtl_pipeline.py):
lambda_GC=3.25 con un pool di 300 varianti nulle, e i candidati selezionati
allo Step 4 sono in maggioranza falsi positivi (14/300 nulle) mentre solo
1/7 varianti causali viene recuperata.

Il metodo alternativo gia' presente, `VQTL_SE_METHOD=bootstrap` (non
default), risulta invece ben calibrato nella stessa simulazione (sd Z=0.95,
lambda_GC=1.20): ricampiona dosaggio+residuo insieme e rifitta l'intero
`beta_QI` ad ogni replica, catturando correttamente la correlazione fra i
`tau`. E' pero' molto piu' costoso (K x bootstrap_k fit invece di K), quindi
non banale da adottare come default per uno scan genoma-wide su milioni di
varianti senza discuterne il trade-off costo/correttezza.

**Non ho cambiato il default ne' la formula**: e' una decisione che tocca la
validita' scientifica dei risultati (manoscritti in corso inclusi), non un
bug di battitura -- da qui la scelta di fermarmi e chiedere prima di agire.
Nota anche che la correzione P_gc del Step 4 (genomic control) attenua in
parte l'inflazione, ma assumendo che il fattore di inflazione sia uniforme
su tutte le varianti/MAF, cosa non verificata.
