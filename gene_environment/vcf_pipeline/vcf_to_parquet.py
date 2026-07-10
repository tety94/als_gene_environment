"""
Converte i VCF filtrati in un'unica matrice genotipica (parquet), sostituendo
la catena originale: vcf_to_csv.py -> create_chr_csv.py -> create_full_csv.py
-> csv_to_parquet.py (4 script, 3 formati CSV intermedi enormi su disco).

BUG PIÙ GRAVE TROVATO E CORRETTO (create_full_csv.py):
  Il merge tra i CSV dei vari cromosomi avveniva così:
    for rows in zip(*files):
        base = rows[0].strip()          # id dal primo file
        ... rest = colonne genotipo degli altri file, prese "così come sono"
  cioè si assumeva che la riga N-esima di OGNI file di cromosoma
  corrispondesse allo STESSO campione, basandosi solo sull'ORDINE delle
  righe, non sull'id. C'era un controllo preliminare (ref_ids == ids) che
  falliva rumorosamente se l'ordine differiva fra file — quindi non è un bug
  "silenzioso" nella versione attuale, ma è comunque un approccio fragile:
  qualunque riordino, anche solo di un file, blocca l'intera pipeline con
  un RuntimeError e nessuna possibilità di recovery parziale, e il parsing è
  fatto a mano con `.split(",")` (nessuna gestione di quoting/escaping).

  Qui il merge fra cromosomi è invece un JOIN basato sull'id campione
  (pandas, allineamento per indice, non per posizione di riga): funziona
  indipendentemente dall'ordine delle righe nei singoli file, ed è quello
  che ci si aspetterebbe da un'operazione di merge.

FIX (10 luglio 2026 - crash "Exceeded size limit" su parquet, VCF di input
enormi e corruzione a catena):
  - L'errore veniva da un file .raw.parquet scritto PARZIALMENTE (worker
    ucciso / run interrotto): pyarrow non riesce a leggere il footer thrift
    di un parquet troncato e lancia un OSError poco chiaro. Ora:
      1) OGNI scrittura di parquet (raw per-file, merge per-cromosoma,
         genoma finale) è ATOMICA: si scrive su "<path>.tmp" e poi si fa
         os.replace(), quindi un processo ucciso a metà non lascia mai un
         file corrotto sul path definitivo.
      2) Prima di leggere/skippare un parquet esistente lo si VALIDA
         (apertura del footer via pyarrow.parquet.ParquetFile). Se non è
         valido viene cancellato e rigenerato, invece di far esplodere
         tutta la pipeline con un traceback in fondo alla catena.
      3) merge_chromosome non muore più al primo file corrotto: elimina il
         file corrotto (verrà rigenerato al prossimo run) e segnala quel
         cromosoma come "da rifare", ma lascia proseguire gli altri.
      4) La pipeline è quindi RIPRENDIBILE: rilanciando lo stesso comando,
         tutto ciò che è già valido viene saltato (niente ricalcolo dei
         290 campioni × 1M+ varianti per cromosoma già fatti), solo ciò che
         manca o è rotto viene rigenerato.
  - Il matching "quale file appartiene a quale cromosoma" usava
    `"chr1." in nome or "chr1_" in nome`, che può dare falsi positivi (es.
    varianti di naming multiple per lo stesso cromosoma). Sostituito con
    una regex che richiede che dopo il numero di cromosoma non segua
    un'altra cifra (niente più confusione chr1/chr11/chr12...).
  - Il logging ora scrive anche su file (non solo su console), sia nel
    processo principale sia in ogni worker (i worker sono processi separati
    e non ereditano gli handler di logging del padre).
  - NUOVO - VCF di input (.vcf.gz) corrotti/troncati: filter_vcf.py scrive
    ora i propri output in bgzip (*_filtered.vcf.gz) invece di VCF testuale
    enorme; il glob qui è stato aggiornato di conseguenza. Se uno di questi
    .vcf.gz risulta troncato (run precedente interrotto), invece di far
    fallire cyvcf2 con un errore criptico in mezzo alla conversione, il
    file viene VALIDATO prima di essere letto (decompressione completa) e,
    se non valido, viene ELIMINATO: al prossimo rilancio di filter_vcf.py
    verrà rigenerato automaticamente (idempotenza), e questo script segnala
    il file come "da rifare" invece di bloccare l'intero batch.
  - NUOVO - statistiche numeriche: oltre ai log testuali già presenti
    (varianti scartate per missing, shape di ogni output), viene ora
    scritto anche un CSV riepilogativo per cromosoma in
    <log_dir>/vcf_to_parquet_stats.csv (campioni, varianti totali, varianti
    scartate per missing rate, varianti finali).

ALTRE OTTIMIZZAZIONI (invariate):
  - Niente più CSV intermedi giganti: si scrive direttamente in Parquet
    (compresso, colonnare, molto più leggero/veloce da rileggere) ad ogni
    stadio (per-file, per-cromosome, genoma intero).
  - Parallelizzazione a livello di file VCF (ProcessPoolExecutor).
  - Il filtro sulla percentuale di missing per SNP viene calcolato PRIMA
    della binarizzazione (come nell'originale), altrimenti l'informazione
    "quanti missing aveva questo SNP" andrebbe persa una volta forzati a 0.
  - La scelta "genotipo mancante -> 0 (non mutato)" dell'originale è una
    decisione di modellazione, non un dettaglio tecnico: qui è esplicita e
    configurabile (MISSING_GENOTYPE_STRATEGY), di default "zero" per
    compatibilità con le analisi precedenti, ma segnalata chiaramente nel
    log e nei commenti perché è la scelta più delicata di tutta la
    conversione dei dati (trattare un dato mancante come "wild type" può
    introdurre bias se il missing non è casuale).
"""
from __future__ import annotations

import csv
import gzip
import logging
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from glob import glob

import numpy as np
import pandas as pd

from gene_environment.config import get_config, get_generation_vcf_folders
from gene_environment.logging_utils import configure_logging, get_logger
from gene_environment.utils.id_utils import clean_sample_id

log = get_logger(__name__)

CHROMOSOMES = [str(i) for i in range(1, 23)]

LOG_FILENAME = "vcf_to_parquet.log"
STATS_FILENAME = "vcf_to_parquet_stats.csv"


class CorruptParquetError(Exception):
    """Sollevata quando uno o più file .raw.parquet risultano corrotti/troncati.
    Il file corrotto viene eliminato prima di sollevare l'eccezione, quindi
    un semplice rilancio della pipeline lo rigenera."""


class CorruptInputVCFError(Exception):
    """Sollevata quando uno o più *_filtered.vcf.gz (output di filter_vcf.py)
    risultano corrotti/troncati. Il file viene eliminato prima di sollevare
    l'eccezione: al prossimo rilancio di filter_vcf.py verrà rigenerato
    automaticamente grazie alla sua idempotenza."""


def _add_file_logging(log_dir: str) -> None:
    """Aggiunge (una sola volta per processo) un FileHandler al root logger,
    così i log finiscono sia su console sia su <log_dir>/vcf_to_parquet.log.
    Va richiamata sia nel processo principale sia in ogni worker (sono
    processi separati e non ereditano gli handler di logging del padre)."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.abspath(os.path.join(log_dir, LOG_FILENAME))

    root = logging.getLogger()
    for h in root.handlers:
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == log_path:
            return  # già aggiunto in questo processo

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [pid=%(process)d] %(name)s: %(message)s"))
    fh.setLevel(logging.INFO)
    root.addHandler(fh)
    if root.level > logging.INFO or root.level == logging.NOTSET:
        root.setLevel(logging.INFO)
    log.info("Logging su file abilitato: %s", log_path)


def _is_valid_parquet(path: str) -> bool:
    """True se il file esiste ed è un parquet leggibile (footer thrift ok).
    Non legge i dati per intero, solo i metadati: è veloce anche su file
    grandi."""
    if not os.path.exists(path):
        return False
    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        _ = pf.metadata  # forza la lettura/validazione del footer
        return True
    except Exception as e:
        log.warning("Parquet non valido/corrotto, verrà rigenerato: %s (%s)", path, e)
        return False


def _is_valid_bgzip_vcf(path: str) -> bool:
    """True se il *_filtered.vcf.gz è un bgzip/gzip leggibile per intero
    (nessun blocco troncato). Simmetrico al controllo fatto lato
    filter_vcf.py prima di scrivere il file: qui serve a distinguere un VCF
    di input genuinamente corrotto da un errore diverso durante il parsing
    con cyvcf2."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        with gzip.open(path, "rb") as f:
            while f.read(1024 * 1024):
                pass
        return True
    except Exception as e:
        log.warning("VCF di input non valido/troncato: %s (%s)", path, e)
        return False


def _write_parquet_atomic(df: pd.DataFrame, out_path: str, **to_parquet_kwargs) -> None:
    """Scrive su <out_path>.<pid>.tmp e poi rinomina atomicamente su out_path.
    Garantisce che, se il processo viene ucciso a metà scrittura (OOM,
    kill, crash del worker, disco pieno...), il path finale non contenga
    MAI un parquet troncato: o non esiste, o è quello vecchio (valido), o
    è quello nuovo completo. Il pid nel nome evita che due processi/run
    concorrenti si scrivano addosso lo stesso file temporaneo."""
    tmp_path = f"{out_path}.{os.getpid()}.tmp"
    try:
        df.to_parquet(tmp_path, engine="pyarrow", **to_parquet_kwargs)
        os.replace(tmp_path, out_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

def _genotype_to_dosage(gt) -> int:
    """gt: tupla cyvcf2 (allele1, allele2, phased). -1 = missing."""
    if gt is None or gt[0] is None or gt[1] is None:
        return -1
    a, b = gt[0], gt[1]
    if a < 0 or b < 0:
        return -1
    return a + b


def vcf_file_to_dosage_df(vcf_path: str) -> pd.DataFrame:
    """Legge un VCF filtrato (bgzip) e ritorna un DataFrame (samples x
    varianti) di dosaggi grezzi (0/1/2/-1=missing). Import di cyvcf2 fatto
    qui dentro per non richiederlo come dipendenza hard di tutto il
    pacchetto. cyvcf2 legge .vcf.gz bgzip nativamente, nessuna
    decompressione manuale necessaria qui."""
    from cyvcf2 import VCF

    vcf = VCF(vcf_path)
    samples = vcf.samples

    variant_ids = []
    columns = []  # una colonna (np.array) per variante

    for variant in vcf:
        alt_allele = variant.ALT[0] if variant.ALT else "."
        var_id = f"{variant.CHROM}_{variant.POS}_{variant.REF}_{alt_allele}"
        variant_ids.append(var_id)
        col = np.fromiter((_genotype_to_dosage(gt) for gt in variant.genotypes), dtype=np.int8, count=len(samples))
        columns.append(col)

    if not columns:
        return pd.DataFrame(index=samples)

    arr = np.column_stack(columns)
    return pd.DataFrame(arr, index=samples, columns=variant_ids)


def _process_single_vcf_worker(args) -> tuple[str, int, list[str], bool]:
    """Ritorna (out_parquet, generation, samples, input_corrotto).
    Se l'ultimo elemento è True, out_parquet/samples sono vuoti/non
    validi: il file di input è stato eliminato e va segnalato al
    chiamante."""
    vcf_path, out_parquet, log_dir, generation = args
    configure_logging(log_dir)
    _add_file_logging(log_dir)

    if _is_valid_parquet(out_parquet):
        log.info("Skip (già convertito e valido): %s", out_parquet)
        samples = pd.read_parquet(out_parquet, columns=[]).index.tolist()
        return out_parquet, generation, samples, False

    if os.path.exists(out_parquet):
        log.warning("File esistente ma corrotto/troncato, lo rigenero: %s", out_parquet)
        os.remove(out_parquet)

    if not _is_valid_bgzip_vcf(vcf_path):
        log.error(
            "VCF di input corrotto/troncato, lo elimino per forzare la rigenerazione da "
            "filter_vcf.py al prossimo run: %s", vcf_path,
        )
        os.remove(vcf_path)
        return "", generation, [], True

    log.info("Converto VCF -> parquet (generazione %d): %s", generation, vcf_path)
    try:
        df = vcf_file_to_dosage_df(vcf_path)
    except Exception as e:
        # cyvcf2 può fallire anche su un bgzip "valido" a livello di blocchi
        # ma con contenuto VCF troncato/malformato a metà (es. riga tagliata
        # a metà scrittura non intercettata dal controllo gzip). Trattato
        # allo stesso modo: elimino e segnalo per rigenerazione.
        log.error(
            "Errore leggendo il VCF con cyvcf2, lo considero corrotto e lo elimino: %s (%s)",
            vcf_path, e,
        )
        os.remove(vcf_path)
        return "", generation, [], True

    _write_parquet_atomic(df, out_parquet, compression="zstd")
    log.info("Scritto %s (%d campioni, %d varianti)", out_parquet, df.shape[0], df.shape[1])
    return out_parquet, generation, df.index.tolist(), False


def _detect_duplicate_chrom_sources(vcf_paths: list[str]) -> None:
    """Segnala (senza bloccare) se più di un VCF filtrato sembra riferirsi
    allo stesso cromosoma nella stessa cartella: capita quando restano in
    giro file di run precedenti con naming diverso (es.
    'chr1_filtered.vcf.gz' e 'chr1.vcf_filtered.vcf.gz' insieme), e porta a
    processare due volte lo stesso cromosoma inutilmente."""
    by_chrom: dict[str, list[str]] = {}
    for p in vcf_paths:
        name = os.path.basename(p)
        m = re.search(r"chr(\d+)(?!\d)", name)
        if m:
            by_chrom.setdefault(m.group(1), []).append(p)
    for chrom, paths in by_chrom.items():
        if len(paths) > 1:
            log.warning(
                "chr%s: trovati %d file VCF filtrati nella stessa cartella (possibili "
                "residui di run precedenti con naming diverso) - verranno processati "
                "TUTTI, controlla se è voluto: %s",
                chrom, len(paths), paths,
            )


def convert_filtered_vcfs_to_parquet() -> tuple[list[str], dict[str, int]]:
    """Step 1: ogni *_filtered.vcf.gz -> un parquet grezzo (dosaggi 0/1/2/-1).

    Ritorna anche la mappa id_campione -> generazione, costruita in base a
    QUALE cartella (VCF_DIR_GENn) proviene ogni file VCF: è l'unica fonte
    affidabile della coorte di un paziente quando il file ambientale non
    contiene alcuna informazione di generazione (il join fra ambiente e
    genetica avviene solo per id).

    Se uno o più VCF di input risultano corrotti vengono eliminati (così
    che filter_vcf.py li rigeneri al prossimo run) e viene sollevata
    CorruptInputVCFError alla fine, DOPO aver comunque processato tutti gli
    altri file: un input corrotto non blocca più l'intero batch."""
    cfg = get_config()
    _add_file_logging(cfg.log_dir)

    jobs = []
    for generation, folder in get_generation_vcf_folders(cfg).items():
        vcf_filtered_dir = os.path.join(folder, "vcf_filtered")
        vcf_files = glob(os.path.join(vcf_filtered_dir, "*_filtered.vcf.gz"))
        _detect_duplicate_chrom_sources(vcf_files)
        for vcf_path in vcf_files:
            out_parquet = vcf_path + ".raw.parquet"
            jobs.append((vcf_path, out_parquet, cfg.log_dir, generation))

    jobs.sort(key=lambda j: os.path.getsize(j[0]), reverse=True)
    log.info("Conversione VCF filtrati -> parquet grezzo: %d file, %d worker", len(jobs), cfg.max_workers)
    out_paths = []
    sample_generation: dict[str, int] = {}
    conflicts = []
    corrupt_inputs = []

    max_workers = min(cfg.max_workers, 6)
    log.info("Uso %d worker (limitati da 16 a %d per evitare OOM su cromosomi grandi)", max_workers, max_workers)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_process_single_vcf_worker, job) for job in jobs]
        for fut in as_completed(futures):
            out_path, generation, samples, was_corrupt = fut.result()
            if was_corrupt:
                corrupt_inputs.append(out_path)
                continue
            out_paths.append(out_path)
            for raw_sample in samples:
                sid = clean_sample_id(str(raw_sample))
                prev = sample_generation.get(sid)
                if prev is not None and prev != generation:
                    conflicts.append((sid, prev, generation))
                sample_generation[sid] = generation

    if conflicts:
        log.warning(
            "%d id campione risultano presenti in PIÙ di una generazione (tenuta l'ultima "
            "vista): %s%s",
            len(conflicts), conflicts[:10], " ..." if len(conflicts) > 10 else "",
        )

    if corrupt_inputs:
        raise CorruptInputVCFError(
            f"{len(corrupt_inputs)} VCF di input erano corrotti/illeggibili e sono stati "
            f"eliminati. Rilancia prima filter_vcf.py (li rigenererà grazie alla sua "
            f"idempotenza), poi rilancia questa pipeline: i file già convertiti in questo "
            f"run ({len(out_paths)}) non verranno ricalcolati."
        )

    return out_paths, sample_generation


def save_sample_generation_map(sample_generation: dict[str, int], out_path: str) -> None:
    df = pd.DataFrame({"id": list(sample_generation.keys()), "generation": list(sample_generation.values())})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = out_path + ".tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, out_path)
    log.info("Mappa id->generazione salvata in %s (%d campioni)", out_path, len(df))


def _files_for_chromosome(chrom: str, raw_parquet_paths: list[str]) -> list[str]:
    """Match robusto: 'chr<chrom>' seguito da un separatore ('.', '_') e MAI
    da un'altra cifra, per evitare che chr1 catturi chr11, chr12, ... chr19."""
    pattern = re.compile(rf"chr{re.escape(chrom)}(?!\d)[._]")
    return [p for p in raw_parquet_paths if pattern.search(os.path.basename(p))]


@dataclass
class ChromStats:
    chrom: str
    n_samples: int = 0
    n_variants_total: int = 0
    n_variants_dropped_missing: int = 0
    n_variants_final: int = 0
    elapsed_seconds: float = 0.0


def merge_chromosome(chrom: str, raw_parquet_paths: list[str], out_folder: str, null_percentage: float,
                      missing_strategy: str = "zero", force: bool = False) -> tuple[str | None, ChromStats | None]:
    """Step 2: unisce (per id, non per posizione di riga) tutti i parquet
    grezzi relativi allo stesso cromosoma, filtra per missing rate e
    binarizza. Ritorna (path del parquet di cromosoma o None, statistiche
    numeriche o None se non c'era nulla da fare).

    Se uno o più file raw sono corrotti/troncati: vengono cancellati (così
    che convert_filtered_vcfs_to_parquet li rigeneri al prossimo run) e
    viene sollevata CorruptParquetError, in modo che il chiamante possa
    saltare SOLO questo cromosoma senza far fallire tutta la pipeline."""
    t0 = time.monotonic()
    out_path = os.path.join(out_folder, f"chr{chrom}_merged.parquet")
    if not force and _is_valid_parquet(out_path):
        log.info("chr%s: già presente e valido, salto (%s)", chrom, out_path)
        return out_path, None

    chrom_files = _files_for_chromosome(chrom, raw_parquet_paths)
    chrom_files = [
        p for p in chrom_files
        if "_selected" not in os.path.basename(p)
    ]
    if not chrom_files:
        log.warning("Nessun file trovato per chr%s", chrom)
        return None, None

    dfs = []
    corrupt_files = []
    for p in chrom_files:
        if not _is_valid_parquet(p):
            corrupt_files.append(p)
            continue
        dfs.append(pd.read_parquet(p))

    if corrupt_files:
        for p in corrupt_files:
            log.error(
                "chr%s: file raw corrotto/troncato, lo elimino per forzare la "
                "rigenerazione al prossimo run: %s", chrom, p,
            )
            os.remove(p)
        raise CorruptParquetError(
            f"chr{chrom}: {len(corrupt_files)} file .raw.parquet erano corrotti e sono "
            f"stati eliminati. Rilancia la pipeline: verranno rigenerati automaticamente "
            f"(i file già validi non vengono ricalcolati)."
        )

    if not dfs:
        log.warning("chr%s: nessun file valido trovato dopo la validazione", chrom)
        return None, None

    # concat verticale (nuovi campioni), allineamento sulle COLONNE (varianti)
    # per nome, non per posizione -> outer join implicito di pandas.
    merged = pd.concat(dfs, axis=0, join="outer")
    merged = merged[~merged.index.duplicated(keep="first")]

    n_variants_total = merged.shape[1]
    # Stesso motivo del fix più sotto: il -1 non è l'unico modo in cui una
    # variante può risultare "mancante" per un campione, dopo il concat con
    # join="outer" fra batch con set di varianti leggermente diversi.
    missing_frac = ((merged < 0) | merged.isna()).mean()
    keep_cols = missing_frac[missing_frac < null_percentage].index
    dropped = merged.shape[1] - len(keep_cols)
    if dropped:
        log.info("chr%s: %d/%d varianti scartate per missing rate >= %.0f%%", chrom, dropped, merged.shape[1], null_percentage * 100)
    merged = merged[keep_cols]

    if merged.shape[1] == 0:
        log.warning("chr%s: nessuna variante superstite dopo il filtro missing", chrom)
        return None, None

    arr = merged.to_numpy(dtype=np.float32, copy=True)
    # Oltre al -1 (missing esplicito da _genotype_to_dosage), il concat con
    # join="outer" qui sopra introduce NaN "genuini" ogni volta che una
    # variante è presente nel raw parquet di un batch ma assente in un
    # altro: ogni batch fa MAF+LD pruning indipendentemente in
    # filter_vcf.py, quindi il set di varianti superstiti per lo stesso
    # cromosoma può differire leggermente da batch a batch. Va trattato
    # come dato mancante alla pari del -1: se si usasse solo `arr < 0`, il
    # NaN passerebbe indenne (in numpy un confronto con NaN è sempre False,
    # quindi né `arr < 0` né `arr > 0` lo intercettano) e sopravvivrebbe
    # fino all'astype(int) finale, che esplode con IntCastingNaNError - è
    # esattamente il crash su chr21 di questo run.
    missing_mask = (arr < 0) | np.isnan(arr)
    if missing_strategy == "zero":
        arr[missing_mask] = 0
    else:  # "nan": missing esplicito, NON silenziosamente trattato come wild-type
        arr[missing_mask] = np.nan
    arr[arr > 0] = 1
    merged[:] = arr

    os.makedirs(out_folder, exist_ok=True)
    _write_parquet_atomic(
        merged.astype("Int8" if missing_strategy == "nan" else np.int8),
        out_path, compression="zstd",
    )
    log.info("chr%s: salvato %s (%d campioni, %d varianti)", chrom, out_path, *merged.shape)

    stats = ChromStats(
        chrom=chrom,
        n_samples=merged.shape[0],
        n_variants_total=n_variants_total,
        n_variants_dropped_missing=dropped,
        n_variants_final=merged.shape[1],
        elapsed_seconds=time.monotonic() - t0,
    )
    return out_path, stats


def build_full_genome_parquet(chrom_parquet_paths: list[str], out_path: str, force: bool = False) -> None:
    """Step 3: merge finale genoma intero, per ID (sostituisce
    create_full_csv.py). pd.concat(axis=1) allinea automaticamente per
    indice (id campione), indipendentemente dall'ordine delle righe nei
    singoli file di cromosoma."""
    if not force and _is_valid_parquet(out_path):
        log.info("Genoma completo già presente e valido, salto: %s", out_path)
        return

    log.info("Merge finale di %d file di cromosoma per id campione", len(chrom_parquet_paths))

    corrupt = [p for p in chrom_parquet_paths if not _is_valid_parquet(p)]
    if corrupt:
        for p in corrupt:
            log.error("File di cromosoma corrotto trovato in fase di merge finale, lo elimino: %s", p)
            os.remove(p)
        raise CorruptParquetError(
            f"{len(corrupt)} file di cromosoma erano corrotti e sono stati eliminati. "
            f"Rilancia la pipeline per rigenerarli prima del merge finale."
        )

    frames = [pd.read_parquet(p) for p in chrom_parquet_paths]
    full = pd.concat(frames, axis=1, join="outer")
    full.index.name = "id"
    _write_parquet_atomic(full, out_path, compression="zstd")
    log.info("Genoma completo salvato in %s (%d campioni, %d varianti)", out_path, *full.shape)


def _write_stats_csv(all_stats: list[ChromStats], log_dir: str) -> str:
    """Scrive un CSV riepilogativo per cromosoma in
    <log_dir>/vcf_to_parquet_stats.csv. Scrittura atomica (tmp + rename)."""
    os.makedirs(log_dir, exist_ok=True)
    out_path = os.path.join(log_dir, STATS_FILENAME)
    tmp_path = out_path + ".tmp"

    fieldnames = list(asdict(all_stats[0]).keys()) if all_stats else [
        "chrom", "n_samples", "n_variants_total", "n_variants_dropped_missing",
        "n_variants_final", "elapsed_seconds",
    ]
    with open(tmp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in all_stats:
            writer.writerow(asdict(s))
    os.replace(tmp_path, out_path)
    return out_path


def run_vcf_to_parquet_pipeline(missing_strategy: str = "zero", force: bool = False) -> str:
    cfg = get_config()
    configure_logging(cfg.log_dir)
    _add_file_logging(cfg.log_dir)

    t_start = time.monotonic()
    raw_paths, sample_generation = convert_filtered_vcfs_to_parquet()

    map_path = cfg.sample_generation_map or os.path.join(cfg.output_folder, "sample_generation_map.csv")
    save_sample_generation_map(sample_generation, map_path)

    chrom_paths = []
    chrom_stats: list[ChromStats] = []
    failed_chroms = []
    for chrom in CHROMOSOMES:
        try:
            p, stats = merge_chromosome(chrom, raw_paths, cfg.output_folder, cfg.null_percentage, missing_strategy, force=force)
            if p:
                chrom_paths.append(p)
            if stats:
                chrom_stats.append(stats)
        except CorruptParquetError as e:
            log.error(str(e))
            failed_chroms.append(chrom)

    if chrom_stats:
        stats_path = _write_stats_csv(chrom_stats, cfg.log_dir)
        tot_variants_total = sum(s.n_variants_total for s in chrom_stats)
        tot_variants_dropped = sum(s.n_variants_dropped_missing for s in chrom_stats)
        tot_variants_final = sum(s.n_variants_final for s in chrom_stats)
        log.info(
            "Riepilogo merge per cromosoma (%d cromosomi processati in questo run): "
            "%d varianti totali -> %d scartate per missing -> %d finali. "
            "Statistiche per-cromosoma salvate in %s",
            len(chrom_stats), tot_variants_total, tot_variants_dropped, tot_variants_final, stats_path,
        )

    if failed_chroms:
        raise RuntimeError(
            f"Cromosomi con file corrotti (eliminati e da rigenerare): {failed_chroms}. "
            f"Rilancia la pipeline: i file già validi ({len(chrom_paths)} cromosomi) "
            f"verranno saltati e solo questi {len(failed_chroms)} verranno rifatti. "
            f"Il parquet genoma finale non viene costruito finché non sono tutti a posto, "
            f"per non produrre silenziosamente un dataset con cromosomi mancanti."
        )

    out_path = os.path.join(cfg.output_folder, "gen.parquet")
    build_full_genome_parquet(chrom_paths, out_path, force=force)

    log.info("Pipeline vcf_to_parquet completata in %.1fs -> %s", time.monotonic() - t_start, out_path)
    return out_path


if __name__ == "__main__":
    run_vcf_to_parquet_pipeline()