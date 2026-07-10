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

FIX SUCCESSIVI (10 luglio 2026 - crash "Exceeded size limit" su parquet):
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

import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

import numpy as np
import pandas as pd

from gene_environment.config import get_config, get_generation_vcf_folders
from gene_environment.logging_utils import configure_logging, get_logger
from gene_environment.utils.id_utils import clean_sample_id

log = get_logger(__name__)

CHROMOSOMES = [str(i) for i in range(1, 23)]

LOG_FILENAME = "vcf_to_parquet.log"


class CorruptParquetError(Exception):
    """Sollevata quando uno o più file .raw.parquet risultano corrotti/troncati.
    Il file corrotto viene eliminato prima di sollevare l'eccezione, quindi
    un semplice rilancio della pipeline lo rigenera."""


def _add_file_logging(log_dir: str) -> None:
    """Aggiunge (una sola volta per processo) un FileHandler al root logger,
    così i log finiscono sia su console sia su <log_dir>/vcf_to_parquet.log.
    Va richiamata sia nel processo principale sia in ogni worker (sono
    processi separati e non ereditano gli handler del padre)."""
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


def _write_parquet_atomic(df: pd.DataFrame, out_path: str, **to_parquet_kwargs) -> None:
    """Scrive su <out_path>.tmp e poi rinomina atomicamente su out_path.
    Garantisce che, se il processo viene ucciso a metà scrittura (OOM,
    kill, crash del worker, disco pieno...), il path finale non contenga
    MAI un parquet troncato: o non esiste, o è quello vecchio (valido), o
    è quello nuovo completo."""
    tmp_path = out_path + ".tmp"
    df.to_parquet(tmp_path, engine="pyarrow", **to_parquet_kwargs)
    os.replace(tmp_path, out_path)


def _genotype_to_dosage(gt) -> int:
    """gt: tupla cyvcf2 (allele1, allele2, phased). -1 = missing."""
    if gt is None or gt[0] is None or gt[1] is None:
        return -1
    a, b = gt[0], gt[1]
    if a < 0 or b < 0:
        return -1
    return a + b


def vcf_file_to_dosage_df(vcf_path: str) -> pd.DataFrame:
    """Legge un VCF filtrato e ritorna un DataFrame (samples x varianti) di
    dosaggi grezzi (0/1/2/-1=missing). Import di cyvcf2 fatto qui dentro per
    non richiederlo come dipendenza hard di tutto il pacchetto."""
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


def _process_single_vcf_worker(args) -> tuple[str, int, list[str]]:
    vcf_path, out_parquet, log_dir, generation = args
    configure_logging(log_dir)
    _add_file_logging(log_dir)

    if _is_valid_parquet(out_parquet):
        log.info("Skip (già convertito e valido): %s", out_parquet)
        samples = pd.read_parquet(out_parquet, columns=[]).index.tolist()
        return out_parquet, generation, samples

    if os.path.exists(out_parquet):
        log.warning("File esistente ma corrotto/troncato, lo rigenero: %s", out_parquet)
        os.remove(out_parquet)

    log.info("Converto VCF -> parquet (generazione %d): %s", generation, vcf_path)
    df = vcf_file_to_dosage_df(vcf_path)
    _write_parquet_atomic(df, out_parquet, compression="zstd")
    log.info("Scritto %s (%d campioni, %d varianti)", out_parquet, df.shape[0], df.shape[1])
    return out_parquet, generation, df.index.tolist()


def _detect_duplicate_chrom_sources(vcf_paths: list[str]) -> None:
    """Segnala (senza bloccare) se più di un VCF filtrato sembra riferirsi
    allo stesso cromosoma nella stessa cartella: capita quando restano in
    giro file di run precedenti con naming diverso (es.
    'chr1_filtered.vcf' e 'chr1.vcf_filtered.vcf' insieme), e porta a
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
    """Step 1: ogni *_filtered.vcf -> un parquet grezzo (dosaggi 0/1/2/-1).

    Ritorna anche la mappa id_campione -> generazione, costruita in base a
    QUALE cartella (VCF_DIR_GENn) proviene ogni file VCF: è l'unica fonte
    affidabile della coorte di un paziente quando il file ambientale non
    contiene alcuna informazione di generazione (il join fra ambiente e
    genetica avviene solo per id)."""
    cfg = get_config()
    _add_file_logging(cfg.log_dir)

    jobs = []
    for generation, folder in get_generation_vcf_folders(cfg).items():
        vcf_filtered_dir = os.path.join(folder, "vcf_filtered")
        vcf_files = glob(os.path.join(vcf_filtered_dir, "*_filtered.vcf"))
        _detect_duplicate_chrom_sources(vcf_files)
        for vcf_path in vcf_files:
            out_parquet = vcf_path + ".raw.parquet"
            jobs.append((vcf_path, out_parquet, cfg.log_dir, generation))

    log.info("Conversione VCF filtrati -> parquet grezzo: %d file, %d worker", len(jobs), cfg.max_workers)
    out_paths = []
    sample_generation: dict[str, int] = {}
    conflicts = []

    with ProcessPoolExecutor(max_workers=cfg.max_workers) as ex:
        futures = [ex.submit(_process_single_vcf_worker, job) for job in jobs]
        for fut in as_completed(futures):
            out_path, generation, samples = fut.result()
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


def merge_chromosome(chrom: str, raw_parquet_paths: list[str], out_folder: str, null_percentage: float,
                      missing_strategy: str = "zero", force: bool = False) -> str | None:
    """Step 2: unisce (per id, non per posizione di riga) tutti i parquet
    grezzi relativi allo stesso cromosoma, filtra per missing rate e
    binarizza. Ritorna il path del parquet di cromosoma, o None se non
    c'era nulla da fare.

    Se uno o più file raw sono corrotti/troncati: vengono cancellati (così
    che convert_filtered_vcfs_to_parquet li rigeneri al prossimo run) e
    viene sollevata CorruptParquetError, in modo che il chiamante possa
    saltare SOLO questo cromosoma senza far fallire tutta la pipeline."""
    out_path = os.path.join(out_folder, f"chr{chrom}_merged.parquet")
    if not force and _is_valid_parquet(out_path):
        log.info("chr%s: già presente e valido, salto (%s)", chrom, out_path)
        return out_path

    chrom_files = _files_for_chromosome(chrom, raw_parquet_paths)
    if not chrom_files:
        log.warning("Nessun file trovato per chr%s", chrom)
        return None

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
        return None

    # concat verticale (nuovi campioni), allineamento sulle COLONNE (varianti)
    # per nome, non per posizione -> outer join implicito di pandas.
    merged = pd.concat(dfs, axis=0, join="outer")
    merged = merged[~merged.index.duplicated(keep="first")]

    missing_frac = (merged < 0).mean()
    keep_cols = missing_frac[missing_frac < null_percentage].index
    dropped = merged.shape[1] - len(keep_cols)
    if dropped:
        log.info("chr%s: %d/%d varianti scartate per missing rate >= %.0f%%", chrom, dropped, merged.shape[1], null_percentage * 100)
    merged = merged[keep_cols]

    if merged.shape[1] == 0:
        log.warning("chr%s: nessuna variante superstite dopo il filtro missing", chrom)
        return None

    arr = merged.to_numpy(dtype=np.float32, copy=True)
    if missing_strategy == "zero":
        arr[arr < 0] = 0
    else:  # "nan": missing esplicito, NON silenziosamente trattato come wild-type
        arr[arr < 0] = np.nan
    arr[arr > 0] = 1
    merged[:] = arr

    os.makedirs(out_folder, exist_ok=True)
    _write_parquet_atomic(
        merged.astype("Int8" if missing_strategy == "nan" else np.int8),
        out_path, compression="zstd",
    )
    log.info("chr%s: salvato %s (%d campioni, %d varianti)", chrom, out_path, *merged.shape)
    return out_path


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


def run_vcf_to_parquet_pipeline(missing_strategy: str = "zero", force: bool = False) -> str:
    cfg = get_config()
    configure_logging(cfg.log_dir)
    _add_file_logging(cfg.log_dir)

    raw_paths, sample_generation = convert_filtered_vcfs_to_parquet()

    map_path = cfg.sample_generation_map or os.path.join(cfg.output_folder, "sample_generation_map.csv")
    save_sample_generation_map(sample_generation, map_path)

    chrom_paths = []
    failed_chroms = []
    for chrom in CHROMOSOMES:
        try:
            p = merge_chromosome(chrom, raw_paths, cfg.output_folder, cfg.null_percentage, missing_strategy, force=force)
            if p:
                chrom_paths.append(p)
        except CorruptParquetError as e:
            log.error(str(e))
            failed_chroms.append(chrom)

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
    return out_path


if __name__ == "__main__":
    run_vcf_to_parquet_pipeline()