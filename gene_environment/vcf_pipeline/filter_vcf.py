"""
Filtraggio VCF -> PLINK binario -> MAF filter -> LD pruning -> VCF filtrato.
(ex gene_reduction.py)

Fix rispetto all'originale:
  - Il loop sui file VCF era completamente SEQUENZIALE (un file dopo l'altro),
    nonostante MAX_WORKERS fosse già configurato e usato altrove nella
    pipeline. Con decine/centinaia di file VCF per cromosoma questo è lo
    step più lento di tutta la pipeline "a monte". Ora è parallelizzato con
    ProcessPoolExecutor, un processo per file VCF (ogni chiamata plink2 è
    già mono-processo pesante in I/O+CPU, quindi parallelizzare a livello di
    file ha senso ed è sicuro).
  - Idempotenza: se l'output finale (_filtered.vcf.gz) esiste già ED È
    VALIDO (vedi sotto), il file viene saltato invece di essere ricalcolato
    da capo ad ogni rilancio.
  - Il prefisso "ACH" usato per escludere campioni dal .fam era hardcoded
    senza alcun commento sul perché. Ora è un parametro di configurazione
    esplicito (EXCLUDE_ID_PREFIXES), documentato, con default vuoto: se non
    configurato non viene rimosso silenziosamente nessun campione.
  - Ogni subprocess.run(..., check=True) ora logga comando ed esito; un
    fallimento di plink2 su UN file non blocca più necessariamente l'intero
    batch (l'errore viene loggato e si passa al file successivo, il
    riepilogo finale elenca i falliti).

FIX (10 luglio 2026 - VCF filtrati enormi e corruzione a valle):
  - CAUSA #1 dei file enormi: l'ultimo step chiamava
    `plink2 --recode vcf`, che scrive VCF TESTUALE non compresso (da qui i
    GB, a fronte di VCF di input compressi). Ora si usa
    `plink2 --export vcf bgz`, che scrive direttamente in bgzip (blocked
    gzip): file molto più piccoli, e bgzip è letto nativamente da cyvcf2
    nello step successivo (vcf_to_parquet.py) senza bisogno di
    decompressione manuale. L'output finale è quindi *_filtered.vcf.gz
    invece di *_filtered.vcf.
  - CAUSA #2 della corruzione a valle: il VCF finale veniva scritto da
    plink2 DIRETTAMENTE sul path definitivo (*_filtered.vcf). Se il
    processo veniva ucciso a metà (OOM, kill manuale, disco pieno - reso
    più probabile proprio dai VCF non compressi), restava un file
    TRONCATO con il nome "finale": al run successivo l'idempotenza lo
    considerava completo (os.path.exists) e lo saltava, e vcf_to_parquet.py
    andava in crash provando a leggerlo (i controlli di corruzione lì
    presenti coprono solo i file .parquet, non i VCF di input). Ora ogni
    file viene scritto su un prefisso temporaneo e poi spostato sul path
    finale con os.replace() SOLO se la scrittura è andata a buon fine
    (stesso pattern "scrivi su tmp poi rinomina" già usato per i parquet in
    vcf_to_parquet.py): un processo ucciso a metà non lascia mai un
    .vcf.gz troncato sul path definitivo.
  - Il controllo di idempotenza ora valida anche che il .vcf.gz esistente
    sia un bgzip leggibile per intero (non solo che il file esista): un
    .vcf.gz troncato da un run precedente a QUESTA fix viene rilevato e
    rigenerato invece di essere scambiato per completo.
  - Log numerici: per ogni file vengono ora loggati (e salvati in un CSV
    riepilogativo in <log_dir>/filter_vcf_stats.csv) campioni totali,
    campioni esclusi per prefisso id, varianti prima del filtro MAF,
    varianti dopo il filtro MAF, varianti dopo l'LD pruning. Il log va
    anche su file (<log_dir>/filter_vcf.log), non solo su console, sia nel
    processo principale sia in ogni worker (processi separati non
    ereditano gli handler di logging del padre).
"""
from __future__ import annotations

import csv
import gzip
import logging
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass

from gene_environment.config import get_config, get_generation_vcf_folders
from gene_environment.logging_utils import configure_logging, get_logger

log = get_logger(__name__)

OUTPUT_SUBFOLDER = "vcf_filtered"
LOG_FILENAME = "filter_vcf.log"
STATS_FILENAME = "filter_vcf_stats.csv"


def _add_file_logging(log_dir: str) -> None:
    """Aggiunge (una sola volta per processo) un FileHandler al root logger,
    così i log finiscono sia su console sia su <log_dir>/filter_vcf.log.
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


def _is_valid_bgzip(path: str) -> bool:
    """True se il file esiste, non è vuoto, ed è un bgzip/gzip leggibile per
    intero (nessun blocco troncato). Decomprimere per intero ha un costo,
    ma questi sono i VCF già filtrati/pruned quindi relativamente piccoli;
    è lo stesso principio del controllo footer usato per i parquet
    nell'altro script, adattato al formato gzip che non ha un footer
    comodo da leggere in isolamento."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        with gzip.open(path, "rb") as f:
            while f.read(1024 * 1024):
                pass
        return True
    except Exception as e:
        log.warning("VCF filtrato non valido/troncato, verrà rigenerato: %s (%s)", path, e)
        return False


def _run(cmd: list[str], log_prefix: str) -> None:
    log.debug("%s: eseguo: %s", log_prefix, " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _count_lines(path: str) -> int:
    with open(path) as f:
        return sum(1 for _ in f)


@dataclass
class FilterStats:
    vcf_file: str
    n_samples_total: int = 0
    n_samples_removed: int = 0
    n_variants_raw: int = 0
    n_variants_after_maf: int = 0
    n_variants_after_pruning: int = 0
    elapsed_seconds: float = 0.0
    skipped: bool = False


def filter_single_vcf(
    input_path: str, output_vcf_folder: str, cfg_dict: dict
) -> tuple[str, bool, str | None, FilterStats]:
    """Esegue l'intera catena plink2 per un singolo VCF. Ritorna
    (nome_file, successo, messaggio_errore, statistiche_numeriche)."""
    from gene_environment.logging_utils import configure_logging as _cfg_log

    _cfg_log(cfg_dict["log_dir"])  # necessario nei worker separati
    _add_file_logging(cfg_dict["log_dir"])

    t0 = time.monotonic()
    vcf_file = os.path.basename(input_path)
    base_name = os.path.splitext(os.path.splitext(vcf_file)[0])[0]  # rimuove .vcf.gz
    stats = FilterStats(vcf_file=vcf_file)

    final_vcf_path = os.path.join(output_vcf_folder, base_name + "_filtered.vcf.gz")
    if _is_valid_bgzip(final_vcf_path):
        log.info("[%s] output già presente e valido, salto: %s", vcf_file, final_vcf_path)
        stats.skipped = True
        stats.elapsed_seconds = time.monotonic() - t0
        return vcf_file, True, None, stats

    try:
        plink_prefix = os.path.join(output_vcf_folder, base_name + "_plink")
        _run(["plink2", "--vcf", input_path, "--make-bed", "--out", plink_prefix], vcf_file)

        fam_file = plink_prefix + ".fam"
        bim_file = plink_prefix + ".bim"
        remove_file = plink_prefix + "_remove.txt"
        exclude_prefixes = tuple(cfg_dict["exclude_id_prefixes"])

        stats.n_samples_total = _count_lines(fam_file)
        stats.n_variants_raw = _count_lines(bim_file)

        n_removed = 0
        with open(fam_file) as f, open(remove_file, "w") as out:
            for line in f:
                fid, iid = line.strip().split()[:2]
                if exclude_prefixes and iid.startswith(exclude_prefixes):
                    out.write(f"{fid} {iid}\n")
                    n_removed += 1
        stats.n_samples_removed = n_removed
        if n_removed:
            log.info("[%s] %d campioni esclusi per prefisso id (%s)", vcf_file, n_removed, exclude_prefixes)

        plink_maf_prefix = os.path.join(output_vcf_folder, base_name + "_maf")
        remove_args = ["--remove", remove_file] if n_removed else []
        _run(
            ["plink2", "--bfile", plink_prefix, *remove_args,
             "--maf", str(cfg_dict["maf_threshold"]), "--make-bed", "--out", plink_maf_prefix],
            vcf_file,
        )
        stats.n_variants_after_maf = _count_lines(plink_maf_prefix + ".bim")
        log.info(
            "[%s] varianti: %d (raw) -> %d (dopo MAF >= %s)",
            vcf_file, stats.n_variants_raw, stats.n_variants_after_maf, cfg_dict["maf_threshold"],
        )

        plink_prune_prefix = os.path.join(output_vcf_folder, base_name + "_pruned")
        _run(
            ["plink2", "--bfile", plink_maf_prefix, "--indep-pairwise",
             str(cfg_dict["ld_window_size"]), str(cfg_dict["ld_step"]), str(cfg_dict["ld_r2_threshold"]),
             "--out", plink_prune_prefix],
            vcf_file,
        )
        stats.n_variants_after_pruning = _count_lines(plink_prune_prefix + ".prune.in")
        log.info(
            "[%s] varianti: %d (dopo MAF) -> %d (dopo LD pruning, window=%s step=%s r2=%s)",
            vcf_file, stats.n_variants_after_maf, stats.n_variants_after_pruning,
            cfg_dict["ld_window_size"], cfg_dict["ld_step"], cfg_dict["ld_r2_threshold"],
        )

        # Scrittura ATOMICA: si esporta su un prefisso temporaneo e solo se
        # plink2 termina con successo si sposta il .vcf.gz risultante sul
        # path finale. Se il processo viene ucciso a metà, il path finale
        # non esiste ancora (niente file troncato con nome "definitivo").
        tmp_prefix = os.path.join(output_vcf_folder, base_name + "_filtered_tmp")
        tmp_vcf_gz = tmp_prefix + ".vcf.gz"
        if os.path.exists(tmp_vcf_gz):
            os.remove(tmp_vcf_gz)  # residuo di un run interrotto precedente

        _run(
            ["plink2", "--bfile", plink_maf_prefix, "--extract", plink_prune_prefix + ".prune.in",
             "--export", "vcf", "bgz", "--out", tmp_prefix],
            vcf_file,
        )

        if not _is_valid_bgzip(tmp_vcf_gz):
            raise RuntimeError(f"plink2 ha prodotto un .vcf.gz non valido: {tmp_vcf_gz}")

        os.replace(tmp_vcf_gz, final_vcf_path)

        log.info(
            "[%s] filtrato con successo -> %s (%d campioni [-%d], %d varianti finali)",
            vcf_file, final_vcf_path, stats.n_samples_total - stats.n_samples_removed,
            stats.n_samples_removed, stats.n_variants_after_pruning,
        )
        stats.elapsed_seconds = time.monotonic() - t0
        return vcf_file, True, None, stats

    except subprocess.CalledProcessError as e:
        err = f"{e}\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"
        log.error("[%s] fallito: %s", vcf_file, err)
        stats.elapsed_seconds = time.monotonic() - t0
        return vcf_file, False, err, stats
    except Exception as e:  # es. RuntimeError sopra per bgzip non valido
        log.error("[%s] fallito: %s", vcf_file, e)
        stats.elapsed_seconds = time.monotonic() - t0
        return vcf_file, False, str(e), stats


def _write_stats_csv(all_stats: list[FilterStats], log_dir: str) -> str:
    """Scrive un CSV riepilogativo con le statistiche numeriche di ogni
    file processato in questo run, in <log_dir>/filter_vcf_stats.csv.
    Scrittura atomica (tmp + rename) come per gli altri output della
    pipeline."""
    os.makedirs(log_dir, exist_ok=True)
    out_path = os.path.join(log_dir, STATS_FILENAME)
    tmp_path = out_path + ".tmp"

    fieldnames = list(asdict(all_stats[0]).keys()) if all_stats else [
        "vcf_file", "n_samples_total", "n_samples_removed", "n_variants_raw",
        "n_variants_after_maf", "n_variants_after_pruning", "elapsed_seconds", "skipped",
    ]
    with open(tmp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in all_stats:
            writer.writerow(asdict(s))
    os.replace(tmp_path, out_path)
    return out_path


def run_filter_vcf(exclude_id_prefixes: list[str] | None = None) -> None:
    cfg = get_config()
    configure_logging(cfg.log_dir)
    _add_file_logging(cfg.log_dir)

    # default: legge da config (EXCLUDE_ID_PREFIXES), non più una lista vuota fissa
    exclude_id_prefixes = exclude_id_prefixes if exclude_id_prefixes is not None else cfg.exclude_id_prefixes
    if exclude_id_prefixes:
        log.info("Prefissi id da escludere dal filtraggio: %s", exclude_id_prefixes)
    else:
        log.warning(
            "EXCLUDE_ID_PREFIXES non configurato: nessun campione verrà escluso per prefisso id. "
            "Lo script originale escludeva sempre i campioni con id che iniziano per 'ACH' — "
            "se è ancora il comportamento voluto, imposta EXCLUDE_ID_PREFIXES=ACH nel .env."
        )

    cfg_dict = {
        "maf_threshold": cfg.maf_threshold,
        "ld_window_size": cfg.ld_window_size,
        "ld_step": cfg.ld_step,
        "ld_r2_threshold": cfg.ld_r2_threshold,
        "exclude_id_prefixes": exclude_id_prefixes,
        "log_dir": cfg.log_dir,
    }

    jobs = []
    for generation, input_folder in get_generation_vcf_folders(cfg).items():
        output_vcf_folder = os.path.join(input_folder, OUTPUT_SUBFOLDER)
        os.makedirs(output_vcf_folder, exist_ok=True)
        vcf_files = [
            f for f in os.listdir(input_folder)
            if f.endswith(".vcf.gz") and not f.startswith("._")
        ]
        log.info("Generazione %d: %d VCF trovati in %s", generation, len(vcf_files), input_folder)
        for vcf_file in vcf_files:
            jobs.append((os.path.join(input_folder, vcf_file), output_vcf_folder))

    log.info("Filtraggio VCF: %d file da processare con %d worker", len(jobs), cfg.max_workers)

    t_start = time.monotonic()
    failed = []
    all_stats: list[FilterStats] = []
    with ProcessPoolExecutor(max_workers=cfg.max_workers) as ex:
        futures = {ex.submit(filter_single_vcf, inp, outp, cfg_dict): inp for inp, outp in jobs}
        for fut in as_completed(futures):
            vcf_file, ok, err, stats = fut.result()
            all_stats.append(stats)
            if not ok:
                failed.append((vcf_file, err))

    elapsed_total = time.monotonic() - t_start
    stats_path = _write_stats_csv(all_stats, cfg.log_dir)

    n_ok = sum(1 for s in all_stats if s.vcf_file not in {f for f, _ in failed})
    n_skipped = sum(1 for s in all_stats if s.skipped)
    tot_variants_raw = sum(s.n_variants_raw for s in all_stats)
    tot_variants_after_maf = sum(s.n_variants_after_maf for s in all_stats)
    tot_variants_final = sum(s.n_variants_after_pruning for s in all_stats)
    tot_samples_removed = sum(s.n_samples_removed for s in all_stats)

    log.info(
        "Riepilogo filtraggio VCF: %d file totali, %d ok (%d saltati perché già presenti), "
        "%d falliti, tempo totale %.1fs. Varianti (somma su tutti i file processati in questo "
        "run, esclusi gli skip): %d raw -> %d dopo MAF -> %d finali dopo LD pruning. "
        "Campioni esclusi per prefisso id (somma): %d. Statistiche per-file salvate in %s",
        len(jobs), n_ok, n_skipped, len(failed), elapsed_total,
        tot_variants_raw, tot_variants_after_maf, tot_variants_final,
        tot_samples_removed, stats_path,
    )

    if failed:
        log.error("Filtraggio completato con %d errori su %d file: %s", len(failed), len(jobs), [f for f, _ in failed])
    else:
        log.info("Filtraggio VCF completato senza errori (%d file).", len(jobs))


if __name__ == "__main__":
    run_filter_vcf()