"""VCF filtering -> binary PLINK -> MAF filter -> LD pruning -> filtered VCF.

Design notes:
  - Files are processed in PARALLEL with ProcessPoolExecutor, one process
    per VCF file (each plink2 call is already a heavy single-process I/O+CPU
    job, so parallelizing at the file level is safe and effective).
  - Idempotency: if the final output (_filtered.vcf.gz) already exists AND
    IS VALID (see _is_valid_bgzip), the file is skipped instead of being
    recomputed on every rerun.
  - The id prefixes used to exclude samples from the .fam file are an
    explicit, documented configuration parameter (EXCLUDE_ID_PREFIXES),
    defaulting to empty: if not configured, no sample is silently removed.
  - Every subprocess.run(..., check=True) logs the command and outcome; a
    plink2 failure on ONE file doesn't necessarily block the whole batch
    (the error is logged and the next file is processed; the final summary
    lists the failures).

Output format and atomic writes: the final export uses
`plink2 --export vcf bgz`, which writes directly in bgzip (blocked gzip)
format -- much smaller than uncompressed VCF, and natively readable by
cyvcf2 in the next step (vcf_to_parquet.py) with no manual decompression
needed. The output is therefore *_filtered.vcf.gz. Each file is written to
a temporary prefix and only moved to the final path with os.replace() once
the write succeeds (same "write to tmp then rename" pattern used for the
parquet files in vcf_to_parquet.py): a process killed midway never leaves
a truncated .vcf.gz at the final path. The idempotency check also
validates that an existing .vcf.gz is a fully readable bgzip (not just
that the file exists), so a truncated file from an interrupted run is
detected and regenerated rather than mistaken for complete.

Numeric logging: for each file, total samples, samples excluded by id
prefix, variants before/after the MAF filter, and variants after LD
pruning are logged and saved to a summary CSV at
<log_dir>/filter_vcf_stats.csv. Logging also goes to file
(<log_dir>/filter_vcf.log), not just console, in both the main process and
every worker (separate processes don't inherit the parent's logging
handlers).

Intermediate file cleanup: each input VCF file produces, besides the final
*_filtered.vcf.gz result, a chain of intermediate plink2 files
(*_plink.bed/.bim/.fam, *_maf.bed/.bim/.fam, *_pruned.*,
*_plink_remove.txt) that no later pipeline step needs: vcf_to_parquet.py
reads exclusively *_filtered.vcf.gz (see the "*_filtered.vcf.gz" glob in
convert_filtered_vcfs_to_parquet). These intermediates are also by far the
heaviest files produced by this script (a chromosome's *_plink.bed can
exceed 2GB), so leaving them on disk needlessly multiplies the space used
per processed chromosome.
Right after the final *_filtered.vcf.gz has been written AND VALIDATED
(never before, and never if validation fails), all intermediates for that
single file are removed with _cleanup_intermediates(). This is safe
because the script's idempotency relies ONLY on the existence/validity of
the final .vcf.gz (see _is_valid_bgzip below): if the final file is
missing or invalid, the script restarts from scratch from
plink2 --vcf <original input>, NEVER from the intermediates. Cleanup also
runs on the "skip" path (output already present and valid), to
retroactively clean up runs where intermediates were left on disk.
Configurable via cfg.keep_intermediate_files (defaults to "clean up" via
getattr, so it doesn't break configs that don't define this field).
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

# Patterns (relative to <output_vcf_folder>/<base_name>) of the
# intermediate files to remove once *_filtered.vcf.gz is confirmed valid.
# Listed explicitly (instead of a catch-all like "everything except
# *_filtered.*") to avoid accidentally deleting something unexpected if
# the script produces other files with different naming in the future.
_INTERMEDIATE_SUFFIXES = [
    "_plink.bed", "_plink.bim", "_plink.fam", "_plink.log",
    "_maf.bed", "_maf.bim", "_maf.fam", "_maf.log",
    "_pruned.log", "_pruned.prune.in", "_pruned.prune.out",
    "_plink_remove.txt",
    "_filtered_tmp.log",
]


def _add_file_logging(log_dir: str) -> None:
    """Adds (once per process) a FileHandler to the root logger, so logs go
    both to console and to <log_dir>/filter_vcf.log. Must be called in both
    the main process and every worker (separate processes don't inherit
    the parent's logging handlers)."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.abspath(os.path.join(log_dir, LOG_FILENAME))

    root = logging.getLogger()
    for h in root.handlers:
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == log_path:
            return  # already added in this process

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [pid=%(process)d] %(name)s: %(message)s"))
    fh.setLevel(logging.INFO)
    root.addHandler(fh)
    if root.level > logging.INFO or root.level == logging.NOTSET:
        root.setLevel(logging.INFO)
    log.info("File logging enabled: %s", log_path)


def _is_valid_bgzip(path: str) -> bool:
    """True if the file exists, is non-empty, and is a fully readable
    bgzip/gzip (no truncated block). Decompressing fully has a cost, but
    these are already filtered/pruned VCFs so relatively small; same
    principle as the footer check used for the parquet files elsewhere,
    adapted to the gzip format which has no convenient footer to read in
    isolation."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        with gzip.open(path, "rb") as f:
            while f.read(1024 * 1024):
                pass
        return True
    except Exception as e:
        log.warning("Filtered VCF invalid/truncated, will be regenerated: %s (%s)", path, e)
        return False


def _cleanup_intermediates(output_vcf_folder: str, base_name: str, vcf_file: str) -> int:
    """Removes the plink2 intermediate files (plink/maf/pruned/remove-list/
    tmp log) for <base_name>, AFTER *_filtered.vcf.gz has been validated.
    Never touches *_filtered.vcf.gz or any file not listed in
    _INTERMEDIATE_SUFFIXES. Returns the number of files actually removed
    (0 if there was nothing to clean, e.g. already cleaned by an earlier
    run)."""
    n_removed = 0
    for suffix in _INTERMEDIATE_SUFFIXES:
        path = os.path.join(output_vcf_folder, base_name + suffix)
        if os.path.exists(path):
            try:
                os.remove(path)
                n_removed += 1
            except OSError as e:
                log.warning("[%s] could not remove intermediate %s: %s", vcf_file, path, e)
    if n_removed:
        log.info("[%s] cleaned up %d intermediate files (plink/maf/pruned)", vcf_file, n_removed)
    return n_removed


def _run(cmd: list[str], log_prefix: str) -> None:
    log.debug("%s: running: %s", log_prefix, " ".join(cmd))
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
    intermediates_cleaned: int = 0


def filter_single_vcf(
    input_path: str, output_vcf_folder: str, cfg_dict: dict
) -> tuple[str, bool, str | None, FilterStats]:
    """Runs the full plink2 chain for a single VCF. Returns
    (file_name, success, error_message, numeric_stats)."""
    from gene_environment.logging_utils import configure_logging as _cfg_log

    _cfg_log(cfg_dict["log_dir"])  # needed in separate worker processes
    _add_file_logging(cfg_dict["log_dir"])

    t0 = time.monotonic()
    vcf_file = os.path.basename(input_path)
    base_name = os.path.splitext(os.path.splitext(vcf_file)[0])[0]  # strip .vcf.gz
    stats = FilterStats(vcf_file=vcf_file)
    keep_intermediates = cfg_dict["keep_intermediate_files"]

    final_vcf_path = os.path.join(output_vcf_folder, base_name + "_filtered.vcf.gz")
    if _is_valid_bgzip(final_vcf_path):
        log.info("[%s] output already present and valid, skipping: %s", vcf_file, final_vcf_path)
        stats.skipped = True
        # Even on skip, clean up any leftover intermediates from earlier
        # runs (idempotent: if there's nothing to remove,
        # _cleanup_intermediates simply returns 0).
        if not keep_intermediates:
            stats.intermediates_cleaned = _cleanup_intermediates(output_vcf_folder, base_name, vcf_file)
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
            log.info("[%s] %d samples excluded by id prefix (%s)", vcf_file, n_removed, exclude_prefixes)

        plink_maf_prefix = os.path.join(output_vcf_folder, base_name + "_maf")
        remove_args = ["--remove", remove_file] if n_removed else []
        _run(
            ["plink2", "--bfile", plink_prefix, *remove_args,
             "--maf", str(cfg_dict["maf_threshold"]), "--make-bed", "--out", plink_maf_prefix],
            vcf_file,
        )
        stats.n_variants_after_maf = _count_lines(plink_maf_prefix + ".bim")
        log.info(
            "[%s] variants: %d (raw) -> %d (after MAF >= %s)",
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
            "[%s] variants: %d (after MAF) -> %d (after LD pruning, window=%s step=%s r2=%s)",
            vcf_file, stats.n_variants_after_maf, stats.n_variants_after_pruning,
            cfg_dict["ld_window_size"], cfg_dict["ld_step"], cfg_dict["ld_r2_threshold"],
        )

        # ATOMIC write: export to a temporary prefix and only move the
        # resulting .vcf.gz to the final path once plink2 succeeds. If the
        # process is killed midway, the final path doesn't exist yet (no
        # truncated file with the "final" name).
        tmp_prefix = os.path.join(output_vcf_folder, base_name + "_filtered_tmp")
        tmp_vcf_gz = tmp_prefix + ".vcf.gz"
        if os.path.exists(tmp_vcf_gz):
            os.remove(tmp_vcf_gz)  # leftover from a previous interrupted run

        _run(
            ["plink2", "--bfile", plink_maf_prefix, "--extract", plink_prune_prefix + ".prune.in",
             "--export", "vcf", "bgz", "--out", tmp_prefix],
            vcf_file,
        )

        if not _is_valid_bgzip(tmp_vcf_gz):
            raise RuntimeError(f"plink2 produced an invalid .vcf.gz: {tmp_vcf_gz}")

        os.replace(tmp_vcf_gz, final_vcf_path)

        log.info(
            "[%s] filtered successfully -> %s (%d samples [-%d], %d final variants)",
            vcf_file, final_vcf_path, stats.n_samples_total - stats.n_samples_removed,
            stats.n_samples_removed, stats.n_variants_after_pruning,
        )

        # Intermediate cleanup: ONLY now, after the final .vcf.gz has been
        # written atomically and validated by the bgzip check above. If
        # any earlier step fails, execution ends up in the except block
        # and the intermediates stay on disk (useful for debugging the
        # failed run).
        if not keep_intermediates:
            stats.intermediates_cleaned = _cleanup_intermediates(output_vcf_folder, base_name, vcf_file)

        stats.elapsed_seconds = time.monotonic() - t0
        return vcf_file, True, None, stats

    except subprocess.CalledProcessError as e:
        err = f"{e}\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"
        log.error("[%s] failed: %s", vcf_file, err)
        stats.elapsed_seconds = time.monotonic() - t0
        return vcf_file, False, err, stats
    except Exception as e:  # e.g. the RuntimeError above for an invalid bgzip
        log.error("[%s] failed: %s", vcf_file, e)
        stats.elapsed_seconds = time.monotonic() - t0
        return vcf_file, False, str(e), stats


def _write_stats_csv(all_stats: list[FilterStats], log_dir: str) -> str:
    """Writes a summary CSV with the numeric statistics for every file
    processed in this run, to <log_dir>/filter_vcf_stats.csv. Atomic write
    (tmp + rename), same as other pipeline outputs."""
    os.makedirs(log_dir, exist_ok=True)
    out_path = os.path.join(log_dir, STATS_FILENAME)
    tmp_path = out_path + ".tmp"

    fieldnames = list(asdict(all_stats[0]).keys()) if all_stats else [
        "vcf_file", "n_samples_total", "n_samples_removed", "n_variants_raw",
        "n_variants_after_maf", "n_variants_after_pruning", "elapsed_seconds", "skipped",
        "intermediates_cleaned",
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

    # default: read from config (EXCLUDE_ID_PREFIXES)
    exclude_id_prefixes = exclude_id_prefixes if exclude_id_prefixes is not None else cfg.exclude_id_prefixes
    if exclude_id_prefixes:
        log.info("Id prefixes to exclude from filtering: %s", exclude_id_prefixes)
    else:
        log.warning(
            "EXCLUDE_ID_PREFIXES not configured: no sample will be excluded by id prefix. "
            "If you need to exclude samples with a specific id prefix, "
            "set EXCLUDE_ID_PREFIXES in the .env."
        )

    # keep_intermediate_files: optional project config field. If it
    # doesn't exist, the default is False, i.e. "clean up the
    # intermediates" -- the desired behavior since they're orders of
    # magnitude heavier than the final result and aren't read by any other
    # pipeline step.
    keep_intermediates = getattr(cfg, "keep_intermediate_files", False)
    if keep_intermediates:
        log.info("keep_intermediate_files=True: the plink/maf/pruned intermediate files will NOT be removed.")
    else:
        log.info("The plink/maf/pruned intermediate files will be removed automatically after each file completes successfully.")

    cfg_dict = {
        "maf_threshold": cfg.maf_threshold,
        "ld_window_size": cfg.ld_window_size,
        "ld_step": cfg.ld_step,
        "ld_r2_threshold": cfg.ld_r2_threshold,
        "exclude_id_prefixes": exclude_id_prefixes,
        "log_dir": cfg.log_dir,
        "keep_intermediate_files": keep_intermediates,
    }

    jobs = []
    for generation, input_folder in get_generation_vcf_folders(cfg).items():
        output_vcf_folder = os.path.join(input_folder, OUTPUT_SUBFOLDER)
        os.makedirs(output_vcf_folder, exist_ok=True)
        vcf_files = [
            f for f in os.listdir(input_folder)
            if f.endswith(".vcf.gz") and not f.startswith("._")
        ]
        log.info("Generation %d: %d VCFs found in %s", generation, len(vcf_files), input_folder)
        for vcf_file in vcf_files:
            jobs.append((os.path.join(input_folder, vcf_file), output_vcf_folder))

    log.info("VCF filtering: %d files to process with %d workers", len(jobs), cfg.max_workers)

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
    tot_intermediates_cleaned = sum(s.intermediates_cleaned for s in all_stats)

    log.info(
        "VCF filtering summary: %d total files, %d ok (%d skipped as already present), "
        "%d failed, total time %.1fs. Variants (summed across all files processed in this "
        "run, skips excluded): %d raw -> %d after MAF -> %d final after LD pruning. "
        "Samples excluded by id prefix (sum): %d. Intermediate files removed (sum): %d. "
        "Per-file statistics saved to %s",
        len(jobs), n_ok, n_skipped, len(failed), elapsed_total,
        tot_variants_raw, tot_variants_after_maf, tot_variants_final,
        tot_samples_removed, tot_intermediates_cleaned, stats_path,
    )

    if failed:
        log.error("Filtering complete with %d errors out of %d files: %s", len(failed), len(jobs), [f for f, _ in failed])
    else:
        log.info("VCF filtering complete with no errors (%d files).", len(jobs))


if __name__ == "__main__":
    run_filter_vcf()