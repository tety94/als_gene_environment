"""Converts the filtered VCFs into a single genotype matrix (parquet).

Chromosome merging is a JOIN based on the sample id (pandas, aligning by
index, not by row position): it works regardless of row order in the
individual files.

Robustness against partial/corrupted files:
  - EVERY parquet write (per-file raw, per-chromosome merge, final genome)
    is ATOMIC: it writes to "<path>.tmp" and then does os.replace(), so a
    process killed midway never leaves a corrupted file at the final path.
  - Before reading/skipping an existing parquet file, it is VALIDATED
    (opening the footer via pyarrow.parquet.ParquetFile). If invalid, it
    is deleted and regenerated, instead of blowing up the whole pipeline
    with a traceback deep in the chain.
  - merge_chromosome doesn't die on the first corrupted file: it removes
    the corrupted file (regenerated on the next run) and flags that
    chromosome as "needs redo", letting the others continue.
  - The pipeline is therefore RESUMABLE: rerunning the same command skips
    everything already valid (no recomputing the samples x 1M+ variants
    per chromosome already done), regenerating only what's missing or
    broken.
  - Chromosome-to-file matching uses a regex requiring that the chromosome
    number isn't followed by another digit (no more chr1/chr11/chr12...
    confusion).
  - Logging also goes to file (not just console), in both the main
    process and every worker (workers are separate processes and don't
    inherit the parent's logging handlers).
  - Corrupted/truncated input VCFs (.vcf.gz): filter_vcf.py writes its
    outputs in bgzip (*_filtered.vcf.gz); the glob here matches that. If
    one of these .vcf.gz files is truncated (an earlier run interrupted),
    instead of letting cyvcf2 fail with a cryptic error mid-conversion,
    the file is VALIDATED before being read (full decompression) and, if
    invalid, is DELETED: the next run of filter_vcf.py will regenerate it
    automatically (idempotency), and this script flags the file as "needs
    redo" instead of blocking the whole batch.
  - Numeric statistics: besides the existing text logs (variants dropped
    for missingness, shape of each output), a per-chromosome summary CSV
    is also written to <log_dir>/vcf_to_parquet_stats.csv (samples, total
    variants, variants dropped for missing rate, final variants).

Other design choices:
  - No giant intermediate CSVs: writes directly to Parquet (compressed,
    columnar, much lighter/faster to re-read) at every stage (per-file,
    per-chromosome, whole genome).
  - File-level parallelization (ProcessPoolExecutor).
  - The per-SNP missing-percentage filter is computed BEFORE
    binarization, otherwise the information "how much missingness this
    SNP had" would be lost once forced to 0.
  - The "missing genotype -> 0 (non-mutant)" choice is a modeling
    decision, not a technical detail: it's explicit and configurable here
    (MISSING_GENOTYPE_STRATEGY), defaulting to "zero" for compatibility
    with earlier analyses, but clearly flagged in the log and comments
    because it's the most delicate choice in the whole data conversion
    (treating a missing value as "wild type" can introduce bias if the
    missingness isn't random).
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
import pyarrow.parquet as pq

from gene_environment.config import get_config, get_generation_vcf_folders
from gene_environment.logging_utils import configure_logging, get_logger
from gene_environment.utils.id_utils import clean_sample_id

log = get_logger(__name__)

CHROMOSOMES = [str(i) for i in range(1, 23)]

LOG_FILENAME = "vcf_to_parquet.log"
STATS_FILENAME = "vcf_to_parquet_stats.csv"


class CorruptParquetError(Exception):
    """Raised when one or more .raw.parquet files are corrupted/truncated.
    The corrupted file is deleted before raising the exception, so simply
    rerunning the pipeline regenerates it."""


class CorruptInputVCFError(Exception):
    """Raised when one or more *_filtered.vcf.gz files (filter_vcf.py's
    output) are corrupted/truncated. The file is deleted before raising
    the exception: the next run of filter_vcf.py will regenerate it
    automatically thanks to its idempotency."""


def _add_file_logging(log_dir: str) -> None:
    """Adds (once per process) a FileHandler to the root logger, so logs go
    both to console and to <log_dir>/vcf_to_parquet.log. Must be called in
    both the main process and every worker (separate processes don't
    inherit the parent's logging handlers)."""
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


def _is_valid_parquet(path: str) -> bool:
    """True if the file exists and is a readable parquet (thrift footer
    ok). Doesn't read the whole data, just the metadata: fast even on
    large files."""
    if not os.path.exists(path):
        return False
    try:

        pf = pq.ParquetFile(
            path,
            thrift_string_size_limit=2_000_000_000,
            thrift_container_size_limit=2_000_000_000,
        )
        _ = pf.metadata  # forces reading/validating the footer
        return True
    except Exception as e:
        log.warning("Invalid/corrupted parquet, will be regenerated: %s (%s)", path, e)
        return False


def _is_valid_bgzip_vcf(path: str) -> bool:
    """True if the *_filtered.vcf.gz is a fully readable bgzip/gzip (no
    truncated block). Symmetric to the check done on the filter_vcf.py
    side before writing the file: here it's used to distinguish a
    genuinely corrupted input VCF from a different error during parsing
    with cyvcf2."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        with gzip.open(path, "rb") as f:
            while f.read(1024 * 1024):
                pass
        return True
    except Exception as e:
        log.warning("Invalid/truncated input VCF: %s (%s)", path, e)
        return False


def _write_parquet_atomic(df: pd.DataFrame, out_path: str, **to_parquet_kwargs) -> None:
    """Writes to <out_path>.<pid>.tmp and then atomically renames to
    out_path. Guarantees that, if the process is killed mid-write (OOM,
    kill, worker crash, disk full...), the final path NEVER contains a
    truncated parquet: it either doesn't exist, is the old (valid) one, or
    is the new complete one. The pid in the name prevents two
    concurrent processes/runs from overwriting each other's temp file."""
    tmp_path = f"{out_path}.{os.getpid()}.tmp"
    try:
        df.to_parquet(tmp_path, engine="pyarrow", **to_parquet_kwargs)
        os.replace(tmp_path, out_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

def _genotype_to_dosage(gt) -> int:
    """gt: cyvcf2 tuple (allele1, allele2, phased). -1 = missing."""
    if gt is None or gt[0] is None or gt[1] is None:
        return -1
    a, b = gt[0], gt[1]
    if a < 0 or b < 0:
        return -1
    return a + b


def vcf_file_to_dosage_df(vcf_path: str) -> pd.DataFrame:
    """Reads a filtered (bgzip) VCF and returns a DataFrame (samples x
    variants) of raw dosages (0/1/2/-1=missing). cyvcf2 is imported inside
    this function so it isn't a hard dependency of the whole package.
    cyvcf2 reads .vcf.gz bgzip natively, no manual decompression needed
    here."""
    from cyvcf2 import VCF

    vcf = VCF(vcf_path)
    samples = vcf.samples

    variant_ids = []
    columns = []  # one column (np.array) per variant

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
    """Returns (out_parquet, generation, samples, input_was_corrupt).
    If the last element is True, out_parquet/samples are empty/invalid:
    the input file was deleted and the caller must be notified."""
    vcf_path, out_parquet, log_dir, generation = args
    configure_logging(log_dir)
    _add_file_logging(log_dir)

    if _is_valid_parquet(out_parquet):
        log.info("Skip (already converted and valid): %s", out_parquet)
        samples = pq.ParquetFile(
            out_parquet, thrift_string_size_limit=2_000_000_000, thrift_container_size_limit=2_000_000_000,
        ).read(columns=[], use_pandas_metadata=True).to_pandas().index.tolist()
        return out_parquet, generation, samples, False

    if os.path.exists(out_parquet):
        log.warning("Existing but corrupted/truncated file, regenerating: %s", out_parquet)
        os.remove(out_parquet)

    if not _is_valid_bgzip_vcf(vcf_path):
        log.error(
            "Corrupted/truncated input VCF, deleting it to force regeneration by "
            "filter_vcf.py on the next run: %s", vcf_path,
        )
        os.remove(vcf_path)
        return "", generation, [], True

    log.info("Converting VCF -> parquet (generation %d): %s", generation, vcf_path)
    try:
        df = vcf_file_to_dosage_df(vcf_path)
    except Exception as e:
        # cyvcf2 can also fail on a bgzip that's "valid" at the block level
        # but with VCF content truncated/malformed midway (e.g. a line cut
        # off mid-write that the gzip check doesn't catch). Handled the
        # same way: delete and flag for regeneration.
        log.error(
            "Error reading the VCF with cyvcf2, treating it as corrupted and deleting it: %s (%s)",
            vcf_path, e,
        )
        os.remove(vcf_path)
        return "", generation, [], True

    _write_parquet_atomic(df, out_parquet, compression="zstd")
    log.info("Wrote %s (%d samples, %d variants)", out_parquet, df.shape[0], df.shape[1])
    return out_parquet, generation, df.index.tolist(), False


def _detect_duplicate_chrom_sources(vcf_paths: list[str]) -> None:
    """Flags (without blocking) if more than one filtered VCF seems to
    refer to the same chromosome in the same folder: happens when files
    from earlier runs with different naming are left around (e.g.
    'chr1_filtered.vcf.gz' and 'chr1.vcf_filtered.vcf.gz' together), and
    leads to needlessly processing the same chromosome twice."""
    by_chrom: dict[str, list[str]] = {}
    for p in vcf_paths:
        name = os.path.basename(p)
        m = re.search(r"chr(\d+)(?!\d)", name)
        if m:
            by_chrom.setdefault(m.group(1), []).append(p)
    for chrom, paths in by_chrom.items():
        if len(paths) > 1:
            log.warning(
                "chr%s: found %d filtered VCF files in the same folder (possibly "
                "leftovers from earlier runs with different naming) - ALL of them "
                "will be processed, check if this is intended: %s",
                chrom, len(paths), paths,
            )


def convert_filtered_vcfs_to_parquet() -> tuple[list[str], dict[str, int]]:
    """Step 1: each *_filtered.vcf.gz -> one raw parquet (dosages 0/1/2/-1).

    Also returns the sample_id -> generation map, built from WHICH folder
    (VCF_DIR_GENn) each VCF file comes from: it's the only reliable source
    of a patient's cohort when the environmental file has no generation
    information (the environment/genetics join happens on id alone).

    If one or more input VCFs turn out to be corrupted they are deleted
    (so filter_vcf.py regenerates them on the next run) and
    CorruptInputVCFError is raised at the end, AFTER still processing all
    the other files: a corrupted input no longer blocks the whole batch."""
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
    log.info("Converting filtered VCFs -> raw parquet: %d files, %d workers", len(jobs), cfg.max_workers)
    out_paths = []
    sample_generation: dict[str, int] = {}
    conflicts = []
    corrupt_inputs = []

    max_workers = min(cfg.max_workers, 6)
    log.info("Using %d workers (capped at %d to avoid OOM on large chromosomes)", max_workers, max_workers)
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
            "%d sample ids are present in MORE than one generation (kept the last "
            "one seen): %s%s",
            len(conflicts), conflicts[:10], " ..." if len(conflicts) > 10 else "",
        )

    if corrupt_inputs:
        raise CorruptInputVCFError(
            f"{len(corrupt_inputs)} input VCFs were corrupted/unreadable and were "
            f"deleted. Rerun filter_vcf.py first (it will regenerate them thanks to "
            f"its idempotency), then rerun this pipeline: the files already converted "
            f"in this run ({len(out_paths)}) will not be recomputed."
        )

    return out_paths, sample_generation


def save_sample_generation_map(sample_generation: dict[str, int], out_path: str) -> None:
    df = pd.DataFrame({"id": list(sample_generation.keys()), "generation": list(sample_generation.values())})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = out_path + ".tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, out_path)
    log.info("id->generation map saved to %s (%d samples)", out_path, len(df))


def _files_for_chromosome(chrom: str, raw_parquet_paths: list[str]) -> list[str]:
    """Robust match: 'chr<chrom>' followed by a separator ('.', '_') and
    NEVER by another digit, to prevent chr1 from matching chr11, chr12,
    ... chr19."""
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
    """Step 2: merges (by id, not row position) all the raw parquet files
    for the same chromosome, filters by missing rate and binarizes.
    Returns (chromosome parquet path or None, numeric statistics or None
    if there was nothing to do).

    If one or more raw files are corrupted/truncated: they are deleted (so
    convert_filtered_vcfs_to_parquet regenerates them on the next run) and
    CorruptParquetError is raised, so the caller can skip ONLY this
    chromosome without failing the whole pipeline."""
    t0 = time.monotonic()
    out_path = os.path.join(out_folder, f"chr{chrom}_merged.parquet")
    if not force and _is_valid_parquet(out_path):
        log.info("chr%s: already present and valid, skipping (%s)", chrom, out_path)
        return out_path, None

    chrom_files = _files_for_chromosome(chrom, raw_parquet_paths)
    chrom_files = [
        p for p in chrom_files
        if "_selected" not in os.path.basename(p)
    ]
    if not chrom_files:
        log.warning("No file found for chr%s", chrom)
        return None, None

    dfs = []
    corrupt_files = []
    for p in chrom_files:
        if not _is_valid_parquet(p):
            corrupt_files.append(p)
            continue
        dfs.append(
            pq.ParquetFile(p, thrift_string_size_limit=2_000_000_000,
                           thrift_container_size_limit=2_000_000_000).read(use_pandas_metadata=True).to_pandas()
        )

    if corrupt_files:
        for p in corrupt_files:
            log.error(
                "chr%s: corrupted/truncated raw file, deleting it to force "
                "regeneration on the next run: %s", chrom, p,
            )
            os.remove(p)
        raise CorruptParquetError(
            f"chr{chrom}: {len(corrupt_files)} .raw.parquet files were corrupted and "
            f"were deleted. Rerun the pipeline: they will be regenerated automatically "
            f"(already-valid files are not recomputed)."
        )

    if not dfs:
        log.warning("chr%s: no valid file found after validation", chrom)
        return None, None

    # vertical concat (new samples), aligning on COLUMNS (variants) by
    # name, not by position -> pandas' implicit outer join.
    merged = pd.concat(dfs, axis=0, join="outer")
    merged = merged[~merged.index.duplicated(keep="first")]

    n_variants_total = merged.shape[1]
    # -1 isn't the only way a variant can be "missing" for a sample after
    # the join="outer" concat above, across batches with slightly
    # different variant sets.
    missing_frac = ((merged < 0) | merged.isna()).mean()
    keep_cols = missing_frac[missing_frac < null_percentage].index
    dropped = merged.shape[1] - len(keep_cols)
    if dropped:
        log.info("chr%s: %d/%d variants dropped for missing rate >= %.0f%%", chrom, dropped, merged.shape[1], null_percentage * 100)
    merged = merged[keep_cols]

    if merged.shape[1] == 0:
        log.warning("chr%s: no variants survived the missing-rate filter", chrom)
        return None, None

    arr = merged.to_numpy(dtype=np.float32, copy=True)
    # Besides -1 (explicit missing from _genotype_to_dosage), the
    # join="outer" concat above introduces "genuine" NaNs whenever a
    # variant is present in one batch's raw parquet but absent in
    # another: each batch does MAF+LD pruning independently in
    # filter_vcf.py, so the set of surviving variants for the same
    # chromosome can differ slightly batch to batch. This must be treated
    # as missing data just like -1: using only `arr < 0` would let NaN
    # through untouched (in numpy a comparison with NaN is always False,
    # so neither `arr < 0` nor `arr > 0` catches it) and it would survive
    # to the final astype(int), which blows up with IntCastingNaNError.
    missing_mask = (arr < 0) | np.isnan(arr)
    if missing_strategy == "zero":
        arr[missing_mask] = 0
    else:  # "nan": explicit missing, NOT silently treated as wild type
        arr[missing_mask] = np.nan
    arr[arr > 0] = 1
    merged[:] = arr

    os.makedirs(out_folder, exist_ok=True)
    _write_parquet_atomic(
        merged.astype("Int8" if missing_strategy == "nan" else np.int8),
        out_path, compression="zstd",
    )
    log.info("chr%s: saved %s (%d samples, %d variants)", chrom, out_path, *merged.shape)

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
    """Step 3: final whole-genome merge, by ID. pd.concat(axis=1)
    automatically aligns by index (sample id), regardless of row order in
    the individual chromosome files."""
    if not force and _is_valid_parquet(out_path):
        log.info("Full genome already present and valid, skipping: %s", out_path)
        return

    log.info("Final merge of %d chromosome files by sample id", len(chrom_parquet_paths))

    corrupt = [p for p in chrom_parquet_paths if not _is_valid_parquet(p)]
    if corrupt:
        for p in corrupt:
            log.error("Corrupted chromosome file found during the final merge, deleting it: %s", p)
            os.remove(p)
        raise CorruptParquetError(
            f"{len(corrupt)} chromosome files were corrupted and were deleted. "
            f"Rerun the pipeline to regenerate them before the final merge."
        )

    frames = [pq.ParquetFile(p, thrift_string_size_limit=2_000_000_000, thrift_container_size_limit=2_000_000_000).read(
        use_pandas_metadata=True).to_pandas()
              for p in chrom_parquet_paths]
    full = pd.concat(frames, axis=1, join="outer")
    full.index.name = "id"
    _write_parquet_atomic(full, out_path, compression="zstd")
    log.info("Full genome saved to %s (%d samples, %d variants)", out_path, *full.shape)


def _write_stats_csv(all_stats: list[ChromStats], log_dir: str) -> str:
    """Writes a per-chromosome summary CSV to
    <log_dir>/vcf_to_parquet_stats.csv. Atomic write (tmp + rename)."""
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
            "Per-chromosome merge summary (%d chromosomes processed in this run): "
            "%d total variants -> %d dropped for missing -> %d final. "
            "Per-chromosome statistics saved to %s",
            len(chrom_stats), tot_variants_total, tot_variants_dropped, tot_variants_final, stats_path,
        )

    if failed_chroms:
        raise RuntimeError(
            f"Chromosomes with corrupted files (deleted, need regeneration): {failed_chroms}. "
            f"Rerun the pipeline: the already-valid files ({len(chrom_paths)} chromosomes) "
            f"will be skipped and only these {len(failed_chroms)} will be redone. "
            f"The final genome parquet is not built until all of them are in order, "
            f"to avoid silently producing a dataset with missing chromosomes."
        )

    out_path = os.path.join(cfg.output_folder, "gen.parquet")
    build_full_genome_parquet(chrom_paths, out_path, force=force)

    log.info("vcf_to_parquet pipeline complete in %.1fs -> %s", time.monotonic() - t_start, out_path)
    return out_path


if __name__ == "__main__":
    run_vcf_to_parquet_pipeline()