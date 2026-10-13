#!/usr/bin/env python3
"""Builds a single (samples x variants) parquet for a FIXED list of variants,
merging several cohorts (gen2 + gen3), starting from the FULL per-chromosome
VCFs (not the *_selected ones).

Where the pieces come from
  - variant list : column names of merged_csv/gen.parquet (chr_pos_ref_alt)
  - genotypes    : <cohort_dir>/<cohort>_vcf_chr<N>.vcf.gz  (needs .tbi/.csi)

Encoding (same as vcf_to_parquet.py)
  - 0 = no alternative allele, 1 = at least one alternative allele
  - genotype missing inside a VCF that HAS the variant -> 0 (default,
    --genotype-missing zero) or NaN (--genotype-missing nan)
  - variant NOT PRESENT in a cohort's VCF -> NaN for all samples of that cohort
    (never 0: absent is not the same as wild type)

Stages (all resumable; rerun the same command and finished work is skipped)
  1. per (cohort, chromosome): region-query the VCF only at the wanted
     positions, match on exact CHROM_POS_REF_ALT (ALT[0], like the old code;
     REF/ALT-swapped records are detected and the genotype is flipped),
     write <work_dir>/<cohort>_chr<N>.found.parquet  (raw: -1 = missing GT)
  2. per chromosome: align the cohorts, stack samples, absent -> NaN
  3. concatenate chromosomes by column (zero-copy in Arrow) and write one
     parquet, columns Int8 (nullable), index "id" (pandas metadata written so
     pq.ParquetFile(...).read(use_pandas_metadata=True).to_pandas() works as
     with the old file).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

THRIFT = dict(thrift_string_size_limit=2_000_000_000, thrift_container_size_limit=2_000_000_000)
CHROMS = [str(i) for i in range(1, 23)]
NON_GEN_COLS = {"FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE", "id"}
VARIANT_RE = re.compile(r"^(?:chr)?(\d+)_(\d+)_([^_]+)_([^_]+)$")
ABSENT = -2  # variant not found in cohort VCF
log = logging.getLogger("build_selected_variants")


# ----------------------------------------------------------------- utilities
def setup_logging(log_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] [pid=%(process)d] %(message)s")
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_path) for h in root.handlers):
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)


def is_valid_parquet(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        _ = pq.ParquetFile(path, **THRIFT).metadata
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("Invalid parquet, will be regenerated: %s (%s)", path, e)
        return False


def atomic_write_text(path: str, text: str) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def atomic_write_parquet(obj, path: str, **kw) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        if isinstance(obj, pd.DataFrame):
            obj.to_parquet(tmp, engine="pyarrow", **kw)
        else:
            pq.write_table(obj, tmp, **kw)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def parquet_index_name(pf: pq.ParquetFile) -> str | None:
    names = pf.schema_arrow.names
    if "id" in names:
        return "id"
    for n in names:
        if n.startswith("__index_level_"):
            return n
    return None


# ------------------------------------------------------------ variant list
def load_variant_list(path: str):
    """Returns (variants_by_chrom, rejected). variants_by_chrom[chrom] is a list
    of (name, pos, ref, alt) in the original column order."""
    pf = pq.ParquetFile(path, **THRIFT)
    idx_name = parquet_index_name(pf)
    names = [n for n in pf.schema_arrow.names if n != idx_name and n not in NON_GEN_COLS]
    log.info("Variant list from %s: %d columns", path, len(names))
    by_chrom: dict[str, list] = {c: [] for c in CHROMS}
    rejected = []
    seen = set()
    for n in names:
        m = VARIANT_RE.match(n)
        if not m or m.group(1) not in by_chrom:
            rejected.append(n)
            continue
        if n in seen:
            rejected.append(n)
            continue
        seen.add(n)
        by_chrom[m.group(1)].append((n, int(m.group(2)), m.group(3), m.group(4)))
    log.info("Parsed %d variants (%d rejected: not chr_pos_ref_alt on chr1-22, or duplicated)",
             len(seen), len(rejected))
    return by_chrom, rejected


# ----------------------------------------------------------------- stage 1
def _resolve_contig(seqnames: list[str], chrom: str) -> str | None:
    for s in seqnames:
        s2 = s[3:] if s.lower().startswith("chr") else s
        if s2 == chrom:
            return s
    return None


def stage1_worker(args):
    """Region-queries one VCF at the wanted positions. Returns a stats dict."""
    cohort, chrom, vcf_path, variants, out_path, log_path = args
    setup_logging(log_path)
    stats_path = out_path + ".stats.json"

    if is_valid_parquet(out_path) and os.path.exists(stats_path):
        log.info("[%s chr%s] already done, skipping", cohort, chrom)
        return json.load(open(stats_path))

    from cyvcf2 import VCF

    t0 = time.monotonic()
    stats = dict(cohort=cohort, chrom=chrom, requested=len(variants), found=0, swapped=0,
                 duplicate_hits=0, n_samples=0, vcf=vcf_path, status="ok")
    if not os.path.exists(vcf_path):
        log.warning("[%s chr%s] VCF not found: %s -> all variants absent for this cohort", cohort, chrom, vcf_path)
        stats["status"] = "vcf_missing"
        return stats  # no output file: stage 2 treats it as absent

    vcf = VCF(vcf_path)
    samples = list(vcf.samples)
    stats["n_samples"] = len(samples)
    contig = _resolve_contig(list(vcf.seqnames), chrom)
    if contig is None:
        log.warning("[%s chr%s] contig not found in header of %s", cohort, chrom, vcf_path)
        stats["status"] = "contig_missing"
        return stats

    wanted: dict[int, list] = {}
    for j, (_, pos, ref, alt) in enumerate(variants):
        wanted.setdefault(pos, []).append((j, ref, alt))

    mat = np.full((len(samples), len(variants)), ABSENT, dtype=np.int8)
    filled = np.zeros(len(variants), dtype=bool)

    for pos in sorted(wanted):
        for rec in vcf(f"{contig}:{pos}-{pos}"):
            if rec.POS != pos:  # region query also returns deletions that span pos
                continue
            alts = rec.ALT
            if not alts:
                continue
            gt = None
            for j, ref, alt in wanted[pos]:
                exact = rec.REF == ref and alts[0] == alt
                swapped = (not exact) and len(alts) == 1 and rec.REF == alt and alts[0] == ref
                if not (exact or swapped):
                    continue
                if filled[j]:
                    stats["duplicate_hits"] += 1
                    continue
                if gt is None:
                    g = np.asarray(rec.genotypes, dtype=np.int16)
                    a, b = g[:, 0], g[:, 1]
                    miss = (a < 0) | (b < 0)
                gt_bin = ((a == 0) | (b == 0)) if swapped else ((a > 0) | (b > 0))
                gt = True
                mat[:, j] = np.where(miss, -1, gt_bin).astype(np.int8)
                filled[j] = True
                stats["swapped"] += int(swapped)

    stats["found"] = int(filled.sum())
    cols = [variants[j][0] for j in np.flatnonzero(filled)]
    df = pd.DataFrame(mat[:, filled], index=pd.Index(samples, name="id"), columns=cols)
    atomic_write_parquet(df, out_path, compression="zstd")
    atomic_write_text(stats_path, json.dumps(stats))
    stats["elapsed"] = round(time.monotonic() - t0, 1)
    log.info("[%s chr%s] %d/%d variants found (%d REF/ALT swapped, %d duplicate hits), %d samples, %.1fs",
             cohort, chrom, stats["found"], stats["requested"], stats["swapped"], stats["duplicate_hits"],
             len(samples), stats["elapsed"])
    return stats


# ----------------------------------------------------------------- stage 2
def read_found(path: str) -> pd.DataFrame:
    return pq.ParquetFile(path, **THRIFT).read(use_pandas_metadata=True).to_pandas()


def stage2_chrom(chrom, variants, cohorts, kept_samples, work_dir, genotype_missing, keep_all_null):
    """Returns (names, list of pa.Array) for one chromosome, rows = concat of
    kept_samples[cohort] in cohort order."""
    names_all = [v[0] for v in variants]
    pos_in_v = {n: i for i, n in enumerate(names_all)}
    blocks = []
    for cohort in cohorts:
        ids = kept_samples[cohort]
        blk = np.full((len(ids), len(names_all)), ABSENT, dtype=np.int8)
        path = os.path.join(work_dir, f"{cohort}_chr{chrom}.found.parquet")
        if is_valid_parquet(path):
            df = read_found(path)
            rows = df.index.get_indexer(ids)
            ok = rows >= 0
            if (~ok).any():
                log.warning("[%s chr%s] %d kept samples not present in this chromosome's VCF -> NaN", cohort, chrom, int((~ok).sum()))
            cols = np.array([pos_in_v[c] for c in df.columns], dtype=np.int64)
            if len(cols):
                sub = np.full((len(ids), len(cols)), ABSENT, dtype=np.int8)
                sub[ok] = df.to_numpy()[rows[ok]]
                blk[:, cols] = sub
        else:
            log.warning("[%s chr%s] no intermediate file -> all variants NaN for this cohort", cohort, chrom)
        blocks.append(blk)
    arr = np.vstack(blocks) if len(blocks) > 1 else blocks[0]
    del blocks

    null_mask = arr == ABSENT
    if genotype_missing == "nan":
        null_mask |= arr == -1
    else:
        arr[arr == -1] = 0
    arr[null_mask] = 0

    all_null = null_mask.all(axis=0)
    keep = np.arange(len(names_all)) if keep_all_null else np.flatnonzero(~all_null)
    arr = np.asfortranarray(arr)
    null_mask = np.asfortranarray(null_mask)
    names = [names_all[j] for j in keep]
    arrays = [pa.array(arr[:, j], mask=null_mask[:, j], type=pa.int8()) for j in keep]
    dropped = [names_all[j] for j in np.flatnonzero(all_null)] if not keep_all_null else []
    return names, arrays, dropped


# ----------------------------------------------------------------- stage 3
def pandas_metadata(index_name: str, names: list[str]) -> bytes:
    """pandas metadata so that to_pandas() restores index `id` and Int8 dtypes."""
    meta = {
        "index_columns": [index_name],
        "column_indexes": [{"name": None, "field_name": None, "pandas_type": "unicode",
                            "numpy_type": "object", "metadata": {"encoding": "UTF-8"}}],
        "columns": [{"name": n, "field_name": n, "pandas_type": "int8", "numpy_type": "Int8", "metadata": None}
                    for n in names]
                   + [{"name": index_name, "field_name": index_name, "pandas_type": "unicode",
                       "numpy_type": "object", "metadata": None}],
        "creator": {"library": "pyarrow", "version": pa.__version__},
        "pandas_version": pd.__version__,
    }
    return json.dumps(meta).encode()


def cohort_generation(name: str) -> int | None:
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else None


def main() -> None:
    base = "/mnt/cresla_prod/genome_datasets"
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants-from", default=f"{base}/merged_csv/gen.parquet",
                    help="parquet whose column names are the variants (chr_pos_ref_alt)")
    ap.add_argument("--cohort", action="append", metavar="NAME=DIR",
                    help="repeatable; default gen2=<base>/gen2 gen3=<base>/gen3. VCFs are DIR/NAME_vcf_chr<N>.vcf.gz")
    ap.add_argument("--output", default=f"{base}/gen3/gen2_gen3_selected_variants.parquet")
    ap.add_argument("--work-dir", default=None, help="intermediate files (default: <output>.work)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--chroms", nargs="*", default=CHROMS)
    ap.add_argument("--exclude-id-prefix", action="append", default=[],
                    help="drop samples whose id starts with this (same as EXCLUDE_ID_PREFIXES in filter_vcf.py); repeatable")
    ap.add_argument("--genotype-missing", choices=["zero", "nan"], default="zero")
    ap.add_argument("--keep-all-null", action="store_true",
                    help="keep variants absent from ALL cohorts as all-NaN columns (default: drop and list them)")
    ap.add_argument("--force", action="store_true", help="redo stage 1 even if intermediates exist")
    args = ap.parse_args()

    cohorts = dict(c.split("=", 1) for c in (args.cohort or [f"gen2={base}/gen2", f"gen3={base}/gen3"]))
    work_dir = args.work_dir or args.output + ".work"
    os.makedirs(work_dir, exist_ok=True)
    log_path = os.path.join(work_dir, "build_selected_variants.log")
    setup_logging(log_path)
    t_start = time.monotonic()
    log.info("Cohorts: %s | output: %s | genotype_missing=%s", cohorts, args.output, args.genotype_missing)

    by_chrom, rejected = load_variant_list(args.variants_from)
    if rejected:
        atomic_write_text(os.path.join(work_dir, "rejected_variant_names.txt"), "\n".join(rejected) + "\n")
    chroms = [c for c in args.chroms if by_chrom.get(c)]

    # ---- stage 1
    jobs = []
    for cohort, cdir in cohorts.items():
        for chrom in chroms:
            out = os.path.join(work_dir, f"{cohort}_chr{chrom}.found.parquet")
            if args.force:
                for p in (out, out + ".stats.json"):
                    if os.path.exists(p):
                        os.remove(p)
            vcf = os.path.join(cdir, f"{cohort}_vcf_chr{chrom}.vcf.gz")
            jobs.append((cohort, chrom, vcf, by_chrom[chrom], out, log_path))
    jobs.sort(key=lambda j: -len(j[3]))
    log.info("Stage 1: %d jobs, %d workers", len(jobs), args.workers)
    all_stats = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(stage1_worker, j) for j in jobs]):
            all_stats.append(fut.result())
    pd.DataFrame(all_stats).sort_values(["cohort", "chrom"], key=lambda s: s.map(lambda x: int(x) if str(x).isdigit() else x)
                                        ).to_csv(os.path.join(work_dir, "stage1_stats.tsv"), sep="\t", index=False)

    # ---- sample sets per cohort (from the first available intermediate)
    kept_samples: dict[str, list[str]] = {}
    sample_gen: list[tuple[str, int | None]] = []
    seen_ids: set[str] = set()
    for cohort in cohorts:
        ids = None
        for chrom in chroms:
            p = os.path.join(work_dir, f"{cohort}_chr{chrom}.found.parquet")
            if is_valid_parquet(p):
                ids = pq.ParquetFile(p, **THRIFT).read(columns=[], use_pandas_metadata=True).to_pandas().index.astype(str).tolist()
                break
        if ids is None:
            raise SystemExit(f"No VCF data found for cohort {cohort}")
        n0 = len(ids)
        if args.exclude_id_prefix:
            ids = [i for i in ids if not i.startswith(tuple(args.exclude_id_prefix))]
        dup = [i for i in ids if i in seen_ids]
        if dup:
            log.warning("%d sample ids of %s already present in an earlier cohort (kept the first): %s", len(dup), cohort, dup[:10])
            ids = [i for i in ids if i not in seen_ids]
        seen_ids.update(ids)
        kept_samples[cohort] = ids
        sample_gen += [(i, cohort_generation(cohort)) for i in ids]
        log.info("Cohort %s: %d samples in VCF, %d kept", cohort, n0, len(ids))

    # ---- stage 2 + 3
    all_names: list[str] = []
    all_arrays: list[pa.Array] = []
    dropped_all: list[str] = []
    for chrom in chroms:
        names, arrays, dropped = stage2_chrom(chrom, by_chrom[chrom], list(cohorts), kept_samples, work_dir,
                                              args.genotype_missing, args.keep_all_null)
        all_names += names
        all_arrays += arrays
        dropped_all += dropped
        log.info("chr%s: %d variants in output (%d absent from all cohorts)", chrom, len(names), len(dropped))

    if dropped_all:
        atomic_write_text(os.path.join(work_dir, "variants_absent_everywhere.txt"), "\n".join(dropped_all) + "\n")
    index_ids = [i for c in cohorts for i in kept_samples[c]]
    id_arr = pa.array(index_ids, type=pa.string())
    table = pa.Table.from_arrays(all_arrays + [id_arr], names=all_names + ["id"])
    table = table.replace_schema_metadata({b"pandas": pandas_metadata("id", all_names)})
    log.info("Writing %s (%d samples x %d variants)", args.output, table.num_rows, len(all_names))
    atomic_write_parquet(table, args.output, compression="zstd", write_statistics=False)

    gm = pd.DataFrame(sample_gen, columns=["id", "generation"])
    gm.to_csv(args.output + ".generation_map.csv", index=False)

    # ---- quick sanity report
    summ = pd.DataFrame(all_stats)
    per_c = summ.groupby("cohort")[["requested", "found", "swapped"]].sum()
    per_c["absent(NaN)"] = per_c["requested"] - per_c["found"]
    log.info("Per-cohort summary:\n%s", per_c.to_string())
    log.info("Done in %.1fs -> %s", time.monotonic() - t_start, args.output)


if __name__ == "__main__":
    main()