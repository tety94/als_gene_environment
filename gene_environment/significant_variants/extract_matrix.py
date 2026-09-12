#!/usr/bin/env python3
"""Extracts, for the significant variants, the binary genotype (0/1) for
every patient in gen1/gen2/gen3, directly from the indexed VCFs (bcftools).

Key design points:
  1) INCREMENTAL WRITES: the combined CSV is written/appended immediately
     after each generation completes (not all at once at the end). If the
     script stops midway (e.g. gen3 fails), gen1 and gen2 are already on disk.
  2) RESUME CHECKPOINT: a JSON state file next to the CSV tracks which
     generations are already completed; a rerun skips those instead of
     starting over.
  3) Per-chromosome parallelization within each generation
     (ProcessPoolExecutor): bcftools queries for different chromosomes are
     independent.
  4) Config (VCF paths, output folder) lives in gene_environment.config
     instead of being hardcoded at the top of the script.

Checkpoint consistency: the checkpoint tracks completed generations, and
ALSO freezes the list of significant variants used to build the CSV
columns. If the DB changed between runs (e.g. modeling.py re-run, new
significant variants added for another exposure) and the checkpoint became
out of sync with what's on disk (e.g. a crash between writing the CSV and
saving the state), a rerun could otherwise append a generation with a
different number of columns than what's already on disk, producing a CSV
with rows of inconsistent length (unreadable by pandas: "Expected N
fields, saw M"). To prevent this:
    - the variant list is frozen into the checkpoint on the first run (or
      the first run with force=True);
    - a rerun without force=True verifies the variant set in the DB hasn't
      changed relative to the checkpoint: if it has, it stops with an
      explicit error instead of producing an inconsistent CSV;
    - before every append, the header already on disk is validated against
      the columns expected for that run.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from gene_environment.config import get_config
from gene_environment.db.repository import get_significant_results
from gene_environment.logging_utils import configure_logging, get_logger
from gene_environment.utils.id_utils import build_variant_label

log = get_logger(__name__)

BCFTOOLS = "bcftools"
VCF_FILENAME_PATTERN = {
    1: "gen1_onlycases_vcf_chr{chrom}.vcf.gz",
    2: "gen2_vcf_chr{chrom}.vcf.gz",
    3: "gen3_vcf_chr{chrom}.vcf.gz",
}


def _vcf_dirs(cfg) -> dict[int, str]:
    return {1: cfg.vcf_dir_gen1, 2: cfg.vcf_dir_gen2, 3: cfg.vcf_dir_gen3}


def vcf_path_for(cfg, generation: int, chrom: str) -> str:
    filename = VCF_FILENAME_PATTERN[generation].format(chrom=chrom)
    return os.path.join(_vcf_dirs(cfg)[generation], filename)


def detect_chrom_naming(vcf_path: str) -> str | None:
    try:
        result = subprocess.run([BCFTOOLS, "view", "-h", vcf_path], capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        log.warning("Impossibile leggere l'header di %s: %s", vcf_path, e)
        return None
    for line in result.stdout.splitlines():
        if line.startswith("##contig"):
            m = re.search(r"ID=([^,>]+)", line)
            if m:
                return m.group(1)
    return None


def resolve_chrom_name(chrom_raw: str, vcf_path: str, _cache: dict = {}) -> str:
    if vcf_path in _cache:
        contig_example = _cache[vcf_path]
    else:
        contig_example = detect_chrom_naming(vcf_path)
        _cache[vcf_path] = contig_example
    if contig_example is None:
        return chrom_raw
    if contig_example.startswith("chr") and not chrom_raw.startswith("chr"):
        return f"chr{chrom_raw}"
    if not contig_example.startswith("chr") and chrom_raw.startswith("chr"):
        return chrom_raw[3:]
    return chrom_raw


def get_samples(vcf_path: str) -> list[str]:
    result = subprocess.run([BCFTOOLS, "query", "-l", vcf_path], capture_output=True, text=True, check=True)
    return result.stdout.strip().split("\n") if result.stdout.strip() else []


def query_positions(vcf_path: str, chrom: str, positions: list[int]) -> list[str]:
    region_list = ",".join(f"{chrom}:{pos}-{pos}" for pos in positions)
    cmd = [BCFTOOLS, "query", "-r", region_list, "-f", "%CHROM\t%POS\t%REF\t%ALT[\t%GT]\n", vcf_path]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return [line for line in result.stdout.splitlines() if line.strip()]


def parse_gt_row(line: str, target_ref: str, target_alt: str) -> list[int] | None:
    parts = line.split("\t")
    ref, alt_field = parts[2], parts[3]
    gts = parts[4:]
    alt_list = alt_field.split(",")
    if ref != target_ref or target_alt not in alt_list:
        return None
    target_idx = str(alt_list.index(target_alt) + 1)
    dosages = []
    for gt in gts:
        alleles = re.split(r"[/|]", gt)
        if any(a == "." for a in alleles):
            dosages.append(-1)
        else:
            dosages.append(sum(1 for a in alleles if a == target_idx))
    return dosages


def binarize_vectorized(df_dosage: pd.DataFrame) -> pd.DataFrame:
    df_num = df_dosage.apply(pd.to_numeric, errors="coerce")
    arr = np.select([df_num.eq(0), df_num.isin([1, 2])], [0, 1], default=np.nan)
    return pd.DataFrame(arr, index=df_dosage.index, columns=df_dosage.columns)


def _extract_chrom_worker(args) -> tuple[str, dict, list[str]]:
    """Extract a single (generation, chromosome) combination. Runs in a
    separate process -> returns serializable data (dict), not a DataFrame."""
    generation, chrom, group_records, vcf_path, log_dir = args
    configure_logging(log_dir)

    if not os.path.exists(vcf_path):
        log.warning("VCF not found for generation %d, chr%s: %s", generation, chrom, vcf_path)
        return chrom, {}, [r["label"] for r in group_records]

    chrom_name = resolve_chrom_name(str(chrom), vcf_path)
    samples = get_samples(vcf_path)

    positions = [r["pos"] for r in group_records]
    t0 = time.perf_counter()
    rows = query_positions(vcf_path, chrom_name, positions)
    log.info("gen%d chr%s: %d variants requested, %d rows found (%.2fs)",
              generation, chrom, len(positions), len(rows), time.perf_counter() - t0)

    rows_by_pos = defaultdict(list)
    for line in rows:
        rows_by_pos[int(line.split("\t")[1])].append(line)

    dosages_by_label: dict[str, dict[str, int]] = {}
    not_found = []

    for rec in group_records:
        candidate_rows = rows_by_pos.get(rec["pos"], [])
        dosages = None
        for line in candidate_rows:
            dosages = parse_gt_row(line, rec["ref"], rec["alt"])
            if dosages is not None:
                break
        if dosages is None:
            if candidate_rows:
                log.warning("[MISMATCH] %s: REF/ALT don't match (expected %s>%s)", rec["label"], rec["ref"], rec["alt"])
            not_found.append(rec["label"])
            continue
        dosages_by_label[rec["label"]] = dict(zip(samples, dosages))

    return chrom, dosages_by_label, not_found


def extract_generation(cfg, generation: int, variants_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    t_start = time.perf_counter()
    log.info("=== Generation %d: starting ===", generation)

    jobs = []
    for chrom, group in variants_df.groupby("chrom"):
        vcf_path = vcf_path_for(cfg, generation, chrom)
        group_records = group.to_dict("records")
        jobs.append((generation, chrom, group_records, vcf_path, cfg.log_dir))

    all_dosages: dict[str, dict[str, int]] = {}
    labels_not_found: list[str] = []

    with ProcessPoolExecutor(max_workers=min(cfg.max_workers, max(1, len(jobs)))) as ex:
        futures = [ex.submit(_extract_chrom_worker, job) for job in jobs]
        for fut in as_completed(futures):
            chrom, dosages_by_label, not_found = fut.result()
            all_dosages.update(dosages_by_label)
            labels_not_found.extend(not_found)

    if not all_dosages:
        log.error("Generation %d: no variants extracted.", generation)
        return pd.DataFrame(), labels_not_found

    df = pd.DataFrame(all_dosages)
    df_bin = binarize_vectorized(df)

    log.info("=== Generation %d: completed in %.1fs (%d patients, %d variants found) ===",
              generation, time.perf_counter() - t_start, len(df_bin), df_bin.shape[1])
    return df_bin, labels_not_found


def _load_checkpoint(state_path: str) -> dict:
    if os.path.exists(state_path):
        with open(state_path) as f:
            return json.load(f)
    return {"completed_generations": [], "variant_labels": None}


def _save_checkpoint(state_path: str, state: dict) -> None:
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)


def _read_existing_header(out_path: str) -> list[str] | None:
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        return None
    with open(out_path) as f:
        first_line = f.readline().rstrip("\n")
    return first_line.split(",") if first_line else None


def run_extract_significant_matrices(force: bool = True, exposure: str | None = None) -> str | None:
    cfg = get_config()
    configure_logging(cfg.log_dir)

    log.info("Fetching significant variants from the DB%s", f" (exposure={exposure})" if exposure else "")
    sig = get_significant_results()
    if sig.empty:
        log.info("No significant variants found. Exiting.")
        return None

    sig["ref"] = sig["mutation"].apply(lambda m: m.split("_", 1)[0])
    sig["alt"] = sig["mutation"].apply(lambda m: m.split("_", 1)[1])
    sig["chrom"] = sig["chromosome"].astype(str)
    sig["pos"] = sig["position"].astype(int)
    sig["label"] = sig.apply(lambda r: build_variant_label(r["chromosome"], r["position"], r["mutation"]), axis=1)
    variants_df = sig[["chrom", "pos", "ref", "alt", "label"]].drop_duplicates(subset="label")
    all_variant_labels = sorted(variants_df["label"].unique())
    log.info("%d unique variants to extract.", len(variants_df))

    os.makedirs(cfg.significant_matrix_dir, exist_ok=True)
    out_path = os.path.join(cfg.significant_matrix_dir, "combined_significant_variants.csv")
    state_path = os.path.join(cfg.significant_matrix_dir, "extract_state.json")

    if force:
        state = {"completed_generations": [], "variant_labels": all_variant_labels}
        if os.path.exists(out_path):
            os.remove(out_path)
        if os.path.exists(state_path):
            os.remove(state_path)
    else:
        state = _load_checkpoint(state_path)

        if state.get("variant_labels") is None:
            # No prior checkpoint (or a checkpoint from an older script
            # version without variant_labels): freeze it now.
            state["variant_labels"] = all_variant_labels
        elif state["variant_labels"] != all_variant_labels:
            # The significant-variant set in the DB changed since the
            # checkpoint was written: continuing would produce a CSV with
            # inconsistent columns across generations. Stop explicitly
            # instead of silently corrupting the output.
            old_n = len(state["variant_labels"])
            new_n = len(all_variant_labels)
            raise RuntimeError(
                f"The significant-variant set in the DB has changed relative to the "
                f"existing checkpoint ({old_n} variants in the checkpoint, {new_n} now). "
                f"Rerun with force=True to regenerate '{out_path}' from scratch, or "
                f"restore the DB to its previous state if the change was unintended."
            )

        # Even if the checkpoint is consistent with the DB, verify that the
        # header actually written to disk matches: protects against
        # manually-edited CSVs or abnormally interrupted runs (e.g. a crash
        # during the CSV write, before the state was saved).
        existing_header = _read_existing_header(out_path)
        expected_header = ["id", "generation"] + all_variant_labels
        if existing_header is not None and existing_header != expected_header:
            raise RuntimeError(
                f"The header of '{out_path}' ({len(existing_header)} columns) doesn't "
                f"match the expected variants ({len(expected_header)} columns). "
                f"The file may have been corrupted by an earlier inconsistent run. "
                f"Rerun with force=True to regenerate it from scratch."
            )

    header_written = os.path.exists(out_path) and os.path.getsize(out_path) > 0

    for gen in (1, 2, 3):
        if gen in state["completed_generations"]:
            log.info("Generation %d already completed (checkpoint), skipping.", gen)
            continue

        df_bin, not_found = extract_generation(cfg, gen, variants_df)
        if not_found:
            log.warning("Generation %d: %d/%d variants not found: %s",
                        gen, len(not_found), len(all_variant_labels), not_found)

        if df_bin.empty:
            log.warning("Generation %d: no data produced, not added to checkpoint (can be retried).", gen)
            continue

        df_bin = df_bin.reindex(columns=all_variant_labels)
        df_bin.index = [f"gen{gen}_{sample_id}" for sample_id in df_bin.index]
        df_bin.index.name = "id"
        df_bin.insert(0, "generation", gen)

        # --- INCREMENTAL WRITE: append immediately, not at the end of the script ---
        df_bin.to_csv(out_path, mode="a", header=not header_written)
        header_written = True
        log.info("Generation %d appended to %s (%d patients)", gen, out_path, len(df_bin))

        state["completed_generations"].append(gen)
        _save_checkpoint(state_path, state)

    log.info("Significant variant extraction complete. Output: %s", out_path)
    return out_path


if __name__ == "__main__":
    run_extract_significant_matrices()