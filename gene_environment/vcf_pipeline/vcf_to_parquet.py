"""
Converte i VCF filtrati in un'unica matrice genotipica (parquet), sostituendo
la catena originale: vcf_to_csv.py -> create_chr_csv.py -> create_full_csv.py
-> csv_to_parquet.py (4 script, 3 formati CSV intermedi enormi su disco).
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

import numpy as np
import pandas as pd

from gene_environment.config import get_config
from gene_environment.logging_utils import configure_logging, get_logger

log = get_logger(__name__)

CHROMOSOMES = [str(i) for i in range(1, 23)]


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


def _process_single_vcf_worker(args) -> str:
    vcf_path, out_parquet, log_dir = args
    configure_logging(log_dir)
    if os.path.exists(out_parquet):
        log.info("Skip (già convertito): %s", out_parquet)
        return out_parquet

    log.info("Converto VCF -> parquet: %s", vcf_path)
    df = vcf_file_to_dosage_df(vcf_path)
    df.to_parquet(out_parquet, engine="pyarrow", compression="zstd")
    log.info("Scritto %s (%d campioni, %d varianti)", out_parquet, df.shape[0], df.shape[1])
    return out_parquet


def convert_filtered_vcfs_to_parquet() -> list[str]:
    """Step 1: ogni *_filtered.vcf -> un parquet grezzo (dosaggi 0/1/2/-1)."""
    cfg = get_config()
    jobs = []
    for folder in cfg.vcf_folders:
        vcf_filtered_dir = os.path.join(folder, "vcf_filtered")
        for vcf_path in glob(os.path.join(vcf_filtered_dir, "*_filtered.vcf")):
            out_parquet = vcf_path + ".raw.parquet"
            jobs.append((vcf_path, out_parquet, cfg.log_dir))

    log.info("Conversione VCF filtrati -> parquet grezzo: %d file, %d worker", len(jobs), cfg.max_workers)
    out_paths = []
    with ProcessPoolExecutor(max_workers=cfg.max_workers) as ex:
        futures = [ex.submit(_process_single_vcf_worker, job) for job in jobs]
        for fut in as_completed(futures):
            out_paths.append(fut.result())
    return out_paths


def merge_chromosome(chrom: str, raw_parquet_paths: list[str], out_folder: str, null_percentage: float,
                      missing_strategy: str = "zero") -> str | None:
    """Step 2: unisce (per id, non per posizione di riga) tutti i parquet
    grezzi relativi allo stesso cromosoma, filtra per missing rate e
    binarizza. Ritorna il path del parquet di cromosoma, o None se non
    c'era nulla da fare."""
    chrom_files = [p for p in raw_parquet_paths if f"chr{chrom}." in os.path.basename(p) or f"chr{chrom}_" in os.path.basename(p)]
    if not chrom_files:
        log.warning("Nessun file trovato per chr%s", chrom)
        return None

    dfs = [pd.read_parquet(p) for p in chrom_files]
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
    out_path = os.path.join(out_folder, f"chr{chrom}_merged.parquet")
    merged.astype("Int8" if missing_strategy == "nan" else np.int8).to_parquet(out_path, engine="pyarrow", compression="zstd")
    log.info("chr%s: salvato %s (%d campioni, %d varianti)", chrom, out_path, *merged.shape)
    return out_path


def build_full_genome_parquet(chrom_parquet_paths: list[str], out_path: str) -> None:
    """Step 3: merge finale genoma intero, per ID (sostituisce
    create_full_csv.py). pd.concat(axis=1) allinea automaticamente per
    indice (id campione), indipendentemente dall'ordine delle righe nei
    singoli file di cromosoma."""
    log.info("Merge finale di %d file di cromosoma per id campione", len(chrom_parquet_paths))
    frames = [pd.read_parquet(p) for p in chrom_parquet_paths]
    full = pd.concat(frames, axis=1, join="outer")
    full.index.name = "id"
    full.to_parquet(out_path, engine="pyarrow", compression="zstd")
    log.info("Genoma completo salvato in %s (%d campioni, %d varianti)", out_path, *full.shape)


def run_vcf_to_parquet_pipeline(missing_strategy: str = "zero") -> str:
    cfg = get_config()
    configure_logging(cfg.log_dir)

    raw_paths = convert_filtered_vcfs_to_parquet()

    chrom_paths = []
    for chrom in CHROMOSOMES:
        p = merge_chromosome(chrom, raw_paths, cfg.output_folder, cfg.null_percentage, missing_strategy)
        if p:
            chrom_paths.append(p)

    out_path = os.path.join(cfg.output_folder, "gen.parquet")
    build_full_genome_parquet(chrom_paths, out_path)
    return out_path


if __name__ == "__main__":
    run_vcf_to_parquet_pipeline()
