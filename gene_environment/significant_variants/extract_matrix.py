#!/usr/bin/env python3
"""
Estrae, per le varianti significative, il genotipo binario (0/1) per ciascun
paziente in gen1/gen2/gen3, direttamente dai VCF indicizzati (bcftools).
(ex extract_significant_variant_matrices.py — supera anche
get_variants_after_gen1.py, che faceva la stessa cosa scansionando i VCF
riga per riga senza usare l'indice .tbi: molto più lento, mantenuto solo
come riferimento storico e non incluso in questo refactor).

NOVITÀ rispetto all'originale (come richiesto):
  1) SCRITTURA INCREMENTALE: il CSV combinato viene scritto/appesto SUBITO
     dopo ogni generazione completata (non solo tutto insieme alla fine).
     Se lo script si interrompe a metà (es. gen3 fallisce), gen1 e gen2 sono
     già su disco.
  2) CHECKPOINT DI RIPRESA: un file di stato JSON accanto al CSV tiene
     traccia delle generazioni già completate; un rilancio salta quelle già
     fatte invece di ripartire da zero.
  3) Parallelizzazione per cromosoma dentro ogni generazione
     (ProcessPoolExecutor): le query bcftools per cromosomi diversi sono
     indipendenti.
  4) Config (path VCF, cartella output) spostata in gene_environment.config
     invece di hardcoded in testa allo script.

FIX (rispetto alla versione precedente):
  Il checkpoint teneva traccia solo delle generazioni completate, ma NON
  congelava l'elenco delle varianti significative usato per costruire le
  colonne del CSV. Se tra un run e l'altro il DB cambiava (es. modeling.py
  rieseguito, nuove varianti significative aggiunte per un'altra exposure)
  e per qualche motivo il checkpoint non era sincronizzato con lo stato del
  file su disco (es. crash tra la scrittura del CSV e il salvataggio dello
  stato), un rilancio poteva riscrivere/appendere una generazione con un
  numero di colonne diverso da quello già su disco, producendo un CSV con
  righe di lunghezza diversa (illeggibile da pandas: "Expected N fields,
  saw M"). Ora:
    - la lista di varianti viene congelata nel checkpoint alla prima
      esecuzione (o alla prima con force=True);
    - un rilancio senza force=True verifica che il set di varianti nel DB
      non sia cambiato rispetto al checkpoint: se è cambiato, si ferma con
      un errore esplicito invece di produrre un CSV incoerente;
    - prima di ogni append, si valida che l'header già presente su disco
      corrisponda esattamente alle colonne attese per quel run.
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
    """Estrae una singola combinazione (generazione, cromosoma). Eseguito in
    processo separato -> ritorna dati serializzabili (dict), non DataFrame."""
    generation, chrom, group_records, vcf_path, log_dir = args
    configure_logging(log_dir)

    if not os.path.exists(vcf_path):
        log.warning("VCF non trovato per generazione %d, chr%s: %s", generation, chrom, vcf_path)
        return chrom, {}, [r["label"] for r in group_records]

    chrom_name = resolve_chrom_name(str(chrom), vcf_path)
    samples = get_samples(vcf_path)

    positions = [r["pos"] for r in group_records]
    t0 = time.perf_counter()
    rows = query_positions(vcf_path, chrom_name, positions)
    log.info("gen%d chr%s: %d varianti richieste, %d righe trovate (%.2fs)",
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
                log.warning("[MISMATCH] %s: REF/ALT non corrispondono (atteso %s>%s)", rec["label"], rec["ref"], rec["alt"])
            not_found.append(rec["label"])
            continue
        dosages_by_label[rec["label"]] = dict(zip(samples, dosages))

    return chrom, dosages_by_label, not_found


def extract_generation(cfg, generation: int, variants_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    t_start = time.perf_counter()
    log.info("=== Generazione %d: inizio ===", generation)

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
        log.error("Generazione %d: nessuna variante estratta.", generation)
        return pd.DataFrame(), labels_not_found

    df = pd.DataFrame(all_dosages)
    df_bin = binarize_vectorized(df)

    log.info("=== Generazione %d: completata in %.1fs (%d pazienti, %d varianti trovate) ===",
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

    log.info("Recupero varianti significative dal DB%s", f" (exposure={exposure})" if exposure else "")
    sig = get_significant_results()
    if sig.empty:
        log.info("Nessuna variante significativa trovata. Esco.")
        return None

    sig["ref"] = sig["mutation"].apply(lambda m: m.split("_", 1)[0])
    sig["alt"] = sig["mutation"].apply(lambda m: m.split("_", 1)[1])
    sig["chrom"] = sig["chromosome"].astype(str)
    sig["pos"] = sig["position"].astype(int)
    sig["label"] = sig.apply(lambda r: build_variant_label(r["chromosome"], r["position"], r["mutation"]), axis=1)
    variants_df = sig[["chrom", "pos", "ref", "alt", "label"]].drop_duplicates(subset="label")
    all_variant_labels = sorted(variants_df["label"].unique())
    log.info("%d varianti uniche da estrarre.", len(variants_df))

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
            # Nessun checkpoint precedente (o checkpoint da versione vecchia
            # dello script, senza variant_labels): lo congeliamo ora.
            state["variant_labels"] = all_variant_labels
        elif state["variant_labels"] != all_variant_labels:
            # Il set di varianti significative nel DB è cambiato rispetto a
            # quando è stato scritto il checkpoint: continuare produrrebbe
            # un CSV con colonne incoerenti tra generazioni. Ci si ferma
            # esplicitamente invece di corrompere silenziosamente l'output.
            old_n = len(state["variant_labels"])
            new_n = len(all_variant_labels)
            raise RuntimeError(
                f"Il set di varianti significative nel DB e' cambiato rispetto al "
                f"checkpoint esistente ({old_n} varianti nel checkpoint, {new_n} ora). "
                f"Rilancia con force=True per rigenerare '{out_path}' da zero, oppure "
                f"ripristina il DB allo stato precedente se il cambiamento non era voluto."
            )

        # Anche se il checkpoint e' coerente con il DB, verifichiamo che
        # l'header effettivamente scritto su disco corrisponda: protegge da
        # CSV toccati a mano o da run interrotti in modo anomalo (es. crash
        # durante la scrittura del CSV, prima del salvataggio dello stato).
        existing_header = _read_existing_header(out_path)
        expected_header = ["id", "generation"] + all_variant_labels
        if existing_header is not None and existing_header != expected_header:
            raise RuntimeError(
                f"L'header di '{out_path}' ({len(existing_header)} colonne) non "
                f"corrisponde alle varianti attese ({len(expected_header)} colonne). "
                f"Il file potrebbe essere corrotto da un run precedente incoerente. "
                f"Rilancia con force=True per rigenerarlo da zero."
            )

    header_written = os.path.exists(out_path) and os.path.getsize(out_path) > 0

    for gen in (1, 2, 3):
        if gen in state["completed_generations"]:
            log.info("Generazione %d già completata (checkpoint), salto.", gen)
            continue

        df_bin, not_found = extract_generation(cfg, gen, variants_df)
        if not_found:
            log.warning("Generazione %d: %d/%d varianti non trovate: %s",
                        gen, len(not_found), len(all_variant_labels), not_found)

        if df_bin.empty:
            log.warning("Generazione %d: nessun dato prodotto, non aggiunta al checkpoint (si può ritentare).", gen)
            continue

        df_bin = df_bin.reindex(columns=all_variant_labels)
        df_bin.index = [f"gen{gen}_{sample_id}" for sample_id in df_bin.index]
        df_bin.index.name = "id"
        df_bin.insert(0, "generation", gen)

        # --- SCRITTURA INCREMENTALE: append subito, non a fine script ---
        df_bin.to_csv(out_path, mode="a", header=not header_written)
        header_written = True
        log.info("Generazione %d appesa a %s (%d pazienti)", gen, out_path, len(df_bin))

        state["completed_generations"].append(gen)
        _save_checkpoint(state_path, state)

    log.info("Estrazione varianti significative completata. Output: %s", out_path)
    return out_path


if __name__ == "__main__":
    run_extract_significant_matrices()