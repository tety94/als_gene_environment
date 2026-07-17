#!/usr/bin/env python3
"""
build_cohort_mapping.py

Se gen.parquet continua a dare problemi (thrift size limit / file corrotto),
questa alternativa ricostruisce la mappatura id -> generazione leggendo
direttamente l'header dei VCF filtrati (i sample id sono le colonne dopo
#CHROM POS ID REF ALT QUAL FILTER INFO FORMAT).

Basta UN file per generazione per ottenere la lista completa dei sample id
di quella coorte (le colonne sample sono le stesse su tutti i cromosomi).
Uso bcftools se disponibile (legge solo l'header, è istantaneo anche su
file da GB), altrimenti fallback su zcat+parsing.

Output: output/table1/id_generation_mapping.csv  (colonne: id, generazione)

Poi in generate_table1.py imposta:
    COHORT_SOURCE = "csv"
    COHORT_MAPPING_CSV = "output/table1/id_generation_mapping.csv"
"""

import shutil
import subprocess
import sys
import gzip
from pathlib import Path

import pandas as pd

# ============================================================
# CONFIG — modifica qui
# ============================================================

# Per ogni generazione: directory con i VCF filtrati, e QUALE file
# rappresentativo usare per leggere l'header (basta un cromosoma,
# uso quello "_filtered.vcf.gz" NON "_selected_filtered", che dovrebbe
# contenere il set campione completo di quella generazione).
GENERATION_VCF_DIRS = {
    "gen1": "/mnt/cresla_prod/genome_datasets/gen1/vcf_filtered",
    "gen2": "/mnt/cresla_prod/genome_datasets/gen2/vcf_filtered",
}

# Pattern del file rappresentativo da usare per estrarre i sample id
# (viene cercato dentro GENERATION_VCF_DIRS[generazione])
REPRESENTATIVE_CHR_PATTERN = "*_vcf_chr1_filtered.vcf.gz"  # evita '*_selected_*'

OUTPUT_CSV = Path("output/table1/id_generation_mapping.csv")

# ============================================================


def find_representative_file(vcf_dir):
    vcf_dir = Path(vcf_dir)
    if not vcf_dir.exists():
        sys.exit(f"ERRORE: directory non trovata: {vcf_dir}")

    candidates = sorted(
        p for p in vcf_dir.glob(REPRESENTATIVE_CHR_PATTERN)
        if "_selected_" not in p.name
    )
    if not candidates:
        # fallback: qualsiasi *_filtered.vcf.gz non selected, primo trovato
        candidates = sorted(
            p for p in vcf_dir.glob("*_filtered.vcf.gz")
            if "_selected_" not in p.name and not p.name.endswith(".raw.parquet")
        )
    if not candidates:
        sys.exit(f"ERRORE: nessun VCF filtrato trovato in {vcf_dir}")

    return candidates[0]


def extract_sample_ids_bcftools(vcf_path):
    result = subprocess.run(
        ["bcftools", "query", "-l", str(vcf_path)],
        capture_output=True, text=True, check=True,
    )
    ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return ids


def extract_sample_ids_manual(vcf_path):
    """Fallback senza bcftools: legge riga per riga finché non trova #CHROM."""
    with gzip.open(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#CHROM"):
                cols = line.rstrip("\n").split("\t")
                # colonne fisse VCF: CHROM POS ID REF ALT QUAL FILTER INFO FORMAT, poi i sample
                fixed = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]
                return cols[len(fixed):]
            if not line.startswith("##") and not line.startswith("#CHROM"):
                # header terminato senza trovare #CHROM: file anomalo
                break
    sys.exit(f"ERRORE: riga #CHROM non trovata nell'header di {vcf_path}")


def extract_sample_ids(vcf_path):
    if shutil.which("bcftools"):
        try:
            return extract_sample_ids_bcftools(vcf_path)
        except subprocess.CalledProcessError as e:
            print(f"  bcftools ha fallito ({e}), provo parsing manuale...")
    return extract_sample_ids_manual(vcf_path)


def main():
    rows = []
    for generazione, vcf_dir in GENERATION_VCF_DIRS.items():
        rep_file = find_representative_file(vcf_dir)
        print(f"[{generazione}] uso file rappresentativo: {rep_file}")
        sample_ids = extract_sample_ids(rep_file)
        print(f"[{generazione}] trovati {len(sample_ids)} sample id")
        for sid in sample_ids:
            rows.append({"id": sid, "generazione": generazione})

    if not rows:
        sys.exit("ERRORE: nessun sample id estratto da nessuna generazione.")

    mapping = pd.DataFrame(rows)

    # controllo duplicati id tra generazioni diverse (non dovrebbe succedere)
    mapping["id"] = mapping["id"].str.split("_").str[0]
    dup = mapping[mapping.duplicated("id", keep=False)]
    if not dup.empty:
        print(f"ATTENZIONE: {dup['id'].nunique()} id compaiono in più di una generazione:")
        print(dup.sort_values("id").to_string(index=False))

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(OUTPUT_CSV, index=False)
    print(f"\nMappatura salvata in: {OUTPUT_CSV.resolve()}")
    print(mapping["generazione"].value_counts())


if __name__ == "__main__":
    main()