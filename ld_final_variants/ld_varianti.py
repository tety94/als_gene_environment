#!/usr/bin/env python3
"""
Calcola l'LD (r2) fra tutte le coppie di varianti DISTINTE di variants_input.csv,
sullo stesso cromosoma, separatamente in ciascun dataset genetico (gen1, gen2),
e scrive un unico CSV con una riga per coppia e le colonne r2 per generazione.

Requisiti: bcftools nel PATH (VCF indicizzati con tabix), numpy, pandas.
(plink2 non serve piu': r2 = correlazione di Pearson fra dosaggi dell'allele
ALT, sulle sole coppie di campioni con genotipo non mancante in entrambe.)

Uso:
    python3 ld_varianti.py --input variants_input.csv --output ld_pairs.csv

Per cambiare i VCF:
    --gen gen1=/path/gen1_onlycases_vcf_chr{chrom}.vcf.gz
    --gen gen2=/path/gen2_vcf_chr{chrom}.vcf.gz
(si puo' ripetere --gen per aggiungere altre generazioni)

Input: colonne usate = variant (formato chrom_pos_ref_alt), chromosome,
position, exposure, gene. Le righe ripetute (diverse exposure/generation/test)
vengono collassate sulla variante distinta.

Output 1 (--output), una riga per coppia:
    chromosome, variant_1, variant_2, pos_1, pos_2, distance_bp,
    gene_1, gene_2, exposures_1, exposures_2, shared_exposures,
    r2_<gen>..., n_<gen>..., r2_min, r2_max
  (r2_min/r2_max sono calcolati sulle generazioni dove r2 e' disponibile)

Output 2 (--output-variants), una riga per variante:
    variant, chromosome, position, gene, exposures, found_<gen>...,
    n_partner_<soglia>_tutte_gen, max_r2_min, partner_max_r2_min
"""

import argparse
import itertools
import subprocess
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

DEFAULT_GENS = [
    "gen1=/mnt/cresla_prod/genome_datasets/gen1/gen1_onlycases_vcf_chr{chrom}.vcf.gz",
    "gen2=/mnt/cresla_prod/genome_datasets/gen2/gen2_vcf_chr{chrom}.vcf.gz",
]


def load_distinct_variants(path):
    d = pd.read_csv(path, dtype=str)
    d["gene"] = d["gene"].fillna("") if "gene" in d else ""
    parts = d["variant"].str.split("_", n=3, expand=True)
    d["ref"], d["alt"] = parts[2], parts[3]
    g = d.groupby("variant", sort=False)
    out = g.agg(
        chromosome=("chromosome", "first"),
        position=("position", "first"),
        ref=("ref", "first"),
        alt=("alt", "first"),
        gene=("gene", "first"),
        exposures=("exposure", lambda s: ";".join(sorted(set(s)))),
    ).reset_index()
    out["position"] = out["position"].astype(int)
    return out


def query_records(vcf_path, regions):
    """Genera (chrom, pos, ref, alt, [GT...]) per le posizioni richieste.
    regions: lista di (chrom, pos). Prova con e senza prefisso 'chr'."""
    for prefix in ("", "chr"):
        reg_file = f"/tmp/_ld_regions_{abs(hash(vcf_path))}.tsv"
        with open(reg_file, "w") as f:
            for c, p in regions:
                f.write(f"{prefix}{c}\t{p}\n")
        cmd = ["bcftools", "query", "-R", reg_file,
               "-f", "%CHROM\t%POS\t%REF\t%ALT[\t%GT]\n", vcf_path]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"  bcftools errore ({prefix or 'senza chr'}): {res.stderr.strip()[:200]}",
                  file=sys.stderr)
            continue
        lines = [l for l in res.stdout.split("\n") if l]
        if lines:
            for l in lines:
                f = l.split("\t")
                yield f[0].replace("chr", "", 1), f[1], f[2], f[3], f[4:]
            return


def gt_to_dosage(gt, alt_idx):
    """Numero di copie dell'allele alt_idx (1-based); NaN se manca un allele."""
    alleles = gt.replace("|", "/").split("/")
    if "." in alleles:
        return np.nan
    return float(sum(1 for a in alleles if a == str(alt_idx)))


def read_dosages(vcf_path, variants):
    """variants: DataFrame di un cromosoma. Ritorna {variant: array dosaggi}."""
    wanted = {(r.chromosome, str(r.position), r.ref, r.alt): r.variant
              for r in variants.itertuples()}
    regions = sorted({(r.chromosome, r.position) for r in variants.itertuples()})
    dos = {}
    for chrom, pos, ref, alts, gts in query_records(vcf_path, regions):
        for i, a in enumerate(alts.split(","), start=1):
            key = (chrom, pos, ref, a)
            if key in wanted and wanted[key] not in dos:
                dos[wanted[key]] = np.array([gt_to_dosage(g, i) for g in gts])
    return dos


def r2_pair(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    n = int(m.sum())
    if n < 3:
        return np.nan, n
    xs, ys = x[m], y[m]
    if xs.std() == 0 or ys.std() == 0:
        return np.nan, n
    return float(np.corrcoef(xs, ys)[0, 1] ** 2), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="variants_input.csv")
    ap.add_argument("--output", default="ld_pairs.csv")
    ap.add_argument("--output-variants", default="ld_varianti.csv")
    ap.add_argument("--gen", action="append",
                    help="nome=pattern_vcf con {chrom}; ripetibile")
    ap.add_argument("--soglia", type=float, default=0.8,
                    help="soglia r2 usata solo per il conteggio partner nel file varianti")
    args = ap.parse_args()

    gens = []
    for s in (args.gen or DEFAULT_GENS):
        name, pattern = s.split("=", 1)
        gens.append((name, pattern))

    var = load_distinct_variants(args.input)
    print(f"{len(var)} varianti distinte su {var['chromosome'].nunique()} cromosomi")
    info = var.set_index("variant")

    # (gen, variant_1, variant_2) -> (r2, n)
    res = {}
    found = {g: set() for g, _ in gens}

    for gname, pattern in gens:
        for chrom, sub in var.groupby("chromosome", sort=False):
            if len(sub) < 2:
                continue
            path = pattern.format(chrom=chrom)
            print(f"[{gname} chr{chrom}] {len(sub)} varianti: {path}")
            dos = read_dosages(path, sub)
            found[gname].update(dos)
            miss = [v for v in sub["variant"] if v not in dos]
            if miss:
                print(f"  non trovate ({len(miss)}): {', '.join(miss)} "
                      f"(posizione assente, REF/ALT diversi o build diversa)")
            for a, b in itertools.combinations(sorted(dos), 2):
                res[(gname, a, b)] = r2_pair(dos[a], dos[b])

    rows = []
    for chrom, sub in var.groupby("chromosome", sort=False):
        for a, b in itertools.combinations(sorted(sub["variant"]), 2):
            ia, ib = info.loc[a], info.loc[b]
            row = {
                "chromosome": chrom, "variant_1": a, "variant_2": b,
                "pos_1": ia.position, "pos_2": ib.position,
                "distance_bp": abs(ia.position - ib.position),
                "gene_1": ia.gene, "gene_2": ib.gene,
                "exposures_1": ia.exposures, "exposures_2": ib.exposures,
                "shared_exposures": ";".join(sorted(
                    set(ia.exposures.split(";")) & set(ib.exposures.split(";")))),
            }
            vals = []
            for gname, _ in gens:
                r2, n = res.get((gname, a, b), (np.nan, 0))
                row[f"r2_{gname}"] = r2
                row[f"n_{gname}"] = n
                if not np.isnan(r2):
                    vals.append(r2)
            row["r2_min"] = min(vals) if vals else np.nan
            row["r2_max"] = max(vals) if vals else np.nan
            rows.append(row)

    pairs = pd.DataFrame(rows)
    pairs.to_csv(args.output, index=False)
    print(f"\n{len(pairs)} coppie scritte in {args.output}")

    # riepilogo per variante
    vrows = []
    for v in var.itertuples():
        mine = pairs[(pairs.variant_1 == v.variant) | (pairs.variant_2 == v.variant)].copy()
        mine["partner"] = np.where(mine.variant_1 == v.variant, mine.variant_2, mine.variant_1)
        row = {"variant": v.variant, "chromosome": v.chromosome, "position": v.position,
               "gene": v.gene, "exposures": v.exposures}
        for gname, _ in gens:
            row[f"found_{gname}"] = v.variant in found[gname]
        # forte in TUTTE le generazioni = r2_min >= soglia (richiede r2 in tutte)
        ok = mine.dropna(subset=[f"r2_{g}" for g, _ in gens])
        row[f"n_partner_r2min_ge_{args.soglia}"] = int((ok["r2_min"] >= args.soglia).sum())
        if len(ok):
            best = ok.loc[ok["r2_min"].idxmax()]
            row["max_r2_min"], row["partner_max_r2_min"] = best["r2_min"], best["partner"]
        else:
            row["max_r2_min"], row["partner_max_r2_min"] = np.nan, ""
        vrows.append(row)
    pd.DataFrame(vrows).to_csv(args.output_variants, index=False)
    print(f"Riepilogo per variante in {args.output_variants}")


if __name__ == "__main__":
    main()
