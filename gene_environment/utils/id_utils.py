"""
Funzioni condivise per la gestione degli ID campione e dei nomi variante.
"""
from __future__ import annotations

import re

_DUP_ID_RE = re.compile(r"^(.+)_\1$")


def clean_sample_id(raw_id: str) -> str:
    """
    Normalizza un id campione:
      - rimuove un eventuale prefisso "genN_" (N=1,2,3)
      - se l'id è nella forma "XXX_XXX" (stessa stringa ripetuta, separata da
        underscore) lo riduce a "XXX". Questo pattern è stato osservato nei
        dati sorgente (es. "RES02977_RES02977") ed è l'unica ragione per cui
        prima si "duplicava" artificialmente l'id nel loader ambientale: qui
        lo normalizziamo una volta sola, alla fonte, invece di duplicare
        l'id altrove per farlo combaciare.
    """
    if raw_id is None:
        return raw_id
    val = raw_id
    m = re.match(r"^gen[123]_(.+)$", val)
    if m:
        val = m.group(1)
    m2 = _DUP_ID_RE.match(val)
    if m2:
        val = m2.group(1)
    return val


def build_variant_label(chromosome: str, position, mutation: str) -> str:
    return f"{chromosome}_{position}_{mutation}"


def parse_variant_label(variant_label: str) -> tuple[str | None, int | None, str | None]:
    """
    Split robusto di un label variante nel formato "CHROM_POS_MUTATION"
    (dove MUTATION può a sua volta contenere underscore, es. "A_G").
    Usa split(max=2) per non troncare la mutazione, come già faceva il
    codice originale in vari punti (ora unificato qui).
    """
    parts = variant_label.split("_", 2)
    chrom = parts[0] if len(parts) > 0 else None
    pos = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
    mutation = parts[2] if len(parts) > 2 else None
    return chrom, pos, mutation
