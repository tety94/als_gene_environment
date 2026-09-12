"""Shared helpers for normalizing sample ids and building/parsing variant labels."""
from __future__ import annotations

import re

_DUP_ID_RE = re.compile(r"^(.+)_\1$")


def clean_sample_id(raw_id: str) -> str:
    """Normalize a sample id:
      - strips an optional "genN_" prefix (N=1,2,3)
      - if the id has the form "XXX_XXX" (same string repeated, separated by
        an underscore), reduces it to "XXX". This pattern has been observed
        in the source data (e.g. "RES02977_RES02977")."""
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
    """Robustly split a variant label of the form "CHROM_POS_MUTATION"
    (where MUTATION may itself contain underscores, e.g. "A_G").
    Uses split(max=2) to avoid truncating the mutation."""
    parts = variant_label.split("_", 2)
    chrom = parts[0] if len(parts) > 0 else None
    pos = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
    mutation = parts[2] if len(parts) > 2 else None
    return chrom, pos, mutation
