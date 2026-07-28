"""
db_utils.py

Shared helpers for report modules that pull data from MySQL stored
routines: calling a stored routine and normalizing its resultset into a
DataFrame, and parsing/sorting/slugifying the `variant` -> chromosome
convention used across Table 2 and Table 2b.
"""

from __future__ import annotations

import re
import sys
from typing import List, Optional

import pandas as pd


def call_stored_routine_to_df(
    astore_name: str, get_connection, cursor_scope, columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """Call the stored routine `astore_name()` and return a DataFrame built
    from the first resultset. Handles drivers that require iterating
    nextset() to reach the actual data. Returns an empty DataFrame (with
    `columns` if given) if the routine returns no rows."""
    rows: List[dict] = []

    with get_connection() as conn:
        with cursor_scope(conn, dictionary=True) as cur:
            try:
                cur.execute(f"CALL {astore_name}()")
            except Exception as e:
                raise RuntimeError(f"Failed to CALL {astore_name}(). DB error: {e}") from e

            try:
                fetched = cur.fetchall()
                if fetched:
                    rows = fetched
            except Exception as e:
                print(f"[warn] initial fetchall() failed, will try nextset(): {e}", file=sys.stderr)
                rows = []

            try:
                while (not rows) and cur.nextset():
                    try:
                        fetched = cur.fetchall()
                        if fetched:
                            rows = fetched
                            break
                    except Exception as e:
                        print(f"[warn] fetchall() on a later resultset failed: {e}", file=sys.stderr)
                        continue
            except Exception as e:
                print(f"[warn] nextset() not supported by this driver: {e}", file=sys.stderr)

    if not rows:
        return pd.DataFrame(columns=columns or [])

    df = pd.DataFrame(rows)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    return df


def extract_chromosome(variant) -> str:
    """
    Extract chromosome from variant strings like:
      - chr1:12345_A/T
      - 1:12345_A/T
      - chrX:...
      - X:...
    """
    if pd.isna(variant):
        return "NA"
    if not isinstance(variant, str):
        variant = str(variant)
    m = re.match(r"^(?:chr)?([^:]+):", variant, flags=re.IGNORECASE)
    chrom = m.group(1) if m else re.split(r"[:_\-]", variant)[0]
    chrom = chrom.lower().lstrip("chr")
    if chrom.isalpha():
        chrom = chrom.upper()
    return chrom


def normalize_chrom_label(chrom) -> str:
    """Same normalization as `extract_chromosome`'s tail, applied to a raw
    DB chromosome value so DB counts and variant-string-derived
    chromosomes are guaranteed to line up (e.g. 'chr1' / '1' both become
    '1')."""
    if chrom is None:
        return "NA"
    c = str(chrom).lower().lstrip("chr")
    if c.isalpha():
        c = c.upper()
    return c


def chrom_sort_key(ch):
    """Sort key that orders numeric chromosomes (1, 2, ... 22) before
    non-numeric ones (X, Y, MT, NA...), each group alphabetically/numerically."""
    try:
        return (0, int(ch))
    except ValueError:
        return (1, ch)


def slugify(text) -> str:
    """Turn a label (e.g. a translated exposure name) into a
    filesystem-safe filename fragment."""
    s = re.sub(r"[^0-9A-Za-z_\-]+", "_", str(text)).strip("_")
    return s or "item"
