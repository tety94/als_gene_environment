"""
exposure_labels.py

Single source of truth for translating the environmental "exposure"
category from the source dataset's Italian land-use terms into the
English labels used in every report (Word tables, figures, CSV
exports).

Every report module that displays an `exposure` value -- as a table
column, a chart title, or a filename fragment -- should route it
through `translate_exposure` (DataFrame column) or
`translate_exposure_value` (single value) rather than keeping a local
copy of this mapping.
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

# Italian land-use category (as stored in the source data / DB) -> English label.
EXPOSURE_LABELS = {
    "seminativi_1500": "Arable land 1500mt",
    "vigneti_1500": "Vineyards 1500mt",
    "risaie_1500": "Rice fields 1500mt",
    "seminativi_1000": "Arable land 1000mt",
    "vigneti_1000": "Vineyards 1000mt",
    "risaie_1000": "Rice fields 1000mt",
}


def translate_exposure_value(value):
    """Translate a single exposure value. Values with no mapping are
    returned unchanged (not silently dropped)."""
    return EXPOSURE_LABELS.get(value, value)


def translate_exposure(df: pd.DataFrame, column: str = "exposure") -> pd.DataFrame:
    """Return a copy of `df` with `column` mapped from Italian land-use
    terms to English. Values not found in EXPOSURE_LABELS are left
    unchanged, with a warning so unmapped categories don't silently slip
    into the paper. No-op if `column` isn't present."""
    if column not in df.columns:
        return df
    df = df.copy()
    unmapped = sorted(set(df[column].dropna()) - set(EXPOSURE_LABELS))
    if unmapped:
        log.warning("Exposure values with no English mapping (left as-is): %s", unmapped)
    df[column] = df[column].map(translate_exposure_value)
    return df
