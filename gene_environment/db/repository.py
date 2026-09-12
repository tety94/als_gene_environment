"""Data-access layer for the variant_results table and related tables (gene annotation, significant results)."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
import json

from gene_environment.config import get_config
from gene_environment.db.connection import get_connection, cursor_scope
from gene_environment.logging_utils import get_logger

log = get_logger(__name__)


def safe_val(x):
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, (np.floating,)) and np.isnan(x):
        return None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


# --------------------------------------------------------------------------
# Variant status / cooperative lock for parallel execution
# --------------------------------------------------------------------------

def variant_already_done(conn, variant: str) -> bool:
    with cursor_scope(conn) as cur:
        cur.execute("SELECT completed, in_progress FROM variant_results WHERE variant=%s", (variant,))
        r = cur.fetchone()
    if r is None:
        return False
    completed, in_progress = r
    return bool(completed) or bool(in_progress)


def mark_variant_in_progress(conn, variant: str) -> bool:
    with cursor_scope(conn) as cur:
        cur.execute(
            "UPDATE variant_results SET in_progress=1 WHERE variant=%s AND completed=0 AND in_progress=0",
            (variant,),
        )
        updated = cur.rowcount > 0
    conn.commit()
    return updated


def reset_variant_in_progress(conn, variant: str, success: bool = True) -> None:
    with cursor_scope(conn) as cur:
        if success:
            cur.execute(
                "UPDATE variant_results SET completed=1, in_progress=0 WHERE variant=%s", (variant,)
            )
        else:
            cur.execute("UPDATE variant_results SET in_progress=0 WHERE variant=%s", (variant,))
    conn.commit()


# --------------------------------------------------------------------------
# Inserting/updating results
# --------------------------------------------------------------------------

_UPDATE_SQL = """
UPDATE variant_results
SET
    gene = %(gene)s,
    chromosome = %(chromosome)s,
    position = %(position)s,
    mutation = %(mutation)s,
    mutati = %(mutati)s,
    non_mutati = %(non_mutati)s,
    obs_coef = %(obs_coef)s,
    mean_coef = %(mean_coef)s,
    sd_coef = %(sd_coef)s,
    empirical_p = %(empirical_p)s,
    iterations = %(iterations)s,
    balance = %(balance)s,
    onset_n_mutati = %(onset_n_mutati)s,
    onset_n_non_mutati = %(onset_n_non_mutati)s,
    onset_median_mutati = %(onset_median_mutati)s,
    onset_median_non_mutati = %(onset_median_non_mutati)s,
    onset_delta_median = %(onset_delta_median)s,
    onset_ci_low = %(onset_ci_low)s,
    onset_ci_high = %(onset_ci_high)s,
    onset_p_value = %(onset_p_value)s,
    onset_effect_size = %(onset_effect_size)s,
    onset_low_power = %(onset_low_power)s,
    onset_method = %(onset_method)s,
    full_model_json = %(full_model_json)s,
    completed = 1
WHERE variant = %(variant)s AND exposure = %(exposure)s AND generation = %(generation)s AND test = %(test)s
"""


def _row_to_params(res: dict, exposure: str, generation: int, test: str) -> dict:
    variant = res["variant"]
    chrom, pos, mutation = variant.split("_", 2) if variant.count("_") >= 2 else (None, None, None)
    onset = res.get("onset") or {}

    params = {
        "gene": None,
        "chromosome": chrom,
        "position": int(pos) if pos is not None else None,
        "mutation": mutation,
        "mutati": safe_val(res.get("n_treated")),
        "non_mutati": safe_val(res.get("n_control")),
        "obs_coef": safe_val(res.get("obs_coef")),
        "mean_coef": safe_val(res.get("perm_mean")),
        "sd_coef": safe_val(res.get("perm_std")),
        "empirical_p": safe_val(res.get("p_emp")),
        "iterations": safe_val(res.get("iterations")),
        "balance": safe_val(res.get("max_smd")),
        "onset_n_mutati": safe_val(onset.get("n_mutati")),
        "onset_n_non_mutati": safe_val(onset.get("n_non_mutati")),
        "onset_median_mutati": safe_val(onset.get("median_mutati")),
        "onset_median_non_mutati": safe_val(onset.get("median_non_mutati")),
        "onset_delta_median": safe_val(onset.get("delta_median")),
        "onset_ci_low": safe_val(onset.get("ci_low")),
        "onset_ci_high": safe_val(onset.get("ci_high")),
        "onset_p_value": safe_val(onset.get("p_value")),
        "onset_effect_size": safe_val(onset.get("effect_size")),
        "onset_low_power": safe_val(onset.get("low_power")),
        "onset_method": onset.get("method"),
        "variant": variant,
        "exposure": exposure,
        "generation": generation,
        "test": test,
    }

    full_model = res.get("full_model")
    params["full_model_json"] = json.dumps(full_model) if full_model is not None else None
    return params


def save_variant_result(conn, res: dict, exposure: str, generation: int, test: str) -> None:
    """Save the result for ONE variant (used for single flushes/fallback)."""
    params = _row_to_params(res, exposure, generation, test)
    with cursor_scope(conn) as cur:
        cur.execute(_UPDATE_SQL, params)


def save_variant_results_bulk(results: Iterable[dict], exposure: str, generation: int, test: str) -> int:
    """Save a list of variant results in a single transaction with
    executemany (much faster than N individual updates)."""
    rows = [_row_to_params(r, exposure, generation, test) for r in results]
    if not rows:
        return 0
    with get_connection() as conn:
        with cursor_scope(conn) as cur:
            cur.executemany(_UPDATE_SQL, rows)
            n = cur.rowcount
    log.info("Saved/updated %d variant results to DB (bulk)", len(rows))
    return n


def load_variant_results(exposure: str, iterations: int) -> pd.DataFrame:
    with get_connection() as conn:
        with cursor_scope(conn) as cur:
            cur.execute(
                "SELECT variant, obs_coef, empirical_p FROM variant_results "
                "WHERE exposure=%s AND completed=1 AND iterations=%s",
                (exposure, iterations),
            )
            rows = cur.fetchall()
    return pd.DataFrame(rows, columns=["variant", "obs_coef", "empirical_p"])


def delete_variants(conn, variant_list: list[str]) -> None:
    if not variant_list:
        return
    with cursor_scope(conn) as cur:
        fmt = ",".join(["%s"] * len(variant_list))
        cur.execute(f"DELETE FROM variant_results WHERE variant IN ({fmt})", tuple(variant_list))


def insert_new_variants(variants: list[dict], exposure: str, generation: int, test: str, chunk_size: int = 5000) -> int:
    """Insert new variants into variant_results, chunked to avoid packets
    that are too large for MySQL (max_allowed_packet)."""
    if not variants:
        return 0

    sql = """
        INSERT IGNORE INTO variant_results
            (variant, chromosome, position, mutation, exposure, generation, test)
        VALUES (%s,%s,%s,%s,%s,%s,%s)
    """
    total = 0
    with get_connection() as conn:
        with cursor_scope(conn) as cur:
            for i in range(0, len(variants), chunk_size):
                chunk = variants[i:i + chunk_size]
                data = []
                for v in chunk:
                    chrom = v.get("chromosome")
                    chrom = str(chrom) if chrom is not None else None
                    pos = v.get("position")
                    pos = int(pos) if pos is not None and str(pos).isdigit() else None
                    data.append((v["variant"], chrom, pos, v.get("mutation"), exposure, generation, test))
                cur.executemany(sql, data)
                total += cur.rowcount
    log.info("Inserted/ignored %d variants (exposure=%s, generation=%s, test=%s)", total, exposure, generation, test)
    return total


def get_variants_to_run(mapping: dict, variant_cols_safe: list[str], exposure: str, generation: int) -> list[str]:
    orig_to_safe = {v: k for k, v in mapping.items()}
    with get_connection() as conn:
        with cursor_scope(conn) as cur:
            cur.execute(
                "SELECT variant FROM variant_results "
                "WHERE (completed=1 OR in_progress=1) AND exposure=%s AND generation=%s",
                (exposure, generation),
            )
            done_variants = {row[0] for row in cur.fetchall()}

    done_safe = {orig_to_safe[v] for v in done_variants if v in orig_to_safe}
    to_run = [v for v in variant_cols_safe if v not in done_safe]
    log.info("Variants already done: %d, to process: %d", len(done_safe), len(to_run))
    return to_run


# --------------------------------------------------------------------------
# Genes / annotation
# --------------------------------------------------------------------------
def get_empty_variants_gene() -> pd.DataFrame:
    """Fetch significant variants from the get_significant_results stored
    procedure (no exposure filter) and keep only those that don't yet have
    a gene assigned in variant_results_significant."""
    sig_df = get_significant_results(exposure=None)
    if sig_df.empty:
        return pd.DataFrame(columns=["variant", "mutation", "position", "chromosome"])

    with get_connection() as conn:
        with cursor_scope(conn) as cur:
            cur.execute(
                "SELECT variant, exposure FROM variant_results_significant WHERE gene IS NOT NULL"
            )
            already_assigned = {(row[0], row[1]) for row in cur.fetchall()}

    empty_df = sig_df[
        ~sig_df.apply(lambda r: (r["variant"], r["exposure"]) in already_assigned, axis=1)
    ]

    log.info(f"{len(empty_df)} variants to compute")
    return empty_df[["variant", "mutation", "position", "chromosome"]].reset_index(drop=True)


def update_variant_gene(conn, variant: str, gene_id: str, gene_name: str) -> None:
    with cursor_scope(conn) as cur:
        cur.execute(
            "UPDATE variant_results_significant SET gene=%s, gene_name=%s WHERE variant=%s AND gene IS NULL",
            (gene_id, gene_name, variant),
        )


def get_genes_to_annotate() -> list[str]:
    with get_connection() as conn:
        with cursor_scope(conn) as cur:
            cur.execute("""
                SELECT DISTINCT vr.gene
                FROM variant_results_significant vr
                LEFT JOIN gene_neuro_annotation gna ON vr.gene = gna.gene_id
                WHERE vr.gene IS NOT NULL AND vr.gene != 'NO-GENE'
            """)
            return [row[0] for row in cur.fetchall()]

def upsert_gene_neuro_annotation(data: dict) -> None:
    sql = """
    INSERT INTO gene_neuro_annotation (
        gene_id, gene_symbol, gene_type, expressed_brain, brain_tissues,
        expressed_neurons, expressed_glia, cell_types, go_neuro_processes,
        go_toxic_response, ctd_chemicals, ctd_neuro_diseases,
        ctd_neuro_disease_direct, ctd_neuro_disease_pesticide_mediated,
        als_panelapp_confidence, als_opentargets_score,
        neuro_plausibility_score
    ) VALUES (
        %(gene_id)s, %(gene_symbol)s, %(gene_type)s, %(expressed_brain)s, %(brain_tissues)s,
        %(expressed_neurons)s, %(expressed_glia)s, %(cell_types)s, %(go_neuro_processes)s,
        %(go_toxic_response)s, %(ctd_chemicals)s, %(ctd_neuro_diseases)s,
        %(ctd_neuro_disease_direct)s, %(ctd_neuro_disease_pesticide_mediated)s,
        %(als_panelapp_confidence)s, %(als_opentargets_score)s,
        %(neuro_plausibility_score)s
    )
    ON DUPLICATE KEY UPDATE
        gene_symbol=VALUES(gene_symbol), gene_type=VALUES(gene_type),
        expressed_brain=VALUES(expressed_brain), brain_tissues=VALUES(brain_tissues),
        expressed_neurons=VALUES(expressed_neurons), expressed_glia=VALUES(expressed_glia),
        cell_types=VALUES(cell_types), go_neuro_processes=VALUES(go_neuro_processes),
        go_toxic_response=VALUES(go_toxic_response), ctd_chemicals=VALUES(ctd_chemicals),
        ctd_neuro_diseases=VALUES(ctd_neuro_diseases),
        ctd_neuro_disease_direct=VALUES(ctd_neuro_disease_direct),
        ctd_neuro_disease_pesticide_mediated=VALUES(ctd_neuro_disease_pesticide_mediated),
        als_panelapp_confidence=VALUES(als_panelapp_confidence),
        als_opentargets_score=VALUES(als_opentargets_score),
        neuro_plausibility_score=VALUES(neuro_plausibility_score),
        last_updated=CURRENT_TIMESTAMP
    """
    with get_connection() as conn:
        with cursor_scope(conn, dictionary=True) as cur:
            cur.execute(sql, data)


def get_gene_neuro_annotation(gene_id: str) -> dict | None:
    with get_connection() as conn:
        with cursor_scope(conn, dictionary=True) as cur:
            cur.execute("SELECT * FROM gene_neuro_annotation WHERE gene_id = %s", (gene_id,))
            return cur.fetchone()


def get_significant_results(exposure: str | None = None) -> pd.DataFrame:
    """Call the `get_significant_results()` stored procedure (cohort 1 and 2
    side by side). NB: the result set has duplicated column names, hence the
    positional cursor. The column count is validated so a shape change in
    the stored procedure is caught immediately instead of silently
    misaligning columns.

    `exposure`: if set, filters by the environmental component (requires the
    DB-side stored procedure to accept the IN p_exposure parameter). If
    None, no filter is applied."""
    expected_columns = [
        "exposure",
        "gene_name", "variant",
        "empirical_p_g1", "obs_coef_g1",
        "empirical_p_g2", "obs_coef_g2",
        "mutati_g1", "non_mutati_g1",
        "mutati_g2", "non_mutati_g2",
    ]
    with get_connection() as conn:
        with cursor_scope(conn) as cur:
            if exposure is not None:
                cur.callproc("get_significant_results_by_exposure", (exposure,))
            else:
                cur.callproc("get_significant_results")
            rows = []
            for result in cur.stored_results():
                rows.extend(result.fetchall())

    if rows and len(rows[0]) != len(expected_columns):
        raise RuntimeError(
            f"get_significant_results(): the stored procedure returned {len(rows[0])} columns, "
            f"expected {len(expected_columns)}. Update `expected_columns` in repository.py."
        )

    df = pd.DataFrame(rows, columns=expected_columns)
    if df.empty:
        return df

    parts = df["variant"].str.split("_", n=2, expand=True)
    df["chromosome"] = parts[0]
    df["position"] = parts[1]
    df["mutation"] = parts[2]
    return df

def load_raw_significant_results() -> pd.DataFrame:
    """Thin wrapper: calls the stored procedure and returns the DataFrame unmodified."""
    try:
        df = get_significant_results()
        return df
    except Exception:
        log.exception("Error fetching significant results from the stored procedure.")
        raise


def get_annotated_results() -> pd.DataFrame:
    """
    Calls the `get_annotated_results` stored procedure and returns a
    DataFrame combining:
        vrs1.exposure, vrs1.gene_name, vrs1.variant
    with every column from `gene_neuro_annotation` (gna.*).

    Unlike `get_significant_results`, this result set has no duplicated
    column names, so a dictionary cursor is used directly (no need for the
    positional-cursor workaround).
    """
    with get_connection() as conn:
        with cursor_scope(conn, dictionary=True) as cur:
            cur.callproc("get_annotated_results")
            rows = []
            for result in cur.stored_results():
                rows.extend(result.fetchall())

    df = pd.DataFrame(rows)
    if df.empty:
        log.warning("get_annotated_results(): the stored procedure returned no rows.")
    return df