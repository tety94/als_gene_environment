"""
DB repository for vqtl -- same pattern (and the same connection pool) as
`gene_environment/db/repository.py`: insert placeholders with
status='pending', then bulk-update (executemany) as results are computed.
Replaces the intermediate .tsv files the pipeline used to write
(vqtl_results.tsv, filtered_snps_full.tsv, interaction_results.tsv,
rge_results.tsv, robustness_results.tsv, perm_results.tsv): these are now
tables in `vqtl_scan_results` / `vqtl_interaction_results` /
`vqtl_rge_het_results` / `vqtl_robustness_results` /
`vqtl_permutation_results` (see db/schema.sql). report.md/report.docx/
figures/*.png remain files (they are final deliverables, not intermediate
state to resume from).

Direct reuse of `gene_environment.db.connection` (same MySQL pool,
"PID-aware": any joblib worker process that opens a connection
automatically gets its own pool, no TCP connections shared across a fork --
see that module for details). In practice, though, writes always happen
from the main process (the one consuming joblib's generator in scan.py),
NEVER inside the workers: simpler, and it also avoids opening N parallel
pools for nothing.

The vqtl_interaction_results / vqtl_rge_het_results /
vqtl_robustness_results / vqtl_permutation_results tables all share the
same shape (composite key generation+variant+exposure[+other], a status
column, statistics columns): the generic functions `ensure_placeholders` /
`get_done_keys` / `bulk_update_status` / `fetch_results` cover all four
without duplicating the same logic 4 times. `vqtl_scan_results` has a
different shape (fingerprint, is_candidate, no "exposure" column) and keeps
its own dedicated functions.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from gene_environment.db.connection import cursor_scope, get_connection
from gene_environment.logging_utils import get_logger

log = get_logger(__name__)


def safe_val(x):
    """Converts NaN/numpy types into values compatible with the MySQL
    driver. Identical to gene_environment.db.repository.safe_val
    (duplicated here, rather than imported, so vqtl is not coupled to an
    internal detail of a gene_environment module meant for a different
    table)."""
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, (np.floating,)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


# ============================================================
# Step 3+4: vqtl_scan_results (genome-wide scan + candidate filter)
# ============================================================

def get_scan_fingerprint(generation: int) -> dict | None:
    with get_connection() as conn:
        with cursor_scope(conn) as cur:
            cur.execute("SELECT fingerprint FROM vqtl_scan_runs WHERE generation=%s", (generation,))
            row = cur.fetchone()
    if row is None:
        return None
    return json.loads(row[0]) if isinstance(row[0], str) else row[0]


def reset_scan_run(generation: int, fingerprint: dict) -> None:
    """Deletes all vqtl_scan_results rows for this generation and records
    the new fingerprint -- called only when the saved fingerprint no longer
    matches the current one (statistical parameters changed, or first
    run)."""
    with get_connection() as conn:
        with cursor_scope(conn) as cur:
            cur.execute("DELETE FROM vqtl_scan_results WHERE generation=%s", (generation,))
            cur.execute(
                "INSERT INTO vqtl_scan_runs (generation, fingerprint) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE fingerprint=VALUES(fingerprint)",
                (generation, json.dumps(fingerprint)),
            )
    log.info("vqtl_scan_results cleared for generation=%s (new fingerprint recorded).", generation)


def ensure_scan_placeholders(generation: int, variants: list[dict], chunk_size: int = 5000) -> int:
    """Insert IGNORE of the placeholders (status='pending') for every
    variant in the scan, if they do not already exist. `variants`:
    [{'variant','chromosome','position'}, ...]."""
    if not variants:
        return 0
    sql = (
        "INSERT IGNORE INTO vqtl_scan_results (generation, variant, chromosome, position) "
        "VALUES (%s, %s, %s, %s)"
    )
    total = 0
    with get_connection() as conn:
        with cursor_scope(conn) as cur:
            for i in range(0, len(variants), chunk_size):
                chunk = variants[i:i + chunk_size]
                data = [(generation, v["variant"], v.get("chromosome"), v.get("position")) for v in chunk]
                cur.executemany(sql, data)
                total += cur.rowcount
    log.info("vqtl_scan_results: %d placeholders inserted/already present (generation=%s)", len(variants), generation)
    return total


def get_done_scan_variants(generation: int) -> set[str]:
    with get_connection() as conn:
        with cursor_scope(conn) as cur:
            cur.execute(
                "SELECT variant FROM vqtl_scan_results WHERE generation=%s AND status='done'", (generation,)
            )
            return {row[0] for row in cur.fetchall()}


def mark_scan_in_progress(generation: int, variant_list: list[str]) -> None:
    if not variant_list:
        return
    with get_connection() as conn:
        with cursor_scope(conn) as cur:
            cur.executemany(
                "UPDATE vqtl_scan_results SET status='in_progress' WHERE generation=%s AND variant=%s",
                [(generation, v) for v in variant_list],
            )


def save_scan_chunk_results(generation: int, rows: list[dict]) -> None:
    """Bulk-updates the status/statistics of a chunk of already-processed
    variants. Every row ALWAYS ends up with status 'done' or 'failed'
    (never 'pending'/'in_progress' again after this call): a variant
    discarded by the MAF/call-rate filters, or one for which the quantile
    regression does not converge, is still 'done' (with the statistics
    columns set to NULL), not 'pending' -- otherwise a subsequent run would
    retry it forever, thinking it still needs to be processed."""
    if not rows:
        return
    sql = """
    UPDATE vqtl_scan_results
    SET status=%(status)s, n=%(n)s, maf=%(maf)s, beta_qi=%(beta_qi)s, se=%(se)s,
        z=%(z)s, p=%(p)s, error_message=%(error_message)s
    WHERE generation=%(generation)s AND variant=%(variant)s
    """
    params = []
    for r in rows:
        params.append({
            "generation": generation, "variant": r["variant"], "status": r.get("status", "done"),
            "n": safe_val(r.get("n")), "maf": safe_val(r.get("maf")), "beta_qi": safe_val(r.get("beta_qi")),
            "se": safe_val(r.get("se")), "z": safe_val(r.get("z")), "p": safe_val(r.get("p")),
            "error_message": r.get("error_message"),
        })
    with get_connection() as conn:
        with cursor_scope(conn) as cur:
            cur.executemany(sql, params)


def update_gc_correction(generation: int, rows: list[dict]) -> None:
    """rows: [{'variant','p_gc','fdr_gc'}, ...] for ALL variants with a
    result (not only the candidates) -- the genomic-control correction is
    computed on the whole scan."""
    if not rows:
        return
    sql = "UPDATE vqtl_scan_results SET p_gc=%(p_gc)s, fdr_gc=%(fdr_gc)s WHERE generation=%(generation)s AND variant=%(variant)s"
    params = [{"generation": generation, "variant": r["variant"], "p_gc": safe_val(r["p_gc"]), "fdr_gc": safe_val(r["fdr_gc"])} for r in rows]
    with get_connection() as conn:
        with cursor_scope(conn) as cur:
            cur.executemany(sql, params)


def mark_candidates(generation: int, variant_list: list[str]) -> None:
    with get_connection() as conn:
        with cursor_scope(conn) as cur:
            cur.execute("UPDATE vqtl_scan_results SET is_candidate=0 WHERE generation=%s", (generation,))
            if variant_list:
                cur.executemany(
                    "UPDATE vqtl_scan_results SET is_candidate=1 WHERE generation=%s AND variant=%s",
                    [(generation, v) for v in variant_list],
                )
    log.info("vqtl_scan_results: %d candidates marked (generation=%s)", len(variant_list), generation)


_SCAN_SIG_COLS = ["variant", "chromosome", "position", "n", "maf", "beta_qi", "se", "z", "p", "p_gc", "fdr_gc"]


def count_significant_scan(generation: int) -> int:
    """How many rows already exist in vqtl_scan_results_significant for
    this generation. Used as the short-circuit signal in cli.py: if > 0
    (and --force was not passed), the genome-wide scan and the filter step
    are SKIPPED entirely for this generation, and results are read directly
    from here + from vqtl_scan_results (populated together, in the same run
    that populated this table)."""
    with get_connection() as conn:
        with cursor_scope(conn) as cur:
            cur.execute("SELECT COUNT(*) FROM vqtl_scan_results_significant WHERE generation=%s", (generation,))
            return cur.fetchone()[0]


def sync_scan_significant(generation: int) -> int:
    """Resynchronizes vqtl_scan_results_significant with the current set of
    candidates (is_candidate=1) in vqtl_scan_results for this generation:
    DELETE + INSERT ... SELECT in a single query, so it always stays an
    exact mirror (never stale rows from a previous filter). Called at the
    end of Step 4 (filter)."""
    with get_connection() as conn:
        with cursor_scope(conn) as cur:
            cur.execute("DELETE FROM vqtl_scan_results_significant WHERE generation=%s", (generation,))
            cur.execute(
                f"""
                INSERT INTO vqtl_scan_results_significant (generation, {', '.join(_SCAN_SIG_COLS)})
                SELECT generation, {', '.join(_SCAN_SIG_COLS)}
                FROM vqtl_scan_results
                WHERE generation=%s AND is_candidate=1 AND status='done'
                """,
                (generation,),
            )
            n = cur.rowcount
    log.info("vqtl_scan_results_significant: %d rows synchronized (generation=%s)", n, generation)
    return n


_SCAN_COLUMNS = ["variant", "chromosome", "position", "n", "maf", "beta_qi", "se", "z", "p", "p_gc", "fdr_gc", "is_candidate"]
_SCAN_RENAME = {"chromosome": "CHR", "position": "POS", "n": "N", "beta_qi": "beta_QI", "maf": "MAF", "se": "SE", "z": "Z", "p": "P", "p_gc": "P_gc", "fdr_gc": "fdr_gc"}


def get_scan_results(generation: int, only_done: bool = True) -> pd.DataFrame:
    """only_done=True: only rows with a VALID result (status='done' AND p
    not null) -- a variant that is 'done' but discarded by call-rate/MAF/QR
    still has status='done' (see save_scan_chunk_results) but no p, and it
    makes no sense to include it downstream in the Manhattan/QQ/FDR
    outputs."""
    where = "generation=%s" + (" AND status='done' AND p IS NOT NULL" if only_done else "")
    with get_connection() as conn:
        with cursor_scope(conn, dictionary=True) as cur:
            cur.execute(f"SELECT {', '.join(_SCAN_COLUMNS)} FROM vqtl_scan_results WHERE {where}", (generation,))
            rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=_SCAN_COLUMNS)
    df = df.rename(columns={"variant": "SNP", **_SCAN_RENAME})
    if not df.empty:
        df = df.sort_values("P").reset_index(drop=True)
    return df


def get_candidates(generation: int) -> pd.DataFrame:
    df = get_scan_results(generation, only_done=True)
    if df.empty:
        return df
    return df[df["is_candidate"] == 1].reset_index(drop=True)


# ============================================================
# Generic "keyed" tables (Step 5/6/7): interaction / rge_het /
# robustness / permutation all share the same basic shape.
# ============================================================

# logical name -> (real table name, extra key columns beyond
# generation+variant+exposure, statistics columns updated at the end of the step)
_KEYED_TABLES: dict[str, dict] = {
    "interaction": {
        "table": "vqtl_interaction_results",
        "extra_keys": [],
        "stat_cols": ["beta_i", "se", "pval", "n", "maf"],
    },
    "rge_het": {
        "table": "vqtl_rge_het_results",
        "extra_keys": [],
        "stat_cols": [
            "rge_beta_exposure_on_snp", "rge_se", "rge_pval", "rge_flag",
            "het_bp_lm_stat", "het_bp_lm_pvalue", "het_bp_f_stat", "het_bp_f_pvalue",
            "heteroscedasticity_flag",
        ],
    },
    "robustness": {
        "table": "vqtl_robustness_results",
        "extra_keys": ["phenotype_variant"],
        "stat_cols": ["beta_i", "se", "pval", "n", "maf"],
    },
    "permutation": {
        "table": "vqtl_permutation_results",
        "extra_keys": [],
        "stat_cols": [
            "beta_i_observed", "n_perm_valid", "empirical_pval", "asymptotic_pval",
            "levene_stat_observed", "levene_pval", "levene_n_perm_valid",
        ],
    },
}


def _spec(name: str) -> dict:
    if name not in _KEYED_TABLES:
        raise ValueError(f"Unknown vqtl table: {name!r} (expected one of: {list(_KEYED_TABLES)})")
    return _KEYED_TABLES[name]


def ensure_placeholders(name: str, generation: int, rows: list[dict], chunk_size: int = 2000) -> int:
    """rows: [{'variant','exposure','chromosome','position', <extra_keys>...}, ...]"""
    spec = _spec(name)
    if not rows:
        return 0
    extra = spec["extra_keys"]
    cols = ["generation", "variant", "exposure", "chromosome", "position"] + extra
    placeholders = ", ".join(["%s"] * len(cols))
    sql = f"INSERT IGNORE INTO {spec['table']} ({', '.join(cols)}) VALUES ({placeholders})"
    total = 0
    with get_connection() as conn:
        with cursor_scope(conn) as cur:
            for i in range(0, len(rows), chunk_size):
                chunk = rows[i:i + chunk_size]
                data = [
                    (generation, r["variant"], r["exposure"], r.get("chromosome"), r.get("position"))
                    + tuple(r[k] for k in extra)
                    for r in chunk
                ]
                cur.executemany(sql, data)
                total += cur.rowcount
    log.info("%s: %d placeholders inserted/already present (generation=%s)", spec["table"], len(rows), generation)
    return total


def get_done_keys(name: str, generation: int) -> set[tuple]:
    spec = _spec(name)
    extra = spec["extra_keys"]
    cols = ["variant", "exposure"] + extra
    with get_connection() as conn:
        with cursor_scope(conn) as cur:
            cur.execute(
                f"SELECT {', '.join(cols)} FROM {spec['table']} WHERE generation=%s AND status='done'",
                (generation,),
            )
            return {tuple(row) for row in cur.fetchall()}


def bulk_update_status(name: str, generation: int, rows: list[dict]) -> None:
    """rows: [{'variant','exposure', <extra_keys>..., <stat_cols>..., 'status': 'done'|'failed', 'error_message': ...}]"""
    spec = _spec(name)
    if not rows:
        return
    extra = spec["extra_keys"]
    stat_cols = spec["stat_cols"]
    set_clause = ", ".join([f"{c}=%({c})s" for c in stat_cols]) + ", status=%(status)s, error_message=%(error_message)s"
    where_clause = "generation=%(generation)s AND variant=%(variant)s AND exposure=%(exposure)s"
    where_clause += "".join([f" AND {k}=%({k})s" for k in extra])
    sql = f"UPDATE {spec['table']} SET {set_clause} WHERE {where_clause}"

    params = []
    for r in rows:
        p = {
            "generation": generation, "variant": r["variant"], "exposure": r["exposure"],
            "status": r.get("status", "done"), "error_message": r.get("error_message"),
        }
        for k in extra:
            p[k] = r[k]
        for c in stat_cols:
            p[c] = safe_val(r.get(c))
        params.append(p)

    with get_connection() as conn:
        with cursor_scope(conn) as cur:
            cur.executemany(sql, params)


def fetch_results(name: str, generation: int, only_done: bool = True) -> pd.DataFrame:
    spec = _spec(name)
    extra = spec["extra_keys"]
    cols = ["variant", "exposure", "chromosome", "position"] + extra + spec["stat_cols"]
    where = "generation=%s" + (" AND status='done'" if only_done else "")
    with get_connection() as conn:
        with cursor_scope(conn, dictionary=True) as cur:
            cur.execute(f"SELECT {', '.join(cols)} FROM {spec['table']} WHERE {where}", (generation,))
            rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=cols)
    rename = {"variant": "SNP", "chromosome": "CHR", "position": "POS", "exposure": "exposure"}
    return df.rename(columns=rename)


def clear_downstream_for_variants(generation: int, variant_list: list[str]) -> None:
    """Deletes rows from ALL 'keyed' tables (interaction/rge_het/
    robustness/permutation) for specific variants, for a given generation.
    Called by filter_candidates() when a variant that was a candidate in a
    previous run no longer is (threshold/top_n changed between filter
    runs): without this cleanup, the downstream tables would accumulate
    orphaned 'done' rows for variants that are no longer relevant, and
    fetch_results() would still include them in the results (there is no
    filter on current candidacy in those tables, only on generation+status)."""
    if not variant_list:
        return
    with get_connection() as conn:
        with cursor_scope(conn) as cur:
            for name, spec in _KEYED_TABLES.items():
                cur.executemany(
                    f"DELETE FROM {spec['table']} WHERE generation=%s AND variant=%s",
                    [(generation, v) for v in variant_list],
                )
    log.info("Cleared orphaned rows in all keyed tables for %d variants no longer candidates (generation=%s)", len(variant_list), generation)


# ============================================================
# vqtl_interaction_results_significant: same principle as
# vqtl_scan_results_significant (mirror + resynchronization), but WITHOUT a
# compute short-circuit function -- see the comment in schema.sql for why.
# It only serves as a direct source for Table 2 (Results) of report.docx,
# instead of re-filtering vqtl_interaction_results every time.
# ============================================================

_INTERACTION_SIG_COLS = ["variant", "exposure", "chromosome", "position", "beta_i", "se", "pval", "n", "maf"]


def sync_interaction_significant(generation: int, p_threshold: float) -> int:
    with get_connection() as conn:
        with cursor_scope(conn) as cur:
            cur.execute("DELETE FROM vqtl_interaction_results_significant WHERE generation=%s", (generation,))
            cur.execute(
                f"""
                INSERT INTO vqtl_interaction_results_significant (generation, {', '.join(_INTERACTION_SIG_COLS)})
                SELECT generation, {', '.join(_INTERACTION_SIG_COLS)}
                FROM vqtl_interaction_results
                WHERE generation=%s AND status='done' AND pval IS NOT NULL AND pval < %s
                """,
                (generation, p_threshold),
            )
            n = cur.rowcount
    log.info("vqtl_interaction_results_significant: %d rows synchronized (generation=%s, p<%s)", n, generation, p_threshold)
    return n


def get_interaction_significant(generation: int) -> pd.DataFrame:
    with get_connection() as conn:
        with cursor_scope(conn, dictionary=True) as cur:
            cur.execute(f"SELECT {', '.join(_INTERACTION_SIG_COLS)} FROM vqtl_interaction_results_significant WHERE generation=%s", (generation,))
            rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=_INTERACTION_SIG_COLS)
    return df.rename(columns={"variant": "SNP", "chromosome": "CHR", "position": "POS"})
