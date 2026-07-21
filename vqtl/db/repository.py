"""
Repository DB per vqtl -- stesso pattern (e stesso connection pool) di
`gene_environment/db/repository.py`: insert dei placeholder con
status='pending', poi update in bulk (executemany) man mano che i risultati
vengono calcolati. Sostituisce i .tsv intermedi che la pipeline scriveva in
precedenza (vqtl_results.tsv, filtered_snps_full.tsv, interaction_results.tsv,
rge_results.tsv, robustness_results.tsv, perm_results.tsv): ora sono tabelle
in `vqtl_scan_results` / `vqtl_interaction_results` / `vqtl_rge_het_results` /
`vqtl_robustness_results` / `vqtl_permutation_results` (vedi db/schema.sql).
report.md/report.docx/figures/*.png restano file (sono deliverable finali,
non stato intermedio da riprendere).

Riuso diretto di `gene_environment.db.connection` (stesso pool MySQL,
"PID-aware": ogni processo worker di joblib che aprisse una connessione
otterrebbe automaticamente un proprio pool, niente connessioni TCP
condivise via fork -- vedi il modulo per il dettaglio). In pratica pero' le
scritture avvengono sempre dal processo principale (chi consuma il
generatore di joblib in scan.py), MAI dentro i worker: piu' semplice, ed
evita comunque di aprire N pool paralleli per niente.

Le tabelle vqtl_interaction_results / vqtl_rge_het_results /
vqtl_robustness_results / vqtl_permutation_results condividono tutte la
stessa forma (chiave composta generation+variant+exposure[+altro], colonna
status, colonne di statistiche): le funzioni generiche
`ensure_placeholders` / `get_done_keys` / `bulk_update_status` /
`fetch_results` coprono tutte e quattro senza duplicare la stessa logica
4 volte. `vqtl_scan_results` ha una forma diversa (fingerprint, is_candidate,
niente colonna "exposure") e resta con funzioni dedicate.
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
    """Converte NaN/tipi numpy in valori compatibili col driver MySQL.
    Identica a gene_environment.db.repository.safe_val (duplicata qui,
    invece di importarla, per non accoppiare vqtl a un dettaglio interno di
    un modulo di gene_environment pensato per un'altra tabella)."""
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
# Step 3+4: vqtl_scan_results (scan genoma-wide + filtro candidati)
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
    """Cancella tutte le righe di vqtl_scan_results per questa generazione
    e registra la nuova fingerprint -- chiamata solo quando la fingerprint
    salvata NON corrisponde piu' a quella corrente (parametri statistici
    cambiati, o primo run)."""
    with get_connection() as conn:
        with cursor_scope(conn) as cur:
            cur.execute("DELETE FROM vqtl_scan_results WHERE generation=%s", (generation,))
            cur.execute(
                "INSERT INTO vqtl_scan_runs (generation, fingerprint) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE fingerprint=VALUES(fingerprint)",
                (generation, json.dumps(fingerprint)),
            )
    log.info("vqtl_scan_results ripulita per generation=%s (nuova fingerprint registrata).", generation)


def ensure_scan_placeholders(generation: int, variants: list[dict], chunk_size: int = 5000) -> int:
    """Insert IGNORE dei placeholder (status='pending') per ogni variante
    dello scan, se non esistono gia'. `variants`: [{'variant','chromosome','position'}, ...]."""
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
    log.info("vqtl_scan_results: %d placeholder inseriti/gia' presenti (generation=%s)", len(variants), generation)
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
    """Aggiorna in bulk lo stato/le statistiche di un chunk di varianti gia'
    processate. Ogni riga ha SEMPRE status 'done' o 'failed' (mai piu'
    'pending'/'in_progress' dopo questa chiamata): una variante scartata dai
    filtri MAF/call-rate o per cui la quantile regression non converge e'
    comunque 'done' (con le colonne statistiche a NULL), non 'pending' --
    altrimenti un run successivo la ritenterebbe all'infinito credendola
    ancora da fare."""
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
    """rows: [{'variant','p_gc','fdr_gc'}, ...] per TUTTE le varianti con
    esito (non solo i candidati) -- la correzione genomic-control e' calcolata
    sull'intero scan."""
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
    log.info("vqtl_scan_results: %d candidati marcati (generation=%s)", len(variant_list), generation)


_SCAN_SIG_COLS = ["variant", "chromosome", "position", "n", "maf", "beta_qi", "se", "z", "p", "p_gc", "fdr_gc"]


def count_significant_scan(generation: int) -> int:
    """Quante righe ci sono gia' in vqtl_scan_results_significant per questa
    generazione. Usata come segnale di short-circuit in cli.py: se > 0 (e
    non e' stato passato --force), lo scan genoma-wide e il filtro vengono
    SALTATI del tutto per questa generazione, i risultati si leggono
    direttamente da qui + da vqtl_scan_results (gia' popolata insieme, nello
    stesso run in cui e' stata popolata questa tabella)."""
    with get_connection() as conn:
        with cursor_scope(conn) as cur:
            cur.execute("SELECT COUNT(*) FROM vqtl_scan_results_significant WHERE generation=%s", (generation,))
            return cur.fetchone()[0]


def sync_scan_significant(generation: int) -> int:
    """Risincronizza vqtl_scan_results_significant con l'attuale insieme di
    candidati (is_candidate=1) in vqtl_scan_results per questa generazione:
    DELETE + INSERT ... SELECT in una sola query, cosi' resta sempre uno
    specchio esatto (mai righe stantie di un filtro precedente). Chiamata
    alla fine dello Step 4 (filter)."""
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
    log.info("vqtl_scan_results_significant: %d righe sincronizzate (generation=%s)", n, generation)
    return n


_SCAN_COLUMNS = ["variant", "chromosome", "position", "n", "maf", "beta_qi", "se", "z", "p", "p_gc", "fdr_gc", "is_candidate"]
_SCAN_RENAME = {"chromosome": "CHR", "position": "POS", "n": "N", "beta_qi": "beta_QI", "maf": "MAF", "se": "SE", "z": "Z", "p": "P", "p_gc": "P_gc", "fdr_gc": "fdr_gc"}


def get_scan_results(generation: int, only_done: bool = True) -> pd.DataFrame:
    """only_done=True: solo righe con un risultato VALIDO (status='done' E
    p non nullo) -- una variante 'done' ma scartata da call-rate/MAF/QR ha
    comunque status='done' (vedi save_scan_chunk_results) ma nessun p, e non
    ha senso includerla in Manhattan/QQ/FDR a valle."""
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
# Tabelle "keyed" generiche (Step 5/6/7): interaction / rge_het /
# robustness / permutation condividono la stessa forma di base.
# ============================================================

# nome logico -> (nome tabella reale, colonne chiave extra oltre a
# generation+variant+exposure, colonne di statistiche aggiornate a fine step)
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
        raise ValueError(f"Tabella vqtl sconosciuta: {name!r} (attese: {list(_KEYED_TABLES)})")
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
    log.info("%s: %d placeholder inseriti/gia' presenti (generation=%s)", spec["table"], len(rows), generation)
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
    """Cancella le righe di TUTTE le tabelle 'keyed' (interaction/rge_het/
    robustness/permutation) relative a varianti specifiche, per una
    generazione. Chiamata da filter_candidates() quando una variante che
    era candidata in un run precedente NON lo e' piu' (soglia/top_n cambiati
    tra un filtro e l'altro): senza questa pulizia, le tabelle a valle
    accumulerebbero righe 'done' orfane di varianti non piu' rilevanti, che
    fetch_results() includerebbe comunque nei risultati (non c'e' nessun
    filtro per candidacy corrente in quelle tabelle, solo per generation+status)."""
    if not variant_list:
        return
    with get_connection() as conn:
        with cursor_scope(conn) as cur:
            for name, spec in _KEYED_TABLES.items():
                cur.executemany(
                    f"DELETE FROM {spec['table']} WHERE generation=%s AND variant=%s",
                    [(generation, v) for v in variant_list],
                )
    log.info("Ripulite righe orfane in tutte le tabelle keyed per %d varianti non piu' candidate (generation=%s)", len(variant_list), generation)


# ============================================================
# vqtl_interaction_results_significant: stesso principio di
# vqtl_scan_results_significant (mirror + risincronizzazione), ma qui SENZA
# funzione di short-circuit sul calcolo -- vedi il commento nello schema.sql
# sul perche'. Serve solo come fonte diretta per Table 2 (Results) del
# report.docx, invece di rifiltrare vqtl_interaction_results ogni volta.
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
    log.info("vqtl_interaction_results_significant: %d righe sincronizzate (generation=%s, p<%s)", n, generation, p_threshold)
    return n


def get_interaction_significant(generation: int) -> pd.DataFrame:
    with get_connection() as conn:
        with cursor_scope(conn, dictionary=True) as cur:
            cur.execute(f"SELECT {', '.join(_INTERACTION_SIG_COLS)} FROM vqtl_interaction_results_significant WHERE generation=%s", (generation,))
            rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=_INTERACTION_SIG_COLS)
    return df.rename(columns={"variant": "SNP", "chromosome": "CHR", "position": "POS"})
