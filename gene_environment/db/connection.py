"""
Gestione delle connessioni MySQL.

PROBLEMA nell'originale (db.py):
  - `get_conn()` apriva una NUOVA connessione TCP al DB ogni volta che veniva
    chiamata, e questo veniva fatto in moltissimi punti (spesso dentro
    funzioni chiamate in loop, es. mark_variant_in_progress/reset per ogni
    variante). Aprire/chiudere una connessione per ogni singola query è
    lento e, sotto carico, può esaurire le connessioni disponibili sul
    server MySQL.
  - Nessun retry/backoff su errori transitori di connessione.
  - Cursori aperti senza sempre garantire la chiusura in caso di eccezione
    (niente context manager -> uso di try/finally sparso e incoerente).

SOLUZIONE:
  - Un connection pool (mysql.connector.pooling) condiviso, dimensionato da
    config (DB_POOL_SIZE). Le connessioni vengono riutilizzate invece di
    essere aperte/chiuse in continuazione.
  - Un context manager `get_connection()` che restituisce la connessione al
    pool automaticamente (commit se tutto ok, rollback se eccezione).
  - Un context manager `cursor_scope()` per i cursori, che chiude sempre il
    cursore anche in caso di errore.
"""
from __future__ import annotations

import contextlib
import time

import mysql.connector
from mysql.connector import pooling

from gene_environment.config import DBConfig, get_config
from gene_environment.logging_utils import get_logger

log = get_logger(__name__)

_pool: pooling.MySQLConnectionPool | None = None


def _get_pool() -> pooling.MySQLConnectionPool:
    global _pool
    if _pool is None:
        cfg: DBConfig = get_config().db
        _pool = pooling.MySQLConnectionPool(
            pool_name="gene_env_pool",
            pool_size=cfg.pool_size,
            host=cfg.host,
            port=cfg.port,
            user=cfg.user,
            password=cfg.password,
            database=cfg.name,
            autocommit=False,
        )
        log.info("Connection pool MySQL creato (pool_size=%d, host=%s:%s, db=%s)",
                  cfg.pool_size, cfg.host, cfg.port, cfg.name)
    return _pool


@contextlib.contextmanager
def get_connection(retries: int = 3, retry_delay: float = 1.0):
    """Context manager: prende una connessione dal pool, fa commit a fine
    blocco se non ci sono state eccezioni, rollback altrimenti, e restituisce
    sempre la connessione al pool (mai lasciarla aperta indefinitamente)."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            conn = _get_pool().get_connection()
            break
        except mysql.connector.Error as e:
            last_err = e
            log.warning("Connessione DB fallita (tentativo %d/%d): %s", attempt, retries, e)
            time.sleep(retry_delay * attempt)
    else:
        raise last_err  # tutti i tentativi falliti

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()  # per una connessione dal pool, close() la restituisce al pool


@contextlib.contextmanager
def cursor_scope(conn, dictionary: bool = False):
    cur = conn.cursor(dictionary=dictionary)
    try:
        yield cur
    finally:
        cur.close()
