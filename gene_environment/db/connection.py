"""
Gestione delle connessioni MySQL.
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
