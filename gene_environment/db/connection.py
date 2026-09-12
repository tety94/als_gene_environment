"""MySQL connection pooling: a shared, per-process connection pool plus context managers for connections and cursors."""
from __future__ import annotations

import contextlib
import os
import time

import mysql.connector
from mysql.connector import pooling

from gene_environment.config import DBConfig, get_config
from gene_environment.logging_utils import get_logger

log = get_logger(__name__)

_pool: pooling.MySQLConnectionPool | None = None
_pool_pid: int | None = None


def _get_pool() -> pooling.MySQLConnectionPool:
    # The pool is keyed by PID and recreated whenever the current PID differs
    # from the one that created it. This matters with multiprocessing
    # (ProcessPoolExecutor / fork): if the pool were created in the parent
    # process and then the parent forked worker processes, those workers
    # would inherit a copy of the pool with TCP connections already open in
    # the parent. Multiple processes sharing a forked socket corrupt the
    # MySQL client-side protocol, causing errors like "MySQL Connection not
    # available". Recreating the pool per-PID ensures every process (parent
    # or worker) ends up with its own pool and fresh TCP connections.
    global _pool, _pool_pid
    current_pid = os.getpid()
    if _pool is None or _pool_pid != current_pid:
        cfg: DBConfig = get_config().db
        _pool = pooling.MySQLConnectionPool(
            # Pool name unique per process, to avoid collisions in
            # mysql-connector's internal registry if the pool is recreated
            # after a fork under the same logical name.
            pool_name=f"gene_env_pool_{current_pid}",
            pool_size=cfg.pool_size,
            host=cfg.host,
            port=cfg.port,
            user=cfg.user,
            password=cfg.password,
            database=cfg.name,
            autocommit=False,
        )
        _pool_pid = current_pid
        log.info("MySQL connection pool created (pid=%d, pool_size=%d, host=%s:%s, db=%s)",
                  current_pid, cfg.pool_size, cfg.host, cfg.port, cfg.name)
    return _pool


@contextlib.contextmanager
def get_connection(retries: int = 3, retry_delay: float = 1.0):
    """Get a connection from the pool, commit on success, rollback on exception, and always return it to the pool."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            conn = _get_pool().get_connection()
            break
        except mysql.connector.Error as e:
            last_err = e
            log.warning("DB connection failed (attempt %d/%d): %s", attempt, retries, e)
            time.sleep(retry_delay * attempt)
    else:
        raise last_err  # all retries exhausted

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()  # for a pooled connection, close() returns it to the pool


@contextlib.contextmanager
def cursor_scope(conn, dictionary: bool = False):
    cur = conn.cursor(dictionary=dictionary)
    try:
        yield cur
    finally:
        cur.close()
