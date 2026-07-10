"""
Logging centralizzato.

PRIMA: quasi ogni script usava `print()` per lo stato di avanzamento. Questo
significa: nessun timestamp uniforme, nessun livello (info/warning/error),
niente file di log persistente, e output completamente mischiato/perso nei
worker paralleli (ProcessPoolExecutor).

ORA: un solo punto di configurazione. Ogni modulo fa
    from gene_environment.logging_utils import get_logger
    log = get_logger(__name__)
e ottiene un logger che scrive sia su console sia su file (con rotazione),
con timestamp, livello, nome modulo e (nei worker) il PID, cosi' i log dei
processi paralleli restano distinguibili.
"""
from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

_CONFIGURED = False


def configure_logging(log_dir: str = "./logs", filename: str = "pipeline.log", level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, filename)

    fmt = "%(asctime)s [%(levelname)s] [pid=%(process)d] %(name)s: %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    root = logging.getLogger()
    root.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    file_handler = RotatingFileHandler(log_path, maxBytes=50 * 1024 * 1024, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # librerie terze troppo verbose
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    if not _CONFIGURED:
        # fallback: se qualcuno importa il logger senza aver chiamato
        # configure_logging esplicitamente (es. in un worker figlio), usiamo
        # comunque una config di default cosi' non si perdono i log.
        try:
            from gene_environment.config import get_config
            configure_logging(get_config().log_dir)
        except Exception:
            configure_logging("./logs")
    return logging.getLogger(name)
