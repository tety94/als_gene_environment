"""Centralized logging setup: writes to both console and a rotating log file, with timestamp, level, module name, and PID (to keep parallel worker output distinguishable)."""
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

    # Third-party libraries that are too verbose at INFO level.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    if not _CONFIGURED:
        # Fallback: if a logger is imported without configure_logging having
        # been called explicitly (e.g. in a child worker), use a default
        # config so logs aren't lost.
        try:
            from gene_environment.config import get_config
            configure_logging(get_config().log_dir)
        except Exception:
            configure_logging("./logs")
    return logging.getLogger(name)
