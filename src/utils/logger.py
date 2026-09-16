"""
utils/logger.py
================
A single, reusable logger factory used across every module.

Addresses the LOGGING / MONITORING non-functional requirement: every
significant event (registration, recognition, attendance marking, errors)
is written both to the console and to a rotating log file, which is
essential for auditing who was marked present and diagnosing failures
after the fact.
"""

import logging
from logging.handlers import RotatingFileHandler

from src import config


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger that writes to console + rotating file.

    Parameters
    ----------
    name : str
        Usually ``__name__`` of the calling module, so log lines are
        traceable to their source.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Already configured (avoids duplicate handlers on repeated calls,
        # which would otherwise duplicate every log line).
        return logger

    logger.setLevel(config.LOG_LEVEL)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        config.LOG_FILE, maxBytes=1_000_000, backupCount=3
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False

    return logger
