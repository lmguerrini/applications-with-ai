from __future__ import annotations

import logging


_LOGGING_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    global _LOGGING_CONFIGURED

    resolved_level = getattr(logging, level.upper(), logging.INFO)
    if _LOGGING_CONFIGURED:
        logging.getLogger().setLevel(resolved_level)
        return

    logging.basicConfig(
        level=resolved_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    _LOGGING_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
