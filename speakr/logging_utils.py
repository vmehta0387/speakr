"""Logging configuration helpers."""

import logging
import os


def configure_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    # Avoid resetting handlers if the host process already configured logging.
    if logging.getLogger().handlers:
        logging.getLogger("speakr").setLevel(level)
        return

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
