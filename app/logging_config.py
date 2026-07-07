"""Centralized logging setup.

One place to configure logging, rather than scattering `logging.basicConfig`
calls across modules (which is fragile — whichever import happens first
would silently win). `main.py` calls `configure_logging()` once, during
startup, before any request is served.
"""
from __future__ import annotations

import logging
import os


def configure_logging() -> None:
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Keep third-party library logging at a less chatty level by default;
    # LOG_LEVEL still controls our own `app.*` loggers via the root config.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
