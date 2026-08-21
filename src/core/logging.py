"""Shared logging setup.

One place to configure the log format so every module just does
``logger = get_logger(__name__)`` instead of sprinkling ``print()`` calls
(as the original pipeline did). Never log secret values -- callers are
responsible for passing already-redacted strings, but as a safety net
``truncate`` is provided for bounding potentially huge scraped/LLM text
before it hits the log.
"""

from __future__ import annotations

import logging
import os

_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    _configure_root()
    return logging.getLogger(name)


def truncate(text: str, limit: int = 400) -> str:
    """Bound a string for safe log output. Never used on secrets -- only on
    LLM/tool payloads that may be arbitrarily large."""
    if text is None:
        return ""
    text = str(text)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [+{len(text) - limit} chars]"
