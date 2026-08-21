"""Central configuration: secret resolution + runtime limits.

Every other module that needs a secret or a limit imports from here instead
of calling ``os.getenv`` / ``load_dotenv`` itself. That keeps secret handling
in exactly one place, so it's auditable and consistent between the Streamlit
app, the CLI, and tests.

Resolution order for secrets:
  1. ``st.secrets`` -- used on Streamlit Community Cloud, where secrets are
     configured in the app's "Secrets" panel and never touch disk in the repo.
  2. Process environment (populated by ``.env`` locally via python-dotenv, or
     by the platform's own env-var mechanism in Docker/CI/etc).

Secrets are never written to ``st.session_state``, never logged, and never
included in error messages shown to the user.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

from ..core.errors import ConfigError

# Loaded exactly once, here, at first import of this module. Nothing else in
# the codebase should call load_dotenv().
load_dotenv()


def get_secret(name: str) -> str | None:
    """Return a secret by name, or None if it isn't configured anywhere."""
    try:
        import streamlit as st

        if name in st.secrets:
            value = st.secrets[name]
            if value:
                return str(value)
    except Exception:
        # No Streamlit runtime (CLI/tests) or no secrets.toml configured --
        # fall through to the environment.
        pass

    value = os.environ.get(name)
    return value or None


def require_secret(name: str) -> str:
    """Return a secret by name, raising a user-safe ConfigError if missing."""
    value = get_secret(name)
    if not value:
        raise ConfigError(
            f"Missing required setting '{name}'. Set it in your .env file "
            f"locally, or in the app's Secrets panel when deployed."
        )
    return value


@lru_cache(maxsize=1)
def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# ---- Runtime limits (all overridable via env vars, sane defaults otherwise) ----

REQUEST_TIMEOUT: int = _int_env("REQUEST_TIMEOUT_SECONDS", 30)
MAX_RETRIES: int = _int_env("MAX_LLM_RETRIES", 2)
MAX_TOPIC_CHARS: int = _int_env("MAX_TOPIC_CHARS", 300)
SCRAPE_CHAR_LIMIT: int = _int_env("SCRAPE_CHAR_LIMIT", 5000)
MAX_RUNS_PER_SESSION: int = _int_env("MAX_RUNS_PER_SESSION", 10)
WRITER_MAX_TOKENS: int = _int_env("WRITER_MAX_TOKENS", 2000)
