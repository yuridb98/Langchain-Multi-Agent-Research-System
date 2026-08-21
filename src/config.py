"""App configuration: secrets, runtime limits, and the cheap-model allowlist.

Secrets resolve in one order everywhere: st.secrets (Streamlit Cloud) first,
then the environment (populated locally by .env via python-dotenv). Nothing
else in the codebase reads os.environ/st.secrets or calls load_dotenv().
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


class ConfigError(Exception):
    """Raised when required configuration (e.g. an API key) is missing or invalid."""


def get_secret(name: str) -> str | None:
    try:
        import streamlit as st

        if name in st.secrets:
            value = st.secrets[name]
            if value:
                return str(value)
    except Exception:
        pass  # no Streamlit runtime (CLI/tests), or no secrets.toml configured

    return os.environ.get(name) or None


def require_secret(name: str) -> str:
    value = get_secret(name)
    if not value:
        raise ConfigError(
            f"Missing required setting '{name}'. Set it in your .env file "
            f"locally, or in the app's Secrets panel when deployed."
        )
    return value


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


# Runtime limits (overridable via env vars).
REQUEST_TIMEOUT = _int_env("REQUEST_TIMEOUT_SECONDS", 30)
MAX_RETRIES = _int_env("MAX_LLM_RETRIES", 2)
MAX_TOPIC_CHARS = _int_env("MAX_TOPIC_CHARS", 300)
SCRAPE_CHAR_LIMIT = _int_env("SCRAPE_CHAR_LIMIT", 5000)
MAX_RUNS_PER_SESSION = _int_env("MAX_RUNS_PER_SESSION", 10)
WRITER_MAX_TOKENS = _int_env("WRITER_MAX_TOKENS", 2000)


# ---- Cheap-model allowlist ----
# The UI only ever offers ids from CHEAP_MODELS. resolve_model() is also
# called inside the pipeline itself, so a tampered client value can never
# reach an expensive model -- this is the actual enforcement point.


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    blurb: str


# Keep this to genuinely cheap, fast tiers -- no "pro"/reasoning flagships.
CHEAP_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("gpt-5.4-nano", "GPT-5.4 Nano", "Cheapest and fastest — good default"),
    ModelSpec("gpt-5.4-mini", "GPT-5.4 Mini", "Best quality in the cheap tier"),
    ModelSpec("gpt-5-nano", "GPT-5 Nano", "Previous generation, very cheap"),
    ModelSpec("gpt-5-mini", "GPT-5 Mini", "Previous generation, balanced"),
    ModelSpec("gpt-4.1-mini", "GPT-4.1 Mini", "Non-reasoning, low latency"),
    ModelSpec("gpt-4.1-nano", "GPT-4.1 Nano", "Non-reasoning, cheapest 4.x"),
    ModelSpec("gpt-4o-mini", "GPT-4o Mini", "Legacy fallback"),
)

ALLOWED_MODEL_IDS: frozenset[str] = frozenset(m.id for m in CHEAP_MODELS)
_BY_ID: dict[str, ModelSpec] = {m.id: m for m in CHEAP_MODELS}

AGENT_NAMES: tuple[str, ...] = ("search", "reader", "writer", "critic")

AGENT_DEFAULTS: dict[str, str] = {
    "search": "gpt-5.4-nano",
    "reader": "gpt-5.4-nano",
    "writer": "gpt-5.4-mini",
    "critic": "gpt-5.4-nano",
}

# Named presets so the UI can set all four agents at once.
PRESETS: dict[str, dict[str, str]] = {
    "Cheapest": {agent: "gpt-5.4-nano" for agent in AGENT_NAMES},
    "Balanced": dict(AGENT_DEFAULTS),
}


def model_spec(model_id: str) -> ModelSpec | None:
    return _BY_ID.get(model_id)


def resolve_model(agent: str, requested: str | None) -> str:
    """Return `requested` if allowed, else the agent's default. Never raises."""
    if requested and requested in ALLOWED_MODEL_IDS:
        return requested
    return AGENT_DEFAULTS.get(agent, CHEAP_MODELS[0].id)
