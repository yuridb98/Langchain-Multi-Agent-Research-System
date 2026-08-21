"""LLM construction.

Replaces the old module-level ``llm = ChatOpenAI(...)`` singleton. That
approach meant every agent shared one hardcoded model and constructing the
client at import time -- a missing API key crashed the whole app on page
load instead of failing gracefully when a user actually runs something.

``make_llm`` is lazy (only called when a pipeline actually runs) and cached
(so re-running with the same model/temperature reuses the client instead of
reconnecting). It also re-validates the model id against the allowlist, so
even a caller that bypasses the UI can't reach an expensive model.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from ..config.models import ALLOWED_MODEL_IDS
from ..config.settings import MAX_RETRIES, REQUEST_TIMEOUT, require_secret
from ..core.errors import ConfigError


@lru_cache(maxsize=16)
def make_llm(model_id: str, temperature: float = 0.0) -> ChatOpenAI:
    if model_id not in ALLOWED_MODEL_IDS:
        raise ConfigError(f"Model '{model_id}' is not in the allowed model list.")

    api_key = SecretStr(require_secret("OPENAI_API_KEY"))

    return ChatOpenAI(
        model=model_id,
        temperature=temperature,
        api_key=api_key,
        timeout=REQUEST_TIMEOUT,
        max_retries=MAX_RETRIES,
    )
