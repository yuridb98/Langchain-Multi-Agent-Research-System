"""The cheap-model allowlist.

This is the server-side enforcement point for "only cheap models are
selectable". The UI only ever offers ids from ``CHEAP_MODELS`` as options,
but ``resolve_model`` is also called inside the pipeline itself, so a
tampered/forged session_state value (or a future non-UI caller) can never
reach an expensive model id -- it silently falls back to the agent's default.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    blurb: str


# Keep this list to genuinely cheap, fast tiers. Do not add "pro"/reasoning
# flagship models here -- the whole point is the owner's API bill stays low
# regardless of what a visitor picks in the UI.
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

# Named presets shown in the UI so a user can set all four agents at once.
PRESETS: dict[str, dict[str, str]] = {
    "Cheapest": {agent: "gpt-5.4-nano" for agent in AGENT_NAMES},
    "Balanced": dict(AGENT_DEFAULTS),
}


def model_spec(model_id: str) -> ModelSpec | None:
    return _BY_ID.get(model_id)


def resolve_model(agent: str, requested: str | None) -> str:
    """Return `requested` if it is an allowed model id, otherwise the agent's
    default. Never raises -- callers should always get back a usable id."""
    if requested and requested in ALLOWED_MODEL_IDS:
        return requested
    return AGENT_DEFAULTS.get(agent, CHEAP_MODELS[0].id)
