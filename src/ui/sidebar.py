"""Sidebar: per-agent model selection.

Every selector only ever offers ids from ``CHEAP_MODELS`` -- there is no way
to pick an expensive model from this UI. That allowlist is also re-enforced
inside the pipeline itself (see config.models.resolve_model), so this is a
convenience, not the actual security boundary.
"""

from __future__ import annotations

import streamlit as st

from ..config.models import AGENT_DEFAULTS, CHEAP_MODELS, PRESETS

_AGENT_LABELS = {
    "search": "🔍 Search Agent",
    "reader": "📄 Reader Agent",
    "writer": "✍️ Writer Chain",
    "critic": "🧐 Critic Chain",
}

_MODEL_LABELS = {m.id: m.label for m in CHEAP_MODELS}
_MODEL_BLURBS = {m.id: m.blurb for m in CHEAP_MODELS}
_MODEL_IDS = [m.id for m in CHEAP_MODELS]


def render() -> tuple[dict[str, str], float]:
    """Render the sidebar controls and return (models, temperature)."""
    st.sidebar.header("Model settings")
    st.sidebar.caption("All options are low-cost model tiers.")

    preset = st.sidebar.radio(
        "Preset",
        options=["Cheapest", "Balanced", "Custom"],
        horizontal=True,
    )

    models: dict[str, str] = {}

    if preset in PRESETS:
        models = dict(PRESETS[preset])
        with st.sidebar.expander("Models used", expanded=False):
            for agent, model_id in models.items():
                st.caption(f"{_AGENT_LABELS[agent]}: {_MODEL_LABELS[model_id]}")
    else:
        for agent in AGENT_DEFAULTS:
            default_id = AGENT_DEFAULTS[agent]
            models[agent] = st.sidebar.selectbox(
                _AGENT_LABELS[agent],
                options=_MODEL_IDS,
                index=_MODEL_IDS.index(default_id),
                format_func=lambda mid: _MODEL_LABELS[mid],
                help=_MODEL_BLURBS[default_id],
                key=f"model_{agent}",
            )

    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Higher values make agent output more varied/creative.",
    )

    return models, temperature
