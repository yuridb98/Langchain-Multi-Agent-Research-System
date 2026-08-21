"""Streamlit UI pieces: theme CSS, the sidebar model controls, and results
rendering. Split into three sections below rather than three files -- each
is small enough that finding things by scrolling one file beats finding
which of three files to open.
"""

from __future__ import annotations

import time

import streamlit as st

from .config import AGENT_DEFAULTS, CHEAP_MODELS, PRESETS
from .pipeline import ResearchResult

# ============================================================================
# Theme
#
# A little CSS for what Streamlit's theme tokens can't reach (hero
# typography). Everything else -- buttons, inputs, selects, sliders, tabs,
# containers -- is native Streamlit, styled via .streamlit/config.toml, so
# it's correct in both light and dark automatically. Colors here come from
# Streamlit's CSS custom properties (--text-color, etc.), not hardcoded hex,
# so this follows whichever theme is active.
# ============================================================================

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=DM+Mono:wght@400;500&display=swap');

.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--primary-color);
    margin-bottom: 0.25rem;
}

.hero-title {
    font-family: 'DM Sans', sans-serif;
    font-weight: 700;
    font-size: 2.1rem;
    margin: 0 0 0.4rem 0;
    color: var(--text-color);
}

.hero-sub {
    font-family: 'DM Sans', sans-serif;
    color: var(--text-color);
    opacity: 0.75;
    max-width: 46rem;
    margin-bottom: 0.5rem;
}

.app-divider {
    border: none;
    border-top: 1px solid var(--secondary-background-color);
    margin: 1.25rem 0;
}
</style>
"""


def inject_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, eyebrow: str = "Multi-Agent AI System") -> None:
    st.markdown(
        f"""
        <div class="hero-eyebrow">{eyebrow}</div>
        <div class="hero-title">{title}</div>
        <p class="hero-sub">{subtitle}</p>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# Sidebar: per-agent model selection
#
# Every selector only offers ids from CHEAP_MODELS -- there's no way to pick
# an expensive model from this UI. config.resolve_model() re-enforces the
# same allowlist inside the pipeline, so this is a convenience, not the
# actual security boundary.
# ============================================================================

_AGENT_LABELS = {
    "search": "🔍 Search Agent",
    "reader": "📄 Reader Agent",
    "writer": "✍️ Writer Chain",
    "critic": "🧐 Critic Chain",
}

_MODEL_LABELS = {m.id: m.label for m in CHEAP_MODELS}
_MODEL_BLURBS = {m.id: m.blurb for m in CHEAP_MODELS}
_MODEL_IDS = [m.id for m in CHEAP_MODELS]


def render_sidebar() -> tuple[dict[str, str], float]:
    """Render the sidebar controls and return (models, temperature)."""
    st.sidebar.header("Model settings")
    st.sidebar.caption("All options are low-cost model tiers.")

    preset = st.sidebar.radio(
        "Preset", options=["Cheapest", "Balanced", "Custom"], horizontal=True
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


# ============================================================================
# Results rendering
#
# LLM/scraped output is untrusted text (it can legitimately contain '<',
# '>', markdown-breaking sequences). It's rendered with plain st.markdown
# (no unsafe_allow_html) or st.text, never interpolated into an HTML string.
# ============================================================================

STEP_LABELS = {
    "search": "Search Agent",
    "reader": "Reader Agent",
    "writer": "Writer Chain",
    "critic": "Critic Chain",
}


def render_result(result: ResearchResult) -> None:
    if result.errors:
        for err in result.errors:
            st.error(err)
        if not result.report:
            return

    with st.expander("Models used for this run", expanded=False):
        for agent, model_id in result.models.items():
            st.caption(f"{STEP_LABELS[agent]}: `{model_id}`")

    tab_report, tab_critique, tab_raw = st.tabs(["📝 Report", "🧐 Critique", "🔎 Sources & Raw"])

    with tab_report:
        if result.report:
            with st.container(border=True):
                st.markdown(result.report)
            st.download_button(
                label="⬇ Download Report (.md)",
                data=result.report,
                file_name=f"research_report_{int(time.time())}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        else:
            st.info("No report was produced.")

    with tab_critique:
        if result.feedback:
            with st.container(border=True):
                st.markdown(result.feedback)
        else:
            st.info("No critique was produced.")

    with tab_raw:
        if result.search:
            st.subheader("Search results (raw)")
            st.text(result.search)
        if result.reader:
            st.subheader("Scraped content (raw)")
            st.text(result.reader)
        if not result.search and not result.reader:
            st.info("No raw output available.")
