"""A small amount of CSS for things Streamlit's theme tokens can't reach
(hero typography, letter-spacing, a subtle divider). Everything else --
buttons, inputs, selects, sliders, tabs, containers -- is native Streamlit
styled entirely via .streamlit/config.toml, so it's correct in both light
and dark automatically.

Colors here are taken from Streamlit's CSS custom properties
(--text-color, --secondary-background-color, etc.) rather than hardcoded
hex values, so this CSS follows whichever theme is active instead of being
locked to one look.
"""

from __future__ import annotations

import streamlit as st

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


def inject() -> None:
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
