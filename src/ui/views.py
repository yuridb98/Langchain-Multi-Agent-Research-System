"""Results rendering.

LLM/scraped output is untrusted text (it can legitimately contain '<', '>',
markdown-breaking sequences, etc). It is rendered with plain `st.markdown`
(no unsafe_allow_html) or `st.text`, never interpolated into an HTML string
-- that avoids the layout corruption the old UI had whenever scraped content
contained a stray '<'.
"""

from __future__ import annotations

import time

import streamlit as st

from ..pipeline.pipeline import ResearchResult

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
