"""Streamlit UI.

Thin by design: page config + theme + a small amount of glue. All actual
work happens in src.pipeline.run_research_pipeline; model selection and
result rendering live in src.ui. That split is what lets the CLI (main.py)
and this app share one pipeline implementation instead of duplicating it.
"""

from __future__ import annotations

import logging

import streamlit as st

from src import ui
from src.config import MAX_RUNS_PER_SESSION, MAX_TOPIC_CHARS, ConfigError
from src.pipeline import run_research_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔬",
    layout="wide",
)

ui.inject_theme()

for key, default in (
    ("result", None),
    ("run_count", 0),
):
    if key not in st.session_state:
        st.session_state[key] = default

ui.hero(
    "Researcher<span style='color:var(--primary-color)'>Agent</span>",
    "Four specialized AI agents collaborate — searching, scraping, writing, "
    "and critiquing — to deliver a research report on any topic.",
)
st.markdown("<hr class='app-divider' />", unsafe_allow_html=True)

models, temperature = ui.render_sidebar()

EXAMPLE_TOPICS = [
    "Future of LLM agents in the tech industry",
    "Latest AI agent frameworks in 2026",
    "Roadmap for AGI development in the next 5 years",
]

if "topic_value" not in st.session_state:
    st.session_state.topic_value = ""

chosen_example = st.pills("Try an example", EXAMPLE_TOPICS, label_visibility="collapsed")
if chosen_example:
    st.session_state.topic_value = chosen_example

with st.form("research_form"):
    topic = st.text_input(
        "Research topic",
        value=st.session_state.topic_value,
        placeholder="e.g. Roadmap for AGI development in the next 5 years",
        max_chars=MAX_TOPIC_CHARS,
    )
    submitted = st.form_submit_button("⚡ Run Research Pipeline", use_container_width=True)

if submitted:
    topic = topic.strip()
    if not topic:
        st.warning("Please enter a research topic first.")
    elif st.session_state.run_count >= MAX_RUNS_PER_SESSION:
        st.warning(
            f"You've reached the limit of {MAX_RUNS_PER_SESSION} runs for this session. "
            "Please refresh the page to start a new session."
        )
    else:
        st.session_state.run_count += 1

        try:
            with st.status("Running pipeline…", expanded=True) as status_box:

                def on_event(step: str, status: str, box=status_box) -> None:
                    icon = {"running": "🔄", "done": "✅", "error": "❌"}.get(status, "•")
                    box.write(f"{icon} {ui.STEP_LABELS.get(step, step)}")

                result = run_research_pipeline(
                    topic, models=models, temperature=temperature, on_event=on_event
                )
                if result.ok:
                    status_box.update(label="Pipeline complete", state="complete")
                else:
                    status_box.update(label="Pipeline finished with errors", state="error")
            st.session_state.result = result
        except ConfigError as exc:
            st.error(str(exc))
        except Exception:
            logger.exception("Pipeline run failed")
            st.error(
                "Something went wrong while running the pipeline. "
                "Please try again in a moment."
            )

if st.session_state.result is not None:
    st.markdown("<hr class='app-divider' />", unsafe_allow_html=True)
    st.subheader("Results")
    ui.render_result(st.session_state.result)

st.caption("ResearchAgent · Powered by a LangChain multi-agent pipeline · Built with Streamlit")
