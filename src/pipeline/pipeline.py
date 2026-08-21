"""The single research-pipeline implementation.

Previously this orchestration existed in two places -- here, and again
inline inside app.py -- and had already drifted (the UI copy had a
``"content "`` typo that silently broke the Reader step). Now both the
Streamlit UI and the CLI (main.py) call ``run_research_pipeline`` directly.

Per-agent models are resolved through ``config.models.resolve_model``, so
even if a caller passes an unexpected model id, the pipeline runs with a
safe, allowed default instead of raising.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ..agents.agents import (
    build_critic_chain,
    build_reader_agent,
    build_search_agent,
    build_writer_chain,
)
from ..agents.llm import make_llm
from ..config.models import AGENT_DEFAULTS, resolve_model
from ..config.settings import WRITER_MAX_TOKENS
from ..core.logging import get_logger, truncate

logger = get_logger(__name__)

# (step, status) -- status is one of "running", "done", "error"
EventCallback = Callable[[str, str], None]


@dataclass
class ResearchResult:
    topic: str
    models: dict[str, str]
    search: str = ""
    reader: str = ""
    report: str = ""
    feedback: str = ""
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _emit(on_event: EventCallback | None, step: str, status: str) -> None:
    if on_event:
        on_event(step, status)


def run_research_pipeline(
    topic: str,
    models: dict[str, str] | None = None,
    temperature: float = 0.0,
    on_event: EventCallback | None = None,
) -> ResearchResult:
    """Run the search -> read -> write -> critique pipeline for `topic`.

    `models` maps agent name ("search", "reader", "writer", "critic") to a
    requested model id; unknown/disallowed ids fall back to that agent's
    default. Missing entries also use the default.
    """
    requested = models or {}
    resolved = {
        agent: resolve_model(agent, requested.get(agent)) for agent in AGENT_DEFAULTS
    }
    result = ResearchResult(topic=topic, models=resolved)

    logger.info("Starting pipeline topic=%s models=%s", truncate(topic, 200), resolved)

    # ---- Step 1: search ----
    _emit(on_event, "search", "running")
    try:
        search_llm = make_llm(resolved["search"], temperature)
        search_agent = build_search_agent(search_llm)
        search_response = search_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Find recent, reliable and detailed information about: {topic}",
                    }
                ]
            }
        )
        result.search = search_response["messages"][-1].content
        _emit(on_event, "search", "done")
    except Exception as exc:
        logger.exception("Search step failed")
        result.errors.append(f"Search step failed: {exc}")
        _emit(on_event, "search", "error")
        return result

    # ---- Step 2: reader ----
    _emit(on_event, "reader", "running")
    try:
        reader_llm = make_llm(resolved["reader"], temperature)
        reader_agent = build_reader_agent(reader_llm)
        reader_response = reader_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Based on the following search results about '{topic}', "
                            "pick the most relevant URL and scrape it for deeper content.\n\n"
                            f"Search Results:\n{result.search[:800]}"
                        ),
                    }
                ]
            }
        )
        result.reader = reader_response["messages"][-1].content
        _emit(on_event, "reader", "done")
    except Exception as exc:
        logger.exception("Reader step failed")
        result.errors.append(f"Reader step failed: {exc}")
        _emit(on_event, "reader", "error")
        # The writer can still produce something from search results alone.

    # ---- Step 3: writer ----
    _emit(on_event, "writer", "running")
    try:
        writer_llm = make_llm(resolved["writer"], temperature).bind(
            max_tokens=WRITER_MAX_TOKENS
        )
        writer_chain = build_writer_chain(writer_llm)
        research_combined = (
            f"SEARCH RESULTS:\n{result.search}\n\n"
            f"DETAILED SCRAPED CONTENT:\n{result.reader}"
        )
        result.report = writer_chain.invoke({"topic": topic, "research": research_combined})
        _emit(on_event, "writer", "done")
    except Exception as exc:
        logger.exception("Writer step failed")
        result.errors.append(f"Writer step failed: {exc}")
        _emit(on_event, "writer", "error")
        return result

    # ---- Step 4: critic ----
    _emit(on_event, "critic", "running")
    try:
        critic_llm = make_llm(resolved["critic"], temperature)
        critic_chain = build_critic_chain(critic_llm)
        result.feedback = critic_chain.invoke({"report": result.report})
        _emit(on_event, "critic", "done")
    except Exception as exc:
        logger.exception("Critic step failed")
        result.errors.append(f"Critic step failed: {exc}")
        _emit(on_event, "critic", "error")

    return result
