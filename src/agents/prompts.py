"""Prompt templates for all four agents/chains.

Wording for the writer and critic prompts is unchanged from the original
implementation. The search and reader ReAct agents previously had no system
prompt at all (only an ad-hoc human message built by the caller) which made
their tool-use behavior inconsistent -- system prompts were added for both.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

SEARCH_SYSTEM_PROMPT = (
    "You are a meticulous research assistant. Use the web_search tool to find "
    "recent, reliable, and relevant information about the user's topic. "
    "Prefer primary sources and reputable outlets. After searching, summarize "
    "the most useful findings and include the source URLs."
)

READER_SYSTEM_PROMPT = (
    "You are a careful research assistant. Given a set of search results, pick "
    "the single most relevant and reliable URL and use the scrape_url tool to "
    "read it in depth. Summarize the key facts, figures, and quotes you find, "
    "and note the URL you scraped."
)

WRITER_SYSTEM_PROMPT = (
    "You are an expert research writer. Write clear, structured and insightful reports."
)

WRITER_HUMAN_TEMPLATE = """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""

CRITIC_SYSTEM_PROMPT = (
    "You are a sharp and constructive research critic. Be honest and specific."
)

CRITIC_HUMAN_TEMPLATE = """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""

writer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", WRITER_SYSTEM_PROMPT),
        ("human", WRITER_HUMAN_TEMPLATE),
    ]
)

critic_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CRITIC_SYSTEM_PROMPT),
        ("human", CRITIC_HUMAN_TEMPLATE),
    ]
)
