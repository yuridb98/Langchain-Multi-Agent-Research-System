"""Agent / chain factories.

Each builder takes an already-constructed ``llm`` (see ``llm.make_llm``)
instead of closing over a shared module-level singleton. This is what makes
per-agent model selection possible: the caller (pipeline) decides which
model each agent gets, rather than all four sharing one hardcoded instance.
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from ..tools.scraper import scrape_url
from ..tools.search import web_search
from .prompts import (
    READER_SYSTEM_PROMPT,
    SEARCH_SYSTEM_PROMPT,
    critic_prompt,
    writer_prompt,
)


def build_search_agent(llm: BaseChatModel):
    return create_agent(model=llm, tools=[web_search], system_prompt=SEARCH_SYSTEM_PROMPT)


def build_reader_agent(llm: BaseChatModel):
    return create_agent(model=llm, tools=[scrape_url], system_prompt=READER_SYSTEM_PROMPT)


def build_writer_chain(llm: BaseChatModel | Runnable) -> Runnable:
    return writer_prompt | llm | StrOutputParser()


def build_critic_chain(llm: BaseChatModel | Runnable) -> Runnable:
    return critic_prompt | llm | StrOutputParser()
