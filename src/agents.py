"""LLM client construction and the four agent/chain factories.

Each build_* function takes an already-constructed llm, so the pipeline can
give every agent a different model instead of sharing one instance.
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from .config import ALLOWED_MODEL_IDS, MAX_RETRIES, REQUEST_TIMEOUT, ConfigError, require_secret
from .prompts import READER_SYSTEM_PROMPT, SEARCH_SYSTEM_PROMPT, critic_prompt, writer_prompt
from .tools import scrape_url, web_search


def make_llm(model_id: str, temperature: float = 0.0) -> ChatOpenAI:
    if model_id not in ALLOWED_MODEL_IDS:
        raise ConfigError(f"Model '{model_id}' is not in the allowed model list.")

    return ChatOpenAI(
        model=model_id,
        temperature=temperature,
        api_key=SecretStr(require_secret("OPENAI_API_KEY")),
        timeout=REQUEST_TIMEOUT,
        max_retries=MAX_RETRIES,
    )


def build_search_agent(llm: BaseChatModel):
    return create_agent(model=llm, tools=[web_search], system_prompt=SEARCH_SYSTEM_PROMPT)


def build_reader_agent(llm: BaseChatModel):
    return create_agent(model=llm, tools=[scrape_url], system_prompt=READER_SYSTEM_PROMPT)


def build_writer_chain(llm: BaseChatModel | Runnable) -> Runnable:
    return writer_prompt | llm | StrOutputParser()


def build_critic_chain(llm: BaseChatModel | Runnable) -> Runnable:
    return critic_prompt | llm | StrOutputParser()
