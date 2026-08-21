"""Web search tool (Tavily).

The Tavily client is constructed lazily on first use rather than at import
time, so importing this module never requires ``TAVILY_API_KEY`` to be set
(useful for tests, and avoids crashing app startup on a missing key).
"""

from __future__ import annotations

from functools import lru_cache

from langchain.tools import tool
from tavily import TavilyClient

from ..config.settings import require_secret
from ..core.logging import get_logger, truncate

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _client() -> TavilyClient:
    return TavilyClient(api_key=require_secret("TAVILY_API_KEY"))


@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information on a topic. Returns titles, URLs and snippets."""
    logger.info("web_search query=%s", truncate(query, 200))

    try:
        results = _client().search(query=query, max_results=5)
    except Exception as exc:  # Tavily auth/quota/network errors, etc.
        logger.warning("web_search failed: %s", exc)
        return f"Search failed: {exc}"

    out = []
    for r in results.get("results", []):
        out.append(f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content'][:300]}\n")

    if not out:
        return "No search results found."

    return "\n----\n".join(out)
