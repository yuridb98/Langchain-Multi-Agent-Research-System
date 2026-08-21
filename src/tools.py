"""Web search and URL scraping tools used by the Search and Reader agents.

scrape_url includes an SSRF guard: the URL it fetches is chosen by an LLM
out of search results, so it's attacker-influenceable input. Without the
guard, a crafted search result could steer a fetch at a cloud metadata
endpoint or an internal service. See _assert_public_url.
"""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
from urllib.parse import urljoin, urlparse

import requests
import trafilatura
from bs4 import BeautifulSoup
from langchain.tools import tool
from readability import Document
from tavily import TavilyClient

from .config import REQUEST_TIMEOUT, SCRAPE_CHAR_LIMIT, require_secret

logger = logging.getLogger(__name__)

# ---- Web search (Tavily) ----

_tavily_client: TavilyClient | None = None


def _tavily() -> TavilyClient:
    global _tavily_client
    if _tavily_client is None:
        _tavily_client = TavilyClient(api_key=require_secret("TAVILY_API_KEY"))
    return _tavily_client


@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information on a topic. Returns titles, URLs and snippets."""
    logger.info("web_search query=%s", query[:200])

    try:
        results = _tavily().search(query=query, max_results=5)
    except Exception as exc:  # Tavily auth/quota/network errors, etc.
        logger.warning("web_search failed: %s", exc)
        return f"Search failed: {exc}"

    out = [
        f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content'][:300]}\n"
        for r in results.get("results", [])
    ]
    return "\n----\n".join(out) if out else "No search results found."


# ---- URL scraping ----

_ALLOWED_SCHEMES = {"http", "https"}
_MAX_REDIRECTS = 5
_STRIP_TAGS = ["script", "style", "nav", "footer", "header", "aside", "form"]
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
}


class UnsafeUrlError(Exception):
    """Raised when a URL resolves to a disallowed (private/internal) address."""


def _assert_public_url(url: str) -> None:
    parsed = urlparse(url)

    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise UnsafeUrlError(f"Unsupported URL scheme: {parsed.scheme!r}")

    host = parsed.hostname
    if not host:
        raise UnsafeUrlError("URL has no host")

    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise UnsafeUrlError(f"Could not resolve host: {host}") from exc

    for info in infos:
        addr = info[4][0]
        ip = ipaddress.ip_address(addr)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise UnsafeUrlError(f"URL resolves to a disallowed address: {addr}")


def _safe_get(url: str) -> requests.Response:
    """GET a URL, validating every hop (including redirects) against the SSRF guard."""
    current_url = url
    for _ in range(_MAX_REDIRECTS + 1):
        _assert_public_url(current_url)
        response = requests.get(
            current_url, headers=_HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=False
        )
        if response.is_redirect or response.is_permanent_redirect:
            next_url = response.headers.get("Location")
            if not next_url:
                break
            current_url = urljoin(current_url, next_url)
            continue
        response.raise_for_status()
        return response

    raise UnsafeUrlError("Too many redirects")


def _extract(html: str) -> str | None:
    extracted = trafilatura.extract(html, include_comments=False, include_tables=False)
    if extracted and len(extracted.strip()) > 200:
        return re.sub(r"\s+", " ", extracted)

    doc = Document(html)
    soup = BeautifulSoup(doc.summary(), "html.parser")
    for tag in soup(_STRIP_TAGS):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    if text and len(text.strip()) > 200:
        return re.sub(r"\s+", " ", text)

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_STRIP_TAGS):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    cleaned = re.sub(r"\s+", " ", text)
    return cleaned or None


@tool
def scrape_url(url: str) -> str:
    """Scrape and extract clean readable content from a URL. Uses multiple extraction strategies for better reliability."""
    logger.info("scrape_url url=%s", url[:200])

    try:
        response = _safe_get(url)
    except UnsafeUrlError as exc:
        logger.warning("scrape_url blocked: %s", exc)
        return f"Refused to scrape this URL: {exc}"
    except requests.exceptions.Timeout:
        return "Request timed out while scraping the URL."
    except requests.exceptions.HTTPError as exc:
        return f"HTTP error occurred: {exc}"
    except Exception as exc:
        return f"Could not scrape URL: {exc}"

    text = _extract(response.text)
    if not text:
        return "Could not extract meaningful content from the page."

    return text[:SCRAPE_CHAR_LIMIT]
