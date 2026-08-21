"""URL scraping tool.

Behavior (fetch + three-strategy extraction cascade) is unchanged from the
original implementation. What's new is an SSRF guard: the URL to scrape is
picked by an LLM out of search results, i.e. it is attacker-influenceable
input. Without a guard, a crafted search result (or a malicious/compromised
page linked from one) could steer the reader agent into fetching
``http://169.254.169.254/...`` (cloud metadata endpoints) or an internal
service, and return the response as "scraped content". The guard resolves
the hostname and rejects loopback/private/link-local/reserved addresses and
non-http(s) schemes, and re-validates on every redirect hop.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from urllib.parse import urlparse

import requests
import trafilatura
from bs4 import BeautifulSoup
from langchain.tools import tool
from readability import Document

from ..config.settings import REQUEST_TIMEOUT, SCRAPE_CHAR_LIMIT
from ..core.logging import get_logger, truncate

logger = get_logger(__name__)

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
            current_url = requests.compat.urljoin(current_url, next_url)
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
    logger.info("scrape_url url=%s", truncate(url, 200))

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
