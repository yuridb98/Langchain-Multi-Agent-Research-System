import pytest

from src import tools
from src.tools import UnsafeUrlError, _assert_public_url, scrape_url


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://localhost/",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata endpoint
        "http://10.0.0.5/",
        "http://192.168.1.1/",
        "http://[::1]/",
        "ftp://example.com/",
        "file:///etc/passwd",
    ],
)
def test_assert_public_url_blocks_unsafe_targets(url):
    with pytest.raises(UnsafeUrlError):
        _assert_public_url(url)


def test_assert_public_url_allows_public_ip_literal():
    # 93.184.216.34 is a public unicast address (example.com); no DNS needed.
    _assert_public_url("http://93.184.216.34/")


def test_scrape_url_tool_refuses_unsafe_target(monkeypatch):
    result = scrape_url.invoke({"url": "http://127.0.0.1/admin"})
    assert "Refused" in result


def test_scrape_url_extracts_via_trafilatura(monkeypatch):
    html = "<html><body>" + ("<p>Interesting article content. </p>" * 20) + "</body></html>"

    class FakeResponse:
        text = html

        def raise_for_status(self):
            pass

        is_redirect = False
        is_permanent_redirect = False

    monkeypatch.setattr(tools, "_assert_public_url", lambda url: None)
    monkeypatch.setattr(tools.requests, "get", lambda *a, **k: FakeResponse())

    result = scrape_url.invoke({"url": "http://example.com/article"})
    assert "Interesting article content" in result


def test_scrape_url_reports_timeout(monkeypatch):
    import requests

    monkeypatch.setattr(tools, "_assert_public_url", lambda url: None)

    def raise_timeout(*a, **k):
        raise requests.exceptions.Timeout()

    monkeypatch.setattr(tools.requests, "get", raise_timeout)

    result = scrape_url.invoke({"url": "http://example.com/slow"})
    assert "timed out" in result.lower()
