import pytest

from src.config.settings import get_secret, require_secret
from src.core.errors import ConfigError


def test_get_secret_reads_from_environment(monkeypatch):
    monkeypatch.delenv("MY_TEST_SECRET", raising=False)
    monkeypatch.setenv("MY_TEST_SECRET", "value-from-env")
    assert get_secret("MY_TEST_SECRET") == "value-from-env"


def test_get_secret_returns_none_when_unset(monkeypatch):
    monkeypatch.delenv("DEFINITELY_UNSET_SECRET", raising=False)
    assert get_secret("DEFINITELY_UNSET_SECRET") is None


def test_require_secret_raises_config_error_when_missing(monkeypatch):
    monkeypatch.delenv("DEFINITELY_UNSET_SECRET", raising=False)
    with pytest.raises(ConfigError):
        require_secret("DEFINITELY_UNSET_SECRET")


def test_require_secret_returns_value_when_present(monkeypatch):
    monkeypatch.setenv("MY_TEST_SECRET", "value-from-env")
    assert require_secret("MY_TEST_SECRET") == "value-from-env"


def test_config_error_message_does_not_leak_value(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-appear")
    monkeypatch.delenv("SOME_OTHER_SECRET", raising=False)
    try:
        require_secret("SOME_OTHER_SECRET")
    except ConfigError as exc:
        assert "sk-should-not-appear" not in str(exc)
