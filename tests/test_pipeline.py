from dataclasses import dataclass

from src import pipeline as pipeline_module
from src.config import ALLOWED_MODEL_IDS
from src.pipeline import run_research_pipeline


@dataclass
class _FakeMessage:
    content: str


class _FakeAgent:
    def __init__(self, content: str, fail: bool = False):
        self._content = content
        self._fail = fail

    def invoke(self, _payload):
        if self._fail:
            raise RuntimeError("boom")
        return {"messages": [_FakeMessage(self._content)]}


class _FakeChain:
    def __init__(self, output: str, fail: bool = False):
        self._output = output
        self._fail = fail

    def invoke(self, _payload):
        if self._fail:
            raise RuntimeError("boom")
        return self._output


class _FakeLLM:
    def bind(self, **_kwargs):
        return self


def _patch_common(monkeypatch, *, reader_fails=False, search_fails=False, writer_fails=False):
    monkeypatch.setattr(pipeline_module, "make_llm", lambda model_id, temperature: _FakeLLM())
    monkeypatch.setattr(
        pipeline_module,
        "build_search_agent",
        lambda llm: _FakeAgent("search output", fail=search_fails),
    )
    monkeypatch.setattr(
        pipeline_module,
        "build_reader_agent",
        lambda llm: _FakeAgent("reader output", fail=reader_fails),
    )
    monkeypatch.setattr(
        pipeline_module,
        "build_writer_chain",
        lambda llm: _FakeChain("final report", fail=writer_fails),
    )
    monkeypatch.setattr(
        pipeline_module, "build_critic_chain", lambda llm: _FakeChain("critic feedback")
    )


def test_successful_run_populates_all_fields_and_events(monkeypatch):
    _patch_common(monkeypatch)
    events = []

    result = run_research_pipeline(
        "test topic", on_event=lambda step, status: events.append((step, status))
    )

    assert result.search == "search output"
    assert result.reader == "reader output"
    assert result.report == "final report"
    assert result.feedback == "critic feedback"
    assert result.errors == []
    assert result.ok

    assert events == [
        ("search", "running"),
        ("search", "done"),
        ("reader", "running"),
        ("reader", "done"),
        ("writer", "running"),
        ("writer", "done"),
        ("critic", "running"),
        ("critic", "done"),
    ]


def test_reader_failure_does_not_stop_the_pipeline(monkeypatch):
    _patch_common(monkeypatch, reader_fails=True)

    result = run_research_pipeline("test topic")

    assert result.search == "search output"
    assert result.reader == ""
    assert result.report == "final report"  # writer still ran
    assert any("Reader step failed" in e for e in result.errors)
    assert not result.ok


def test_search_failure_stops_the_pipeline(monkeypatch):
    _patch_common(monkeypatch, search_fails=True)

    result = run_research_pipeline("test topic")

    assert result.report == ""
    assert result.feedback == ""
    assert any("Search step failed" in e for e in result.errors)


def test_disallowed_model_falls_back_to_default(monkeypatch):
    _patch_common(monkeypatch)

    result = run_research_pipeline("test topic", models={"writer": "gpt-5.5-pro"})

    for model_id in result.models.values():
        assert model_id in ALLOWED_MODEL_IDS
    assert result.models["writer"] != "gpt-5.5-pro"
