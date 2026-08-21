from src.config import (
    AGENT_DEFAULTS,
    ALLOWED_MODEL_IDS,
    CHEAP_MODELS,
    resolve_model,
)


def test_every_default_is_allowed():
    for agent, model_id in AGENT_DEFAULTS.items():
        assert model_id in ALLOWED_MODEL_IDS, f"{agent} default {model_id} is not allowlisted"


def test_no_duplicate_model_ids():
    ids = [m.id for m in CHEAP_MODELS]
    assert len(ids) == len(set(ids))


def test_resolve_model_accepts_allowed_id():
    allowed = next(iter(ALLOWED_MODEL_IDS))
    assert resolve_model("writer", allowed) == allowed


def test_resolve_model_rejects_expensive_model():
    assert resolve_model("writer", "gpt-5.5-pro") == AGENT_DEFAULTS["writer"]
    assert resolve_model("search", "o3") == AGENT_DEFAULTS["search"]


def test_resolve_model_rejects_arbitrary_string():
    assert resolve_model("critic", "not-a-real-model") == AGENT_DEFAULTS["critic"]


def test_resolve_model_handles_missing_request():
    assert resolve_model("reader", None) == AGENT_DEFAULTS["reader"]


def test_resolve_model_unknown_agent_falls_back_to_first_cheap_model():
    assert resolve_model("unknown-agent", None) == CHEAP_MODELS[0].id
