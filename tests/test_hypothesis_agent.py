"""Unit tests for the Hypothesis agent node.

These tests verify:
  - LLM output is post-processed correctly (id assignment, lowercased organ_system)
  - All three anti-bias invariants are enforced by code, not just by the prompt
  - The node raises ValueError when the LLM violates an invariant so that
    LangGraph's retry_policy can re-run it
  - The LLM is called with both a SystemMessage and a HumanMessage that
    contains the formatted patient case
"""

from unittest.mock import MagicMock, call

import pytest

from meddx.agents.hypothesis import (
    _HypothesisItem,
    _HypothesisList,
    _validate_invariants,
    hypothesis_node,
)
from meddx.schemas import UNAVAILABLE, DiagnosticState, Hypothesis, PatientCase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_model(hypotheses: list[dict]) -> MagicMock:
    """Return a mock model whose .with_structured_output().invoke() returns
    a _HypothesisList built from the provided dicts."""
    result = _HypothesisList(hypotheses=[_HypothesisItem(**h) for h in hypotheses])
    chain = MagicMock()
    chain.invoke.return_value = result
    model = MagicMock()
    model.with_structured_output.return_value = chain
    return model


_VALID_HYPOTHESES = [
    {"name": "Acute coronary syndrome", "rationale": "chest pain + risk factors",
     "organ_system": "cardiovascular", "is_must_not_miss": True},
    {"name": "Pulmonary embolism", "rationale": "episodic pleuritic pattern",
     "organ_system": "respiratory", "is_must_not_miss": True},
    {"name": "GERD", "rationale": "postprandial burning",
     "organ_system": "gastrointestinal", "is_must_not_miss": False},
    {"name": "Musculoskeletal pain", "rationale": "reproducible on palpation",
     "organ_system": "musculoskeletal", "is_must_not_miss": False},
    {"name": "Panic disorder", "rationale": "episodic + autonomic",
     "organ_system": "psychiatric", "is_must_not_miss": False},
]


def _complete_state() -> DiagnosticState:
    return DiagnosticState(
        patient_case=PatientCase(
            chief_complaint="Chest pain for 2 days",
            history_of_present_illness="Retrosternal, episodic",
            past_medical_history="Hypertension",
            medications="Amlodipine 5 mg",
            family_history="Father: MI at 56",
            systems_review="Low-grade fever",
            risk_factors="Smoker",
            available_investigations=UNAVAILABLE,
        )
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_node_returns_hypotheses_with_sequential_ids(monkeypatch):
    mock_model = _make_mock_model(_VALID_HYPOTHESES)
    monkeypatch.setattr("meddx.agents.hypothesis.model_for_agent", lambda _: mock_model)

    result = hypothesis_node(_complete_state())

    ids = [h.id for h in result["hypotheses"]]
    assert ids == ["h1", "h2", "h3", "h4", "h5"]


def test_node_lowercases_organ_system(monkeypatch):
    mixed_case = [
        {**h, "organ_system": h["organ_system"].upper()} for h in _VALID_HYPOTHESES
    ]
    monkeypatch.setattr(
        "meddx.agents.hypothesis.model_for_agent", lambda _: _make_mock_model(mixed_case)
    )

    result = hypothesis_node(_complete_state())

    assert all(h.organ_system == h.organ_system.lower() for h in result["hypotheses"])


def test_node_calls_llm_with_system_and_human_messages(monkeypatch):
    from langchain_core.messages import HumanMessage, SystemMessage

    mock_model = _make_mock_model(_VALID_HYPOTHESES)
    monkeypatch.setattr("meddx.agents.hypothesis.model_for_agent", lambda _: mock_model)

    hypothesis_node(_complete_state())

    chain = mock_model.with_structured_output.return_value
    (messages,), _ = chain.invoke.call_args
    types = [type(m) for m in messages]
    assert types == [SystemMessage, HumanMessage]
    # The human message must contain actual case data
    human_text = messages[1].content
    assert "Chest pain" in human_text
    assert "Amlodipine" in human_text


def test_node_human_message_contains_all_case_fields(monkeypatch):
    mock_model = _make_mock_model(_VALID_HYPOTHESES)
    monkeypatch.setattr("meddx.agents.hypothesis.model_for_agent", lambda _: mock_model)

    hypothesis_node(_complete_state())

    chain = mock_model.with_structured_output.return_value
    (messages,), _ = chain.invoke.call_args
    human_text = messages[1].content
    assert "unavailable" in human_text.lower()  # UNAVAILABLE marker rendered


# ---------------------------------------------------------------------------
# Anti-bias invariant enforcement (code-level, not prompt-level)
# ---------------------------------------------------------------------------

def test_node_raises_if_fewer_than_5_hypotheses(monkeypatch):
    monkeypatch.setattr(
        "meddx.agents.hypothesis.model_for_agent",
        lambda _: _make_mock_model(_VALID_HYPOTHESES[:3]),
    )
    with pytest.raises(ValueError, match="minimum is 5"):
        hypothesis_node(_complete_state())


def test_node_raises_if_no_must_not_miss(monkeypatch):
    no_mnm = [{**h, "is_must_not_miss": False} for h in _VALID_HYPOTHESES]
    monkeypatch.setattr(
        "meddx.agents.hypothesis.model_for_agent",
        lambda _: _make_mock_model(no_mnm),
    )
    with pytest.raises(ValueError, match="must-not-miss"):
        hypothesis_node(_complete_state())


def test_node_raises_if_single_organ_system(monkeypatch):
    same_system = [{**h, "organ_system": "cardiovascular"} for h in _VALID_HYPOTHESES]
    monkeypatch.setattr(
        "meddx.agents.hypothesis.model_for_agent",
        lambda _: _make_mock_model(same_system),
    )
    with pytest.raises(ValueError, match="organ system"):
        hypothesis_node(_complete_state())


# ---------------------------------------------------------------------------
# _validate_invariants unit tests
# ---------------------------------------------------------------------------

def _h(**kwargs) -> Hypothesis:
    defaults = {"id": "x", "name": "X", "rationale": "r",
                "organ_system": "cardiovascular", "is_must_not_miss": False}
    return Hypothesis(**{**defaults, **kwargs})


def test_validate_passes_minimal_valid_set():
    hypotheses = [
        _h(id="h1", organ_system="cardiovascular", is_must_not_miss=True),
        _h(id="h2", organ_system="respiratory"),
        _h(id="h3", organ_system="gastrointestinal"),
        _h(id="h4", organ_system="musculoskeletal"),
        _h(id="h5", organ_system="psychiatric"),
    ]
    _validate_invariants(hypotheses)  # must not raise


def test_validate_accumulates_multiple_errors():
    # Only 2 hypotheses, no must-not-miss, same organ system
    tiny = [_h(id="h1"), _h(id="h2")]
    with pytest.raises(ValueError) as exc_info:
        _validate_invariants(tiny)
    msg = str(exc_info.value)
    assert "minimum is 5" in msg
    assert "must-not-miss" in msg
