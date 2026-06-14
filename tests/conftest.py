"""Shared pytest fixtures.

LLM mock strategy: tests that run the full graph must not require API keys,
GPU, or a running Qdrant. We patch model_for_agent and get_retriever at the
agents level so each agent that calls the real LLM / retriever gets a mock.
The mocks satisfy the same structural invariants as the real agents — the tests
keep verifying anti-bias contracts, just without network / hardware calls.
"""

from unittest.mock import MagicMock

import pytest

from meddx.agents.hypothesis import _HypothesisItem, _HypothesisList
from meddx.agents.evidence import _EvidenceQueriesResult, _QueryPair

# ---------------------------------------------------------------------------
# Hypothesis mock
# ---------------------------------------------------------------------------

_MOCK_HYPOTHESES = _HypothesisList(
    hypotheses=[
        _HypothesisItem(
            name="Acute coronary syndrome",
            rationale="Retrosternal chest pain + smoking + family history of MI.",
            organ_system="cardiovascular",
            is_must_not_miss=True,
        ),
        _HypothesisItem(
            name="Pulmonary embolism",
            rationale="Episodic pleuritic pain, risk factors present.",
            organ_system="respiratory",
            is_must_not_miss=True,
        ),
        _HypothesisItem(
            name="Gastroesophageal reflux disease",
            rationale="Retrosternal burning worse after meals.",
            organ_system="gastrointestinal",
            is_must_not_miss=False,
        ),
        _HypothesisItem(
            name="Musculoskeletal chest pain",
            rationale="Pain reproducible with movement/palpation.",
            organ_system="musculoskeletal",
            is_must_not_miss=False,
        ),
        _HypothesisItem(
            name="Panic disorder",
            rationale="Episodic pattern with autonomic features.",
            organ_system="psychiatric",
            is_must_not_miss=False,
        ),
    ]
)


def _make_hypothesis_llm_mock():
    chain = MagicMock()
    chain.invoke.return_value = _MOCK_HYPOTHESES
    model = MagicMock()
    model.with_structured_output.return_value = chain
    return model


@pytest.fixture()
def mock_hypothesis_llm(monkeypatch):
    """Patch model_for_agent inside hypothesis module — no API key needed."""
    mock_model = _make_hypothesis_llm_mock()
    monkeypatch.setattr(
        "meddx.agents.hypothesis.model_for_agent",
        lambda _agent_name: mock_model,
    )
    return mock_model.with_structured_output.return_value


# ---------------------------------------------------------------------------
# Evidence mock — LLM query generator
# ---------------------------------------------------------------------------

def _make_evidence_queries_for(hypothesis_ids: list[str]) -> _EvidenceQueriesResult:
    """Build mock query pairs covering every hypothesis id."""
    pairs = [
        _QueryPair(
            hypothesis_id=hid,
            supporting_query=f"supporting evidence for hypothesis {hid}",
            refuting_query=f"refuting evidence against hypothesis {hid}",
        )
        for hid in hypothesis_ids
    ]
    return _EvidenceQueriesResult(pairs=pairs)


# Known IDs from the mock differential above (h1..h5 after hypothesis_node assigns them)
_MOCK_HYP_IDS = ["h1", "h2", "h3", "h4", "h5"]


def _make_evidence_llm_mock():
    result = _make_evidence_queries_for(_MOCK_HYP_IDS)
    chain = MagicMock()
    chain.invoke.return_value = result
    model = MagicMock()
    model.with_structured_output.return_value = chain
    return model


# ---------------------------------------------------------------------------
# Evidence mock — Qdrant retriever
# ---------------------------------------------------------------------------

_MOCK_PAYLOAD_TEMPLATE = {
    "pmid": None,
    "doi": None,
    "title": "Mock Article",
    "journal": "Mock Journal",
    "year": 2024,
    "study_type": "rct",
    "evidence_level": 3,
    "section": "results",
    "text": "Mock text content.",
    "chunk_index": 0,
}


def _make_mock_retriever(payloads_per_call: int = 2):
    """Return a mock HybridRetriever that returns *payloads_per_call* payloads
    per search() call with unique PMIDs so deduplication doesn't collapse them."""
    call_counter = {"n": 0}

    def fake_search(query, k=5, max_evidence_level=None):
        results = []
        for i in range(payloads_per_call):
            call_counter["n"] += 1
            payload = dict(_MOCK_PAYLOAD_TEMPLATE)
            payload["pmid"] = f"mock{call_counter['n']:06d}"
            payload["doi"] = f"10.0000/mock.{call_counter['n']}"
            payload["title"] = f"Mock Article {call_counter['n']}: {query[:30]}"
            results.append(payload)
        return results

    mock = MagicMock()
    mock.is_ready.return_value = True
    mock.search.side_effect = fake_search
    return mock


@pytest.fixture()
def mock_evidence(monkeypatch):
    """Patch both the LLM query generator and the retriever inside the evidence
    module. Returns (mock_llm_chain, mock_retriever) for assertion access."""
    mock_model = _make_evidence_llm_mock()
    monkeypatch.setattr(
        "meddx.agents.evidence.model_for_agent",
        lambda _agent_name: mock_model,
    )
    mock_retriever = _make_mock_retriever()
    monkeypatch.setattr(
        "meddx.agents.evidence.get_retriever",
        lambda: mock_retriever,
    )
    return mock_model.with_structured_output.return_value, mock_retriever
