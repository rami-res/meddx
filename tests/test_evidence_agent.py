"""Unit tests for the Evidence agent node.

Covers:
  - Query generation (LLM call structure, fallback for missing hypotheses)
  - Payload → Citation conversion
  - Deduplication within stance lists
  - Symmetric retrieval invariant (supporting AND refuting for every hypothesis)
  - Graceful degradation when the corpus is empty (is_ready() = False)
  - search() called twice per hypothesis (FOR and AGAINST separately)
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, call, patch

import pytest

# Prevent heavy optional deps from failing on import
for _mod in ("qdrant_client", "qdrant_client.models", "FlagEmbedding"):
    sys.modules.setdefault(_mod, MagicMock())

from meddx.agents.evidence import (
    TOP_K,
    _EvidenceQueriesResult,
    _QueryPair,
    _deduplicate,
    _fallback_queries,
    _format_case_summary,
    _generate_queries,
    _payload_to_citation,
    evidence_node,
)
from meddx.schemas import (
    UNAVAILABLE,
    Citation,
    DiagnosticState,
    Hypothesis,
    PatientCase,
    Phase,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_hypotheses(ids: list[str] | None = None) -> list[Hypothesis]:
    ids = ids or ["h1", "h2", "h3"]
    systems = ["cardiovascular", "respiratory", "gastrointestinal"]
    return [
        Hypothesis(
            id=hid,
            name=f"Hypothesis {hid}",
            rationale=f"Rationale for {hid}",
            organ_system=systems[i % len(systems)],
            is_must_not_miss=(i == 0),
        )
        for i, hid in enumerate(ids)
    ]


def _complete_state(hyp_ids: list[str] | None = None) -> DiagnosticState:
    return DiagnosticState(
        patient_case=PatientCase(
            chief_complaint="Chest pain",
            history_of_present_illness="Episodic, retrosternal",
            past_medical_history="Hypertension",
            medications="Amlodipine",
            family_history="Father: MI",
            systems_review="Fever",
            risk_factors="Smoker",
            available_investigations=UNAVAILABLE,
        ),
        hypotheses=_make_hypotheses(hyp_ids),
    )


def _make_query_result(hypothesis_ids: list[str]) -> _EvidenceQueriesResult:
    return _EvidenceQueriesResult(
        pairs=[
            _QueryPair(
                hypothesis_id=hid,
                supporting_query=f"supporting query {hid}",
                refuting_query=f"refuting query {hid}",
            )
            for hid in hypothesis_ids
        ]
    )


def _make_fake_payload(n: int) -> dict:
    return {
        "pmid": f"PMID{n:06d}",
        "doi": f"10.0000/test.{n}",
        "title": f"Test Article {n}",
        "journal": "Test Journal",
        "year": 2024,
        "study_type": "rct",
        "evidence_level": 3,
        "section": "results",
        "text": f"Content of article {n}.",
        "chunk_index": 0,
    }


def _make_evidence_llm_mock(hypothesis_ids: list[str]):
    result = _make_query_result(hypothesis_ids)
    chain = MagicMock()
    chain.invoke.return_value = result
    model = MagicMock()
    model.with_structured_output.return_value = chain
    return model


def _make_retriever_mock(payloads_per_call: int = 2, ready: bool = True):
    counter = {"n": 0}

    def fake_search(query, k=5, max_evidence_level=None):
        results = []
        for _ in range(payloads_per_call):
            counter["n"] += 1
            p = _make_fake_payload(counter["n"])
            p["title"] = f"Article {counter['n']}: {query[:20]}"
            results.append(p)
        return results

    mock = MagicMock()
    mock.is_ready.return_value = ready
    mock.search.side_effect = fake_search
    return mock


# ---------------------------------------------------------------------------
# _fallback_queries
# ---------------------------------------------------------------------------

def test_fallback_queries_returns_two_english_strings():
    supp, ref = _fallback_queries("Acute coronary syndrome")
    assert "Acute coronary syndrome" in supp
    assert "Acute coronary syndrome" in ref
    assert supp != ref


# ---------------------------------------------------------------------------
# _payload_to_citation
# ---------------------------------------------------------------------------

def test_payload_to_citation_full_payload():
    c = _payload_to_citation(_make_fake_payload(1))
    assert c is not None
    assert c.pmid == "PMID000001"
    assert c.doi == "10.0000/test.1"
    assert c.title == "Test Article 1"
    assert c.year == 2024
    assert c.study_type == "rct"
    assert c.url is not None
    assert "PMID000001" in c.url


def test_payload_to_citation_url_uses_pmid():
    p = _make_fake_payload(5)
    c = _payload_to_citation(p)
    assert c.url == "https://europepmc.org/article/MED/PMID000005"


def test_payload_to_citation_no_pmid_url_is_none():
    p = _make_fake_payload(1)
    p["pmid"] = None
    c = _payload_to_citation(p)
    assert c.url is None


def test_payload_to_citation_missing_title_returns_none():
    p = _make_fake_payload(1)
    p["title"] = ""
    assert _payload_to_citation(p) is None


def test_payload_to_citation_missing_title_key_returns_none():
    p = {k: v for k, v in _make_fake_payload(1).items() if k != "title"}
    assert _payload_to_citation(p) is None


# ---------------------------------------------------------------------------
# _deduplicate
# ---------------------------------------------------------------------------

def _cite(pmid: str) -> Citation:
    return Citation(pmid=pmid, title="T", journal="J", year=2024, study_type="rct")


def test_deduplicate_removes_same_pmid():
    dupes = [_cite("111"), _cite("111")]
    result = _deduplicate(dupes)
    assert len(result) == 1
    assert result[0].pmid == "111"


def test_deduplicate_keeps_different_pmids():
    citations = [_cite("111"), _cite("222"), _cite("333")]
    assert _deduplicate(citations) == citations


def test_deduplicate_preserves_order_of_first_occurrence():
    c1, c2, c3 = _cite("aaa"), _cite("bbb"), _cite("aaa")
    result = _deduplicate([c1, c2, c3])
    assert [c.pmid for c in result] == ["aaa", "bbb"]


def test_deduplicate_no_identifier_citations_always_kept():
    no_id = Citation(pmid=None, doi=None, title="T", journal="J", year=2024, study_type="other")
    result = _deduplicate([no_id, no_id])
    assert len(result) == 2  # no way to deduplicate without identifiers


# ---------------------------------------------------------------------------
# _generate_queries
# ---------------------------------------------------------------------------

def test_generate_queries_returns_pair_for_every_hypothesis(monkeypatch):
    state = _complete_state(["h1", "h2", "h3"])
    mock_model = _make_evidence_llm_mock(["h1", "h2", "h3"])
    monkeypatch.setattr("meddx.agents.evidence.model_for_agent", lambda _: mock_model)

    query_map = _generate_queries(state)

    assert set(query_map) == {"h1", "h2", "h3"}
    for hid, (supp, ref) in query_map.items():
        assert isinstance(supp, str) and supp
        assert isinstance(ref, str) and ref


def test_generate_queries_falls_back_for_missing_hypothesis(monkeypatch):
    """If the LLM omits h3, we get a template fallback for it."""
    state = _complete_state(["h1", "h2", "h3"])
    # LLM only returns pairs for h1, h2
    mock_model = _make_evidence_llm_mock(["h1", "h2"])
    monkeypatch.setattr("meddx.agents.evidence.model_for_agent", lambda _: mock_model)

    query_map = _generate_queries(state)

    assert "h3" in query_map
    supp, ref = query_map["h3"]
    assert "Hypothesis h3" in supp  # from _fallback_queries(h.name)


def test_generate_queries_calls_llm_with_system_and_human_messages(monkeypatch):
    from langchain_core.messages import HumanMessage, SystemMessage

    state = _complete_state(["h1"])
    mock_model = _make_evidence_llm_mock(["h1"])
    monkeypatch.setattr("meddx.agents.evidence.model_for_agent", lambda _: mock_model)

    _generate_queries(state)

    chain = mock_model.with_structured_output.return_value
    (messages,), _ = chain.invoke.call_args
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)


def test_generate_queries_human_message_contains_hypothesis_id(monkeypatch):
    from langchain_core.messages import HumanMessage

    state = _complete_state(["h7"])
    mock_model = _make_evidence_llm_mock(["h7"])
    monkeypatch.setattr("meddx.agents.evidence.model_for_agent", lambda _: mock_model)

    _generate_queries(state)

    chain = mock_model.with_structured_output.return_value
    (messages,), _ = chain.invoke.call_args
    human_text = messages[1].content
    assert "h7" in human_text


# ---------------------------------------------------------------------------
# evidence_node — full node
# ---------------------------------------------------------------------------

def test_evidence_node_calls_search_twice_per_hypothesis(monkeypatch):
    """FOR and AGAINST must be separate retriever calls per hypothesis."""
    state = _complete_state(["h1", "h2"])
    mock_model = _make_evidence_llm_mock(["h1", "h2"])
    monkeypatch.setattr("meddx.agents.evidence.model_for_agent", lambda _: mock_model)
    mock_ret = _make_retriever_mock()
    monkeypatch.setattr("meddx.agents.evidence.get_retriever", lambda: mock_ret)

    evidence_node(state)

    # 2 hypotheses × 2 searches each = 4 calls
    assert mock_ret.search.call_count == 4


def test_evidence_node_supporting_and_refuting_for_every_hypothesis(monkeypatch):
    state = _complete_state(["h1", "h2", "h3"])
    mock_model = _make_evidence_llm_mock(["h1", "h2", "h3"])
    monkeypatch.setattr("meddx.agents.evidence.model_for_agent", lambda _: mock_model)
    mock_ret = _make_retriever_mock(payloads_per_call=2)
    monkeypatch.setattr("meddx.agents.evidence.get_retriever", lambda: mock_ret)

    result = evidence_node(state)

    evidence = result["evidence"]
    assert len(evidence) == 3
    assert {e.hypothesis_id for e in evidence} == {"h1", "h2", "h3"}
    assert all(e.supporting for e in evidence)
    assert all(e.refuting for e in evidence)


def test_evidence_node_phase_is_challenge(monkeypatch):
    state = _complete_state(["h1"])
    mock_model = _make_evidence_llm_mock(["h1"])
    monkeypatch.setattr("meddx.agents.evidence.model_for_agent", lambda _: mock_model)
    mock_ret = _make_retriever_mock()
    monkeypatch.setattr("meddx.agents.evidence.get_retriever", lambda: mock_ret)

    result = evidence_node(state)

    assert result["phase"] == Phase.CHALLENGE


def test_evidence_node_deduplicates_within_stance(monkeypatch):
    """If retriever returns duplicates (same PMID), they must be collapsed."""
    state = _complete_state(["h1"])
    mock_model = _make_evidence_llm_mock(["h1"])
    monkeypatch.setattr("meddx.agents.evidence.model_for_agent", lambda _: mock_model)

    # Make retriever return the same payload twice
    dup_payload = _make_fake_payload(42)
    mock_ret = MagicMock()
    mock_ret.is_ready.return_value = True
    mock_ret.search.return_value = [dup_payload, dup_payload]  # exact duplicate
    monkeypatch.setattr("meddx.agents.evidence.get_retriever", lambda: mock_ret)

    result = evidence_node(state)

    ev = result["evidence"][0]
    assert len(ev.supporting) == 1  # deduplicated
    assert len(ev.refuting) == 1


def test_evidence_node_graceful_degradation_empty_corpus(monkeypatch):
    """When corpus is not ingested, node returns empty lists rather than raising."""
    state = _complete_state(["h1", "h2"])
    mock_model = _make_evidence_llm_mock(["h1", "h2"])
    monkeypatch.setattr("meddx.agents.evidence.model_for_agent", lambda _: mock_model)

    mock_ret = MagicMock()
    mock_ret.is_ready.return_value = False  # corpus empty
    monkeypatch.setattr("meddx.agents.evidence.get_retriever", lambda: mock_ret)

    result = evidence_node(state)

    evidence = result["evidence"]
    assert len(evidence) == 2
    assert all(e.supporting == [] for e in evidence)
    assert all(e.refuting == [] for e in evidence)
    mock_ret.search.assert_not_called()


def test_evidence_node_citations_have_urls_from_pmid(monkeypatch):
    state = _complete_state(["h1"])
    mock_model = _make_evidence_llm_mock(["h1"])
    monkeypatch.setattr("meddx.agents.evidence.model_for_agent", lambda _: mock_model)
    mock_ret = _make_retriever_mock(payloads_per_call=1)
    monkeypatch.setattr("meddx.agents.evidence.get_retriever", lambda: mock_ret)

    result = evidence_node(state)

    ev = result["evidence"][0]
    for c in ev.supporting + ev.refuting:
        if c.pmid:
            assert c.url is not None
            assert c.pmid in c.url
