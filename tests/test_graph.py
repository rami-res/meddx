"""End-to-end graph runs + anti-bias invariants.

Graph-level tests that pass through hypothesis_node and evidence_node use
fixtures from conftest.py so they run without API keys, GPU, or Qdrant.
The anti-bias contract tests remain meaningful — they check the graph's
structural invariants, not whether the LLM / retriever produced real content.
"""

import pytest

from meddx.agents import blind_view
from meddx.graph import build_graph
from meddx.rag.citations import assert_citations_grounded, unknown_citations
from meddx.schemas import (
    UNAVAILABLE,
    Citation,
    DiagnosticState,
    PatientCase,
    Phase,
)


def complete_case() -> PatientCase:
    return PatientCase(
        chief_complaint="Chest pain for 2 days",
        history_of_present_illness="Retrosternal, episodic, worse after meals",
        past_medical_history="Hypertension",
        medications="Amlodipine 5 mg",
        family_history="Father: MI at 56",
        systems_review="Intermittent low-grade fever",
        risk_factors="Smoker, 20 pack-years",
        available_investigations=UNAVAILABLE,
    )


def test_incomplete_case_stops_at_intake_gate(mock_intake_llm):
    graph = build_graph()
    result = graph.invoke(
        DiagnosticState(patient_case=PatientCase(chief_complaint="Chest pain"))
    )
    assert result["phase"] == Phase.AWAITING_DATA
    assert result["missing_fields"]
    assert result["hypotheses"] == []
    assert result.get("synthesis") is None  # LangGraph omits None-valued keys


def test_complete_case_runs_end_to_end(
    mock_hypothesis_llm,
    mock_evidence,
    mock_devils_advocate_llm,
    mock_root_cause_llm,
    mock_synthesis,
):
    graph = build_graph()
    result = graph.invoke(DiagnosticState(patient_case=complete_case()))

    assert result["phase"] == Phase.DONE

    # Anti-anchoring/availability invariants (hypothesis agent)
    hypotheses = result["hypotheses"]
    assert len(hypotheses) >= 5
    assert any(h.is_must_not_miss for h in hypotheses)
    assert len({h.organ_system for h in hypotheses}) >= 2

    # Symmetric evidence per hypothesis (anti confirmation bias — evidence agent)
    evidence = result["evidence"]
    assert {e.hypothesis_id for e in evidence} == {h.id for h in hypotheses}
    assert all(e.supporting and e.refuting for e in evidence)

    # Challenge and root cause produced
    assert result["challenge"].discriminating_test
    assert result["root_cause"] is not None

    # Synthesis ranks every hypothesis and stays citation-grounded
    ranking = result["synthesis"].ranking
    assert len(ranking) == len(hypotheses)
    retrieved = [c for e in evidence for c in e.supporting + e.refuting]
    used = [c for r in ranking for c in r.citations]
    assert unknown_citations(used, retrieved) == []


def test_devils_advocate_view_is_blind(
    mock_hypothesis_llm,
    mock_evidence,
    mock_devils_advocate_llm,
    mock_root_cause_llm,
    mock_synthesis,
):
    """The Devil's Advocate projection must carry no ranking/ordering signal."""
    graph = build_graph()
    result = graph.invoke(DiagnosticState(patient_case=complete_case()))
    state = DiagnosticState(**result)

    view = blind_view(state)
    assert set(view) == {"patient_case", "hypotheses"}  # nothing else leaks
    assert view["hypotheses"] == sorted(view["hypotheses"])  # alphabetical = no rank
    assert all(isinstance(name, str) for name in view["hypotheses"])  # names only


def test_citation_validator_rejects_ungrounded():
    retrieved = [
        Citation(pmid="123", title="A", journal="J", year=2024, study_type="RCT")
    ]
    hallucinated = [
        Citation(pmid="999", title="B", journal="J", year=2023, study_type="RCT")
    ]
    assert unknown_citations(hallucinated, retrieved) == hallucinated
    with pytest.raises(ValueError, match="999"):
        assert_citations_grounded(hallucinated, retrieved)
