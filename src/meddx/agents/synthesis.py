"""Synthesis/Tutor node — final ranked differential with grounded citations.

Stub: ranks must-not-miss hypotheses first, attaches evidence citations, and
runs the programmatic citation check. The Socratic step (student ranks
first) will be a LangGraph interrupt before this node's answer is revealed.
"""

from meddx.rag.citations import assert_citations_grounded
from meddx.schemas import (
    Citation,
    DiagnosticState,
    Phase,
    RankedHypothesis,
    SynthesisResult,
)

# TODO(real implementation):
#   llm = model_for_agent("synthesis"); system = load_prompt("synthesis")
#   + LangGraph interrupt() for the Socratic step before revealing ranking


def synthesis_node(state: DiagnosticState) -> dict:
    evidence_by_hypothesis = {e.hypothesis_id: e for e in state.evidence}

    # Stub ranking: must-not-miss first, then original order.
    ordered = sorted(
        state.hypotheses, key=lambda h: (not h.is_must_not_miss,)
    )

    retrieved: list[Citation] = []
    ranking: list[RankedHypothesis] = []
    for rank, hypothesis in enumerate(ordered, start=1):
        evidence = evidence_by_hypothesis.get(hypothesis.id)
        citations = (evidence.supporting + evidence.refuting) if evidence else []
        retrieved.extend(citations)
        ranking.append(
            RankedHypothesis(
                hypothesis_id=hypothesis.id,
                rank=rank,
                probability_note="[stub] must-not-miss kept on top"
                if hypothesis.is_must_not_miss
                else "[stub]",
                citations=citations,
            )
        )

    # Anti-hallucination invariant: every cited PMID/DOI must be grounded.
    used = [c for r in ranking for c in r.citations]
    assert_citations_grounded(used, retrieved)

    synthesis = SynthesisResult(
        ranking=ranking,
        workup_plan=[
            state.challenge.discriminating_test if state.challenge else "",
            *(state.root_cause.candidate_underlying_conditions if state.root_cause else []),
        ],
        evidence_summary="[stub] Evidence strength marked per citation study_type.",
    )
    return {"synthesis": synthesis, "phase": Phase.DONE}
