"""Root-Cause node — full-organism view (anti search satisficing).

Receives the complete picture: patient case + all hypotheses + evidence
summary + Devil's Advocate challenge. Asks: do the working diagnoses explain
EVERY finding? Could the diagnosis itself be a symptom of something deeper?

The agent's job is to keep the search open even when a satisfying diagnosis
seems to be at hand — countering the tendency to stop searching once a
plausible explanation is found.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from meddx.llm import model_for_agent
from meddx.prompts import load_prompt
from meddx.schemas import DiagnosticState, Phase, RootCauseAssessment


def _format_hypotheses(state: DiagnosticState) -> str:
    lines = []
    for h in state.hypotheses:
        mnm = " [MUST-NOT-MISS]" if h.is_must_not_miss else ""
        lines.append(f"  - {h.name} ({h.organ_system}){mnm}: {h.rationale}")
    return "\n".join(lines)


def _format_evidence_counts(state: DiagnosticState) -> str:
    name_by_id = {h.id: h.name for h in state.hypotheses}
    lines = []
    for ev in state.evidence:
        name = name_by_id.get(ev.hypothesis_id, ev.hypothesis_id)
        lines.append(
            f"  {name}: "
            f"{len(ev.supporting)} supporting, {len(ev.refuting)} refuting citations"
        )
    return "\n".join(lines) if lines else "  No evidence retrieved."


def _format_challenge(state: DiagnosticState) -> str:
    ch = state.challenge
    if not ch:
        return "  No challenge generated."
    parts = ["  Contradictions:"]
    for c in ch.contradictions:
        parts.append(f"    - {c}")
    parts.append("  Discriminating test: " + (ch.discriminating_test or "none"))
    return "\n".join(parts)


def _format_case(state: DiagnosticState) -> str:
    lines = []
    for field_name, value in state.patient_case:
        if value and str(value).strip():
            label = field_name.replace("_", " ").title()
            lines.append(f"  {label}: {value}")
    return "\n".join(lines)


def root_cause_node(state: DiagnosticState) -> dict:
    llm = model_for_agent("root_cause").with_structured_output(RootCauseAssessment)
    system = load_prompt("root_cause")

    assessment: RootCauseAssessment = llm.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(
                content=(
                    f"Patient case:\n{_format_case(state)}\n\n"
                    f"Current hypotheses:\n{_format_hypotheses(state)}\n\n"
                    f"Evidence retrieved:\n{_format_evidence_counts(state)}\n\n"
                    f"Devil's Advocate findings:\n{_format_challenge(state)}\n\n"
                    "Assess whether the current differential explains ALL findings. "
                    "Identify any unexplained findings and candidate underlying conditions."
                )
            ),
        ]
    )
    return {"root_cause": assessment, "phase": Phase.SYNTHESIS}
