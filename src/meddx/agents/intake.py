"""Intake node — completeness gate + conversational data collection.

Two responsibilities:
  1. Gate (deterministic): PatientCase.missing_fields() decides whether the
     pipeline may proceed. This is a code-level invariant — the LLM is never
     asked to decide if the data is sufficient (anti premature closure).
  2. Conversation (LLM): when the case is incomplete, generates a pedagogical
     question for the student explaining WHY each missing field matters.

The UI flow:
  graph.invoke(state) → AWAITING_DATA + intake_message
  student fills missing fields in the UI / types answer
  graph.invoke(updated_state) → HYPOTHESES (gate passed)
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from meddx.llm import model_for_agent
from meddx.prompts import load_prompt
from meddx.schemas import DiagnosticState, Phase


def _format_collected(state: DiagnosticState) -> str:
    lines = []
    for field_name, value in state.patient_case:
        if value and str(value).strip():
            label = field_name.replace("_", " ").title()
            lines.append(f"  {label}: {value}")
    return "\n".join(lines) if lines else "  (nothing collected yet)"


def intake_node(state: DiagnosticState) -> dict:
    missing = state.patient_case.missing_fields()

    # ── Gate: all fields filled → proceed ────────────────────────────────
    if not missing:
        return {"missing_fields": [], "phase": Phase.HYPOTHESES, "intake_message": None}

    # ── Conversation: ask the student for the missing fields ──────────────
    llm = model_for_agent("intake")
    system = load_prompt("intake")

    missing_labels = [f.replace("_", " ") for f in missing]
    collected_text = _format_collected(state)

    message: str = llm.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(
                content=(
                    f"Language: {state.user_language}\n\n"
                    f"Collected so far:\n{collected_text}\n\n"
                    f"Still missing: {', '.join(missing_labels)}\n\n"
                    "Generate your next question asking the student for the "
                    "missing information. Be educational — briefly explain why "
                    "each missing item matters for the differential diagnosis."
                )
            ),
        ]
    ).content

    return {
        "missing_fields": missing,
        "phase": Phase.AWAITING_DATA,
        "intake_message": message,
    }


def route_after_intake(state: DiagnosticState) -> str:
    """Conditional edge: block pipeline until case is complete."""
    return "ask_student" if state.missing_fields else "generate_hypotheses"
