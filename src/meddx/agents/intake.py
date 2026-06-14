"""Intake node — completeness gate (anti premature closure).

The gate itself is deterministic code (not an LLM decision): the graph will
not proceed to hypothesis generation while required PatientCase fields are
neither filled in nor explicitly marked unavailable. The LLM part of this
agent (conversational data collection) plugs in later.
"""

from meddx.schemas import DiagnosticState, Phase

# TODO(real implementation): conversational collection of missing fields:
#   llm = model_for_agent("intake"); system = load_prompt("intake")


def intake_node(state: DiagnosticState) -> dict:
    missing = state.patient_case.missing_fields()
    if missing:
        return {"missing_fields": missing, "phase": Phase.AWAITING_DATA}
    return {"missing_fields": [], "phase": Phase.HYPOTHESES}


def route_after_intake(state: DiagnosticState) -> str:
    """Conditional edge: block the pipeline until the case is complete."""
    return "ask_student" if state.missing_fields else "generate_hypotheses"
