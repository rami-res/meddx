"""Hypothesis node — broad unranked differential (anti anchoring/availability bias).

Three anti-bias invariants are enforced by code after every LLM call:
  1. ≥5 hypotheses  (anti anchoring — no premature narrowing)
  2. ≥1 must-not-miss flagged  (life-threatening states never silently dropped)
  3. ≥2 distinct organ systems  (anti availability bias)

If the LLM violates any invariant, hypothesis_node raises ValueError so that
LangGraph's built-in retry mechanism (set node retry_policy) can re-run it.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from meddx.llm import model_for_agent
from meddx.prompts import load_prompt
from meddx.schemas import DiagnosticState, Hypothesis, PatientCase, Phase


# ---------------------------------------------------------------------------
# LLM output schema  (id is assigned programmatically — the LLM must not
# invent identifiers, as they need to be stable within a session)
# ---------------------------------------------------------------------------

class _HypothesisItem(BaseModel):
    name: str = Field(
        description="Diagnosis or condition name, e.g. 'Acute coronary syndrome'"
    )
    rationale: str = Field(
        description=(
            "One paragraph tying this hypothesis to specific findings in the "
            "patient case. Do not use generic text — reference the patient's "
            "actual data."
        )
    )
    organ_system: str = Field(
        description=(
            "Primary organ system in lowercase, e.g. cardiovascular, respiratory, "
            "gastrointestinal, musculoskeletal, neurological, endocrine, psychiatric, "
            "haematological, renal, infectious"
        )
    )
    is_must_not_miss: bool = Field(
        default=False,
        description=(
            "True if failing to diagnose this condition could be immediately "
            "life-threatening or cause irreversible harm at low prior probability"
        ),
    )


class _HypothesisList(BaseModel):
    hypotheses: list[_HypothesisItem] = Field(
        description=(
            "Broad differential — at least 5 items. "
            "Output order carries NO meaning; do not rank."
        )
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_case(case: PatientCase) -> str:
    lines = []
    for field_name, value in case:
        label = field_name.replace("_", " ").title()
        display = str(value) if value else "not provided"
        lines.append(f"{label}: {display}")
    return "\n".join(lines)


def _validate_invariants(hypotheses: list[Hypothesis]) -> None:
    errors: list[str] = []

    if len(hypotheses) < 5:
        errors.append(f"only {len(hypotheses)} hypotheses — minimum is 5 (anti anchoring)")

    if not any(h.is_must_not_miss for h in hypotheses):
        errors.append("no must-not-miss condition flagged")

    systems = {h.organ_system.lower() for h in hypotheses}
    if len(systems) < 2:
        errors.append(
            f"all hypotheses from one organ system ({systems}) — "
            "availability bias not countered"
        )

    if errors:
        raise ValueError("Hypothesis invariant violation: " + "; ".join(errors))


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def hypothesis_node(state: DiagnosticState) -> dict:
    llm = model_for_agent("hypothesis").with_structured_output(_HypothesisList)
    system = load_prompt("hypothesis")
    case_text = _format_case(state.patient_case)

    result: _HypothesisList = llm.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(
                content=(
                    f"Reply language: {state.user_language}\n\n"
                    f"Patient case:\n\n{case_text}"
                )
            ),
        ]
    )

    hypotheses = [
        Hypothesis(
            id=f"h{i}",
            name=item.name,
            rationale=item.rationale,
            organ_system=item.organ_system.lower(),
            is_must_not_miss=item.is_must_not_miss,
        )
        for i, item in enumerate(result.hypotheses, start=1)
    ]

    _validate_invariants(hypotheses)

    return {"hypotheses": hypotheses, "phase": Phase.EVIDENCE}
