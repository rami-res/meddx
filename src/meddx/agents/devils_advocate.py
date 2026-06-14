"""Devil's Advocate node — blind adversarial critique (anti anchoring /
confirmation bias).

ANTI-BIAS INVARIANT (CLAUDE.md / ADR-0005):
The LLM receives ONLY blind_view(state) — patient case + alphabetically sorted
hypothesis names. No ranking, no rationales, no must-not-miss flags, no
probability estimates. This is enforced structurally (by the projection), not
by prompt discipline alone.

blind_view() is also exported for use in tests that verify no information leaks.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from meddx.llm import model_for_agent
from meddx.prompts import load_prompt
from meddx.schemas import ChallengeReport, DiagnosticState, Phase


# ---------------------------------------------------------------------------
# LLM output schema
# ---------------------------------------------------------------------------
# dict[str, str] is technically valid JSON but less reliable for structured
# output across providers. Using a list-of-pairs is more portable.

class _AlternativeExplanation(BaseModel):
    symptom: str = Field(description="A key symptom or case finding")
    alternative: str = Field(
        description="A plausible alternative explanation for this finding that "
                    "differs from the most obvious hypothesis"
    )


class _ChallengeOutput(BaseModel):
    contradictions: list[str] = Field(
        description=(
            "For each hypothesis: which specific case findings contradict it. "
            "Format: 'Hypothesis name: contradiction'. Be concrete, not generic."
        )
    )
    alternative_explanations: list[_AlternativeExplanation] = Field(
        description=(
            "For each key symptom: at least one alternative explanation that "
            "points away from the most obvious hypothesis."
        )
    )
    discriminating_test: str = Field(
        description=(
            "The SINGLE investigation that would best separate the two most "
            "competitive hypotheses. Include brief reasoning."
        )
    )


# ---------------------------------------------------------------------------
# State projection (the only permitted view for the LLM)
# ---------------------------------------------------------------------------

def blind_view(state: DiagnosticState) -> dict:
    """Projection: patient case + alphabetically sorted hypothesis names only.
    No ranking, no rationales, no must-not-miss flags — anti-anchoring."""
    return {
        "patient_case": state.patient_case.model_dump(),
        "hypotheses": sorted(h.name for h in state.hypotheses),
    }


def _format_blind_view(view: dict) -> str:
    case = view["patient_case"]
    case_lines = [
        f"  {k.replace('_', ' ').title()}: {v}"
        for k, v in case.items()
        if v and str(v).strip()
    ]
    hyp_lines = [f"  - {name}" for name in view["hypotheses"]]
    return (
        "Patient case:\n" + "\n".join(case_lines) + "\n\n"
        "Hypotheses (alphabetical — no implied ranking):\n" + "\n".join(hyp_lines)
    )


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def devils_advocate_node(state: DiagnosticState) -> dict:
    view = blind_view(state)   # ONLY permitted input — enforced here structurally

    llm = model_for_agent("devils_advocate").with_structured_output(_ChallengeOutput)
    system = load_prompt("devils_advocate")

    output: _ChallengeOutput = llm.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(content=_format_blind_view(view)),
        ]
    )

    challenge = ChallengeReport(
        contradictions=output.contradictions,
        alternative_explanations={
            ae.symptom: ae.alternative for ae in output.alternative_explanations
        },
        discriminating_test=output.discriminating_test,
    )
    return {"challenge": challenge, "phase": Phase.ROOT_CAUSE}
