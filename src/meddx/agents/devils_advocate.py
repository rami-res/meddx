"""Devil's Advocate node — blind adversarial critique (anti confirmation
bias / anchoring).

ANTI-BIAS INVARIANT (CLAUDE.md): this agent must never see hypothesis
ranking or any ordering signal. The projection below is the ONLY input that
may be passed to the LLM — it strips ranking, rationale, and randomizes
away ordering by sorting names alphabetically.
"""

from meddx.schemas import ChallengeReport, DiagnosticState, Phase

# TODO(real implementation):
#   llm = model_for_agent("devils_advocate").with_structured_output(ChallengeReport)
#   system = load_prompt("devils_advocate")
#   llm input is built EXCLUSIVELY from blind_view(state)


def blind_view(state: DiagnosticState) -> dict:
    """State projection for the Devil's Advocate: case + sorted hypothesis
    names only. No ranking, no rationales, no must-not-miss flags."""
    return {
        "patient_case": state.patient_case.model_dump(),
        "hypotheses": sorted(h.name for h in state.hypotheses),
    }


def devils_advocate_node(state: DiagnosticState) -> dict:
    view = blind_view(state)  # the only permitted input for the LLM call
    challenge = ChallengeReport(
        contradictions=[
            f"[stub] Case finding X is atypical for {name}" for name in view["hypotheses"][:2]
        ],
        alternative_explanations={
            "[stub] chest pain": "could be referred pain from an abdominal source",
        },
        discriminating_test=(
            "[stub] High-sensitivity troponin + D-dimer separates the two most "
            "competitive hypotheses"
        ),
    )
    return {"challenge": challenge, "phase": Phase.ROOT_CAUSE}
