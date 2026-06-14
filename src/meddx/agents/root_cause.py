"""Root-Cause node — full-organism view (anti search satisficing).

Stub: returns a canned assessment. The real implementation re-reads ALL case
findings against the differential and searches for underlying conditions.
"""

from meddx.schemas import DiagnosticState, Phase, RootCauseAssessment

# TODO(real implementation):
#   llm = model_for_agent("root_cause").with_structured_output(RootCauseAssessment)
#   system = load_prompt("root_cause")


def root_cause_node(state: DiagnosticState) -> dict:
    assessment = RootCauseAssessment(
        all_findings_explained=False,
        unexplained_findings=["[stub] intermittent low-grade fever not explained"],
        candidate_underlying_conditions=[
            "[stub] occult infection or inflammatory condition to rule out",
        ],
        comment="[stub] Working diagnoses do not yet explain every finding; "
        "keep the search open.",
    )
    return {"root_cause": assessment, "phase": Phase.SYNTHESIS}
