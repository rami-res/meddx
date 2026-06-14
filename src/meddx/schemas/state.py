"""LangGraph state for the diagnostic session.

The state is the single source of truth flowing through the graph. Anti-bias
visibility rules are enforced by *projections* of this state (each agent node
builds its LLM input from a subset of fields), not by prompt discipline alone.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from meddx.schemas.models import (
    ChallengeReport,
    Hypothesis,
    HypothesisEvidence,
    PatientCase,
    RootCauseAssessment,
    SynthesisResult,
)


class Phase(str, Enum):
    INTAKE = "intake"
    AWAITING_DATA = "awaiting_data"  # intake gate failed; waiting for the student
    HYPOTHESES = "hypotheses"
    EVIDENCE = "evidence"
    CHALLENGE = "challenge"
    ROOT_CAUSE = "root_cause"
    SYNTHESIS = "synthesis"
    DONE = "done"


class DiagnosticState(BaseModel):
    # Input
    user_language: str = "uk"
    patient_case: PatientCase = Field(default_factory=PatientCase)

    # Intake gate
    phase: Phase = Phase.INTAKE
    missing_fields: list[str] = Field(default_factory=list)

    # Agent outputs (hypotheses stay UNRANKED until synthesis — anti anchoring)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    evidence: list[HypothesisEvidence] = Field(default_factory=list)
    challenge: ChallengeReport | None = None
    root_cause: RootCauseAssessment | None = None
    synthesis: SynthesisResult | None = None
