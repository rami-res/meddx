"""Pydantic models shared by agents, the graph state, and the RAG layer."""

from __future__ import annotations

from pydantic import BaseModel, Field

#: Marker a student can use for data that genuinely cannot be obtained.
#: An explicitly unavailable field counts as "collected" for the intake gate.
UNAVAILABLE = "unavailable"


class PatientCase(BaseModel):
    """Structured case data. Every field is required for the intake gate:
    it must be filled in or explicitly marked UNAVAILABLE."""

    chief_complaint: str | None = None
    history_of_present_illness: str | None = None
    past_medical_history: str | None = None
    medications: str | None = None
    family_history: str | None = None
    systems_review: str | None = None
    risk_factors: str | None = None
    available_investigations: str | None = None

    def missing_fields(self) -> list[str]:
        """Fields still blocking the intake completeness gate (anti premature
        closure): empty and not explicitly marked unavailable."""
        return [
            name
            for name, value in self
            if value is None or not str(value).strip()
        ]


class Hypothesis(BaseModel):
    id: str
    name: str
    rationale: str
    organ_system: str
    is_must_not_miss: bool = False


class Citation(BaseModel):
    pmid: str | None = None
    doi: str | None = None
    title: str
    journal: str
    year: int
    study_type: str  # meta-analysis | systematic review | RCT | cohort | case report | ...
    url: str | None = None

    def identifiers(self) -> set[str]:
        return {i for i in (self.pmid, self.doi) if i}


class HypothesisEvidence(BaseModel):
    """Symmetric evidence for one hypothesis (anti confirmation bias):
    supporting and refuting citations are retrieved separately."""

    hypothesis_id: str
    supporting: list[Citation] = Field(default_factory=list)
    refuting: list[Citation] = Field(default_factory=list)


class ChallengeReport(BaseModel):
    """Devil's Advocate output (produced blind to any ranking)."""

    contradictions: list[str] = Field(default_factory=list)
    alternative_explanations: dict[str, str] = Field(
        default_factory=dict, description="key symptom -> alternative explanation"
    )
    discriminating_test: str = ""


class RootCauseAssessment(BaseModel):
    all_findings_explained: bool = False
    unexplained_findings: list[str] = Field(default_factory=list)
    candidate_underlying_conditions: list[str] = Field(default_factory=list)
    comment: str = ""


class RankedHypothesis(BaseModel):
    hypothesis_id: str
    rank: int
    probability_note: str = ""
    citations: list[Citation] = Field(default_factory=list)


class SynthesisResult(BaseModel):
    ranking: list[RankedHypothesis] = Field(default_factory=list)
    workup_plan: list[str] = Field(default_factory=list)
    evidence_summary: str = ""
