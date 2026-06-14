"""Unit tests for Intake, Devil's Advocate, Root-Cause, and Synthesis agents.

All LLM calls and LangGraph interrupt() are patched so tests run without API
keys, GPU, or a live Qdrant instance. Anti-bias invariants are tested as
structural properties of each agent's code — not just its output.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from meddx.agents.devils_advocate import (
    _AlternativeExplanation,
    _ChallengeOutput,
    blind_view,
    devils_advocate_node,
)
from meddx.agents.intake import intake_node, route_after_intake
from meddx.agents.root_cause import root_cause_node
from meddx.agents.synthesis import _RankedEntry, _SynthesisLLMOutput, synthesis_node
from meddx.rag.citations import unknown_citations
from meddx.schemas import (
    UNAVAILABLE,
    ChallengeReport,
    Citation,
    DiagnosticState,
    Hypothesis,
    HypothesisEvidence,
    PatientCase,
    Phase,
    RootCauseAssessment,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _complete_case() -> PatientCase:
    return PatientCase(
        chief_complaint="Chest pain",
        history_of_present_illness="Retrosternal, episodic",
        past_medical_history="Hypertension",
        medications="Amlodipine",
        family_history="Father: MI",
        systems_review="Fever",
        risk_factors="Smoker",
        available_investigations=UNAVAILABLE,
    )


def _incomplete_case() -> PatientCase:
    return PatientCase(chief_complaint="Chest pain")


def _hypotheses() -> list[Hypothesis]:
    return [
        Hypothesis(
            id="h1",
            name="Acute coronary syndrome",
            rationale="Retrosternal pain, risk factors.",
            organ_system="cardiovascular",
            is_must_not_miss=True,
        ),
        Hypothesis(
            id="h2",
            name="Pulmonary embolism",
            rationale="Episodic, pleuritic component.",
            organ_system="respiratory",
            is_must_not_miss=True,
        ),
        Hypothesis(
            id="h3",
            name="GERD",
            rationale="Post-prandial retrosternal burning.",
            organ_system="gastrointestinal",
        ),
    ]


def _citation(n: int) -> Citation:
    return Citation(
        pmid=str(n),
        doi=f"10.0000/test.{n}",
        title=f"Test Article {n}",
        journal="Test Journal",
        year=2024,
        study_type="rct",
    )


def _evidence(hypotheses: list[Hypothesis]) -> list[HypothesisEvidence]:
    return [
        HypothesisEvidence(
            hypothesis_id=h.id,
            supporting=[_citation(i * 10 + 1), _citation(i * 10 + 2)],
            refuting=[_citation(i * 10 + 3)],
        )
        for i, h in enumerate(hypotheses)
    ]


def _challenge() -> ChallengeReport:
    return ChallengeReport(
        contradictions=["ACS: normal ECG in initial presentation"],
        alternative_explanations={"chest pain": "musculoskeletal origin"},
        discriminating_test="Serial troponin I at 0h and 3h",
    )


def _root_cause() -> RootCauseAssessment:
    return RootCauseAssessment(
        all_findings_explained=True,
        unexplained_findings=[],
        candidate_underlying_conditions=["none identified"],
        comment="All findings consistent with ACS.",
    )


def _state_with_hypotheses() -> DiagnosticState:
    return DiagnosticState(
        patient_case=_complete_case(),
        hypotheses=_hypotheses(),
        phase=Phase.EVIDENCE,
    )


def _state_with_evidence() -> DiagnosticState:
    hyps = _hypotheses()
    return DiagnosticState(
        patient_case=_complete_case(),
        hypotheses=hyps,
        evidence=_evidence(hyps),
        phase=Phase.CHALLENGE,
    )


def _state_with_challenge() -> DiagnosticState:
    hyps = _hypotheses()
    return DiagnosticState(
        patient_case=_complete_case(),
        hypotheses=hyps,
        evidence=_evidence(hyps),
        challenge=_challenge(),
        phase=Phase.ROOT_CAUSE,
    )


def _state_for_synthesis() -> DiagnosticState:
    hyps = _hypotheses()
    return DiagnosticState(
        patient_case=_complete_case(),
        hypotheses=hyps,
        evidence=_evidence(hyps),
        challenge=_challenge(),
        root_cause=_root_cause(),
        phase=Phase.SYNTHESIS,
    )


# ---------------------------------------------------------------------------
# Intake agent tests
# ---------------------------------------------------------------------------

class TestIntakeGate:
    def test_gate_passes_with_complete_case(self):
        state = DiagnosticState(patient_case=_complete_case())
        result = intake_node(state)

        assert result["phase"] == Phase.HYPOTHESES
        assert result["missing_fields"] == []
        assert result["intake_message"] is None

    def test_gate_blocks_with_incomplete_case(self, mock_intake_llm):
        state = DiagnosticState(patient_case=_incomplete_case())
        result = intake_node(state)

        assert result["phase"] == Phase.AWAITING_DATA
        assert result["missing_fields"]
        assert isinstance(result["intake_message"], str)
        assert len(result["intake_message"]) > 0

    def test_llm_not_called_when_case_complete(self):
        # Gate must short-circuit before any LLM call
        with patch("meddx.agents.intake.model_for_agent") as mock_factory:
            state = DiagnosticState(patient_case=_complete_case())
            intake_node(state)
            mock_factory.assert_not_called()

    def test_llm_called_when_case_incomplete(self, mock_intake_llm):
        state = DiagnosticState(patient_case=_incomplete_case())
        intake_node(state)
        mock_intake_llm.invoke.assert_called_once()

    def test_missing_fields_identifies_all_gaps(self):
        case = PatientCase(chief_complaint="Chest pain")
        missing = case.missing_fields()
        expected_missing = {
            "history_of_present_illness",
            "past_medical_history",
            "medications",
            "family_history",
            "systems_review",
            "risk_factors",
            "available_investigations",
        }
        assert set(missing) == expected_missing

    def test_unavailable_marker_satisfies_gate(self):
        case = PatientCase(
            chief_complaint="Chest pain",
            history_of_present_illness="Worsening over 2 days",
            past_medical_history=UNAVAILABLE,
            medications=UNAVAILABLE,
            family_history=UNAVAILABLE,
            systems_review=UNAVAILABLE,
            risk_factors=UNAVAILABLE,
            available_investigations=UNAVAILABLE,
        )
        assert case.missing_fields() == []

    def test_route_after_intake_when_complete(self):
        state = DiagnosticState(patient_case=_complete_case(), missing_fields=[])
        assert route_after_intake(state) == "generate_hypotheses"

    def test_route_after_intake_when_incomplete(self):
        state = DiagnosticState(
            patient_case=_incomplete_case(),
            missing_fields=["history_of_present_illness"],
        )
        assert route_after_intake(state) == "ask_student"


# ---------------------------------------------------------------------------
# Devil's Advocate agent tests
# ---------------------------------------------------------------------------

class TestDevilsAdvocateBlindView:
    def test_view_contains_only_permitted_keys(self):
        state = _state_with_hypotheses()
        view = blind_view(state)
        assert set(view.keys()) == {"patient_case", "hypotheses"}

    def test_view_hypotheses_are_names_only(self):
        state = _state_with_hypotheses()
        view = blind_view(state)
        # Must be strings (names), not Hypothesis objects or dicts with rationale
        assert all(isinstance(name, str) for name in view["hypotheses"])

    def test_view_hypotheses_alphabetically_sorted(self):
        hyps = [
            Hypothesis(id="h1", name="GERD", rationale="...", organ_system="GI"),
            Hypothesis(id="h2", name="ACS", rationale="...", organ_system="CV"),
            Hypothesis(id="h3", name="PE", rationale="...", organ_system="resp"),
        ]
        state = DiagnosticState(patient_case=_complete_case(), hypotheses=hyps)
        view = blind_view(state)
        names = view["hypotheses"]
        assert names == sorted(names)

    def test_view_contains_no_rationale(self):
        state = _state_with_hypotheses()
        view = blind_view(state)
        for name in view["hypotheses"]:
            assert "rationale" not in name.lower()
            assert "Retrosternal" not in name
            assert "Episodic" not in name

    def test_view_contains_no_must_not_miss_flag(self):
        state = _state_with_hypotheses()
        view = blind_view(state)
        # The view must not expose is_must_not_miss — that's ranking-adjacent
        assert "must_not_miss" not in str(view["hypotheses"])
        assert "must-not-miss" not in str(view["hypotheses"]).lower()

    def test_view_sorting_removes_original_order_signal(self):
        # Input is in order h1=ACS, h2=PE, h3=GERD;
        # blind view must alphabetize → GERD < PE < ACS is wrong, correct is ACS < GERD < PE
        hyps = _hypotheses()
        state = DiagnosticState(patient_case=_complete_case(), hypotheses=hyps)
        view = blind_view(state)
        assert view["hypotheses"][0] == "Acute coronary syndrome"
        assert view["hypotheses"][1] == "GERD"
        assert view["hypotheses"][2] == "Pulmonary embolism"

    def test_patient_case_present_in_view(self):
        state = _state_with_hypotheses()
        view = blind_view(state)
        assert "chief_complaint" in view["patient_case"]


class TestDevilsAdvocateNode:
    def test_node_produces_challenge_report(self, mock_devils_advocate_llm):
        state = _state_with_hypotheses()
        result = devils_advocate_node(state)
        assert "challenge" in result
        challenge = result["challenge"]
        assert challenge.contradictions
        assert challenge.discriminating_test

    def test_node_transitions_to_root_cause_phase(self, mock_devils_advocate_llm):
        state = _state_with_hypotheses()
        result = devils_advocate_node(state)
        assert result["phase"] == Phase.ROOT_CAUSE

    def test_alternative_explanations_dict(self, mock_devils_advocate_llm):
        state = _state_with_hypotheses()
        result = devils_advocate_node(state)
        # ChallengeReport.alternative_explanations is dict[str, str]
        assert isinstance(result["challenge"].alternative_explanations, dict)


# ---------------------------------------------------------------------------
# Root Cause agent tests
# ---------------------------------------------------------------------------

class TestRootCauseNode:
    def test_node_produces_assessment(self, mock_root_cause_llm):
        state = _state_with_challenge()
        result = root_cause_node(state)
        assert "root_cause" in result
        rc = result["root_cause"]
        assert isinstance(rc, RootCauseAssessment)

    def test_node_transitions_to_synthesis_phase(self, mock_root_cause_llm):
        state = _state_with_challenge()
        result = root_cause_node(state)
        assert result["phase"] == Phase.SYNTHESIS

    def test_node_runs_without_challenge(self, mock_root_cause_llm):
        # Devil's Advocate may not have run (partial state); root_cause must be resilient
        state = _state_with_evidence()
        state = state.model_copy(update={"challenge": None})
        result = root_cause_node(state)
        assert result["root_cause"] is not None

    def test_assessment_has_required_fields(self, mock_root_cause_llm):
        state = _state_with_challenge()
        result = root_cause_node(state)
        rc = result["root_cause"]
        assert isinstance(rc.all_findings_explained, bool)
        assert isinstance(rc.unexplained_findings, list)
        assert isinstance(rc.candidate_underlying_conditions, list)


# ---------------------------------------------------------------------------
# Synthesis agent tests
# ---------------------------------------------------------------------------

class TestSynthesisNode:
    def _make_synthesis_output(self, hypotheses: list[Hypothesis]) -> _SynthesisLLMOutput:
        return _SynthesisLLMOutput(
            ranked_hypotheses=[
                _RankedEntry(
                    hypothesis_id=h.id,
                    rank=i + 1,
                    reasoning=f"Evidence supports {h.name}",
                )
                for i, h in enumerate(hypotheses)
            ],
            workup_plan=["ECG", "Troponin", "D-dimer"],
            evidence_summary="Backed by RCT-level evidence.",
            socratic_feedback="Your ranking was close to the system's.",
        )

    def test_interrupt_called_before_llm(self):
        state = _state_for_synthesis()
        events: list[str] = []

        def fake_interrupt(question):
            events.append("interrupt")
            return "1, 2, 3"

        mock_chain = MagicMock()

        def fake_llm_invoke(msgs):
            events.append("llm")
            return self._make_synthesis_output(_hypotheses())

        mock_chain.invoke.side_effect = fake_llm_invoke
        mock_model = MagicMock()
        mock_model.with_structured_output.return_value = mock_chain

        with (
            patch("meddx.agents.synthesis.interrupt", side_effect=fake_interrupt),
            patch("meddx.agents.synthesis.model_for_agent", return_value=mock_model),
        ):
            synthesis_node(state)

        assert events == ["interrupt", "llm"]

    def test_interrupt_question_lists_all_hypotheses(self):
        state = _state_for_synthesis()
        captured_questions = []

        def fake_interrupt(question):
            captured_questions.append(question)
            return "1, 2, 3"

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = self._make_synthesis_output(state.hypotheses)
        mock_model = MagicMock()
        mock_model.with_structured_output.return_value = mock_chain

        with (
            patch("meddx.agents.synthesis.interrupt", side_effect=fake_interrupt),
            patch("meddx.agents.synthesis.model_for_agent", return_value=mock_model),
        ):
            synthesis_node(state)

        assert captured_questions
        question = captured_questions[0]
        for h in state.hypotheses:
            assert h.name in question

    def test_student_ranking_passed_to_llm_context(self):
        state = _state_for_synthesis()
        captured_messages = []

        mock_chain = MagicMock()

        def capture_invoke(msgs):
            captured_messages.extend(msgs)
            return self._make_synthesis_output(state.hypotheses)

        mock_chain.invoke.side_effect = capture_invoke
        mock_model = MagicMock()
        mock_model.with_structured_output.return_value = mock_chain

        student_answer = "PE first, then ACS, then GERD"
        with (
            patch("meddx.agents.synthesis.interrupt", return_value=student_answer),
            patch("meddx.agents.synthesis.model_for_agent", return_value=mock_model),
        ):
            synthesis_node(state)

        # Student's ranking must appear in the HumanMessage content
        human_contents = [m.content for m in captured_messages if hasattr(m, "content")]
        assert any(student_answer in c for c in human_contents)

    def test_citations_come_from_state_not_llm(self, mock_synthesis):
        """Programmatic attachment: LLM never fabricates citation identifiers."""
        state = _state_for_synthesis()
        result = synthesis_node(state)

        ranking = result["synthesis"].ranking
        # All citations in ranking must originate from state.evidence
        ev_by_id = {e.hypothesis_id: e for e in state.evidence}
        for ranked_h in ranking:
            ev = ev_by_id.get(ranked_h.hypothesis_id)
            state_cits = (ev.supporting + ev.refuting) if ev else []
            state_ids = {i for c in state_cits for i in c.identifiers()}
            for c in ranked_h.citations:
                assert c.identifiers() & state_ids, (
                    f"Citation {c.pmid}/{c.doi} not from state.evidence"
                )

    def test_citation_grounding_validates(self, mock_synthesis):
        """assert_citations_grounded must not raise when citations are grounded."""
        state = _state_for_synthesis()
        result = synthesis_node(state)

        ranking = result["synthesis"].ranking
        retrieved = [c for ev in state.evidence for c in ev.supporting + ev.refuting]
        used = [c for r in ranking for c in r.citations]
        assert unknown_citations(used, retrieved) == []

    def test_node_transitions_to_done_phase(self, mock_synthesis):
        state = _state_for_synthesis()
        result = synthesis_node(state)
        assert result["phase"] == Phase.DONE

    def test_synthesis_result_has_all_hypotheses_ranked(self):
        state = _state_for_synthesis()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = self._make_synthesis_output(state.hypotheses)
        mock_model = MagicMock()
        mock_model.with_structured_output.return_value = mock_chain

        with (
            patch("meddx.agents.synthesis.interrupt", return_value="1, 2, 3"),
            patch("meddx.agents.synthesis.model_for_agent", return_value=mock_model),
        ):
            result = synthesis_node(state)

        ranking = result["synthesis"].ranking
        ranked_ids = {r.hypothesis_id for r in ranking}
        hypothesis_ids = {h.id for h in state.hypotheses}
        assert ranked_ids == hypothesis_ids

    def test_synthesis_includes_workup_plan(self, mock_synthesis):
        state = _state_for_synthesis()
        result = synthesis_node(state)
        assert result["synthesis"].workup_plan

    def test_synthesis_includes_socratic_feedback(self, mock_synthesis):
        state = _state_for_synthesis()
        result = synthesis_node(state)
        assert result["synthesis"].socratic_feedback

    def test_synthesis_includes_evidence_summary(self, mock_synthesis):
        state = _state_for_synthesis()
        result = synthesis_node(state)
        assert result["synthesis"].evidence_summary
