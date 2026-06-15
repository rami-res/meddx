"""High-level facade for Streamlit UI → database writes.

All methods catch exceptions and return them as strings so the diagnostic
flow continues even when MySQL is unavailable.
"""

from __future__ import annotations

from meddx.db.base import get_session
from meddx.db.repositories import (
    CaseRepository,
    SessionRepository,
    StudentAnswerRepository,
)
from meddx.schemas import PatientCase, SynthesisResult


class DiagnosticSessionStore:
    """Write student diagnostic sessions to MySQL.

    Thread-safe: each method opens its own short-lived session from the shared
    factory (no long-lived session held on the object).
    """

    def __init__(self, session_factory) -> None:
        self._factory = session_factory

    # ------------------------------------------------------------------
    # Public API — all return None on success, error string on failure
    # ------------------------------------------------------------------

    def open_session(self, thread_id: str, patient_case: PatientCase) -> str | None:
        """Create or update the session + case records (idempotent).

        Called each time the patient-case form is submitted (including
        AWAITING_DATA retries). get_or_create + save_case upsert make it safe
        to call multiple times for the same thread_id.
        """
        try:
            with get_session(self._factory) as s:
                SessionRepository(s).get_or_create(thread_id)
                CaseRepository(s).save_case(thread_id, patient_case)
            return None
        except Exception as exc:
            return str(exc)

    def record_phase(self, thread_id: str, result: dict) -> str | None:
        """Update session phase; persist hypotheses when available."""
        try:
            phase = result.get("phase")
            phase_str: str = phase.value if hasattr(phase, "value") else str(phase)
            status = _phase_to_status(phase_str)

            with get_session(self._factory) as s:
                SessionRepository(s).update_phase(thread_id, phase=phase_str, status=status)
                hypotheses = result.get("hypotheses") or []
                if hypotheses:
                    case = CaseRepository(s).get_case(thread_id)
                    if case is not None:
                        CaseRepository(s).save_hypotheses(case.id, hypotheses)
            return None
        except Exception as exc:
            return str(exc)

    def record_answer(
        self,
        thread_id: str,
        raw_ranking: str,
        synthesis: SynthesisResult,
    ) -> str | None:
        """Save student answer, update final ranks, mark session done."""
        try:
            with get_session(self._factory) as s:
                StudentAnswerRepository(s).save(thread_id, raw_ranking, synthesis)
                case = CaseRepository(s).get_case(thread_id)
                if case is not None:
                    CaseRepository(s).update_final_ranks(case.id, synthesis.ranking)
                SessionRepository(s).update_phase(thread_id, phase="done", status="done")
            return None
        except Exception as exc:
            return str(exc)


def _phase_to_status(phase_str: str) -> str:
    if phase_str == "synthesis":
        return "interrupted"
    if phase_str == "done":
        return "done"
    return "active"
