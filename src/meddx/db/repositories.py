"""Repository layer — all DB reads/writes go through these classes.

Each repository takes an open SQLAlchemy Session and wraps the query logic
so callers never write raw ORM queries. The pattern also makes unit-testing
trivial: pass a Session bound to an in-memory SQLite engine.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from meddx.db.models import (
    Case,
    CaseHypothesis,
    DiagnosticSession,
    StudentAnswer,
    User,
)
from meddx.schemas import Hypothesis, PatientCase, RankedHypothesis, SynthesisResult


# ---------------------------------------------------------------------------
# UserRepository
# ---------------------------------------------------------------------------

class UserRepository:
    def __init__(self, session: Session) -> None:
        self._s = session

    def get_by_email(self, email: str) -> User | None:
        return self._s.scalar(select(User).where(User.email == email))

    def get_or_create(self, email: str, name: str, locale: str = "uk") -> User:
        """Idempotent: fetch by e-mail or insert a new row."""
        user = self.get_by_email(email)
        if user is None:
            user = User(email=email, name=name, locale=locale)
            self._s.add(user)
            self._s.flush()  # populate auto-incremented id without committing
        return user

    def get_by_id(self, user_id: int) -> User | None:
        return self._s.get(User, user_id)


# ---------------------------------------------------------------------------
# SessionRepository
# ---------------------------------------------------------------------------

class SessionRepository:
    def __init__(self, session: Session) -> None:
        self._s = session

    def create(
        self,
        thread_id: str,
        user_id: int | None = None,
        phase: str = "intake",
    ) -> DiagnosticSession:
        db_session = DiagnosticSession(id=thread_id, user_id=user_id, phase=phase)
        self._s.add(db_session)
        self._s.flush()
        return db_session

    def get(self, thread_id: str) -> DiagnosticSession | None:
        return self._s.get(DiagnosticSession, thread_id)

    def get_or_create(
        self,
        thread_id: str,
        user_id: int | None = None,
    ) -> DiagnosticSession:
        sess = self.get(thread_id)
        if sess is None:
            sess = self.create(thread_id, user_id=user_id)
        return sess

    def update_phase(
        self,
        thread_id: str,
        phase: str,
        status: str | None = None,
    ) -> None:
        sess = self.get(thread_id)
        if sess is None:
            raise ValueError(f"Session {thread_id!r} not found")
        sess.phase = phase
        if status is not None:
            sess.status = status
        self._s.flush()

    def list_for_user(self, user_id: int) -> list[DiagnosticSession]:
        return list(
            self._s.scalars(
                select(DiagnosticSession)
                .where(DiagnosticSession.user_id == user_id)
                .order_by(DiagnosticSession.created_at.desc())
            )
        )


# ---------------------------------------------------------------------------
# CaseRepository
# ---------------------------------------------------------------------------

class CaseRepository:
    def __init__(self, session: Session) -> None:
        self._s = session

    def save_case(self, session_id: str, patient_case: PatientCase) -> Case:
        """Upsert the patient case for a session.

        If a Case already exists for this session (e.g. from an AWAITING_DATA
        retry), replace the JSON rather than inserting a duplicate.
        """
        existing: Case | None = self._s.scalar(
            select(Case).where(Case.session_id == session_id)
        )
        if existing is not None:
            existing.patient_case_json = patient_case.model_dump()
            self._s.flush()
            return existing

        case = Case(
            session_id=session_id,
            patient_case_json=patient_case.model_dump(),
        )
        self._s.add(case)
        self._s.flush()
        return case

    def save_hypotheses(
        self,
        case_id: int,
        hypotheses: list[Hypothesis],
    ) -> list[CaseHypothesis]:
        """Insert hypothesis rows; skip any already present (idempotent)."""
        existing_ids = set(
            self._s.scalars(
                select(CaseHypothesis.hypothesis_id).where(
                    CaseHypothesis.case_id == case_id
                )
            )
        )
        rows = []
        for h in hypotheses:
            if h.id not in existing_ids:
                row = CaseHypothesis(
                    case_id=case_id,
                    hypothesis_id=h.id,
                    name=h.name,
                    organ_system=h.organ_system,
                    is_must_not_miss=h.is_must_not_miss,
                )
                self._s.add(row)
                rows.append(row)
        self._s.flush()
        return rows

    def update_final_ranks(self, case_id: int, ranking: list[RankedHypothesis]) -> None:
        """Set rank_final once the synthesis node completes."""
        rank_by_id = {r.hypothesis_id: r.rank for r in ranking}
        rows = list(
            self._s.scalars(
                select(CaseHypothesis).where(CaseHypothesis.case_id == case_id)
            )
        )
        for row in rows:
            if row.hypothesis_id in rank_by_id:
                row.rank_final = rank_by_id[row.hypothesis_id]
        self._s.flush()

    def get_case(self, session_id: str) -> Case | None:
        return self._s.scalar(select(Case).where(Case.session_id == session_id))


# ---------------------------------------------------------------------------
# StudentAnswerRepository
# ---------------------------------------------------------------------------

class StudentAnswerRepository:
    def __init__(self, session: Session) -> None:
        self._s = session

    def save(
        self,
        session_id: str,
        raw_ranking: str,
        synthesis: SynthesisResult | None = None,
    ) -> StudentAnswer:
        """Persist the student's ranking text + the system's synthesis output.

        ranking_json stores both the raw string and the parsed order so future
        analytics can work without re-parsing.
        """
        ordered_ids: list[str] = []
        if synthesis:
            ordered_ids = [
                r.hypothesis_id
                for r in sorted(synthesis.ranking, key=lambda r: r.rank)
            ]

        ranking_json = {"raw": raw_ranking, "ordered_ids": ordered_ids}

        feedback_json: dict | None = None
        if synthesis:
            feedback_json = {
                "socratic_feedback": synthesis.socratic_feedback,
                "evidence_summary": synthesis.evidence_summary,
                "workup_plan": synthesis.workup_plan,
            }

        answer = StudentAnswer(
            session_id=session_id,
            ranking_json=ranking_json,
            feedback_json=feedback_json,
        )
        self._s.add(answer)
        self._s.flush()
        return answer

    def list_for_session(self, session_id: str) -> list[StudentAnswer]:
        return list(
            self._s.scalars(
                select(StudentAnswer)
                .where(StudentAnswer.session_id == session_id)
                .order_by(StudentAnswer.created_at.asc())
            )
        )
