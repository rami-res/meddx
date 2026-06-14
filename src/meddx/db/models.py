"""SQLAlchemy ORM models for MedDx (MySQL 8 production, SQLite in tests).

Schema (from architecture overview §5):
  users            — registered students
  sessions         — one diagnostic run (= one LangGraph thread_id)
  cases            — de-identified patient case JSON per session
  hypotheses       — individual hypothesis records (final rank set at DONE)
  student_answers  — student's ranking submission + Socratic feedback
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from meddx.db.base import Base


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# users
# ---------------------------------------------------------------------------

class User(Base):
    """A student (or instructor) account.

    locale: BCP-47 code (e.g. 'uk', 'en') — used to set the response language.
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    locale: Mapped[str] = mapped_column(String(10), nullable=False, default="uk")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)

    # Sessions survive user deletion (user_id → NULL via DB ON DELETE SET NULL).
    # passive_deletes=True tells SQLAlchemy to rely on the DB FK instead of
    # loading and nullifying rows itself.
    sessions: Mapped[list[DiagnosticSession]] = relationship(
        "DiagnosticSession", back_populates="user", passive_deletes=True
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email!r}>"


# ---------------------------------------------------------------------------
# sessions
# ---------------------------------------------------------------------------

class DiagnosticSession(Base):
    """One diagnostic run: tracks the LangGraph thread_id + current phase.

    id = thread_id UUID string — same key used by the LangGraph checkpointer,
    so no secondary lookup is needed to correlate graph state with DB record.

    status: active | interrupted | done | abandoned
    phase:  mirrors meddx.schemas.Phase values
    """

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # thread_id UUID
    user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="active")
    phase: Mapped[str] = mapped_column(String(20), nullable=False, default="intake")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now, onupdate=_now)

    user: Mapped[User | None] = relationship("User", back_populates="sessions")
    case: Mapped[Case | None] = relationship(
        "Case", back_populates="session", uselist=False, cascade="all, delete-orphan"
    )
    student_answers: Mapped[list[StudentAnswer]] = relationship(
        "StudentAnswer", back_populates="session", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<DiagnosticSession id={self.id!r} phase={self.phase!r} status={self.status!r}>"


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------

class Case(Base):
    """De-identified patient case collected during the intake phase.

    patient_case_json: serialised PatientCase Pydantic model.
    One-to-one with DiagnosticSession.
    """

    __tablename__ = "cases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("sessions.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    patient_case_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)

    session: Mapped[DiagnosticSession] = relationship("DiagnosticSession", back_populates="case")
    hypotheses: Mapped[list[CaseHypothesis]] = relationship(
        "CaseHypothesis", back_populates="case", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Case id={self.id} session_id={self.session_id!r}>"


# ---------------------------------------------------------------------------
# hypotheses
# ---------------------------------------------------------------------------

class CaseHypothesis(Base):
    """One hypothesis generated for a case.

    hypothesis_id: agent-assigned ID ('h1' … 'hN').
    rank_final: NULL until the synthesis node completes; set to the system's
                final rank so student progress can be analysed later.
    """

    __tablename__ = "hypotheses"
    __table_args__ = (
        UniqueConstraint("case_id", "hypothesis_id", name="uq_case_hypothesis"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    case_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True
    )
    hypothesis_id: Mapped[str] = mapped_column(String(10), nullable=False)   # 'h1', 'h2', …
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    organ_system: Mapped[str] = mapped_column(String(100), nullable=False)
    is_must_not_miss: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    rank_final: Mapped[int | None] = mapped_column(Integer, nullable=True)   # set at DONE

    case: Mapped[Case] = relationship("Case", back_populates="hypotheses")

    def __repr__(self) -> str:
        return (
            f"<CaseHypothesis id={self.id} case_id={self.case_id} "
            f"name={self.name!r} rank={self.rank_final}>"
        )


# ---------------------------------------------------------------------------
# student_answers
# ---------------------------------------------------------------------------

class StudentAnswer(Base):
    """Student's hypothesis ranking submission and the system's Socratic feedback.

    ranking_json: raw student input + parsed order, e.g.
        {"raw": "1, 3, 2", "ordered_ids": ["h1", "h3", "h2"]}
    feedback_json: Socratic feedback text + metadata from SynthesisResult, e.g.
        {"text": "...", "evidence_summary": "...", "workup_plan": [...]}
    """

    __tablename__ = "student_answers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    ranking_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    feedback_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)

    session: Mapped[DiagnosticSession] = relationship(
        "DiagnosticSession", back_populates="student_answers"
    )

    def __repr__(self) -> str:
        return f"<StudentAnswer id={self.id} session_id={self.session_id!r}>"
