"""DB layer tests — all run against an in-memory SQLite engine.

No MySQL container is required. SQLAlchemy's JSON type maps to TEXT in SQLite
and round-trips correctly. Foreign-key enforcement is enabled via PRAGMA.

Covers:
  - Base: engine creation, session context manager, FK enforcement
  - Models: table creation, column defaults, repr strings
  - UserRepository: get_or_create idempotency
  - SessionRepository: create/get/update_phase/list_for_user
  - CaseRepository: save_case upsert, save_hypotheses idempotency, rank update
  - StudentAnswerRepository: save with/without synthesis, list_for_session
  - Cross-repo: cascade deletes, session→case→hypotheses chain
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from meddx.db.base import Base, get_session, make_engine, make_session_factory
from meddx.db.models import (
    Case,
    CaseHypothesis,
    DiagnosticSession,
    StudentAnswer,
    User,
)
from meddx.db.repositories import (
    CaseRepository,
    SessionRepository,
    StudentAnswerRepository,
    UserRepository,
)
from meddx.schemas import (
    Hypothesis,
    PatientCase,
    RankedHypothesis,
    SynthesisResult,
    UNAVAILABLE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    eng = make_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)


@pytest.fixture()
def db(engine) -> Session:
    """Open a session; roll back after each test to keep tests isolated."""
    factory = make_session_factory(engine)
    session: Session = factory()
    yield session
    session.rollback()
    session.close()


# Convenience lambdas for repo construction
def _users(s): return UserRepository(s)
def _sessions(s): return SessionRepository(s)
def _cases(s): return CaseRepository(s)
def _answers(s): return StudentAnswerRepository(s)


def _complete_case() -> PatientCase:
    return PatientCase(
        chief_complaint="Chest pain",
        history_of_present_illness="Retrosternal, episodic",
        past_medical_history="Hypertension",
        medications="Amlodipine",
        family_history="Father: MI",
        systems_review="Mild fever",
        risk_factors="Smoker",
        available_investigations=UNAVAILABLE,
    )


def _hypotheses() -> list[Hypothesis]:
    return [
        Hypothesis(id="h1", name="Acute coronary syndrome", rationale="...",
                   organ_system="cardiovascular", is_must_not_miss=True),
        Hypothesis(id="h2", name="Pulmonary embolism", rationale="...",
                   organ_system="respiratory", is_must_not_miss=True),
        Hypothesis(id="h3", name="GERD", rationale="...",
                   organ_system="gastrointestinal"),
    ]


def _synthesis() -> SynthesisResult:
    return SynthesisResult(
        ranking=[
            RankedHypothesis(hypothesis_id="h1", rank=1),
            RankedHypothesis(hypothesis_id="h2", rank=2),
            RankedHypothesis(hypothesis_id="h3", rank=3),
        ],
        workup_plan=["ECG", "Troponin"],
        evidence_summary="Supported by RCT evidence.",
        socratic_feedback="Good ranking!",
    )


# ---------------------------------------------------------------------------
# Engine / base tests
# ---------------------------------------------------------------------------

class TestEngine:
    def test_tables_created(self, engine):
        from sqlalchemy import inspect
        insp = inspect(engine)
        tables = insp.get_table_names()
        assert "users" in tables
        assert "sessions" in tables
        assert "cases" in tables
        assert "hypotheses" in tables
        assert "student_answers" in tables

    def test_sqlite_fk_pragma_enabled(self, engine):
        with engine.connect() as conn:
            result = conn.execute(text("PRAGMA foreign_keys"))
            assert result.fetchone()[0] == 1

    def test_get_session_commits_on_success(self, engine):
        factory = make_session_factory(engine)
        with get_session(factory) as s:
            user = User(email="test_commit@example.com", name="Test")
            s.add(user)

        with get_session(factory) as s:
            found = s.query(User).filter_by(email="test_commit@example.com").first()
            assert found is not None
            s.delete(found)  # clean up

    def test_get_session_rolls_back_on_exception(self, engine):
        factory = make_session_factory(engine)
        try:
            with get_session(factory) as s:
                s.add(User(email="rollback@example.com", name="Rollback"))
                raise RuntimeError("forced error")
        except RuntimeError:
            pass

        with get_session(factory) as s:
            found = s.query(User).filter_by(email="rollback@example.com").first()
            assert found is None


# ---------------------------------------------------------------------------
# UserRepository
# ---------------------------------------------------------------------------

class TestUserRepository:
    def test_create_user(self, db):
        user = _users(db).get_or_create("alice@example.com", "Alice")
        assert user.id is not None
        assert user.email == "alice@example.com"
        assert user.locale == "uk"

    def test_get_or_create_idempotent(self, db):
        r = _users(db)
        u1 = r.get_or_create("bob@example.com", "Bob")
        u2 = r.get_or_create("bob@example.com", "Robert")  # same email, different name
        assert u1.id == u2.id
        assert u2.name == "Bob"  # original name preserved

    def test_get_by_email_returns_none_when_missing(self, db):
        assert _users(db).get_by_email("nobody@example.com") is None

    def test_get_by_id(self, db):
        user = _users(db).get_or_create("carol@example.com", "Carol")
        fetched = _users(db).get_by_id(user.id)
        assert fetched is not None
        assert fetched.email == "carol@example.com"

    def test_locale_default(self, db):
        user = _users(db).get_or_create("dave@example.com", "Dave")
        assert user.locale == "uk"

    def test_custom_locale(self, db):
        user = _users(db).get_or_create("eve@example.com", "Eve", locale="en")
        assert user.locale == "en"

    def test_repr(self, db):
        user = _users(db).get_or_create("frank@example.com", "Frank")
        assert "frank@example.com" in repr(user)


# ---------------------------------------------------------------------------
# SessionRepository
# ---------------------------------------------------------------------------

THREAD_A = "aaaa-1111-bbbb-2222-cccc"
THREAD_B = "bbbb-1111-aaaa-3333-dddd"


class TestSessionRepository:
    def test_create_session(self, db):
        sess = _sessions(db).create(THREAD_A)
        assert sess.id == THREAD_A
        assert sess.phase == "intake"
        assert sess.status == "active"

    def test_get_session(self, db):
        _sessions(db).create(THREAD_A)
        fetched = _sessions(db).get(THREAD_A)
        assert fetched is not None
        assert fetched.id == THREAD_A

    def test_get_returns_none_for_unknown(self, db):
        assert _sessions(db).get("nonexistent-id") is None

    def test_get_or_create_creates_if_missing(self, db):
        sess = _sessions(db).get_or_create(THREAD_A)
        assert sess.id == THREAD_A

    def test_get_or_create_idempotent(self, db):
        s1 = _sessions(db).get_or_create(THREAD_A)
        s2 = _sessions(db).get_or_create(THREAD_A)
        assert s1.id == s2.id

    def test_update_phase(self, db):
        _sessions(db).create(THREAD_A)
        _sessions(db).update_phase(THREAD_A, phase="synthesis", status="interrupted")
        sess = _sessions(db).get(THREAD_A)
        assert sess.phase == "synthesis"
        assert sess.status == "interrupted"

    def test_update_phase_unknown_raises(self, db):
        with pytest.raises(ValueError, match="not found"):
            _sessions(db).update_phase("nonexistent", phase="done")

    def test_list_for_user(self, db):
        user = _users(db).get_or_create("henry@example.com", "Henry")
        _sessions(db).create(THREAD_A, user_id=user.id)
        _sessions(db).create(THREAD_B, user_id=user.id)
        sessions = _sessions(db).list_for_user(user.id)
        assert len(sessions) == 2
        ids = {s.id for s in sessions}
        assert THREAD_A in ids and THREAD_B in ids

    def test_session_with_user_link(self, db):
        user = _users(db).get_or_create("ivan@example.com", "Ivan")
        sess = _sessions(db).create(THREAD_A, user_id=user.id)
        assert sess.user_id == user.id

    def test_repr(self, db):
        sess = _sessions(db).create(THREAD_A)
        assert THREAD_A in repr(sess)
        assert "intake" in repr(sess)


# ---------------------------------------------------------------------------
# CaseRepository
# ---------------------------------------------------------------------------

class TestCaseRepository:
    def _session(self, db) -> DiagnosticSession:
        return _sessions(db).create(THREAD_A)

    def test_save_case(self, db):
        self._session(db)
        case = _cases(db).save_case(THREAD_A, _complete_case())
        assert case.id is not None
        assert case.session_id == THREAD_A
        assert case.patient_case_json["chief_complaint"] == "Chest pain"

    def test_save_case_upsert(self, db):
        self._session(db)
        case1 = _cases(db).save_case(THREAD_A, _complete_case())
        updated = PatientCase(
            chief_complaint="Acute headache",
            history_of_present_illness="Sudden onset",
            past_medical_history=UNAVAILABLE,
            medications=UNAVAILABLE,
            family_history=UNAVAILABLE,
            systems_review=UNAVAILABLE,
            risk_factors=UNAVAILABLE,
            available_investigations=UNAVAILABLE,
        )
        case2 = _cases(db).save_case(THREAD_A, updated)
        assert case1.id == case2.id
        assert case2.patient_case_json["chief_complaint"] == "Acute headache"

    def test_patient_case_json_roundtrip(self, db):
        self._session(db)
        original = _complete_case()
        case = _cases(db).save_case(THREAD_A, original)
        fetched = _cases(db).get_case(THREAD_A)
        assert fetched.patient_case_json == original.model_dump()

    def test_get_case_returns_none_when_missing(self, db):
        assert _cases(db).get_case("nonexistent") is None

    def test_save_hypotheses(self, db):
        self._session(db)
        case = _cases(db).save_case(THREAD_A, _complete_case())
        rows = _cases(db).save_hypotheses(case.id, _hypotheses())
        assert len(rows) == 3
        assert rows[0].hypothesis_id == "h1"
        assert rows[0].is_must_not_miss is True

    def test_save_hypotheses_idempotent(self, db):
        self._session(db)
        case = _cases(db).save_case(THREAD_A, _complete_case())
        _cases(db).save_hypotheses(case.id, _hypotheses())
        rows2 = _cases(db).save_hypotheses(case.id, _hypotheses())
        assert rows2 == []  # no duplicates inserted

    def test_update_final_ranks(self, db):
        self._session(db)
        case = _cases(db).save_case(THREAD_A, _complete_case())
        _cases(db).save_hypotheses(case.id, _hypotheses())
        ranking = [
            RankedHypothesis(hypothesis_id="h1", rank=1),
            RankedHypothesis(hypothesis_id="h2", rank=2),
            RankedHypothesis(hypothesis_id="h3", rank=3),
        ]
        _cases(db).update_final_ranks(case.id, ranking)

        hyps = db.query(CaseHypothesis).filter_by(case_id=case.id).all()
        rank_map = {h.hypothesis_id: h.rank_final for h in hyps}
        assert rank_map == {"h1": 1, "h2": 2, "h3": 3}

    def test_rank_starts_as_none(self, db):
        self._session(db)
        case = _cases(db).save_case(THREAD_A, _complete_case())
        _cases(db).save_hypotheses(case.id, _hypotheses())
        hyps = db.query(CaseHypothesis).filter_by(case_id=case.id).all()
        assert all(h.rank_final is None for h in hyps)

    def test_repr(self, db):
        self._session(db)
        case = _cases(db).save_case(THREAD_A, _complete_case())
        assert THREAD_A in repr(case)


# ---------------------------------------------------------------------------
# StudentAnswerRepository
# ---------------------------------------------------------------------------

class TestStudentAnswerRepository:
    def _session_with_case(self, db) -> tuple[str, int]:
        _sessions(db).create(THREAD_A)
        case = _cases(db).save_case(THREAD_A, _complete_case())
        _cases(db).save_hypotheses(case.id, _hypotheses())
        return THREAD_A, case.id

    def test_save_without_synthesis(self, db):
        _sessions(db).create(THREAD_A)
        answer = _answers(db).save(THREAD_A, "1, 2, 3")
        assert answer.id is not None
        assert answer.ranking_json["raw"] == "1, 2, 3"
        assert answer.ranking_json["ordered_ids"] == []
        assert answer.feedback_json is None

    def test_save_with_synthesis(self, db):
        self._session_with_case(db)
        synthesis = _synthesis()
        answer = _answers(db).save(THREAD_A, "h1 > h2 > h3", synthesis=synthesis)
        assert answer.ranking_json["raw"] == "h1 > h2 > h3"
        assert answer.ranking_json["ordered_ids"] == ["h1", "h2", "h3"]
        assert answer.feedback_json["socratic_feedback"] == "Good ranking!"
        assert answer.feedback_json["workup_plan"] == ["ECG", "Troponin"]

    def test_feedback_json_roundtrip(self, db):
        self._session_with_case(db)
        synthesis = _synthesis()
        answer = _answers(db).save(THREAD_A, "1, 2, 3", synthesis=synthesis)
        fetched = _answers(db).list_for_session(THREAD_A)
        assert len(fetched) == 1
        assert fetched[0].feedback_json["evidence_summary"] == "Supported by RCT evidence."

    def test_list_for_session(self, db):
        _sessions(db).create(THREAD_A)
        _answers(db).save(THREAD_A, "3, 1, 2")
        _answers(db).save(THREAD_A, "1, 2, 3")
        answers = _answers(db).list_for_session(THREAD_A)
        assert len(answers) == 2
        # ordered by created_at ascending
        assert answers[0].ranking_json["raw"] == "3, 1, 2"

    def test_repr(self, db):
        _sessions(db).create(THREAD_A)
        answer = _answers(db).save(THREAD_A, "1, 2, 3")
        assert THREAD_A in repr(answer)


# ---------------------------------------------------------------------------
# Cross-repo: cascade deletes
# ---------------------------------------------------------------------------

class TestCascadeDeletes:
    def test_deleting_case_deletes_hypotheses(self, db):
        _sessions(db).create(THREAD_A)
        case = _cases(db).save_case(THREAD_A, _complete_case())
        _cases(db).save_hypotheses(case.id, _hypotheses())

        db.delete(case)
        db.flush()

        hyps = db.query(CaseHypothesis).filter_by(case_id=case.id).all()
        assert hyps == []

    def test_deleting_session_cascades_to_case_and_answers(self, db):
        sess = _sessions(db).create(THREAD_A)
        _cases(db).save_case(THREAD_A, _complete_case())
        _answers(db).save(THREAD_A, "1, 2, 3")

        db.delete(sess)
        db.flush()

        assert _cases(db).get_case(THREAD_A) is None
        assert _answers(db).list_for_session(THREAD_A) == []

    def test_deleting_user_sets_session_user_id_null(self, db):
        user = _users(db).get_or_create("zara@example.com", "Zara")
        _sessions(db).create(THREAD_A, user_id=user.id)

        db.delete(user)
        db.flush()

        sess = _sessions(db).get(THREAD_A)
        assert sess is not None
        assert sess.user_id is None
