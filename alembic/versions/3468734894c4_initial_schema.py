"""Initial schema: users, sessions, cases, hypotheses, student_answers.

Revision ID: 3468734894c4
Revises:
Create Date: 2026-06-14
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers used by Alembic
revision: str = "3468734894c4"
down_revision: str | None = None
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


def upgrade() -> None:
    # ── users ────────────────────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("locale", sa.String(10), nullable=False, server_default="uk"),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    # ── sessions ─────────────────────────────────────────────────────────────
    op.create_table(
        "sessions",
        sa.Column("id", sa.String(36), primary_key=True, comment="LangGraph thread_id UUID"),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="active"),
        sa.Column("phase", sa.String(20), nullable=False, server_default="intake"),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"),
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_sessions_user_id",
            ondelete="SET NULL",
        ),
    )
    op.create_index("ix_sessions_user_id", "sessions", ["user_id"])

    # ── cases ─────────────────────────────────────────────────────────────────
    op.create_table(
        "cases",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.String(36), nullable=False),
        sa.Column("patient_case_json", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.id"],
            name="fk_cases_session_id",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("session_id", name="uq_cases_session_id"),
    )

    # ── hypotheses ────────────────────────────────────────────────────────────
    op.create_table(
        "hypotheses",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("case_id", sa.Integer(), nullable=False),
        sa.Column("hypothesis_id", sa.String(10), nullable=False, comment="h1, h2, …"),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("organ_system", sa.String(100), nullable=False),
        sa.Column("is_must_not_miss", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("rank_final", sa.Integer(), nullable=True, comment="NULL until synthesis"),
        sa.ForeignKeyConstraint(
            ["case_id"],
            ["cases.id"],
            name="fk_hypotheses_case_id",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("case_id", "hypothesis_id", name="uq_case_hypothesis"),
    )
    op.create_index("ix_hypotheses_case_id", "hypotheses", ["case_id"])

    # ── student_answers ───────────────────────────────────────────────────────
    op.create_table(
        "student_answers",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.String(36), nullable=False),
        sa.Column("ranking_json", sa.JSON(), nullable=False),
        sa.Column("feedback_json", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.id"],
            name="fk_student_answers_session_id",
            ondelete="CASCADE",
        ),
    )
    op.create_index("ix_student_answers_session_id", "student_answers", ["session_id"])


def downgrade() -> None:
    op.drop_table("student_answers")
    op.drop_table("hypotheses")
    op.drop_table("cases")
    op.drop_table("sessions")
    op.drop_table("users")
