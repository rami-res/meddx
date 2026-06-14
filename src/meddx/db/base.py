"""SQLAlchemy engine + session factory.

Production target: MySQL 8 (via settings.mysql_url).
Tests use an in-memory SQLite engine passed in directly, avoiding the MySQL
dependency — see tests/test_db.py for the fixture pattern.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


def make_engine(url: str, echo: bool = False):
    """Create a SQLAlchemy engine.

    For SQLite URLs, enables foreign-key enforcement (off by default in SQLite).
    """
    engine = create_engine(url, echo=echo, future=True)

    if url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_connection, connection_record):  # noqa: ANN001
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def make_session_factory(engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@contextmanager
def get_session(session_factory: sessionmaker[Session]) -> Generator[Session, None, None]:
    """Yield a transactional session, rolling back on exception."""
    session: Session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
