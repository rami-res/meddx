"""Alembic env.py — loads DB URL from meddx.config.settings so the
same .env file drives both the application and migrations.

Usage:
  alembic upgrade head          # apply all migrations
  alembic revision --autogenerate -m "description"   # generate new migration
  alembic downgrade -1          # roll back one revision
"""

import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool

from alembic import context

# Make 'src/' importable when running alembic from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# ── app imports ─────────────────────────────────────────────────────────────
from meddx.config import settings          # noqa: E402
from meddx.db.base import Base             # noqa: E402
import meddx.db.models  # noqa: E402, F401 — registers all models on Base.metadata

# ── alembic config ──────────────────────────────────────────────────────────
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Override the URL from alembic.ini with the one from pydantic-settings / .env
config.set_main_option("sqlalchemy.url", settings.mysql_url)

target_metadata = Base.metadata


# ── offline mode (generates SQL without a live DB connection) ────────────────
def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


# ── online mode (applies migrations to a live DB) ───────────────────────────
def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
