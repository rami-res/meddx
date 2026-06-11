# 0007. MySQL for users and diagnostic case records

Date: 2026-06-11
Status: Accepted

## Context

Besides the knowledge corpus (vectors — ADR-0002), the system stores
transactional data: user accounts, diagnostic sessions and their phase
state, anonymized training cases, generated hypotheses, student answers and
feedback. MySQL is an explicit project requirement for this data. A clear
separation of concerns is desired: the vector store holds *knowledge*, the
relational store holds *application state*.

## Decision

Use **MySQL 8** (Docker container in `docker-compose.yml`) with
**SQLAlchemy 2.0** ORM and **Alembic** migrations (`src/meddx/db/`).

Initial schema (overview):

```
users(id, email, name, locale, created_at)
sessions(id, user_id, status, phase, created_at)
cases(id, session_id, patient_case_json, created_at)
hypotheses(id, case_id, name, is_must_not_miss, rank_final)
student_answers(id, session_id, ranking_json, score, feedback_json)
```

Semi-structured payloads (case data, rankings, feedback) are stored as JSON
columns — they are read/written whole by the application and validated by
Pydantic schemas, so relational decomposition adds no value at this stage.

The LangGraph checkpointer (ADR-0005) may persist graph state into the same
MySQL instance (separate tables), keeping all session state in one place.

### Alternatives considered

- **SQLite** — zero infrastructure, but fails the explicit MySQL
  requirement and complicates concurrent access from Streamlit sessions.
  Still used for throwaway local tests where convenient.
- **PostgreSQL** — technically equal or better (and already present inside
  the Langfuse stack), but the requirement names MySQL; running app data in
  Langfuse's internal Postgres would couple unrelated lifecycles.
- **Storing everything in Qdrant payloads** — abuses the vector store as an
  application database: no transactions, no relational integrity.

## Consequences

- Clean separation: Qdrant = knowledge corpus, MySQL = users/sessions/records.
- Alembic migrations document schema evolution (useful for the course
  write-up).
- One more container and connection configuration (`MYSQL_*` in `.env`);
  trivial on the dev machine.
