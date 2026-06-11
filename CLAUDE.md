# CLAUDE.md

## Project

MedDx — a multi-agent RAG system that counters cognitive biases (anchoring,
premature closure, confirmation bias, search satisficing) in differential
diagnosis training for medical students. Six LangGraph agents (Intake,
Hypothesis, Evidence/RAG, Devil's Advocate, Root-Cause, Synthesis/Tutor)
walk a diagnostic case through a strict state machine; hypotheses are backed
by citations from open scientific sources (Europe PMC / PubMed / PMC / DOAJ /
BMC / PLOS / Cureus), ranked by evidence level rather than any single
country's protocols.

**Educational tool only — not for clinical use.** Keep the disclaimer in the
UI; never produce self-treatment advice paths.

Course project ("RAG Agentic AI engineer"); single git repo for code, docs,
and ADRs.

## Tech stack

- **Orchestration:** LangGraph `StateGraph` (phases: INTAKE → HYPOTHESES →
  EVIDENCE → CHALLENGE → ROOT_CAUSE → SYNTHESIS), checkpointer for resumable
  sessions.
- **LLM:** LangChain `init_chat_model`, multi-provider — OpenAI (default),
  Ollama (local, RTX 4080), Google Gemini. Per-agent model map in
  `src/meddx/config.py`. **Never hard-code a model/provider in agent code.**
- **RAG:** BGE-M3 embeddings locally on GPU (dense + sparse) → Qdrant hybrid
  search (RRF) with payload filters (`year`, `study_type`, `evidence_level`).
- **Storage:** Qdrant = literature corpus; MySQL 8 (SQLAlchemy 2 + Alembic) =
  users, sessions, cases, student answers.
- **Observability:** Langfuse v3 self-hosted (docker-compose), wired via
  LangChain `CallbackHandler`.
- **UI:** Streamlit.

## Repo map

```
docs/architecture/   architecture overview (Ukrainian)
docs/adr/            ADRs (English, Nygard format) — README.md is the index
src/meddx/config.py  pydantic-settings: provider keys, per-agent model map
src/meddx/llm/       init_chat_model factory + Langfuse callback
src/meddx/graph/     LangGraph StateGraph assembly, phase transitions
src/meddx/agents/    agent nodes (intake, hypothesis, evidence, devils_advocate, root_cause, synthesis)
src/meddx/rag/       BGE-M3 embeddings, Qdrant store, hybrid retriever
src/meddx/ingestion/ Europe PMC / NCBI E-utilities clients, chunking
src/meddx/db/        SQLAlchemy models, repositories (MySQL)
src/meddx/schemas/   Pydantic: PatientCase, Hypothesis, Citation, graph state
src/meddx/prompts/   agent system prompts (one file per agent)
app/streamlit_app.py UI entry point
scripts/             ingestion and maintenance scripts
data/                raw corpus cache (gitignored)
```

## Commands

```bash
docker compose up -d                  # Qdrant :6333, MySQL :3306, Langfuse :3000
pip install -e ".[dev]"               # install package + dev tools
streamlit run app/streamlit_app.py    # run UI
pytest                                # tests
ruff check src tests                  # lint
alembic upgrade head                  # DB migrations (once set up)
```

## Conventions

- **ADRs in English** in `docs/adr/` (Nygard format). Any significant
  technology or architecture decision gets a new numbered ADR; update the
  index in `docs/adr/README.md`. Accepted ADRs are immutable — supersede,
  don't edit.
- **Architecture docs in Ukrainian** (course deliverable) in
  `docs/architecture/`.
- Agent prompts live in `src/meddx/prompts/`, one file per agent — not
  inline in code.
- All structured agent outputs are Pydantic models in `src/meddx/schemas/`.
- **Anti-bias invariants** (do not weaken when editing agents):
  - Intake completeness gate blocks phase transition until required
    `PatientCase` fields are filled or explicitly marked unavailable.
  - Hypothesis agent returns ≥5 unranked hypotheses incl. must-not-miss.
  - Evidence agent retrieves FOR and AGAINST evidence separately per
    hypothesis; literature queries are normalized to English.
  - Devil's Advocate never receives hypothesis ranking (enforce via graph
    state projection, not prompts).
  - Citation validation is programmatic: every PMID/DOI in an answer must
    exist in the retrieved context.
- Secrets only via `.env` (see `.env.example`); never commit keys, never
  put keys in code or prompts.
- Students may write in any language; reply in the user's language, search
  literature in English.

## Environment notes

- Dev machine: Manjaro Linux, i9-14900HX, RTX 4080 Mobile 12GB VRAM, 94GB
  RAM, docker + docker-compose. BGE-M3 and Ollama models run locally on GPU.
- Full local stack (incl. Langfuse: web, worker, Postgres, ClickHouse,
  Redis, MinIO) starts with one `docker compose up -d`.
