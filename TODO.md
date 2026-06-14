# TODO — запуск і розвиток проєкту

Стан на 2026-06-14:
- ✅ Каркас LangGraph (6 stub-агентів, intake-gate, валідатор цитат)
- ✅ Hypothesis-агент — реальний LLM-виклик з інваріантною валідацією
- ✅ Інжест корпусу (Europe PMC → chunker → BGE-M3 → Qdrant), 65 тестів
- ⏳ Evidence Agent — stub (retrieval не підключений)
- ⏳ MySQL-моделі, Streamlit UI

## Старт середовища (разово, ~20 хв)

- [ ] **0. Зафіксувати каркас у git**
  ```bash
  git add -A && git commit -m "Project skeleton: architecture, ADRs, LLM factory, LangGraph pipeline with stub agents"
  ```
- [ ] **1. Перевірити, що каркас живий** (без жодних ключів)
  ```bash
  source .venv/bin/activate
  pytest                        # очікувано: 14 passed
  python scripts/run_demo.py    # повний прогін пайплайна на stub-агентах
  ```
- [ ] **2. Повне встановлення залежностей**
  ```bash
  pip install -e ".[dev]"
  ```
  ⚠️ `FlagEmbedding` (BGE-M3) тягне PyTorch (~2.5 GB). Щоб відкласти:
  ```bash
  pip install streamlit sqlalchemy pymysql alembic langfuse qdrant-client httpx \
      langchain-openai langchain-ollama langchain-google-genai
  ```
  а FlagEmbedding — коли дійде до інжесту (крок 7).
- [ ] **3. Інфраструктура**
  ```bash
  cp .env.example .env
  docker compose up -d          # Qdrant :6333, MySQL :3306, Langfuse :3000
  ```
  Разово для Langfuse: відкрити http://localhost:3000 → створити
  акаунт/organization/project → скопіювати `LANGFUSE_PUBLIC_KEY` і
  `LANGFUSE_SECRET_KEY` у `.env`. Без ключів трейсинг просто вимкнений
  (no-op) — нічого не ламається.
- [ ] **4. Ключі LLM-провайдерів** (достатньо одного з трьох)
  | Режим | Що потрібно |
  |---|---|
  | Хмарний (основний) | `OPENAI_API_KEY` у `.env` |
  | Порівняння | `GOOGLE_API_KEY` у `.env` |
  | Офлайн/безкоштовно | `ollama serve` + `ollama pull llama3.1:8b` (або `qwen2.5:14b` — влізе у 12GB VRAM); у `src/meddx/config.py` переключити model map на `ollama:...` |

## Реалізація (рекомендований порядок)

- [x] **5. Перший реальний агент — Hypothesis**: `with_structured_output`,
  `load_prompt`, 3 інваріанти кодом (≥5 гіпотез, must-not-miss, ≥2 системи
  органів), `ValueError` → LangGraph retry. Тести: 9 unit + conftest-мок.
- [x] **6. Інжест корпусу**: реалізовано повний пайплайн:
  - `src/meddx/ingestion/europe_pmc.py` — cursor-пагінація, класифікація
    study_type → evidence_level (1=meta-analysis … 8=other)
  - `src/meddx/ingestion/chunker.py` — section-aware split (regex) +
    sliding-window fallback, детермінований point_id (MD5 → UUID)
  - `src/meddx/rag/embedder.py` — BGE-M3 (deferred import, GPU)
  - `src/meddx/rag/store.py` — Qdrant named vectors (dense+sparse, IDF),
    idempotent upsert
  - `scripts/ingest.py` — CLI: `--query --limit --min-year --batch-size`
  - 42 нових тести (без мережі, GPU, Qdrant)
  ```bash
  # Для запуску реального інжесту (потрібні: Qdrant + pip install FlagEmbedding):
  python scripts/ingest.py --query "chest pain differential diagnosis" --limit 100
  python scripts/ingest.py --query "fever of unknown origin" --limit 50
  ```
- [x] **7. Evidence Agent на реальному retrieval**: реалізовано:
  - `src/meddx/rag/retriever.py` — `HybridRetriever(client, collection, embedder)`
    з DI; `search(query, k, max_evidence_level)` → RRF(dense+sparse) + fallback
    без фільтру якщо результатів замало; `get_retriever()` lru_cache singleton
  - `src/meddx/agents/evidence.py` — LLM генерує EN-запити для ВСІХ гіпотез за
    один виклик (`_EvidenceQueriesResult`); fallback-шаблони для пропущених ID;
    симетричний retrieval (2 пошуки × N гіпотез); `_deduplicate()`; graceful
    degradation коли корпус порожній (`is_ready() = False`)
  - 20 нових тестів (без GPU, Qdrant, API-ключів)
- [x] **8. Решта агентів**: Intake-діалог (допит відсутніх полів),
  Devil's Advocate (тільки через `blind_view()`), Root-Cause,
  Synthesis із сократівським кроком (LangGraph `interrupt()`).
  Реалізовано:
  - `src/meddx/agents/intake.py` — completeness gate (code, not LLM) +
    conversational LLM question for missing fields; `route_after_intake()`
  - `src/meddx/agents/devils_advocate.py` — `blind_view()` projection
    (patient_case + alphabetically sorted names only, no rationale/ranking);
    `_ChallengeOutput` list-based schema → `ChallengeReport`
  - `src/meddx/agents/root_cause.py` — full context (case + hypotheses +
    evidence counts + challenge) → `RootCauseAssessment`
  - `src/meddx/agents/synthesis.py` — `interrupt()` Socratic step →
    LLM ranking → programmatic citation attachment from `state.evidence` →
    `assert_citations_grounded()` → `SynthesisResult`
  - `tests/test_remaining_agents.py` — 39 unit tests for all 4 agents
  - `tests/conftest.py` + `tests/test_graph.py` — updated with mocks for
    all new agents; 117 tests, 0 failures
- [ ] **9. Streamlit UI**: чат + форма кейсу + цикл `AWAITING_DATA` →
  допит → перезапуск графа (checkpointer з `thread_id`).
- [ ] **10. MySQL**: SQLAlchemy-моделі (`src/meddx/db/`) + `alembic init` —
  коли з'являться користувачі/сесії в UI.

## Пізніше (з architecture overview §11)

- [ ] Evals: датасет кейсів з еталонними диференційними рядами; метрики
  (повнота ряду, must-not-miss recall, citation accuracy) через Langfuse.
- [ ] Локальний reranker (bge-reranker-v2-m3) поверх hybrid retrieval.
- [ ] Веб-пошук свіжих публікацій поверх локального індексу.
- [ ] Персональна аналітика студента: типові упередження → рекомендації.
