# MedDx — багатоагентна RAG-система протидії когнітивним упередженням у диференційній діагностиці

Курсовий проєкт курсу «RAG Agentic AI інженер».

> ⚠️ **Лише для навчання.** Не призначений для клінічного застосування і не замінює лікаря.

---

## Про проєкт

Молоді лікарі та студенти-медики систематично припускаються діагностичних помилок
через когнітивні упередження. **MedDx** архітектурно протидіє кожному з них окремим
механізмом, а не лише нагадуванням у промпті:

| Упередження | Механізм протидії |
|---|---|
| **Anchoring** — перша гіпотеза стає «якорем» | Hypothesis-агент генерує ≥5 гіпотез *до* будь-якого ранжування; Devil's Advocate отримує список *без ранжування* |
| **Premature closure** — діагноз до повного збору даних | Intake-gate: `PatientCase` блокує перехід далі, поки не заповнені всі обов'язкові поля або не позначені «недоступно» |
| **Confirmation bias** — пошук лише підтверджень | Evidence-агент виконує *симетричний retrieval*: окремий пошук доказів ЗА і ПРОТИ кожної гіпотези |
| **Availability bias** — перевага «нещодавно баченим» | Обов'язкова категорія must-not-miss та гіпотеза з іншої системи органів |
| **Search satisficing** — зупинка на першому правдоподібному | Root-Cause агент перевіряє: чи пояснює діагноз *усі* знахідки? чи не є він наслідком глибшого стану? |

Джерела доказової бази — відкриті наукові видання (Europe PMC, PubMed/MEDLINE, PMC,
DOAJ, BMC, PLOS Medicine, Cureus). Ранжування за рівнем доказовості (мета-аналіз → RCT →
когортне → case report), а не за протоколами окремої країни.

**Ключові властивості:**
- 6 спеціалізованих LangGraph-агентів у жорсткому state machine (INTAKE → HYPOTHESES → EVIDENCE → CHALLENGE → ROOT_CAUSE → SYNTHESIS)
- Сократівський крок: студент ранжує самостійно → система порівнює і пояснює розбіжності
- Мультимовність: студент пише будь-якою мовою, пошук у літературі нормалізується до англійської
- Програмна анти-галюцинація: кожен PMID/DOI у відповіді перевіряється проти retrieved-контексту

---

## Архітектура

Детальний опис — [`docs/architecture/system-architecture-overview.md`](docs/architecture/system-architecture-overview.md)  
Рішення з обґрунтуванням альтернатив — [`docs/adr/`](docs/adr/README.md)

```
Streamlit UI
    └── LangGraph StateGraph (6 агентів + checkpointer)
            ├── Intake       — completeness gate
            ├── Hypothesis   — ≥5 unranked hypotheses
            ├── Evidence     — symmetric RAG per hypothesis
            │       └── BGE-M3 (local GPU) → Qdrant hybrid search (RRF)
            ├── Devil's Adv. — blind adversarial critique
            ├── Root-Cause   — full-organism view
            └── Synthesis    — Socratic step + ranked differential + citations
                    └── LangGraph interrupt() → student ranking → resume
```

---

## Залежності

| Бібліотека | Призначення | Версія |
|---|---|---|
| `langgraph` | Оркестрація агентів (StateGraph, checkpointer, interrupt) | ≥0.4 |
| `langchain` | Абстракція LLM-провайдерів, ланцюжки, промпти | ≥0.3 |
| `langchain-openai` | OpenAI-провайдер (GPT-4.1 за замовчуванням) | ≥0.3 |
| `langchain-ollama` | Ollama-провайдер (локальні моделі, Llama / Qwen) | ≥0.3 |
| `langchain-google-genai` | Google Gemini-провайдер | ≥2.0 |
| `langfuse` | Трейсинг, cost-tracking, evals (self-hosted) | ≥2.60 |
| `qdrant-client` | Векторна БД для літературного корпусу | ≥1.12 |
| `FlagEmbedding` | BGE-M3: мультилінгвальні dense+sparse ембедінги (GPU) | ≥1.3 |
| `sqlalchemy` | ORM для MySQL (users, sessions, cases) | ≥2.0 |
| `alembic` | Міграції схеми MySQL | ≥1.13 |
| `pymysql` | MySQL-драйвер для Python | ≥1.1 |
| `streamlit` | Web-інтерфейс студента | ≥1.40 |
| `pydantic` | Схеми стану графа та structured outputs агентів | ≥2.9 |
| `pydantic-settings` | Конфігурація з `.env` | ≥2.5 |
| `httpx` | HTTP-клієнт для Europe PMC / NCBI API | ≥0.27 |

Повний список з версіями — [`pyproject.toml`](pyproject.toml).

---

## Вимоги до оточення

| Компонент | Мінімум | Примітка |
|---|---|---|
| Python | 3.11+ | перевірено на 3.12 / 3.14 |
| Docker + Compose | Docker 24+ | для Qdrant, MySQL, Langfuse |
| GPU (CUDA) | RTX 4080 / 12 GB VRAM | тільки для BGE-M3 ембедінгів; без GPU інжест неможливий |
| RAM | 16 GB+ | 94 GB на dev-машині, але ~16 GB достатньо без GPU-моделей |
| Інтернет | потрібен | Europe PMC API (інжест), OpenAI API (агенти) |

**Без GPU:** інжест корпусу (BGE-M3) не запуститься, але весь граф і UI працюють — Evidence-агент деградує gracefully («корпус порожній»).

**Без API-ключів:** встановіть Ollama і запустіть `make start-ollama` (llama3.1:8b, ~5 GB).

---

## Запуск локально

### Передумови — одноразово

```bash
# 1. Клонуйте репозиторій
git clone <repo-url> && cd kursova

# 2. Створіть і активуйте віртуальне середовище
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Встановіть залежності
pip install -e ".[dev]"
# Якщо потрібен інжест корпусу (BGE-M3, ~2 GB):
# pip install FlagEmbedding

# 4. Створіть .env
cp .env.example .env
# Відкрийте .env і вкажіть хоча б один із:
#   OPENAI_API_KEY=sk-...         ← хмарний (рекомендовано)
#   OLLAMA_BASE_URL=http://...    ← локальний (безкоштовно)
```

### Запуск з Makefile (рекомендовано)

```bash
# Шлях A — OpenAI (найшвидший старт)
make infra          # піднімає Qdrant + MySQL (Docker)
make wait-mysql     # чекає готовності MySQL
make migrate        # Alembic: створює таблиці
make app            # Streamlit UI → http://localhost:8501

# Шлях B — Ollama (без ключів, безкоштовно)
ollama pull llama3.1:8b
make start-ollama   # infra + migrate + app з Ollama-моделями

# Всі кроки одним рядком (після заповнення .env):
make start
```

### Запуск без Makefile

```bash
# Інфраструктура
docker compose up -d qdrant mysql

# Міграція БД (після готовності MySQL ~15 с)
alembic upgrade head

# UI
streamlit run app/streamlit_app.py
```

### Перевірка, що все запущено

```bash
curl http://localhost:6333/healthz          # Qdrant → {"title":"qdrant - ..."}
docker compose ps                           # MySQL → healthy
alembic current                             # → 3468734894c4 (head)
# Streamlit відкрийте в браузері: http://localhost:8501
```

---

## Інжест корпусу (опційно, потрібен GPU)

RAG-компонент працює лише після наповнення Qdrant. Без інжесту Evidence-агент повернє порожній список цитат, але решта пайплайну (Hypothesis, Devil's Advocate, Root-Cause, Synthesis) залишиться функціональною.

```bash
# Швидка перевірка (~5 хв, 3 теми × 50 статей):
make ingest-demo

# Повний навчальний корпус (~30 хв, 15 тем × 100–150 статей):
make ingest-corpus

# Одна довільна тема:
make ingest Q="sepsis diagnosis criteria" L=100

# Стан колекції:
make corpus-status
```

---

## Тести

```bash
make test           # 155 тестів, не потребує API-ключів, GPU, Docker
make test-fast      # без інгест-тестів (~0.15 с)
make lint           # ruff
```

---

## Структура репозиторію

```
app/streamlit_app.py        — Streamlit UI (chat + форма + synthesis interrupt)
src/meddx/
  agents/                   — 6 агентів LangGraph
  graph/                    — StateGraph: вузли, ребра, conditional routing
  llm/                      — фабрика init_chat_model + Langfuse callback
  rag/                      — BGE-M3 embedder, Qdrant store, hybrid retriever
  ingestion/                — Europe PMC client, section-aware chunker
  db/                       — SQLAlchemy моделі, репозиторії, Alembic env
  schemas/                  — Pydantic: PatientCase, Hypothesis, DiagnosticState
  prompts/                  — system prompts агентів (по одному файлу)
  config.py                 — pydantic-settings: ключі, per-agent model map
scripts/
  ingest.py                 — CLI інжесту (Europe PMC → BGE-M3 → Qdrant)
  corpus_status.py          — стан Qdrant-колекції
  corpus_reset.py           — видалення колекції з підтвердженням
  run_demo.py               — демо без API-ключів (stub-агенти)
docs/
  architecture/             — архітектурний огляд (українська)
  adr/                      — Architecture Decision Records (англійська)
alembic/versions/           — міграції схеми MySQL
Makefile                    — всі команди розробки (make help)
docker-compose.yml          — Qdrant + MySQL + Langfuse self-hosted
```

---

## Корисні посилання під час роботи

| Сервіс | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| Qdrant dashboard | http://localhost:6333/dashboard |
| Langfuse (трейсинг) | http://localhost:3000 |
