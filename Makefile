## MedDx — development commands
## Usage: make <target>
##
## Quick start (OpenAI):
##   cp .env.example .env   # add OPENAI_API_KEY
##   make start
##
## Quick start (Ollama, no API key):
##   ollama pull llama3.1:8b
##   make start-ollama

PYTHON     := .venv/bin/python
STREAMLIT  := .venv/bin/streamlit
PYTEST     := .venv/bin/pytest
RUFF       := .venv/bin/ruff
ALEMBIC    := .venv/bin/alembic
PIP        := .venv/bin/pip

# Core infra (Qdrant + MySQL only — no Langfuse).
# Use 'make infra-full' if you also want tracing.
CORE_SERVICES := qdrant mysql

.PHONY: help start start-ollama stop restart \
        infra infra-full infra-down \
        migrate migrate-sql migrate-down \
        app app-debug \
        ingest ingest-demo ingest-corpus \
        corpus-status corpus-reset \
        test test-fast lint \
        install install-dev setup \
        db-shell qdrant-dashboard langfuse

# ---------------------------------------------------------------------------
# Default: print help
# ---------------------------------------------------------------------------

help:
	@sed -n 's/^## //p' Makefile
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## ' Makefile | \
	    awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# High-level workflows
# ---------------------------------------------------------------------------

start: infra wait-mysql migrate app  ## Full start: infra up + migrate + Streamlit UI

start-ollama: ## Start using Ollama instead of OpenAI (no API key needed)
	@echo "Switching agent models to Ollama (llama3.1:8b)..."
	@$(PYTHON) -c "\
import json, pathlib; \
p = pathlib.Path('src/meddx/config.py'); \
print('Set MEDDX_AGENT_MODELS in .env to override — see src/meddx/config.py');\
"
	MEDDX_AGENT_MODELS='{"intake":"ollama:llama3.1:8b","hypothesis":"ollama:llama3.1:8b","evidence":"ollama:llama3.1:8b","devils_advocate":"ollama:llama3.1:8b","root_cause":"ollama:llama3.1:8b","synthesis":"ollama:llama3.1:8b"}' \
	    $(MAKE) start

stop: infra-down  ## Stop all Docker services

restart: stop start  ## Stop then start everything

# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------

infra:  ## Start core infra only: Qdrant + MySQL (no Langfuse)
	docker compose up -d $(CORE_SERVICES)
	@echo "Qdrant dashboard → http://localhost:6333/dashboard"

infra-full:  ## Start full stack including Langfuse observability
	docker compose up -d
	@echo "Langfuse UI      → http://localhost:3000  (create account on first run)"
	@echo "Qdrant dashboard → http://localhost:6333/dashboard"

infra-down:  ## Stop and remove all containers (data volumes are kept)
	docker compose down

wait-mysql:  ## Wait until MySQL is accepting connections
	@echo "Waiting for MySQL to be healthy..."
	@until docker compose exec -T mysql mysqladmin ping -u root -proot --silent 2>/dev/null; do \
	    printf '.'; sleep 2; \
	done
	@echo " MySQL is up."

# ---------------------------------------------------------------------------
# Database migrations
# ---------------------------------------------------------------------------

migrate:  ## Apply all pending Alembic migrations
	$(ALEMBIC) upgrade head

migrate-sql:  ## Print the SQL that 'make migrate' would run (dry run)
	$(ALEMBIC) upgrade head --sql

migrate-down:  ## Roll back the last migration
	$(ALEMBIC) downgrade -1

migrate-history:  ## Show migration history
	$(ALEMBIC) history --verbose

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app:  ## Run the Streamlit UI (http://localhost:8501)
	$(STREAMLIT) run app/streamlit_app.py

app-debug:  ## Run Streamlit with verbose logging
	$(STREAMLIT) run app/streamlit_app.py --logger.level debug

# ---------------------------------------------------------------------------
# Corpus ingestion
# Prerequisites: Qdrant running (make infra) + FlagEmbedding installed
#   pip install FlagEmbedding   # downloads BAAI/bge-m3 ~2 GB on first run
# No LLM keys needed — ingestion uses only BGE-M3 locally.
# ---------------------------------------------------------------------------

# Overridable per-call: make ingest Q="sepsis" L=200 Y=2015 BS=8
Q  ?= chest pain differential diagnosis
L  ?= 100
Y  ?= 2010
BS ?= 12

ingest:  ## Ingest one query  →  make ingest Q="headache" L=200 Y=2015
	$(PYTHON) scripts/ingest.py \
	    --query "$(Q)" \
	    --limit $(L) \
	    --min-year $(Y) \
	    --batch-size $(BS)

# ---------------------------------------------------------------------------
# ingest-demo: quick smoke-test (3 topics, 50 articles each, ~5 min on GPU)
# ---------------------------------------------------------------------------

ingest-demo:  ## Quick corpus smoke-test: 3 topics × 50 articles (~5 min)
	$(PYTHON) scripts/ingest.py --query "chest pain differential diagnosis"  --limit 50 --min-year 2015
	$(PYTHON) scripts/ingest.py --query "fever of unknown origin"             --limit 50 --min-year 2015
	$(PYTHON) scripts/ingest.py --query "dyspnea acute causes"               --limit 50 --min-year 2015
	@$(MAKE) corpus-status

# ---------------------------------------------------------------------------
# ingest-corpus: full training corpus (15 clinical topics, ~1500 articles)
# Estimated time: 25–40 min on RTX 4080 (BGE-M3 batch_size=12).
# Safe to re-run: upsert is idempotent (deduplicates by PMID/chunk_index).
# ---------------------------------------------------------------------------

ingest-corpus:  ## Full corpus: 15 clinical topics × 100–150 articles (~30 min)
	@echo "=== MedDx corpus ingestion — 15 clinical topics ==="
	@echo "    Qdrant: $(shell grep QDRANT_URL .env 2>/dev/null | cut -d= -f2 || echo http://localhost:6333)"
	@echo ""
	@echo "[1/15] Chest pain"
	$(PYTHON) scripts/ingest.py --query "chest pain differential diagnosis myocardial infarction" \
	    --limit 150 --min-year 2010 --batch-size $(BS)
	@echo ""
	@echo "[2/15] Dyspnea / breathlessness"
	$(PYTHON) scripts/ingest.py --query "dyspnea breathlessness acute causes pulmonary embolism" \
	    --limit 150 --min-year 2010 --batch-size $(BS)
	@echo ""
	@echo "[3/15] Fever of unknown origin"
	$(PYTHON) scripts/ingest.py --query "fever of unknown origin etiology diagnosis" \
	    --limit 150 --min-year 2010 --batch-size $(BS)
	@echo ""
	@echo "[4/15] Headache"
	$(PYTHON) scripts/ingest.py --query "headache differential diagnosis secondary causes" \
	    --limit 100 --min-year 2010 --batch-size $(BS)
	@echo ""
	@echo "[5/15] Acute abdominal pain"
	$(PYTHON) scripts/ingest.py --query "acute abdominal pain differential diagnosis appendicitis" \
	    --limit 150 --min-year 2010 --batch-size $(BS)
	@echo ""
	@echo "[6/15] Jaundice"
	$(PYTHON) scripts/ingest.py --query "jaundice etiology hepatitis cholestasis diagnosis" \
	    --limit 100 --min-year 2010 --batch-size $(BS)
	@echo ""
	@echo "[7/15] Hemoptysis"
	$(PYTHON) scripts/ingest.py --query "hemoptysis causes pulmonary tuberculosis lung cancer" \
	    --limit 100 --min-year 2010 --batch-size $(BS)
	@echo ""
	@echo "[8/15] Edema"
	$(PYTHON) scripts/ingest.py --query "peripheral edema lower extremity causes heart failure" \
	    --limit 100 --min-year 2010 --batch-size $(BS)
	@echo ""
	@echo "[9/15] Syncope"
	$(PYTHON) scripts/ingest.py --query "syncope etiology cardiovascular vasovagal diagnosis" \
	    --limit 100 --min-year 2010 --batch-size $(BS)
	@echo ""
	@echo "[10/15] Unexplained weight loss"
	$(PYTHON) scripts/ingest.py --query "unexplained weight loss malignancy cancer differential diagnosis" \
	    --limit 100 --min-year 2010 --batch-size $(BS)
	@echo ""
	@echo "[11/15] Anemia"
	$(PYTHON) scripts/ingest.py --query "anemia differential diagnosis iron deficiency hemolytic" \
	    --limit 100 --min-year 2010 --batch-size $(BS)
	@echo ""
	@echo "[12/15] Hematuria"
	$(PYTHON) scripts/ingest.py --query "hematuria etiology urological kidney diagnosis" \
	    --limit 100 --min-year 2010 --batch-size $(BS)
	@echo ""
	@echo "[13/15] Altered mental status / delirium"
	$(PYTHON) scripts/ingest.py --query "delirium altered mental status encephalopathy etiology" \
	    --limit 100 --min-year 2010 --batch-size $(BS)
	@echo ""
	@echo "[14/15] Joint pain / arthritis"
	$(PYTHON) scripts/ingest.py --query "arthritis joint pain differential diagnosis rheumatoid" \
	    --limit 100 --min-year 2010 --batch-size $(BS)
	@echo ""
	@echo "[15/15] Hypertension secondary causes"
	$(PYTHON) scripts/ingest.py --query "secondary hypertension causes endocrine renal" \
	    --limit 100 --min-year 2010 --batch-size $(BS)
	@echo ""
	@echo "=== Ingestion complete ==="
	@$(MAKE) corpus-status

# ---------------------------------------------------------------------------
# Corpus inspection and maintenance
# ---------------------------------------------------------------------------

corpus-status:  ## Show Qdrant collection stats: doc count, vector config
	$(PYTHON) scripts/corpus_status.py

corpus-reset:  ## DELETE all ingested data from Qdrant (irreversible, prompts for YES)
	$(PYTHON) scripts/corpus_reset.py

# ---------------------------------------------------------------------------
# Tests & linting
# ---------------------------------------------------------------------------

test:  ## Run full test suite (117+ tests, no API keys or GPU needed)
	$(PYTEST) tests/ -q

test-fast:  ## Run tests except the slow ingestion suite
	$(PYTEST) tests/ -q --ignore=tests/test_ingestion.py

test-db:  ## Run only DB layer tests
	$(PYTEST) tests/test_db.py -v

lint:  ## Run ruff linter
	$(RUFF) check src tests

lint-fix:  ## Auto-fix ruff lint errors
	$(RUFF) check --fix src tests

# ---------------------------------------------------------------------------
# Setup (first run)
# ---------------------------------------------------------------------------

setup:  ## Create .env from .env.example if it doesn't exist
	@if [ ! -f .env ]; then \
	    cp .env.example .env; \
	    echo ".env created from .env.example — fill in your API keys."; \
	else \
	    echo ".env already exists."; \
	fi

install:  ## Install production dependencies into .venv
	$(PIP) install -e .

install-dev:  ## Install all dependencies including dev tools
	$(PIP) install -e ".[dev]"

# ---------------------------------------------------------------------------
# Convenience shortcuts
# ---------------------------------------------------------------------------

db-shell:  ## Open a MySQL shell inside the container
	docker compose exec mysql mysql -u meddx -pmeddx meddx

qdrant-dashboard:  ## Open Qdrant dashboard in the browser
	xdg-open http://localhost:6333/dashboard

langfuse:  ## Open Langfuse UI in the browser
	xdg-open http://localhost:3000

demo:  ## Run the graph demo without API keys (stub agents)
	$(PYTHON) scripts/run_demo.py
