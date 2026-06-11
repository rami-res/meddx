# MedDx — багатоагентна RAG-система для протидії когнітивним упередженням у диференційній діагностиці

Курсовий проєкт («RAG Agentic AI інженер»). Система допомагає
студентам-медикам відпрацьовувати дисципліновану диференційну діагностику:
повний збір даних, широкий диференційний ряд, активний пошук спростувань,
пошук першопричини — з цитатами з відкритих наукових джерел (PubMed/MEDLINE,
PMC, Europe PMC, DOAJ, BMC, PLOS Medicine, Cureus).

> ⚠️ Освітній інструмент. Не призначений для клінічного застосування.

## Документація

- **Архітектура:** [docs/architecture/system-architecture-overview.md](docs/architecture/system-architecture-overview.md)
- **Рішення (ADR, англійською):** [docs/adr/](docs/adr/README.md)

## Стек

LangGraph (оркестрація 6 агентів) · LangChain `init_chat_model`
(OpenAI / Ollama / Gemini з перемиканням) · Qdrant (hybrid search) ·
BGE-M3 локально (мультилінгвальні ембедінги) · MySQL (користувачі/записи) ·
Langfuse self-hosted (трейсинг) · Streamlit (UI).

## Quick start

```bash
# 1. Інфраструктура (Qdrant, MySQL, Langfuse)
cp .env.example .env        # заповнити ключі
docker compose up -d

# 2. Залежності
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 3. UI
streamlit run app/streamlit_app.py
```

Langfuse UI: http://localhost:3000 · Qdrant dashboard: http://localhost:6333/dashboard
