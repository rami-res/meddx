# 0003. Multi-provider LLM access via LangChain `init_chat_model`

Date: 2026-06-11
Status: Accepted

## Context

The system runs six reasoning agents with different complexity profiles
(structured intake collection vs. adversarial critique vs. evidence
synthesis). Requirements:

- **Switchable model providers** — an explicit project requirement: the
  system must not be locked to a single vendor, and models must be
  comparable per agent.
- A free **offline mode** for development and demos: the dev machine has an
  RTX 4080 (12 GB VRAM), enough for local ~8–14B models via Ollama.
- Per-agent model selection (cheaper model for simple agents, stronger model
  for Devil's Advocate / Synthesis) driven by configuration, not code.
- Structured outputs (Pydantic) and tool calling must work across providers.

## Decision

Use **LangChain** chat-model abstraction with **`init_chat_model`** as a
single factory (`src/meddx/llm/`). Models are referenced as
`"<provider>:<model>"` strings in configuration (`src/meddx/config.py`,
overridable via `.env`), never hard-coded in agent code.

Supported providers (selected for this project):

| Provider | Package | Role |
|---|---|---|
| **OpenAI** | `langchain-openai` | Default cloud provider for reasoning agents |
| **Ollama** (local, RTX 4080) | `langchain-ollama` | Free offline mode for development/demo (Llama 3.1 8B, Qwen 2.5 14B class) |
| **Google Gemini** | `langchain-google-genai` | Alternative cloud provider for quality comparison |

A per-agent model map lives in configuration, e.g.:

```python
AGENT_MODELS = {
    "intake":          "openai:gpt-4.1-mini",
    "hypothesis":      "openai:gpt-4.1",
    "evidence":        "openai:gpt-4.1",
    "devils_advocate": "openai:gpt-4.1",
    "root_cause":      "openai:gpt-4.1",
    "synthesis":       "openai:gpt-4.1",
}
```

Note: Anthropic Claude was deliberately left out of the initial provider set
by the project owner; adding it later is a one-line configuration change
(`langchain-anthropic` + `"anthropic:claude-opus-4-8"` in the model map) —
no agent code changes required.

### Alternatives considered

- **Direct vendor SDK (single provider)** — simplest, but violates the
  switchability requirement and prevents per-agent provider comparison.
- **LiteLLM proxy** — provider abstraction at the HTTP layer; duplicates
  what LangChain already gives us and adds a running service.
- **Hand-rolled provider adapter** — reimplements `init_chat_model`,
  structured-output and tool-calling glue for no benefit.

## Consequences

- Swapping a provider or model per agent is a config/env change; A/B
  comparison of agent quality across providers becomes trivial.
- LangChain becomes a core dependency (also motivated by LangGraph,
  ADR-0005, and Langfuse integration, ADR-0006).
- Structured-output behavior differs subtly between providers — agent
  schemas must be tested against every enabled provider.
- Local Ollama models are weaker than cloud models; offline mode is for
  development and demos, not for quality benchmarks.
