# 0005. LangGraph for agent orchestration

Date: 2026-06-11
Status: Accepted

## Context

The diagnostic pipeline is a deterministic state machine with fixed phases
(`INTAKE → HYPOTHESES → EVIDENCE → CHALLENGE → ROOT_CAUSE → SYNTHESIS`) and
strict rules about **what each agent is allowed to see** (e.g. the Devil's
Advocate must not see hypothesis ranking; the Intake gate must block phase
transitions until the case data is complete). The session must survive
process restarts (a student can continue a case later), and the UI needs
streaming of intermediate agent output.

LangGraph is also an explicit technology requirement of the project.

## Decision

Orchestrate agents with **LangGraph** (`StateGraph`):

- Each agent is a **node**; phase transitions are **conditional edges**
  (the Intake completeness gate is a conditional edge predicate).
- Information visibility is controlled by the **state schema**: each node
  receives a projection of the shared state, so anti-bias rules ("blind"
  Devil's Advocate) are enforced structurally, not by prompt discipline.
- **Checkpointer** persists graph state per session (SQLite for local dev,
  MySQL-backed for the deployed variant) — resumable sessions and
  human-in-the-loop pauses (the Socratic step in Synthesis) for free.
- Graph assembly lives in `src/meddx/graph/`; nodes in `src/meddx/agents/`.

### Alternatives considered

- **Hand-rolled orchestrator** (plain Python state machine over the SDK) —
  maximum control and transparency, but reimplements checkpointing,
  streaming, retries and HITL interrupts that LangGraph provides; also
  fails the project's framework requirement.
- **CrewAI / AutoGen** — designed for free-form agent collaboration; we need
  the opposite: a rigid state machine with controlled information flow.
- **LlamaIndex Workflows** — viable, but the project standardizes on the
  LangChain ecosystem (ADR-0003, ADR-0006), and LangGraph integrates
  natively with it.

## Consequences

- Checkpointing, streaming, and human-in-the-loop interrupts come built in;
  the Socratic pause in Synthesis maps directly onto LangGraph interrupts.
- Native Langfuse integration via LangChain callbacks (ADR-0006).
- Framework abstraction cost: graph state typing and node contracts must be
  kept disciplined (Pydantic state schema in `src/meddx/schemas/`), or the
  shared state degrades into an untyped dict.
