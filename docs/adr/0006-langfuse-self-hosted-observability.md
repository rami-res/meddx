# 0006. Self-hosted Langfuse for observability

Date: 2026-06-11
Status: Accepted

## Context

A six-agent pipeline is impossible to debug and evaluate from logs alone:
we need per-session traces (every node, prompt, response), token/cost
accounting per provider (we run several — ADR-0003), prompt versioning, and
a base for future evals (differential completeness, must-not-miss recall,
citation accuracy). Langfuse is an explicit technology requirement of the
project. The dev machine (94 GB RAM, Docker) easily hosts the full stack.

## Decision

Run **Langfuse v3 self-hosted** in the project's `docker-compose.yml`
(langfuse-web, langfuse-worker, PostgreSQL, ClickHouse, Redis, MinIO,
following the official compose file). Instrumentation via the Langfuse
**LangChain `CallbackHandler`** attached to LangGraph runs — one integration
point covers all agents and all providers.

### Alternatives considered

- **Langfuse Cloud (free tier)** — simplest setup (keys only), but event
  limits (~50k/month) are easy to exceed during ingestion experiments,
  requires internet, and session data leaves the host. Self-hosting keeps
  everything local and unlimited.
- **LangSmith** — closed SaaS, paid beyond a small free tier; project
  explicitly standardizes on Langfuse.
- **No observability / plain logging** — unacceptable for debugging a
  multi-agent pipeline and provides no eval substrate.

## Consequences

- Full traces, costs, and prompt management with no usage limits; data
  stays on the host — convenient for the course demo.
- The Langfuse stack adds ~6 containers to `docker-compose.yml`; heavier
  local infrastructure (fine on this machine, but documented so it can be
  swapped for Langfuse Cloud by changing `.env` only).
- Langfuse datasets/scores become the natural home for the planned evals.
