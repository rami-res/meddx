# Architecture Decision Records (ADR)

Log of architecture decisions for this project, in the Michael Nygard format
(Context / Decision / Consequences). ADRs are written in English.

## Index

| # | Title | Status |
|---|---|---|
| [0001](0001-record-architecture-decisions.md) | Record architecture decisions | Accepted |
| [0002](0002-qdrant-vector-db.md) | Qdrant as the vector database | Accepted |
| [0003](0003-multi-provider-llm-via-langchain.md) | Multi-provider LLM access via LangChain `init_chat_model` | Accepted |
| [0004](0004-local-bge-m3-embeddings.md) | Local BGE-M3 embeddings for multilingual hybrid search | Accepted |
| [0005](0005-langgraph-orchestration.md) | LangGraph for agent orchestration | Accepted |
| [0006](0006-langfuse-self-hosted-observability.md) | Self-hosted Langfuse for observability | Accepted |
| [0007](0007-mysql-for-users-and-records.md) | MySQL for users and diagnostic case records | Accepted |

## Template for new ADRs

```markdown
# NNNN. Decision title

Date: YYYY-MM-DD
Status: Proposed | Accepted | Deprecated | Superseded by NNNN

## Context
What forces the decision; constraints that apply.

## Decision
What was decided and why. Which alternatives were considered and rejected.

## Consequences
What becomes easier, what becomes harder; risks and mitigations.
```

Rules: ADRs are immutable after acceptance — outdated decisions are marked
`Superseded by NNNN` and a new record is created.
