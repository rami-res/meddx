# 0001. Record architecture decisions

Date: 2026-06-11
Status: Accepted

## Context

This course project involves a number of non-trivial technology choices
(vector database, LLM providers, embedding model, orchestration framework,
observability stack). The reasoning behind these choices is part of the
project evaluation and must be captured at decision time, not reconstructed
afterwards.

## Decision

Keep Architecture Decision Records in the Nygard format under `docs/adr/`,
numbered sequentially (`NNNN-kebab-case-title.md`), written in English.
Accepted ADRs are immutable: superseded decisions are marked
`Superseded by NNNN` and a new record is created.

## Consequences

- Every significant choice has a recorded context and rejected alternatives.
- Small ongoing discipline cost for maintaining the records.
