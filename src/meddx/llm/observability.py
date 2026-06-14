"""Langfuse tracing for LangChain/LangGraph runs (ADR-0006).

Usage: pass callbacks into graph invocation config:
    graph.invoke(state, config={"callbacks": langfuse_callbacks(), ...})
Tracing is a no-op when Langfuse keys are not configured, so local runs and
tests work without the observability stack.
"""

from __future__ import annotations

from meddx.config import settings


def langfuse_callbacks() -> list:
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return []
    try:
        # langfuse v3
        from langfuse.langchain import CallbackHandler  # type: ignore[import-not-found]

        return [CallbackHandler()]
    except ImportError:
        # langfuse v2
        from langfuse.callback import CallbackHandler  # type: ignore[import-not-found]

        return [
            CallbackHandler(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
        ]
