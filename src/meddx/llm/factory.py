"""Multi-provider chat-model factory (ADR-0003).

Models are referenced as "<provider>:<model>" strings configured per agent in
meddx.config.Settings.agent_models. Agent code must call model_for_agent()
and never instantiate a provider class directly.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from meddx.config import settings

#: Providers enabled for this project (see ADR-0003). Adding one (e.g.
#: "anthropic") means: install the langchain-* package, add a kwargs branch
#: below, reference it in the agent model map. No agent code changes.
SUPPORTED_PROVIDERS = ("openai", "ollama", "google_genai")


def split_model_ref(ref: str) -> tuple[str, str]:
    """Split "provider:model" -> (provider, model), validating the provider."""
    provider, sep, model = ref.partition(":")
    if not sep or not model:
        raise ValueError(f"Model reference must look like 'provider:model', got {ref!r}")
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported provider {provider!r}; supported: {', '.join(SUPPORTED_PROVIDERS)}"
        )
    return provider, model


def provider_kwargs(provider: str) -> dict[str, Any]:
    """Provider-specific constructor kwargs taken from settings/.env."""
    if provider == "openai":
        return {"api_key": settings.openai_api_key or None}
    if provider == "ollama":
        return {"base_url": settings.ollama_base_url}
    if provider == "google_genai":
        return {"google_api_key": settings.google_api_key or None}
    raise ValueError(f"Unsupported provider {provider!r}")


@lru_cache(maxsize=None)
def model_for_agent(agent: str) -> BaseChatModel:
    """Chat model for an agent, resolved from the configured model map."""
    try:
        ref = settings.agent_models[agent]
    except KeyError:
        raise KeyError(
            f"No model configured for agent {agent!r}; known agents: "
            f"{', '.join(sorted(settings.agent_models))}"
        ) from None
    provider, model = split_model_ref(ref)
    return init_chat_model(model, model_provider=provider, **provider_kwargs(provider))
