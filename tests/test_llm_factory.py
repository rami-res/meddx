"""LLM factory: model-reference parsing and provider kwargs (no network)."""

import pytest

from meddx.config import settings
from meddx.llm import SUPPORTED_PROVIDERS, provider_kwargs, split_model_ref


def test_split_model_ref():
    assert split_model_ref("openai:gpt-4.1") == ("openai", "gpt-4.1")
    assert split_model_ref("ollama:llama3.1") == ("ollama", "llama3.1")


@pytest.mark.parametrize("bad", ["gpt-4.1", "openai:", "anthropic:claude-opus-4-8"])
def test_split_model_ref_rejects_invalid(bad):
    with pytest.raises(ValueError):
        split_model_ref(bad)


def test_provider_kwargs_cover_all_supported_providers():
    for provider in SUPPORTED_PROVIDERS:
        assert isinstance(provider_kwargs(provider), dict)


def test_ollama_kwargs_use_configured_base_url():
    assert provider_kwargs("ollama")["base_url"] == settings.ollama_base_url


def test_configured_agent_models_are_valid_refs():
    """Every agent in the config map must reference a supported provider."""
    for agent, ref in settings.agent_models.items():
        provider, model = split_model_ref(ref)
        assert provider in SUPPORTED_PROVIDERS, agent
        assert model
