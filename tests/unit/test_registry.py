"""Tests for the failover chain's construction.

The chain is a list of (provider, key, model) candidates rather than a list of vendors,
because free tiers meter per key and per model. When a request is refused, the useful
next attempt is often a different key or a different model on the *same* vendor, so the
chain has to contain those before it moves on.
"""

from __future__ import annotations

from itinera.core.config import LLMProviderName, Settings
from itinera.llm.registry import ModelRegistry


def _settings(**overrides: object) -> Settings:
    base = {
        "_env_file": None,
        "llm_provider_chain": "groq,openrouter,gemini",
        "groq_api_key": "groq-key",
        "groq_models": "model-a,model-b",
        "openrouter_api_key": "or-key",
        "openrouter_models": "or-model",
        "google_api_key": "gemini-key",
        "gemini_models": "gemini-model",
    }
    return Settings(**{**base, **overrides})  # type: ignore[arg-type]


def test_every_key_is_tried_against_every_model() -> None:
    registry = ModelRegistry(_settings(groq_api_key_alt="groq-key-2"))

    groq = [c.label for c in registry.candidates if c.provider.name is LLMProviderName.GROQ]

    assert groq == [
        "groq:model-a",
        "groq:model-b",
        "groq:model-a (key 2)",
        "groq:model-b (key 2)",
    ]


def test_providers_keep_their_configured_order() -> None:
    registry = ModelRegistry(_settings())

    assert [c.provider.name.value for c in registry.candidates] == [
        "groq",
        "groq",
        "openrouter",
        "gemini",
    ]
    assert registry.primary.model == "model-a"


def test_a_provider_without_a_key_is_skipped_entirely() -> None:
    """An unconfigured fallback must cost no latency at call time."""
    registry = ModelRegistry(_settings(openrouter_api_key=None))

    assert not any(
        c.provider.name is LLMProviderName.OPENROUTER for c in registry.candidates
    )


def test_a_repeated_key_is_not_counted_twice() -> None:
    """Pasting the same key into both slots adds no quota, so it adds no candidates."""
    registry = ModelRegistry(_settings(groq_api_key_alt="groq-key"))

    groq = [c for c in registry.candidates if c.provider.name is LLMProviderName.GROQ]

    assert len(groq) == 2


def test_duplicate_model_ids_are_collapsed() -> None:
    registry = ModelRegistry(_settings(groq_models="model-a, model-a ,model-b"))

    groq = [c.model for c in registry.candidates if c.provider.name is LLMProviderName.GROQ]

    assert groq == ["model-a", "model-b"]


def test_no_credentials_at_all_degrades_to_the_offline_provider() -> None:
    registry = ModelRegistry(
        _settings(groq_api_key=None, openrouter_api_key=None, google_api_key=None)
    )

    assert registry.is_offline
    assert registry.describe() == [{"provider": "fake", "model": "fake"}]


def test_every_candidate_is_reachable_as_a_fallback() -> None:
    """The chain must attach all of them, not just the first of each vendor."""
    registry = ModelRegistry(_settings(groq_api_key_alt="groq-key-2"))
    chain = registry.plain_chain()

    assert len(registry.candidates) == 6
    assert len(chain.fallbacks) == len(registry.candidates) - 1


def test_describe_never_leaks_a_credential() -> None:
    described = ModelRegistry(_settings()).describe()

    assert all(set(entry) == {"provider", "model"} for entry in described)
    assert "groq-key" not in str(described)


def test_ollama_needs_no_credential() -> None:
    """A local daemon has no key, so the provider must not be skipped for lacking one."""
    registry = ModelRegistry(
        _settings(
            llm_provider_chain="ollama,groq",
            ollama_models="qwen3:4b,llama3.2:3b",
        )
    )

    labels = [c.label for c in registry.candidates]

    assert labels[:2] == ["ollama:qwen3:4b", "ollama:llama3.2:3b"]
    assert registry.primary.provider.name is LLMProviderName.OLLAMA
    assert not registry.is_offline


def test_ollama_can_run_with_no_hosted_provider_at_all() -> None:
    """The point of local models: a usable chain with no accounts anywhere."""
    registry = ModelRegistry(
        _settings(
            llm_provider_chain="ollama",
            groq_api_key=None,
            openrouter_api_key=None,
            google_api_key=None,
        )
    )

    assert not registry.is_offline, "a local model is a real provider, not the stub"
    assert registry.describe() == [{"provider": "ollama", "model": "qwen3:4b"}]


def test_ollama_models_are_built_with_the_configured_context_window() -> None:
    """A window wider than the GPU holds spills into RAM and collapses throughput."""
    registry = ModelRegistry(_settings(llm_provider_chain="ollama", ollama_num_ctx=8192))

    model = registry.chat_model(LLMProviderName.OLLAMA)

    assert model.num_ctx == 8192


def test_skipping_advances_past_the_candidate_that_answered_badly() -> None:
    registry = ModelRegistry(_settings())
    labels = [c.label for c in registry.candidates]

    full = registry.plain_chain()
    skipped = registry._chain(lambda model: model, skip=1)

    assert len(labels) == 4
    assert len(full.fallbacks) == 3
    assert len(skipped.fallbacks) == 2


def test_skipping_past_the_end_still_returns_something_callable() -> None:
    """Exhausting the chain must not leave a caller with nothing to invoke."""
    registry = ModelRegistry(_settings())

    chain = registry._chain(lambda model: model, skip=99)

    assert chain is not None
    assert not getattr(chain, "fallbacks", [])
