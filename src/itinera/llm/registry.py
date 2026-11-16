"""Assembly of the chat-model failover chain.

The chain is a list of *candidates*, not a list of vendors. Each candidate is one
(provider, key, model) combination, and the chain tries them in order until one answers.
That distinction matters because free tiers are metered per key and per model: a Groq
key that has spent its tokens-per-minute allowance on one model has spent it for that
key, so the useful next attempt is a different key or a different model - not simply a
different vendor, and certainly not giving up.

Vendors are still ordered deliberately. Candidates are grouped by the provider order in
``LLM_PROVIDER_CHAIN``, and within a provider every key is tried against every model, so
the chain exhausts cheap local alternatives before moving to another vendor.

Failover happens *within* a single call rather than between calls, so an agent
mid-reasoning does not abort because one candidate hit its quota.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import BaseModel, SecretStr

from itinera.core.config import LLMProviderName, Settings
from itinera.core.exceptions import ConfigurationError
from itinera.core.logging import get_logger
from itinera.llm.base import ChatProvider
from itinera.llm.providers.fake import FakeProvider
from itinera.llm.providers.gemini import GeminiProvider
from itinera.llm.providers.groq import GroqProvider
from itinera.llm.providers.ollama import OllamaProvider
from itinera.llm.providers.openrouter import OpenRouterProvider

logger = get_logger(__name__)

_PROVIDER_TYPES: dict[LLMProviderName, type[ChatProvider]] = {
    LLMProviderName.GROQ: GroqProvider,
    LLMProviderName.GEMINI: GeminiProvider,
    LLMProviderName.OPENROUTER: OpenRouterProvider,
    LLMProviderName.OLLAMA: OllamaProvider,
    LLMProviderName.FAKE: FakeProvider,
}


@dataclass(frozen=True)
class Candidate:
    """One way of serving a request: a vendor, a credential and a model."""

    provider: ChatProvider
    model: str
    api_key: SecretStr
    key_index: int

    @property
    def label(self) -> str:
        """Human-readable identifier for logs, never including the credential."""
        suffix = "" if self.key_index == 0 else f" (key {self.key_index + 1})"
        return f"{self.provider.name.value}:{self.model}{suffix}"


class ModelRegistry:
    """Resolves the configured providers into a chain of candidate runnables.

    Built models are cached per candidate so repeated agent turns reuse one client -
    and therefore one rate limiter - instead of resetting the token bucket on every
    call, which would defeat the pacing entirely.
    """

    def __init__(self, settings: Settings) -> None:
        """Resolve which candidates are usable under the current configuration.

        Args:
            settings: Active configuration supplying the chain order and credentials.

        Raises:
            ConfigurationError: If the chain names a provider with no implementation.
        """
        self._settings = settings
        self._model_cache: dict[tuple[LLMProviderName, str, int], BaseChatModel] = {}
        self._candidates = self._resolve_candidates()

    def _resolve_candidates(self) -> list[Candidate]:
        """Expand the configured providers into every usable candidate."""
        resolved: list[Candidate] = []

        for name in self._settings.provider_chain:
            provider_type = _PROVIDER_TYPES.get(name)
            if provider_type is None:  # pragma: no cover - guarded by Settings validation
                raise ConfigurationError(f"No implementation registered for provider {name!r}")

            provider = provider_type(self._settings)
            if not provider.is_configured:
                logger.info(
                    "llm.provider_skipped", provider=name.value, reason="no API key configured"
                )
                continue

            for key_index, api_key in enumerate(provider.api_keys):
                for model in provider.models:
                    resolved.append(Candidate(provider, model, api_key, key_index))

        if not resolved:
            logger.warning(
                "llm.no_providers_configured",
                detail="falling back to the offline provider; plans will be placeholders",
            )
            offline = FakeProvider(self._settings)
            resolved.append(Candidate(offline, "fake", SecretStr("offline"), 0))

        logger.info(
            "llm.chain_resolved",
            candidates=[candidate.label for candidate in resolved],
            count=len(resolved),
        )
        return resolved

    @property
    def candidates(self) -> tuple[Candidate, ...]:
        """The active chain, in failover order."""
        return tuple(self._candidates)

    @property
    def primary(self) -> Candidate:
        """The candidate that serves requests when everything is healthy."""
        return self._candidates[0]

    @property
    def is_offline(self) -> bool:
        """Whether the chain resolved to the offline provider only."""
        return all(
            candidate.provider.name is LLMProviderName.FAKE for candidate in self._candidates
        )

    def chat_model(self, name: LLMProviderName, model: str | None = None) -> BaseChatModel:
        """Return a built chat model for ``name``.

        Exposed so the offline provider's scripted responses can be reached without
        patching module state, and so diagnostics can inspect what was actually built.

        Args:
            name: The provider to look up.
            model: A specific model, or ``None`` for that provider's first candidate.

        Raises:
            ConfigurationError: If no matching candidate is in the active chain.
        """
        for candidate in self._candidates:
            if candidate.provider.name is name and (model is None or candidate.model == model):
                return self._model_for(candidate)
        raise ConfigurationError(
            f"Provider {name.value!r} is not in the active chain"
            if model is None
            else f"Model {model!r} for provider {name.value!r} is not in the active chain"
        )

    def _model_for(self, candidate: Candidate) -> BaseChatModel:
        """Return the cached chat model for ``candidate``, building it on first use."""
        cache_key = (candidate.provider.name, candidate.model, candidate.key_index)
        if cache_key not in self._model_cache:
            self._model_cache[cache_key] = candidate.provider.build(
                candidate.model, candidate.api_key
            )
        return self._model_cache[cache_key]

    def _chain(self, adapt: Any, skip: int = 0) -> Runnable[Any, Any]:
        """Build a failover chain by applying ``adapt`` to every candidate.

        Args:
            adapt: Callable turning a chat model into the runnable the caller wants,
                such as binding tools or a structured-output schema.
            skip: How many leading candidates to leave out. Failover only advances when
                a call *raises*; a model that answers with an empty object has not
                failed by that measure, so retrying it produces the same empty object.
                Skipping lets a caller that judged the answer unusable escalate past the
                candidate that produced it.

        Returns:
            The first remaining candidate's runnable, with the rest attached as
            fallbacks. Skipping past the end keeps the last candidate rather than
            leaving the caller with nothing to call.
        """
        candidates = self._candidates[skip:] or self._candidates[-1:]
        runnables: list[Runnable[Any, Any]] = [
            adapt(self._model_for(candidate)) for candidate in candidates
        ]
        primary, fallbacks = runnables[0], runnables[1:]
        if not fallbacks:
            return primary
        return primary.with_fallbacks(fallbacks)

    def tool_chain(self, tools: Sequence[BaseTool]) -> Runnable[Any, Any]:
        """Return a tool-bound chain for agent loops.

        Args:
            tools: Tools to expose to the model.

        Returns:
            A runnable that falls through to the next candidate on failure.
        """
        return self._chain(lambda model: model.bind_tools(tools))

    def structured_chain(
        self, schema: type[BaseModel], skip: int = 0
    ) -> Runnable[Any, Any]:
        """Return a chain that produces instances of ``schema``.

        Args:
            schema: The Pydantic model the caller wants back.
            skip: Candidates to leave out of the front of the chain, for a caller
                retrying after an answer that was returned successfully but unusable.

        Returns:
            A runnable that falls through to the next candidate on failure.
        """
        return self._chain(lambda model: model.with_structured_output(schema), skip=skip)

    def plain_chain(self) -> Runnable[Any, Any]:
        """Return a chain for free-text completions with no tools or schema."""
        return self._chain(lambda model: model)

    def describe(self) -> list[dict[str, str]]:
        """Return the active chain for health reporting, without credentials."""
        return [
            {"provider": candidate.provider.name.value, "model": candidate.model}
            for candidate in self._candidates
        ]
