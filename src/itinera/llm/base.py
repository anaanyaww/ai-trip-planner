"""Chat-model provider abstraction.

Business logic never names a vendor. Agents ask the registry for a model and receive a
LangChain runnable; which company serves the tokens is a configuration concern. That
boundary is what makes the failover chain a `.env` edit rather than a code change, and
it is what lets the entire test suite run against a deterministic fake without a single
line of agent code knowing the difference.

A provider is a *factory*, not a single model. It builds a chat model for a given model
identifier and a given API key, because the failover chain is built from every
combination of the two: several models per vendor, and several keys per vendor where
the account allows it. A vendor's free tier is usually metered per key and per model,
so a second key or a different model is often the difference between a rejected request
and a served one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from pydantic import SecretStr

from itinera.core.config import LLMProviderName, Settings

_SECONDS_PER_MINUTE = 60.0
_RATE_LIMITER_POLL_SECONDS = 0.1


class ChatProvider(ABC):
    """A chat-model vendor that can supply candidates for the failover chain."""

    name: LLMProviderName

    def __init__(self, settings: Settings) -> None:
        """Store configuration; models are built on demand.

        Args:
            settings: Active configuration supplying keys, model ids and limits.
        """
        self._settings = settings

    @property
    def models(self) -> tuple[str, ...]:
        """Configured model identifiers for this provider, in preference order."""
        return self._settings.llm_models(self.name)

    @property
    def api_keys(self) -> tuple[SecretStr, ...]:
        """Every usable API key configured for this provider."""
        return self._settings.llm_api_keys(self.name)

    @property
    def is_configured(self) -> bool:
        """Whether this provider has at least one key and one model.

        A provider without credentials is skipped when the chain is assembled rather
        than being allowed to fail on first call, so an unconfigured fallback costs no
        latency.
        """
        return bool(self.api_keys) and bool(self.models)

    def _rate_limiter(self) -> InMemoryRateLimiter:
        """Build a limiter pacing this provider below its published free-tier ceiling.

        Applied inside the chat model, so it throttles every call the agent loop makes
        without the agent having to know a quota exists.
        """
        return InMemoryRateLimiter(
            requests_per_second=self._settings.llm_rate_limit_rpm / _SECONDS_PER_MINUTE,
            check_every_n_seconds=_RATE_LIMITER_POLL_SECONDS,
            max_bucket_size=float(self._settings.llm_rate_limit_rpm),
        )

    @abstractmethod
    def build(self, model: str, api_key: SecretStr) -> BaseChatModel:
        """Construct one chat model.

        Args:
            model: The model identifier to serve.
            api_key: The credential to serve it with.

        Returns:
            A ready-to-use LangChain chat model.
        """

    def __repr__(self) -> str:
        """Return a debug representation naming the provider and its models."""
        return f"{type(self).__name__}(models={list(self.models)!r})"
