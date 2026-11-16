"""Groq provider.

Chosen for its throughput and its keyless-signup free tier. Groq serves open-weight
models only, so the configured identifiers are model families rather than proprietary
endpoints.

Its free tier meters tokens per minute rather than only requests, and the ceiling is
low enough that a long revision prompt can be rejected outright. That is why several
models and, where the account allows, several keys are configured: each is metered
separately, so the next candidate in the chain often succeeds where the previous one
was refused.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from pydantic import SecretStr

from itinera.core.config import LLMProviderName
from itinera.llm.base import ChatProvider


class GroqProvider(ChatProvider):
    """Builds Groq-backed chat models."""

    name = LLMProviderName.GROQ

    def build(self, model: str, api_key: SecretStr) -> BaseChatModel:
        """Construct one Groq chat model."""
        return ChatGroq(
            model_name=model,
            groq_api_key=api_key,
            temperature=self._settings.llm_temperature,
            max_tokens=self._settings.llm_max_tokens,
            request_timeout=self._settings.llm_timeout_seconds,
            rate_limiter=self._rate_limiter(),
            max_retries=0,
        )
