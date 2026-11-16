"""OpenRouter provider.

Reached through the OpenAI-compatible client, since OpenRouter exposes that wire
protocol. It aggregates many vendors behind one key, which makes it the most useful
place to list several models: each `:free` identifier is served by a different upstream
with its own limits, so one being exhausted says nothing about the next.

The free roster changes without notice, which is precisely why the identifiers are
configuration rather than constants.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from itinera.core.config import LLMProviderName
from itinera.llm.base import ChatProvider


class OpenRouterProvider(ChatProvider):
    """Builds OpenRouter-backed chat models."""

    name = LLMProviderName.OPENROUTER

    def build(self, model: str, api_key: SecretStr) -> BaseChatModel:
        """Construct one OpenRouter chat model."""
        return ChatOpenAI(
            model_name=model,
            openai_api_key=api_key,
            openai_api_base=self._settings.openrouter_base_url,
            temperature=self._settings.llm_temperature,
            max_tokens=self._settings.llm_max_tokens,
            request_timeout=self._settings.llm_timeout_seconds,
            rate_limiter=self._rate_limiter(),
            max_retries=0,
        )
