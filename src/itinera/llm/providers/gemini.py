"""Google AI Studio provider.

A different vendor from the others rather than a second instance of one of them. A
fallback sharing the primary's infrastructure fails at the same moment the primary
does, which defeats the purpose of having one.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from itinera.core.config import LLMProviderName
from itinera.llm.base import ChatProvider


class GeminiProvider(ChatProvider):
    """Builds Gemini-backed chat models."""

    name = LLMProviderName.GEMINI

    def build(self, model: str, api_key: SecretStr) -> BaseChatModel:
        """Construct one Gemini chat model."""
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=self._settings.llm_temperature,
            max_output_tokens=self._settings.llm_max_tokens,
            timeout=self._settings.llm_timeout_seconds,
            rate_limiter=self._rate_limiter(),
            max_retries=0,
        )
