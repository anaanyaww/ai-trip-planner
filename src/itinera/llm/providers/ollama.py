"""Ollama provider: models running on this machine.

The one provider with no quota. Every hosted free tier meters something - requests per
day, tokens per minute - and a planning run that researches, drafts and then revises can
exhaust an allowance in an afternoon. A local model has none of that, which makes it the
right choice for development, for demos that must not fail in front of an audience, and
for anyone unwilling to send a traveller's itinerary to a third party.

The cost is capability. The agents need two things from a model - tool calling and
structured output - and small local models are markedly worse at both than the hosted
models they replace. So Ollama is not the default: it is opted into through
``LLM_PROVIDER_CHAIN``, and it works best listed *alongside* hosted providers rather
than instead of them, where it can absorb the load until it struggles and hand over.

Unlike the hosted providers this one needs no credential, so it is treated as configured
whenever it is named in the chain. If the daemon is not running the call fails and the
chain moves on, which is the same handling any other unavailable provider gets.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from pydantic import SecretStr

from itinera.core.config import LLMProviderName
from itinera.llm.base import ChatProvider


class OllamaProvider(ChatProvider):
    """Builds chat models served by a local Ollama daemon."""

    name = LLMProviderName.OLLAMA

    @property
    def is_configured(self) -> bool:
        """Available whenever models are configured; a local daemon needs no key."""
        return bool(self.models)

    def build(self, model: str, api_key: SecretStr) -> BaseChatModel:
        """Construct one locally served chat model.

        Args:
            model: The Ollama model tag, such as ``qwen3:4b``.
            api_key: Ignored. Accepted so every provider shares one signature.
        """
        return ChatOllama(
            model=model,
            base_url=self._settings.ollama_base_url,
            temperature=self._settings.llm_temperature,
            num_predict=self._settings.llm_max_tokens,
            num_ctx=self._settings.ollama_num_ctx,
        )
