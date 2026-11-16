"""Deterministic offline provider.

This exists so the system has a fourth mode alongside its three vendors: no network at
all. The test suite runs against it, which keeps tests fast, hermetic and free of
free-tier quota consumption, and the service will boot and serve requests with no API
keys configured, which makes a first clone runnable before signing up for anything.

Tests that need a specific model response push it onto ``structured_responses``;
anything not scripted falls back to a schema-satisfying stub, so a test only has to
specify the part of the conversation it actually cares about.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from enum import Enum
from types import UnionType
from typing import Any, Union, get_args, get_origin

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, SecretStr

from itinera.core.config import LLMProviderName
from itinera.llm.base import ChatProvider

_PLACEHOLDER_TEXT = "unknown"


def _stub_value(annotation: Any) -> Any:
    """Return a minimal value satisfying ``annotation``.

    Handles the shapes the domain models actually use - optionals, unions, containers,
    enums, nested models and the date types - and falls back to ``None`` for anything
    else, letting the model's own validation surface the gap.
    """
    origin = get_origin(annotation)

    if origin in (Union, UnionType):
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        return None if len(args) < len(get_args(annotation)) else _stub_value(args[0])

    if origin in (list, Sequence):
        return []
    if origin is dict:
        return {}

    if isinstance(annotation, type):
        if issubclass(annotation, Enum):
            return next(iter(annotation))
        if issubclass(annotation, BaseModel):
            return build_stub(annotation)
        if issubclass(annotation, bool):
            return False
        if issubclass(annotation, int):
            return 1
        if issubclass(annotation, float):
            return 0.0
        if issubclass(annotation, str):
            return _PLACEHOLDER_TEXT
        if issubclass(annotation, dt.datetime):
            return dt.datetime.now(dt.timezone.utc)
        if issubclass(annotation, dt.date):
            return dt.date.today()

    return None


def build_stub(schema: type[BaseModel]) -> BaseModel:
    """Construct a valid instance of ``schema`` using defaults and placeholders.

    Args:
        schema: The Pydantic model the caller asked the model to produce.

    Returns:
        An instance with every required field populated.
    """
    values: dict[str, Any] = {}
    for field_name, field in schema.model_fields.items():
        if not field.is_required():
            continue
        values[field_name] = _stub_value(field.annotation)
    return schema(**values)


class FakeChatModel(BaseChatModel):
    """A chat model that answers from a script instead of from a network call.

    Attributes:
        canned_text: Content returned for unstructured completions.
        structured_responses: Objects to return from structured calls, matched to a
            call by type and consumed once. An unscripted schema falls back to a
            generated stub.
    """

    canned_text: str = "This is a scripted response from the offline provider."
    structured_responses: list[Any] = []

    def _take_scripted(self, schema: Any) -> Any | None:
        """Remove and return the first scripted response matching ``schema``.

        Matched by type rather than by position, because a single workflow run makes
        several structured calls against different schemas. A test scripting only the
        itinerary should not have its object consumed by the research step that runs
        first.
        """
        if not isinstance(schema, type):
            return None
        for index, response in enumerate(self.structured_responses):
            if isinstance(response, schema):
                return self.structured_responses.pop(index)
        return None

    @property
    def _llm_type(self) -> str:
        """Identifier used by LangChain's tracing and serialisation."""
        return "ai-trip-planner-offline"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Return the canned completion, ignoring the conversation."""
        message = AIMessage(content=self.canned_text)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> BaseChatModel:
        """Accept tool bindings and never call them.

        Returning a model that emits no tool calls makes an agent loop terminate on its
        first turn, which is the behaviour offline runs want.
        """
        return self

    def with_structured_output(
        self, schema: Any, **kwargs: Any
    ) -> Runnable[Any, Any]:
        """Return a runnable yielding the next scripted object, or a stub.

        Args:
            schema: The Pydantic model requested by the caller.
            **kwargs: Ignored; present for signature compatibility.

        Returns:
            A runnable producing an instance of ``schema``.
        """
        def _respond(_input: Any) -> Any:
            scripted = self._take_scripted(schema)
            if scripted is not None:
                return scripted
            return build_stub(schema) if isinstance(schema, type) else None

        return RunnableLambda(_respond)


class FakeProvider(ChatProvider):
    """Builds the offline chat model."""

    name = LLMProviderName.FAKE

    @property
    def is_configured(self) -> bool:
        """Always available: the offline provider needs no credentials."""
        return True

    def build(self, model: str, api_key: SecretStr) -> BaseChatModel:
        """Construct the offline chat model, ignoring the model id and credential."""
        return FakeChatModel()
