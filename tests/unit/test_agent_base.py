"""Tests for the shared tool-calling loop.

The loop's job is to keep a model's misbehaviour from becoming a failed run: a tool
that errors, a tool that does not exist, a model that never stops calling tools. Each
of those is exercised here against a model scripted to misbehave in exactly one way.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, create_model

from itinera.agents.base import ToolCallingAgent
from itinera.core.exceptions import WorkflowError
from itinera.llm.providers.fake import FakeChatModel
from itinera.llm.registry import ModelRegistry


class Answer(BaseModel):
    """The schema the test agent extracts."""

    text: str = ""


class ScriptedModel(FakeChatModel):
    """An offline model that emits a scripted sequence of tool calls."""

    tool_call_script: list[list[dict[str, Any]]] = []
    turns: int = 0

    def bind_tools(self, tools: Any, **kwargs: Any) -> BaseChatModel:
        """Ignore the bound tools; the script decides what is called."""
        return self

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """Return the next scripted turn, then stop asking for tools."""
        index = self.turns
        self.turns += 1
        calls = self.tool_call_script[index] if index < len(self.tool_call_script) else []
        return AIMessage(content="", tool_calls=calls)

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable[Any, Any]:
        """Return the extraction result, ignoring the transcript."""
        return RunnableLambda(lambda _: Answer(text="extracted"))


def _call(name: str, call_id: str = "call-1", **args: Any) -> dict[str, Any]:
    return {"name": name, "id": call_id, "args": args, "type": "tool_call"}


class SingleModelRegistry:
    """A registry standing in for the real one, with no failover and one model.

    The agent takes its chains from a registry, so supplying a stand-in is how a test
    controls what the model does without reaching into the agent's internals.
    """

    def __init__(self, model: BaseChatModel) -> None:
        """Store the single model both chains resolve to."""
        self._model = model

    @property
    def candidates(self) -> tuple[Any, ...]:
        """One candidate, so escalation logging has something to name."""
        return (SimpleNamespace(label="stub:model"),)

    def tool_chain(self, tools: Any) -> Any:
        """Return the model with the tools bound."""
        return self._model.bind_tools(tools)

    def structured_chain(self, schema: Any, skip: int = 0) -> Any:
        """Return the model with the schema bound, ignoring the escalation offset."""
        return self._model.with_structured_output(schema)


def _agent(
    model: BaseChatModel,
    tools: list[BaseTool],
    max_iterations: int = 4,
    extraction_attempts: int = 3,
) -> ToolCallingAgent[Answer]:
    """Build an agent driven by ``model``."""
    return ToolCallingAgent(
        name="test",
        registry=SingleModelRegistry(model),  # type: ignore[arg-type]
        tools=tools,
        output_schema=Answer,
        max_iterations=max_iterations,
        extraction_attempts=extraction_attempts,
    )


def _tool(
    name: str, argument: str = "query", result: str = "ok", fails: bool = False
) -> tuple[BaseTool, list[dict[str, Any]]]:
    """Return a tool and the list recording every call made to it.

    The argument schema is declared explicitly because LangChain drops arguments a
    tool does not declare, which would make every call look empty.
    """
    calls: list[dict[str, Any]] = []
    schema = create_model(f"{name.title()}Input", **{argument: (str, "")})

    async def _run(**kwargs: Any) -> str:
        calls.append(kwargs)
        if fails:
            raise RuntimeError("the upstream refused")
        return result

    tool = StructuredTool.from_function(
        coroutine=_run, name=name, description=name, args_schema=schema
    )
    return tool, calls


async def test_a_model_asking_for_no_tools_extracts_immediately() -> None:
    model = ScriptedModel(tool_call_script=[])
    agent = _agent(model, [_tool("search")[0]])

    result = await agent.run("system", "task", "extract")

    assert result == Answer(text="extracted")
    assert model.turns == 1


async def test_tool_results_are_fed_back_to_the_model() -> None:
    search, calls = _tool("search", result="found it")
    model = ScriptedModel(tool_call_script=[[_call("search", query="kyoto")]])
    agent = _agent(model, [search])

    await agent.run("system", "task", "extract")

    assert calls == [{"query": "kyoto"}]
    assert model.turns == 2, "the model gets a turn to react to the tool output"


async def test_a_failing_tool_does_not_abort_the_run() -> None:
    """A tool that errors should let the agent take another route."""
    model = ScriptedModel(tool_call_script=[[_call("search", query="kyoto")]])
    agent = _agent(model, [_tool("search", fails=True)[0]])

    result = await agent.run("system", "task", "extract")

    assert result == Answer(text="extracted")


async def test_a_call_to_an_unknown_tool_is_reported_not_raised() -> None:
    model = ScriptedModel(tool_call_script=[[_call("teleport")]])
    agent = _agent(model, [_tool("search")[0]])

    assert await agent.run("system", "task", "extract") == Answer(text="extracted")


async def test_the_iteration_ceiling_stops_a_runaway_loop() -> None:
    """An agent that keeps calling tools spends quota belonging to the next request."""
    search, calls = _tool("search")
    model = ScriptedModel(tool_call_script=[[_call("search")]] * 20)
    agent = _agent(model, [search], max_iterations=3)

    result = await agent.run("system", "task", "extract")

    assert result == Answer(text="extracted"), "it still extracts from what it gathered"
    assert len(calls) == 3


async def test_several_tool_calls_in_one_turn_all_run() -> None:
    search, search_calls = _tool("search")
    weather, weather_calls = _tool("weather", argument="city")
    model = ScriptedModel(
        tool_call_script=[[_call("search", query="a"), _call("weather", "call-2", city="b")]]
    )
    agent = _agent(model, [search, weather])

    await agent.run("system", "task", "extract")

    assert search_calls == [{"query": "a"}]
    assert weather_calls == [{"city": "b"}]


async def test_a_total_model_failure_becomes_a_workflow_error() -> None:
    """Unlike a tool failure, a model failure has no alternative route."""

    class DeadModel(ScriptedModel):
        async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
            raise ConnectionError("every provider is down")

    agent = _agent(DeadModel(), [_tool("search")[0]])

    with pytest.raises(WorkflowError, match="could not reach any model provider"):
        await agent.run("system", "task", "extract")


async def test_an_extraction_returning_the_wrong_type_is_rejected() -> None:
    class WrongSchemaModel(ScriptedModel):
        def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable[Any, Any]:
            return RunnableLambda(lambda _: "not a model instance")

    agent = _agent(WrongSchemaModel(), [_tool("search")[0]])

    with pytest.raises(WorkflowError, match="rather than Answer"):
        await agent.run("system", "task", "extract")


def test_the_registry_binds_both_chain_flavours(settings: Any) -> None:
    """Agents need tools bound; extraction needs a schema bound. Both must build."""
    registry = ModelRegistry(settings)

    assert registry.is_offline
    assert registry.tool_chain([_tool("search")[0]]) is not None
    assert registry.structured_chain(Answer) is not None
    assert registry.describe() == [{"provider": "fake", "model": "fake"}]


class EmptyThenFullModel(ScriptedModel):
    """Returns an unusable object first, then a usable one.

    Small models intermittently answer a large structured-output call with an empty
    object, so the agent has to ask again rather than treat the first reply as final.
    """

    extraction_calls: int = 0

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable[Any, Any]:
        """Return an empty answer once, then a usable one."""

        def _respond(_input: Any) -> Any:
            self.extraction_calls += 1
            return Answer(text="" if self.extraction_calls == 1 else "extracted")

        return RunnableLambda(_respond)


async def test_an_unusable_extraction_is_retried() -> None:
    model = EmptyThenFullModel()
    agent = _agent(model, [_tool("search")[0]])

    result = await agent.run(
        "system", "task", "extract", accept=lambda answer: bool(answer.text)
    )

    assert result.text == "extracted"
    assert model.extraction_calls == 2


async def test_retries_are_bounded_and_the_last_result_is_returned() -> None:
    """A permanently unusable extraction degrades rather than failing the run."""

    class AlwaysEmptyModel(ScriptedModel):
        extraction_calls: int = 0

        def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable[Any, Any]:
            def _respond(_input: Any) -> Any:
                self.extraction_calls += 1
                return Answer(text="")

            return RunnableLambda(_respond)

    model = AlwaysEmptyModel()
    agent = ToolCallingAgent(
        name="test",
        registry=SingleModelRegistry(model),  # type: ignore[arg-type]
        tools=[_tool("search")[0]],
        output_schema=Answer,
        max_iterations=4,
        extraction_attempts=3,
    )

    result = await agent.run(
        "system", "task", "extract", accept=lambda answer: bool(answer.text)
    )

    assert result == Answer(text="")
    assert model.extraction_calls == 3


async def test_no_acceptance_test_means_a_single_extraction() -> None:
    model = EmptyThenFullModel()
    agent = _agent(model, [_tool("search")[0]])

    await agent.run("system", "task", "extract")

    assert model.extraction_calls == 1


async def test_a_rejected_extraction_escalates_to_the_next_candidate() -> None:
    """Retrying the model that just returned an empty object gets the same object.

    Failover only advances when a call raises, and an empty answer is a successful call.
    So the retry has to skip past the candidate that produced it.
    """
    skips: list[int] = []

    class RecordingRegistry(SingleModelRegistry):
        """Records which candidate each extraction attempt started from."""

        def structured_chain(self, schema: Any, skip: int = 0) -> Any:
            skips.append(skip)
            attempt = len(skips)
            return RunnableLambda(
                lambda _: Answer(text="" if attempt == 1 else f"from candidate {skip}")
            )

    model = ScriptedModel()
    agent: ToolCallingAgent[Answer] = ToolCallingAgent(
        name="test",
        registry=RecordingRegistry(model),  # type: ignore[arg-type]
        tools=[_tool("search")[0]],
        output_schema=Answer,
        max_iterations=4,
        extraction_attempts=3,
    )

    result = await agent.run(
        "system", "task", "extract", accept=lambda answer: bool(answer.text)
    )

    assert skips == [0, 1], "the retry must start one candidate further down"
    assert result.text == "from candidate 1"
