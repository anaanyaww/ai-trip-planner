"""The tool-calling loop both agents share.

Each agent is a two-phase conversation. First a bounded reasoning loop in which the
model calls tools and reads their output; then a single structured-output call that
turns the accumulated transcript into a typed object. Splitting the phases is what
keeps the domain models clean: the model is never asked to both use tools and emit
valid JSON in the same turn, which is where small free-tier models tend to fail.

The loop is deliberately bounded. An agent that keeps calling tools burns quota that
belongs to the next request, so the iteration ceiling is configuration and hitting it
is a logged event rather than an error - the agent still produces its structured
output from whatever it gathered.

Tool failures are returned to the model as text rather than raised. A search that
fails should let the agent try a different angle; only a failure of the model itself
aborts the run.

The extraction phase reads a *flattened* transcript rather than the raw message list.
An agent records most of its work in tool-call arguments rather than in prose, and
handing a function-calling conversation to a structured-output call - which providers
implement with function calling themselves - loses that work: the model is being asked
to open a new call while an unfinished one is still in view, and answers with an empty
object. Rendering the conversation to text first makes the agent's own reasoning the
material the extraction reads.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from typing import Any, Generic, TypeVar

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from itinera.core.exceptions import AllProvidersFailedError, WorkflowError
from itinera.core.logging import get_logger
from itinera.llm.registry import ModelRegistry
from itinera.services.progress import note

logger = get_logger(__name__)

OutputT = TypeVar("OutputT", bound=BaseModel)

_TOOL_ERROR_PREFIX = "The tool failed"

_MAX_RESULT_CHARS_IN_TRANSCRIPT = 600


def _describe_call(name: str, arguments: dict[str, Any]) -> str:
    """Describe a tool call the way a person would say it out loud.

    The interesting part of a call is its subject - which query, which day - not the
    tool's name, so the most identifying argument is pulled to the front where one is
    recognisable and the tool name is used alone where it is not.
    """
    for key in ("query", "queries", "destination", "day_number", "city"):
        value = arguments.get(key)
        if isinstance(value, list) and value:
            value = value[0]
        if value not in (None, "", []):
            subject = str(value)
            label = name.replace("_", " ")
            return f"{label}: {subject[:90]}"
    return name.replace("_", " ")


class ToolCallingAgent(Generic[OutputT]):
    """Runs a bounded tool-calling conversation and returns a typed result."""

    def __init__(
        self,
        name: str,
        registry: ModelRegistry,
        tools: Sequence[BaseTool],
        output_schema: type[OutputT],
        max_iterations: int,
        extraction_attempts: int = 1,
    ) -> None:
        """Wire the agent to its model chain and tools.

        Args:
            name: Agent identifier used in logs.
            registry: Supplies the failover-backed model chains.
            tools: Tools the agent may call during the reasoning phase.
            output_schema: The model the extraction phase must produce.
            max_iterations: Ceiling on reasoning turns.
            extraction_attempts: How many times to ask for the structured result before
                accepting an unusable one, when the caller supplies an acceptance test.
        """
        self._name = name
        self._registry = registry
        self._tools = {tool.name: tool for tool in tools}
        self._output_schema = output_schema
        self._max_iterations = max_iterations
        self._extraction_attempts = max(extraction_attempts, 1)
        self._tool_chain = registry.tool_chain(list(tools))
        self._structured_chains: dict[int, Runnable[Any, Any]] = {}

    async def run(
        self,
        system_prompt: str,
        task: str,
        extraction_prompt: str,
        accept: Callable[[OutputT], bool] | None = None,
    ) -> OutputT:
        """Run the reasoning loop, then extract the structured result.

        Args:
            system_prompt: The agent's standing instructions.
            task: The specific request for this run.
            extraction_prompt: Instructions for turning the transcript into the schema.
            accept: Predicate deciding whether an extraction is usable. A structured
                call can succeed and still return an empty object - small models do this
                intermittently on large schemas - so a caller that knows what "empty"
                means for its own output can have the extraction retried.

        Returns:
            An instance of the agent's output schema.

        Raises:
            WorkflowError: If every provider failed, or none produced a usable object.
        """
        messages: list[BaseMessage] = [SystemMessage(system_prompt), HumanMessage(task)]
        messages = await self._reason(messages)

        result = await self._extract(messages, system_prompt, extraction_prompt)
        if accept is None:
            return result

        for attempt in range(1, self._extraction_attempts):
            if accept(result):
                return result
            logger.warning(
                "agent.extraction_rejected",
                agent=self._name,
                attempt=attempt,
                schema=self._output_schema.__name__,
                escalating_to=self._registry.candidates[
                    min(attempt, len(self._registry.candidates) - 1)
                ].label,
            )
            result = await self._extract(
                messages, system_prompt, extraction_prompt, skip=attempt
            )

        if not accept(result):
            logger.error(
                "agent.extraction_unusable",
                agent=self._name,
                attempts=self._extraction_attempts,
                schema=self._output_schema.__name__,
            )
        return result

    async def _reason(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """Let the model call tools until it stops asking or the ceiling is reached."""
        for iteration in range(self._max_iterations):
            response = await self._invoke_tool_chain(messages)
            messages.append(response)

            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                logger.info("agent.reasoning_complete", agent=self._name, turns=iteration + 1)
                note(self._name, f"Finished gathering after {iteration + 1} turns")
                return messages

            messages.extend(
                await asyncio.gather(*(self._run_tool(call) for call in tool_calls))
            )
        else:
            logger.warning(
                "agent.iteration_limit_reached",
                agent=self._name,
                limit=self._max_iterations,
            )
        return messages

    async def _invoke_tool_chain(self, messages: list[BaseMessage]) -> AIMessage:
        """Call the model, translating a total provider failure into a workflow error."""
        try:
            response = await self._tool_chain.ainvoke(messages)
        except Exception as exc:
            logger.exception("agent.model_call_failed", agent=self._name)
            raise WorkflowError(
                f"The {self._name} agent could not reach any model provider: {exc}",
                agent=self._name,
            ) from exc

        if not isinstance(response, AIMessage):  # pragma: no cover - defensive
            raise WorkflowError(
                f"The {self._name} agent received an unexpected response type",
                agent=self._name,
            )
        return response

    async def _run_tool(self, call: dict[str, object]) -> ToolMessage:
        """Execute one tool call and wrap its result as a tool message."""
        name = str(call.get("name", ""))
        call_id = str(call.get("id", ""))
        raw_arguments = call.get("args")
        arguments: dict[str, Any] = raw_arguments if isinstance(raw_arguments, dict) else {}
        tool = self._tools.get(name)

        if tool is None:
            logger.warning("agent.unknown_tool", agent=self._name, tool=name)
            return ToolMessage(
                content=f"{_TOOL_ERROR_PREFIX}: no tool named {name!r} exists.",
                tool_call_id=call_id,
            )

        try:
            result = await tool.ainvoke(arguments)
        except Exception as exc:
            logger.warning("agent.tool_failed", agent=self._name, tool=name, error=str(exc))
            return ToolMessage(
                content=f"{_TOOL_ERROR_PREFIX}: {exc}. Continue without it.",
                tool_call_id=call_id,
            )

        logger.info("agent.tool_called", agent=self._name, tool=name)
        note(self._name, _describe_call(name, arguments))
        return ToolMessage(content=str(result), tool_call_id=call_id)

    @staticmethod
    def _text_of(content: Any) -> str:
        """Return message content as plain text.

        Some providers return a list of typed blocks rather than a string, so the text
        parts are collected instead of stringifying the whole structure.
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            return "\n".join(part for part in parts if part)
        return str(content or "")

    def _render_transcript(self, messages: Sequence[BaseMessage]) -> str:
        """Render the reasoning conversation as text the extraction can read.

        Tool calls are rendered with their arguments, because that is where an agent
        actually records its decisions: the itinerary a planner built lives in the
        arguments it passed to the feasibility check, not in any prose it wrote.
        """
        lines: list[str] = []
        for message in messages:
            if isinstance(message, SystemMessage):
                continue
            if isinstance(message, HumanMessage):
                lines.append("## Task\n" + self._text_of(message.content))
            elif isinstance(message, AIMessage):
                text = self._text_of(message.content).strip()
                if text:
                    lines.append("## Reasoning\n" + text)
                for call in message.tool_calls or []:
                    arguments = json.dumps(call.get("args", {}), default=str)
                    lines.append(f"## Called `{call.get('name')}`\n{arguments}")
            elif isinstance(message, ToolMessage):
                lines.append("## Result\n" + self._text_of(message.content))
        return "\n\n".join(lines)

    def _structured_chain(self, skip: int) -> Runnable[Any, Any]:
        """Return the extraction chain that starts ``skip`` candidates in."""
        if skip not in self._structured_chains:
            self._structured_chains[skip] = self._registry.structured_chain(
                self._output_schema, skip=skip
            )
        return self._structured_chains[skip]

    async def _extract(
        self,
        messages: list[BaseMessage],
        system_prompt: str,
        extraction_prompt: str,
        skip: int = 0,
    ) -> OutputT:
        """Turn the transcript into the agent's typed output."""
        request = "\n\n".join(
            [
                extraction_prompt,
                "Everything below is the work you have already done. Transcribe it "
                "faithfully. Do not start over, and do not invent anything that is "
                "not here.",
                self._render_transcript(messages),
            ]
        )
        note(self._name, "Writing up what was found")
        try:
            result = await self._structured_chain(skip).ainvoke(
                [SystemMessage(system_prompt), HumanMessage(request)]
            )
        except AllProvidersFailedError:
            raise
        except Exception as exc:
            logger.exception("agent.extraction_failed", agent=self._name)
            raise WorkflowError(
                f"The {self._name} agent could not produce a structured result: {exc}",
                agent=self._name,
            ) from exc

        if not isinstance(result, self._output_schema):
            raise WorkflowError(
                f"The {self._name} agent returned {type(result).__name__} rather than "
                f"{self._output_schema.__name__}",
                agent=self._name,
            )

        logger.info("agent.extracted", agent=self._name, schema=self._output_schema.__name__)
        return result
