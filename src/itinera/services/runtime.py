"""The application's composition root.

Every long-lived collaborator - the engine, the checkpoint connection, the HTTP pool,
the model registry, the compiled graph - is built here once and torn down here once.
Nothing in the codebase reaches for a global; the API asks the runtime for what it
needs, and tests build a runtime pointed at temporary files and the offline provider.

Construction is split from start-up on purpose. ``__init__`` does only what cannot
fail, and ``start`` does the I/O - opening databases, creating schemas - so a failure
to start is reported as a failure to start rather than as an import-time error.
"""

from __future__ import annotations

from itinera.core.config import Settings
from itinera.core.logging import get_logger
from itinera.graph.builder import PlanningGraph, build_graph
from itinera.graph.nodes import WorkflowDependencies
from itinera.llm.registry import ModelRegistry
from itinera.persistence.checkpointer import CheckpointerService
from itinera.persistence.database import Database
from itinera.prompts.library import PromptLibrary, get_prompt_library
from itinera.tools.base import ToolContext

logger = get_logger(__name__)


class Runtime:
    """Owns the process-wide collaborators and their lifecycle."""

    def __init__(self, settings: Settings) -> None:
        """Build the collaborators that need no I/O."""
        self.settings = settings
        self.registry = ModelRegistry(settings)
        self.prompts: PromptLibrary = get_prompt_library(settings)
        self.tools = ToolContext.create(settings)
        self.database = Database(settings)
        self.checkpoints = CheckpointerService(settings)
        self._graph: PlanningGraph | None = None

    @property
    def graph(self) -> PlanningGraph:
        """The compiled workflow.

        Raises:
            RuntimeError: If accessed before :meth:`start`.
        """
        if self._graph is None:
            raise RuntimeError("The runtime has not been started")
        return self._graph

    @property
    def is_offline(self) -> bool:
        """Whether no model provider is configured and plans will be placeholders."""
        return self.registry.is_offline

    async def start(self) -> None:
        """Open the databases and compile the graph."""
        self.settings.ensure_runtime_directories()
        await self.database.create_schema()
        checkpointer = await self.checkpoints.open()

        dependencies = WorkflowDependencies(
            settings=self.settings,
            registry=self.registry,
            prompts=self.prompts,
            tools=self.tools,
        )
        self._graph = build_graph(dependencies, checkpointer)

        logger.info(
            "runtime.started",
            environment=self.settings.app_env.value,
            providers=self.registry.describe(),
            prompt_version=self.prompts.version,
        )

    async def stop(self) -> None:
        """Release every resource the runtime opened, in reverse order."""
        self._graph = None
        await self.checkpoints.close()
        await self.database.dispose()
        await self.tools.aclose()
        logger.info("runtime.stopped")
