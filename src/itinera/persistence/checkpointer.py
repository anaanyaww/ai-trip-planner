"""The workflow checkpointer.

This is what makes the human-in-the-loop pause durable. When the graph interrupts, the
checkpointer writes the full state to SQLite keyed by the plan's thread id; when a
review arrives - on a different request, possibly after a restart - the graph is
resumed from that record. Without it, the pause would only last as long as the process
that started it.

The connection is opened once at start-up and shared, because SQLite pays a real cost
to open a database and the workflow touches it on every node transition. It is held
here rather than inside the saver so shutdown has something to close.
"""

from __future__ import annotations

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from itinera.core.config import Settings
from itinera.core.logging import get_logger
from itinera.persistence.serialization import build_serializer

logger = get_logger(__name__)


class CheckpointerService:
    """Owns the checkpoint database connection for the process lifetime."""

    def __init__(self, settings: Settings) -> None:
        """Record where the checkpoint database lives."""
        self._path = settings.resolve_path(settings.checkpoint_db_path)
        self._connection: aiosqlite.Connection | None = None
        self._saver: AsyncSqliteSaver | None = None

    @property
    def saver(self) -> AsyncSqliteSaver:
        """The checkpointer.

        Raises:
            RuntimeError: If accessed before :meth:`open`.
        """
        if self._saver is None:
            raise RuntimeError("The checkpointer has not been opened")
        return self._saver

    async def open(self) -> AsyncSqliteSaver:
        """Connect to the checkpoint database and create its tables if needed."""
        if self._saver is not None:
            return self._saver

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = await aiosqlite.connect(str(self._path))
        self._saver = AsyncSqliteSaver(self._connection, serde=build_serializer())
        await self._saver.setup()

        logger.info("checkpointer.ready", path=str(self._path))
        return self._saver

    async def close(self) -> None:
        """Close the checkpoint database connection."""
        if self._connection is not None:
            await self._connection.close()
            self._connection = None
            self._saver = None
            logger.info("checkpointer.closed")
