"""Database engine and session management.

One engine per process, created at start-up and disposed at shutdown, because the
engine owns the connection pool. Sessions are per-unit-of-work and handed out by an
async context manager that commits on success and rolls back on failure, so no call
site has to remember the transaction boundary.

SQLite needs one adjustment to behave: foreign keys are off by default, which would
silently ignore the cascade on the audit trail.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from itinera.core.config import Settings
from itinera.core.logging import get_logger
from itinera.persistence.models import Base

logger = get_logger(__name__)

_SQLITE_PREFIX = "sqlite"


def _enable_sqlite_foreign_keys(engine: AsyncEngine) -> None:
    """Turn on foreign-key enforcement for every SQLite connection in the pool."""

    @event.listens_for(engine.sync_engine, "connect")
    def _set_pragma(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


class Database:
    """Owns the engine and hands out sessions."""

    def __init__(self, settings: Settings) -> None:
        """Create the engine for the configured database URL."""
        self._settings = settings
        self._engine = create_async_engine(
            settings.database_url,
            echo=False,
            future=True,
            pool_pre_ping=True,
        )
        if settings.database_url.startswith(_SQLITE_PREFIX):
            _enable_sqlite_foreign_keys(self._engine)

        self._session_factory = async_sessionmaker(
            self._engine, expire_on_commit=False, class_=AsyncSession
        )

    @property
    def engine(self) -> AsyncEngine:
        """The process-wide engine."""
        return self._engine

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Yield a session, committing on success and rolling back on failure."""
        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def create_schema(self) -> None:
        """Create any missing tables.

        Alembic owns schema evolution; this exists so a fresh clone and the test suite
        can start against an empty database without running a migration first.
        """
        async with self._engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        logger.info("database.schema_ready", url=self._redacted_url())

    async def healthcheck(self) -> bool:
        """Return whether the database answers a trivial query."""
        try:
            async with self._engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
        except Exception as exc:
            logger.warning("database.unhealthy", error=str(exc))
            return False
        return True

    async def dispose(self) -> None:
        """Close the connection pool."""
        await self._engine.dispose()

    def _redacted_url(self) -> str:
        """Return the database URL without any embedded credentials."""
        return self._engine.url.render_as_string(hide_password=True)
