"""Two-layer caching for expensive external calls.

Free-tier quotas are the binding constraint on this system - Serper grants 2,500
searches in total, and a single reject-and-revise cycle re-runs research. Caching is
therefore a quota-preservation measure rather than a latency optimisation, which is
why the persistent layer matters: it survives process restarts, so repeated local
runs against the same destination cost nothing.

Layer 1 is a bounded in-process TTL map serving hot repeats within one run. Layer 2 is
an on-disk store shared across restarts. Disk reads and writes are dispatched to a
worker thread so the event loop is never blocked by file I/O.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from types import TracebackType
from typing import Any, Final, TypeVar

from cachetools import TTLCache
from diskcache import Cache as DiskCache

from itinera.core.config import Settings
from itinera.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

_MISSING: Final = object()
_MEMORY_LAYER_TTL_SECONDS: Final = 300.0


def build_cache_key(namespace: str, payload: Mapping[str, Any]) -> str:
    """Build a stable cache key from a namespace and a parameter mapping.

    The payload is serialised with sorted keys so that logically identical calls made
    with differently ordered arguments share a cache entry. Values are stringified via
    JSON's default encoder, so callers must pass JSON-serialisable parameters.

    Args:
        namespace: Logical grouping, typically ``"<provider>:<operation>"``.
        payload: Call parameters that fully determine the response.

    Returns:
        A key of the form ``"<namespace>:<sha256 prefix>"``.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]
    return f"{namespace}:{digest}"


class TwoLayerCache:
    """In-process TTL cache backed by a persistent on-disk store.

    The cache is intentionally non-generic over value types: entries are whatever the
    caller stored, and callers know the type they wrote.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialise both layers from configuration.

        Args:
            settings: Active configuration supplying the enable flag, directory and
                in-process cache bound.
        """
        self._enabled = settings.cache_enabled
        self._memory: TTLCache[str, Any] = TTLCache(
            maxsize=settings.cache_memory_maxsize, ttl=_MEMORY_LAYER_TTL_SECONDS
        )
        self._disk: DiskCache | None = None
        self._directory: Path = settings.resolve_path(settings.cache_dir)

        if self._enabled:
            self._directory.mkdir(parents=True, exist_ok=True)
            self._disk = DiskCache(str(self._directory))

    async def get(self, key: str) -> Any | None:
        """Return the cached value for ``key``, or ``None`` when absent or expired."""
        if not self._enabled:
            return None

        cached = self._memory.get(key, _MISSING)
        if cached is not _MISSING:
            logger.debug("cache.hit", key=key, layer="memory")
            return cached

        if self._disk is None:
            return None

        value = await asyncio.to_thread(self._disk.get, key, _MISSING)
        if value is _MISSING:
            logger.debug("cache.miss", key=key)
            return None

        self._memory[key] = value
        logger.debug("cache.hit", key=key, layer="disk")
        return value

    async def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        """Store ``value`` under ``key``.

        A ``ttl_seconds`` of zero disables caching for this entry, which is how LLM
        response caching is switched off by default without a separate code path.
        """
        if not self._enabled or ttl_seconds <= 0:
            return

        self._memory[key] = value
        if self._disk is not None:
            await asyncio.to_thread(self._disk.set, key, value, expire=ttl_seconds)

    async def get_or_set(
        self, key: str, factory: Callable[[], Awaitable[T]], ttl_seconds: int
    ) -> T:
        """Return the cached value for ``key``, computing and storing it on a miss.

        Args:
            key: Key produced by :func:`build_cache_key`.
            factory: Zero-argument coroutine function producing the value on a miss.
            ttl_seconds: Lifetime of the stored entry; zero bypasses storage.

        Returns:
            The cached or freshly computed value.
        """
        cached = await self.get(key)
        if cached is not None:
            return cached  # type: ignore[no-any-return]

        value = await factory()
        await self.set(key, value, ttl_seconds)
        return value

    async def clear(self) -> None:
        """Drop every entry from both layers."""
        self._memory.clear()
        if self._disk is not None:
            await asyncio.to_thread(self._disk.clear)

    def close(self) -> None:
        """Release the on-disk store's file handles."""
        if self._disk is not None:
            self._disk.close()
            self._disk = None

    def __enter__(self) -> TwoLayerCache:
        """Enter the context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the on-disk store on exit."""
        self.close()
