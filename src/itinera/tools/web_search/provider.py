"""Search provider interface and its two implementations.

The assignment mandates real web search and names Serper and Exa. Both are
implemented behind one interface: they return different payload shapes, and pinning
the rest of the system to either one would make switching a code change rather than
a configuration change. Serper is the configured default for its larger free
allowance; Exa is a one-line switch away.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import ValidationError

from itinera.core.config import Settings
from itinera.core.http import HttpClient
from itinera.core.logging import get_logger
from itinera.domain.research import SearchResult

logger = get_logger(__name__)

_EXA_SNIPPET_CHARS = 400


def _coerce_results(raw: list[dict[str, Any]], provider: str) -> list[SearchResult]:
    """Build search results, dropping entries that fail validation.

    A single malformed hit - most often a missing or non-HTTP URL - should not fail
    the whole research step, so bad entries are logged and skipped rather than raised.
    """
    results: list[SearchResult] = []
    for entry in raw:
        try:
            results.append(SearchResult(provider=provider, **entry))
        except ValidationError:
            logger.debug("search.result_discarded", provider=provider, entry=entry)
    return results


class SearchProvider(ABC):
    """A web-search backend.

    The fetch and parse halves are kept separate because only the raw payload is safe
    to cache. Caching normalised :class:`SearchResult` objects would persist pickled
    domain models across restarts, so a later change to the model would replay stale
    objects past validation - which is exactly how a field that had become a ``str``
    came back as a ``Url`` and broke checkpointing.
    """

    name: str

    def __init__(self, settings: Settings, http: HttpClient) -> None:
        """Store configuration and the shared HTTP client."""
        self._settings = settings
        self._http = http

    @abstractmethod
    async def fetch(self, query: str, limit: int) -> Any:
        """Call the backend and return its raw JSON payload.

        Args:
            query: The search string.
            limit: Maximum number of results to request.

        Returns:
            The decoded response, containing only JSON primitives.
        """

    @abstractmethod
    def parse(self, payload: Any, limit: int) -> list[SearchResult]:
        """Normalise a raw payload into search results.

        Args:
            payload: A response previously returned by :meth:`fetch`.
            limit: Maximum number of results to return.

        Returns:
            Normalised results, possibly fewer than ``limit``.
        """

    async def search(self, query: str, limit: int) -> list[SearchResult]:
        """Fetch and normalise in one step, for callers that do not cache."""
        return self.parse(await self.fetch(query, limit), limit)


class SerperProvider(SearchProvider):
    """Google results through Serper's API."""

    name = "serper"

    async def fetch(self, query: str, limit: int) -> Any:
        """Query Serper's ``/search`` endpoint."""
        return await self._http.request_json(
            "POST",
            f"{self._settings.serper_base_url}/search",
            json_body={"q": query, "num": limit},
            headers={
                "X-API-KEY": self._settings.require_search_api_key(),
                "Content-Type": "application/json",
            },
            service="serper",
        )

    def parse(self, payload: Any, limit: int) -> list[SearchResult]:
        """Normalise Serper's organic results."""
        organic = payload.get("organic", []) if isinstance(payload, dict) else []
        return _coerce_results(
            [
                {
                    "title": entry.get("title", ""),
                    "url": entry.get("link", ""),
                    "snippet": entry.get("snippet", ""),
                }
                for entry in organic[:limit]
            ],
            provider=self.name,
        )


class ExaProvider(SearchProvider):
    """Neural search with content retrieval through Exa's API."""

    name = "exa"

    async def fetch(self, query: str, limit: int) -> Any:
        """Query Exa's ``/search`` endpoint."""
        return await self._http.request_json(
            "POST",
            f"{self._settings.exa_base_url}/search",
            json_body={
                "query": query,
                "numResults": limit,
                "contents": {"text": {"maxCharacters": _EXA_SNIPPET_CHARS}},
            },
            headers={
                "x-api-key": self._settings.require_search_api_key(),
                "Content-Type": "application/json",
            },
            service="exa",
        )

    def parse(self, payload: Any, limit: int) -> list[SearchResult]:
        """Normalise Exa's results.

        Exa returns retrieved page text rather than a snippet, so it is truncated to a
        comparable length; leaving it whole would swamp the prompt budget.
        """
        results = payload.get("results", []) if isinstance(payload, dict) else []
        return _coerce_results(
            [
                {
                    "title": entry.get("title") or entry.get("url", ""),
                    "url": entry.get("url", ""),
                    "snippet": (entry.get("text") or "")[:_EXA_SNIPPET_CHARS].strip(),
                }
                for entry in results[:limit]
            ],
            provider=self.name,
        )


_PROVIDERS: dict[str, type[SearchProvider]] = {
    SerperProvider.name: SerperProvider,
    ExaProvider.name: ExaProvider,
}


def build_search_provider(settings: Settings, http: HttpClient) -> SearchProvider:
    """Return the search provider selected by configuration."""
    provider_type = _PROVIDERS[settings.search_provider.value]
    return provider_type(settings, http)
