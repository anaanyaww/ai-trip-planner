"""Tests for the web search tool and its provider backends."""

from __future__ import annotations

from dataclasses import replace

import httpx
import pytest

from itinera.core.config import SearchProviderName
from itinera.core.exceptions import MissingCredentialsError
from itinera.domain.research import SearchResult
from itinera.tools.web_search import SourceRecorder, build_search_provider, build_web_search_tool

SERPER_PAYLOAD = {
    "organic": [
        {
            "title": "Kyoto temple etiquette",
            "link": "https://example.com/etiquette",
            "snippet": "Remove your shoes before entering.",
        },
        {
            "title": "Missing a link",
            "snippet": "This entry has no URL and must be discarded.",
        },
    ]
}

EXA_PAYLOAD = {
    "results": [
        {
            "title": "Kyoto in autumn",
            "url": "https://example.com/autumn",
            "text": "  Maple season peaks in late November.  ",
        }
    ]
}


async def test_serper_results_are_normalised(make_context, json_handler) -> None:
    context = make_context(json_handler(SERPER_PAYLOAD))
    provider = build_search_provider(context.settings, context.http)

    results = await provider.search("kyoto etiquette", limit=5)

    assert len(results) == 1, "the entry without a URL should have been discarded"
    assert results[0].title == "Kyoto temple etiquette"
    assert str(results[0].url) == "https://example.com/etiquette"
    assert results[0].provider == "serper"


async def test_serper_sends_the_api_key_and_query(make_context, json_handler) -> None:
    handler = json_handler(SERPER_PAYLOAD)
    context = make_context(handler)
    provider = build_search_provider(context.settings, context.http)

    await provider.search("kyoto etiquette", limit=3)

    request = handler.calls[0]
    assert request.headers["X-API-KEY"] == "test-serper-key"
    assert b'"num":3' in request.content.replace(b" ", b"")


async def test_exa_text_is_trimmed_into_a_snippet(settings, make_context, json_handler) -> None:
    context = make_context(json_handler(EXA_PAYLOAD))
    exa_settings = settings.model_copy(update={"search_provider": SearchProviderName.EXA})
    provider = build_search_provider(exa_settings, context.http)

    results = await provider.search("kyoto autumn", limit=1)

    assert results[0].snippet == "Maple season peaks in late November."
    assert results[0].provider == "exa"


def test_missing_credentials_name_the_variable_to_set(settings) -> None:
    keyless = settings.model_copy(update={"serper_api_key": None})

    with pytest.raises(MissingCredentialsError, match="SERPER_API_KEY"):
        keyless.require_search_api_key()


async def test_tool_records_every_url_it_saw(make_context, json_handler) -> None:
    context = make_context(json_handler(SERPER_PAYLOAD))
    recorder = SourceRecorder()
    tool = build_web_search_tool(context, recorder)

    output = await tool.ainvoke({"query": "kyoto etiquette"})

    assert "https://example.com/etiquette" in output
    assert [str(result.url) for result in recorder.results] == [
        "https://example.com/etiquette"
    ]


def test_recorder_deduplicates_repeated_urls() -> None:
    recorder = SourceRecorder()
    hit = SearchResult(title="A", url="https://example.com/a", provider="serper")
    recorder.record([hit])
    recorder.record([hit, SearchResult(title="B", url="https://example.com/b")])

    assert len(recorder) == 2


async def test_query_budget_is_enforced(settings, make_context, json_handler) -> None:
    handler = json_handler(SERPER_PAYLOAD)
    context = make_context(handler)
    budgeted = context.settings.model_copy(update={"search_max_queries_per_run": 2})
    tool = build_web_search_tool(replace(context, settings=budgeted))

    await tool.ainvoke({"query": "one"})
    await tool.ainvoke({"query": "two"})
    refusal = await tool.ainvoke({"query": "three"})

    assert "budget" in refusal.lower()
    assert len(handler.calls) == 2


async def test_repeated_query_is_served_from_cache(make_context, json_handler) -> None:
    handler = json_handler(SERPER_PAYLOAD)
    context = make_context(handler)
    tool = build_web_search_tool(context)

    first = await tool.ainvoke({"query": "Kyoto Etiquette"})
    second = await tool.ainvoke({"query": "  kyoto etiquette  "})

    assert len(handler.calls) == 1, "case and whitespace must not defeat the cache"
    assert first == second, "a cached payload must parse to the same results"


async def test_only_raw_payloads_are_cached(make_context, json_handler) -> None:
    """Caching domain models would replay them past validation after a model change.

    The cache is on disk and outlives the process, so a pickled SearchResult written by
    an older build comes back with whatever field types that build used. Keeping the
    cache to JSON primitives means every read is re-validated.
    """
    context = make_context(json_handler(SERPER_PAYLOAD))
    tool = build_web_search_tool(context)

    await tool.ainvoke({"query": "kyoto etiquette"})

    cached = [context.cache._memory[key] for key in context.cache._memory]
    assert cached, "the query should have been cached"
    for entry in cached:
        assert _is_json_primitive(entry), f"cached a non-primitive: {type(entry).__name__}"


def _is_json_primitive(value: object) -> bool:
    """Whether ``value`` is built only from types JSON and pickle agree on."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_json_primitive(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(k, str) and _is_json_primitive(v) for k, v in value.items()
        )
    return False


async def test_empty_query_is_rejected_without_a_call(make_context, json_handler) -> None:
    handler = json_handler(SERPER_PAYLOAD)
    tool = build_web_search_tool(make_context(handler))

    output = await tool.ainvoke({"query": "   "})

    assert "empty" in output.lower()
    assert handler.calls == []


async def test_provider_failure_is_reported_as_prose(make_context) -> None:
    def failing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="upstream on fire")

    tool = build_web_search_tool(make_context(failing))

    output = await tool.ainvoke({"query": "kyoto"})

    assert "failed" in output.lower()
    assert "continue with the information you have" in output.lower()
