"""Tests for the budget allocator.

The allocator exists so the budget arithmetic is not left to a language model, so the
first thing worth asserting is that the split is exhaustive: the category lines must
account for the whole planning budget, whatever emphasis is applied.
"""

from __future__ import annotations

import httpx
import pytest

from itinera.domain.enums import BudgetCategory
from itinera.tools.budget_allocator import (
    AllocationRecorder,
    BudgetAllocatorService,
    _extract_rate,
    build_budget_allocator_tool,
)

FX_PAYLOAD = {"amount": 1.0, "base": "USD", "date": "2026-08-24", "rates": {"JPY": 147.25}}


def _no_calls(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"no HTTP call expected, got {request.url}")


async def test_lines_account_for_the_whole_budget(make_context, travel_request) -> None:
    service = BudgetAllocatorService(make_context(_no_calls), travel_request)

    allocation = await service.allocate(emphasis=None, local_currency=None)

    assert allocation.total_budget == pytest.approx(2000.0)
    assert sum(line.total for line in allocation.lines) == pytest.approx(2000.0)
    assert {line.category for line in allocation.lines} == set(BudgetCategory)


async def test_per_person_per_day_divides_by_party_and_duration(
    make_context, travel_request
) -> None:
    service = BudgetAllocatorService(make_context(_no_calls), travel_request)

    allocation = await service.allocate(emphasis=None, local_currency=None)

    assert allocation.activities_per_person_per_day == pytest.approx(50.0)
    assert allocation.daily_envelope_per_person == pytest.approx(250.0)


async def test_emphasis_moves_a_bounded_share_from_lodging(
    make_context, travel_request, settings
) -> None:
    service = BudgetAllocatorService(make_context(_no_calls), travel_request)

    default = await service.allocate(emphasis=None, local_currency=None)
    emphasised = await service.allocate(emphasis=BudgetCategory.FOOD, local_currency=None)

    def total(allocation, category: BudgetCategory) -> float:
        return next(line.total for line in allocation.lines if line.category is category)

    moved = default.total_budget * settings.budget_emphasis_shift
    assert total(emphasised, BudgetCategory.FOOD) == pytest.approx(
        total(default, BudgetCategory.FOOD) + moved
    )
    assert total(emphasised, BudgetCategory.LODGING) == pytest.approx(
        total(default, BudgetCategory.LODGING) - moved
    )
    assert sum(line.total for line in emphasised.lines) == pytest.approx(
        default.total_budget
    )


async def test_emphasising_the_donor_leaves_the_split_untouched(
    make_context, travel_request
) -> None:
    service = BudgetAllocatorService(make_context(_no_calls), travel_request)

    default = await service.allocate(emphasis=None, local_currency=None)
    lodging = await service.allocate(emphasis=BudgetCategory.LODGING, local_currency=None)

    assert [line.total for line in lodging.lines] == [line.total for line in default.lines]


async def test_exchange_rate_is_attached_with_its_quote_date(
    make_context, travel_request, json_handler
) -> None:
    service = BudgetAllocatorService(make_context(json_handler(FX_PAYLOAD)), travel_request)

    allocation = await service.allocate(emphasis=None, local_currency="JPY")

    assert allocation.exchange_rate == pytest.approx(147.25)
    assert allocation.local_currency == "JPY"
    assert allocation.fx_rate_date.isoformat() == "2026-08-24"
    assert allocation.to_local(100.0) == pytest.approx(14725.0)
    assert any("2026-08-24" in note for note in allocation.notes)


async def test_matching_currency_skips_the_fx_call(
    make_context, travel_request, json_handler
) -> None:
    handler = json_handler(FX_PAYLOAD)
    service = BudgetAllocatorService(make_context(handler), travel_request)

    allocation = await service.allocate(emphasis=None, local_currency="USD")

    assert handler.calls == []
    assert allocation.exchange_rate is None


async def test_fx_failure_degrades_to_an_unconverted_allocation(
    make_context, travel_request
) -> None:
    """A missing rate costs a convenience; it must not fail the run."""

    def failing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="fx down")

    service = BudgetAllocatorService(make_context(failing), travel_request)

    allocation = await service.allocate(emphasis=None, local_currency="JPY")

    assert allocation.exchange_rate is None
    assert allocation.local_currency is None
    assert sum(line.total for line in allocation.lines) == pytest.approx(2000.0)


async def test_unsupported_currency_is_treated_as_no_rate(
    make_context, travel_request, json_handler
) -> None:
    handler = json_handler({"amount": 1.0, "base": "USD", "date": "2026-08-24", "rates": {}})
    service = BudgetAllocatorService(make_context(handler), travel_request)

    allocation = await service.allocate(emphasis=None, local_currency="XYZ")

    assert allocation.exchange_rate is None


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {"rates": {"JPY": 147.0}},
        {"date": "2026-08-24"},
        {"date": "2026-08-24", "rates": {"JPY": 0}},
        {"date": "2026-08-24", "rates": {"JPY": "147"}},
    ],
)
def test_malformed_fx_payloads_yield_no_rate(payload) -> None:
    assert _extract_rate(payload, "JPY") is None


async def test_tool_records_the_allocation_and_renders_the_envelope(
    make_context, travel_request, json_handler
) -> None:
    recorder = AllocationRecorder()
    tool = build_budget_allocator_tool(
        make_context(json_handler(FX_PAYLOAD)), travel_request, recorder
    )

    output = await tool.ainvoke({"emphasis": "food", "local_currency": "jpy"})

    assert recorder.allocation is not None
    assert recorder.allocation.local_currency == "JPY", "the code must be normalised"
    assert "per person per day" in output
    assert "JPY" in output


async def test_tool_rejects_a_malformed_currency_code(
    make_context, travel_request, json_handler
) -> None:
    tool = build_budget_allocator_tool(make_context(json_handler(FX_PAYLOAD)), travel_request)

    with pytest.raises(ValueError, match="three-letter"):
        await tool.ainvoke({"local_currency": "japanese yen"})
