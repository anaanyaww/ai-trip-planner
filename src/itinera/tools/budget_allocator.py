"""Budget allocation: the planner agent's first tool.

Asking a language model to divide a budget produces numbers that look reasonable and
do not add up. This tool does the arithmetic instead: the model supplies only the two
judgement calls it is actually good at - which category the traveller's interests
justify favouring, and what currency the destination trades in - and receives a split
that is guaranteed to sum to the budget.

The category ratios live in configuration rather than here, so the house view of what
a trip costs can be retuned without a code change. The emphasis shift is deliberately
small and always taken from lodging: it lets a food-focused trip spend more on food
without letting the model quietly reallocate the whole envelope.

Exchange rates come from Frankfurter, which is keyless and publishes European Central
Bank reference rates. A quote is only useful if it is dated, so the rate's publication
date is carried through to the plan; a rate presented without one invites the reader to
assume it is live.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from itinera.core.exceptions import ItineraError
from itinera.core.logging import get_logger
from itinera.domain.enums import BudgetCategory
from itinera.domain.itinerary import BudgetAllocation, BudgetLine
from itinera.domain.travel import TravelRequest
from itinera.tools.base import ToolContext, guarded_call

logger = get_logger(__name__)

_TOOL_NAME = "budget_allocator"
_EMPHASIS_DONOR = BudgetCategory.LODGING
_ROUNDING = 2


class ExchangeRate(BaseModel):
    """A dated conversion rate from the request currency into the local one."""

    model_config = ConfigDict(frozen=True)

    base: str
    quote: str
    rate: Annotated[float, Field(gt=0)]
    quoted_on: date


class AllocationRecorder:
    """Holds the allocation a run produced, for attachment to the draft itinerary.

    The agent is given prose; the itinerary needs the structured object. Recording it
    here avoids reconstructing the split from the model's rendering of it.
    """

    def __init__(self) -> None:
        """Start with nothing recorded."""
        self.allocation: BudgetAllocation | None = None

    def record(self, allocation: BudgetAllocation) -> None:
        """Store the most recent allocation."""
        self.allocation = allocation


class BudgetAllocatorInput(BaseModel):
    """Arguments accepted by the budget allocator."""

    emphasis: BudgetCategory | None = Field(
        default=None,
        description=(
            "Category the traveller's stated interests justify favouring, for example "
            "'food' for a culinary trip or 'activities' for an adventure trip. Omit "
            "when the interests suggest no particular emphasis."
        ),
    )
    local_currency: str | None = Field(
        default=None,
        description=(
            "ISO 4217 code of the currency spent at the destination, for example 'JPY' "
            "for Japan. Omit if it is the same as the traveller's currency or unknown."
        ),
    )

    @field_validator("local_currency")
    @classmethod
    def _normalise_currency(cls, value: str | None) -> str | None:
        """Uppercase the code, treating blank input as absent."""
        if value is None:
            return None
        normalised = value.strip().upper()
        if not normalised:
            return None
        if len(normalised) != 3 or not normalised.isalpha():
            raise ValueError(f"local_currency must be a three-letter code, got {value!r}")
        return normalised


def _weights(
    ratios: dict[str, float], emphasis: BudgetCategory | None, shift: float
) -> dict[BudgetCategory, float]:
    """Return the category ratios, moving ``shift`` from lodging to ``emphasis``.

    The shift is capped at the donor's own share so no category can be driven negative,
    and is skipped entirely when the emphasis is the donor itself.
    """
    weights = {BudgetCategory(name): value for name, value in ratios.items()}
    if emphasis is None or emphasis is _EMPHASIS_DONOR or shift <= 0:
        return weights

    moved = min(shift, weights[_EMPHASIS_DONOR])
    weights[_EMPHASIS_DONOR] -= moved
    weights[emphasis] += moved
    return weights


def _extract_rate(payload: Any, quote: str) -> float | None:
    """Pull the quoted rate out of a Frankfurter response, if it is present."""
    if not isinstance(payload, dict) or "date" not in payload:
        return None
    rates = payload.get("rates")
    if not isinstance(rates, dict):
        return None
    rate = rates.get(quote)
    if not isinstance(rate, (int, float)) or isinstance(rate, bool) or rate <= 0:
        return None
    return float(rate)


def _build_line(category: BudgetCategory, amount: float, travelers: int, days: int) -> BudgetLine:
    """Derive a category's totals from its share of the budget."""
    per_person = amount / travelers
    return BudgetLine(
        category=category,
        total=round(amount, _ROUNDING),
        per_person=round(per_person, _ROUNDING),
        per_person_per_day=round(per_person / days, _ROUNDING),
    )


class BudgetAllocatorService:
    """Turns a travel request into a spending envelope."""

    def __init__(self, context: ToolContext, request: TravelRequest) -> None:
        """Store the injected dependencies and the request being planned."""
        self._context = context
        self._settings = context.settings
        self._request = request

    async def exchange_rate(self, local_currency: str) -> ExchangeRate | None:
        """Fetch the rate from the request currency into ``local_currency``.

        Returns ``None`` rather than raising when the rate is unavailable: a missing
        conversion costs the plan a convenience and is not worth failing the run for.
        """
        base = self._request.currency
        if local_currency == base:
            return None

        params = {"base": base, "symbols": local_currency}
        try:
            payload = await guarded_call(
                self._context,
                service="frankfurter",
                cache_namespace="frankfurter:latest",
                cache_params=params,
                ttl_seconds=self._settings.cache_ttl_fx_seconds,
                operation=lambda: self._context.http.request_json(
                    "GET",
                    f"{self._settings.frankfurter_base_url}/latest",
                    params=params,
                    service="frankfurter",
                ),
            )
        except ItineraError as exc:
            logger.warning("fx.unavailable", base=base, quote=local_currency, error=str(exc))
            return None

        rate = _extract_rate(payload, local_currency)
        if rate is None:
            logger.info("fx.unsupported_currency", base=base, quote=local_currency)
            return None

        return ExchangeRate(
            base=base,
            quote=local_currency,
            rate=rate,
            quoted_on=date.fromisoformat(payload["date"]),
        )

    async def allocate(
        self, emphasis: BudgetCategory | None, local_currency: str | None
    ) -> BudgetAllocation:
        """Build the allocation for the request.

        Args:
            emphasis: Category to favour, or ``None`` for the default split.
            local_currency: Destination currency to quote alongside, when known.

        Returns:
            An allocation whose lines sum to the planning budget.
        """
        request = self._request
        total = round(request.budget_midpoint, _ROUNDING)
        weights = _weights(
            self._settings.budget_ratios, emphasis, self._settings.budget_emphasis_shift
        )
        lines = [
            _build_line(category, total * weight, request.travelers, request.duration_days)
            for category, weight in weights.items()
        ]

        notes = [
            f"Planning against the midpoint of the {request.budget_min:.0f}-"
            f"{request.budget_max:.0f} {request.currency} range."
        ]
        if emphasis is not None and emphasis is not _EMPHASIS_DONOR:
            notes.append(
                f"{self._settings.budget_emphasis_shift:.0%} of the budget moved from "
                f"{_EMPHASIS_DONOR.value} to {emphasis.value} to match the stated interests."
            )

        rate = await self.exchange_rate(local_currency) if local_currency else None
        if rate is not None:
            notes.append(
                f"1 {rate.base} = {rate.rate:.2f} {rate.quote} as of "
                f"{rate.quoted_on.isoformat()}; treat local prices as indicative."
            )

        return BudgetAllocation(
            currency=request.currency,
            total_budget=total,
            travelers=request.travelers,
            duration_days=request.duration_days,
            lines=lines,
            local_currency=rate.quote if rate else None,
            exchange_rate=rate.rate if rate else None,
            fx_rate_date=rate.quoted_on if rate else None,
            notes=notes,
        )


def _format(allocation: BudgetAllocation) -> str:
    """Render the allocation as prose the planner can schedule against."""
    lines = [
        f"Budget: {allocation.total_budget:.0f} {allocation.currency} total for "
        f"{allocation.travelers} traveller(s) over {allocation.duration_days} day(s).",
        f"That is {allocation.daily_envelope_per_person:.2f} {allocation.currency} per "
        "person per day across all categories.",
        "",
        "Allocation by category (total | per person | per person per day):",
    ]
    lines.extend(
        f"  {line.category.value}: {line.total:.2f} | {line.per_person:.2f} | "
        f"{line.per_person_per_day:.2f} {allocation.currency}"
        for line in allocation.lines
    )

    schedulable = allocation.activities_per_person_per_day
    lines.extend(
        [
            "",
            f"Schedule activities and dining to no more than {schedulable:.2f} "
            f"{allocation.currency} per person per day.",
        ]
    )

    local = allocation.to_local(schedulable)
    if local is not None:
        lines.append(
            f"In local terms that is roughly {local:.2f} {allocation.local_currency} "
            "per person per day."
        )
    lines.extend(allocation.notes)
    return "\n".join(lines)


def build_budget_allocator_tool(
    context: ToolContext, request: TravelRequest, recorder: AllocationRecorder | None = None
) -> BaseTool:
    """Build the budget allocator bound to a run's request.

    Args:
        context: Injected settings, HTTP client, cache and breakers.
        request: The travel request being planned, supplying budget, party and dates.
        recorder: Collector for the structured allocation.

    Returns:
        A LangChain tool the planner agent can call.
    """
    service = BudgetAllocatorService(context, request)

    async def _allocate(
        emphasis: BudgetCategory | None = None, local_currency: str | None = None
    ) -> str:
        """Split the trip budget across categories."""
        allocation = await service.allocate(emphasis, local_currency)
        if recorder is not None:
            recorder.record(allocation)

        logger.info(
            "budget.allocated",
            total=allocation.total_budget,
            emphasis=emphasis.value if emphasis else None,
            local_currency=allocation.local_currency,
        )
        return _format(allocation)

    return StructuredTool.from_function(
        coroutine=_allocate,
        name=_TOOL_NAME,
        description=(
            "Split the traveller's budget across lodging, food, activities, transport "
            "and contingency, and convert it into the destination's currency. Call this "
            "before scheduling anything: it returns the per-person, per-day spend the "
            "itinerary must stay within."
        ),
        args_schema=BudgetAllocatorInput,
    )
