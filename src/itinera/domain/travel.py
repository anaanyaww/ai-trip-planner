"""The travel request: the system's single input.

Validation is split deliberately. Rules that are always true - a trip cannot end
before it starts, a budget ceiling cannot sit below its floor - live on the model, so
an invalid request can never be constructed anywhere in the system. Rules that depend
on configuration, such as the maximum trip length, are checked separately against
:class:`~itinera.core.config.Settings`, because a model must not reach into global
state to validate itself.

Monetary values are plain floats. They are indicative planning envelopes rather than
transactional amounts - nothing here is billed or settled - so the exactness of
``Decimal`` would buy precision the domain does not need. Handling real payments would
warrant revisiting this.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from itinera.core.config import Settings
from itinera.core.exceptions import PlanValidationError

_CURRENCY_PATTERN = re.compile(r"^[A-Z]{3}$")
_MAX_INTERESTS = 15
_MAX_INTEREST_LENGTH = 60
_PAST_DATE_GRACE = timedelta(days=1)


def _today_utc() -> date:
    """Return today's date in UTC.

    Used instead of ``date.today()`` so validation does not shift with the server's
    local timezone.
    """
    return datetime.now(timezone.utc).date()


class TravelRequest(BaseModel):
    """A user's travel preferences, as submitted to ``POST /plan``."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    destination: Annotated[str, Field(min_length=2, max_length=120)]
    start_date: date
    end_date: date
    budget_min: Annotated[float, Field(ge=0)]
    budget_max: Annotated[float, Field(gt=0)]
    currency: Annotated[str, Field(min_length=3, max_length=3)] = "USD"
    interests: Annotated[list[str], Field(default_factory=list)]
    travelers: Annotated[int, Field(ge=1)] = 1

    @field_validator("currency")
    @classmethod
    def _normalise_currency(cls, value: str) -> str:
        normalised = value.strip().upper()
        if not _CURRENCY_PATTERN.match(normalised):
            raise ValueError(
                f"currency must be a three-letter ISO 4217 code, got {value!r}"
            )
        return normalised

    @field_validator("destination")
    @classmethod
    def _validate_destination(cls, value: str) -> str:
        if not any(char.isalpha() for char in value):
            raise ValueError("destination must contain letters")
        return value

    @field_validator("interests")
    @classmethod
    def _normalise_interests(cls, value: list[str]) -> list[str]:
        """Trim, drop blanks and de-duplicate case-insensitively, preserving order."""
        seen: set[str] = set()
        cleaned: list[str] = []
        for raw in value:
            interest = raw.strip()
            if not interest:
                continue
            if len(interest) > _MAX_INTEREST_LENGTH:
                raise ValueError(
                    f"each interest must be at most {_MAX_INTEREST_LENGTH} characters"
                )
            key = interest.casefold()
            if key not in seen:
                seen.add(key)
                cleaned.append(interest)

        if len(cleaned) > _MAX_INTERESTS:
            raise ValueError(f"at most {_MAX_INTERESTS} interests may be supplied")
        return cleaned

    @model_validator(mode="after")
    def _validate_date_range(self) -> TravelRequest:
        if self.end_date < self.start_date:
            raise ValueError("end_date must fall on or after start_date")
        if self.start_date < _today_utc() - _PAST_DATE_GRACE:
            raise ValueError("start_date must not be in the past")
        return self

    @model_validator(mode="after")
    def _validate_budget_range(self) -> TravelRequest:
        if self.budget_max < self.budget_min:
            raise ValueError("budget_max must be greater than or equal to budget_min")
        return self


    @property
    def duration_days(self) -> int:
        """Trip length in days, counting both the arrival and departure days."""
        return (self.end_date - self.start_date).days + 1

    @property
    def budget_midpoint(self) -> float:
        """Centre of the budget range, used as the allocator's planning target."""
        return (self.budget_min + self.budget_max) / 2

    @property
    def days_until_departure(self) -> int:
        """Days between today and departure; negative once the trip has started."""
        return (self.start_date - _today_utc()).days

    def trip_dates(self) -> list[date]:
        """Return every calendar date of the trip, in order."""
        return [self.start_date + timedelta(days=offset) for offset in range(self.duration_days)]

    def validate_against_limits(self, settings: Settings) -> None:
        """Check the request against configured operational ceilings.

        Args:
            settings: Active configuration supplying the trip-length and party-size caps.

        Raises:
            PlanValidationError: If the trip is longer or the party larger than allowed.
        """
        if self.duration_days > settings.max_trip_days:
            raise PlanValidationError(
                f"Trip length of {self.duration_days} days exceeds the maximum of "
                f"{settings.max_trip_days} days.",
                duration_days=self.duration_days,
                limit=settings.max_trip_days,
            )
        if self.travelers > settings.max_travelers:
            raise PlanValidationError(
                f"Party size of {self.travelers} exceeds the maximum of "
                f"{settings.max_travelers} travellers.",
                travelers=self.travelers,
                limit=settings.max_travelers,
            )

    def summary(self) -> dict[str, Any]:
        """Return a compact representation for prompts and log records."""
        return {
            "destination": self.destination,
            "dates": f"{self.start_date.isoformat()} to {self.end_date.isoformat()}",
            "duration_days": self.duration_days,
            "budget": f"{self.budget_min:.0f}-{self.budget_max:.0f} {self.currency}",
            "travelers": self.travelers,
            "interests": ", ".join(self.interests) or "not specified",
        }
