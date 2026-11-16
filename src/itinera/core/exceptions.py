"""Exception hierarchy for AI Trip Planner.

Every failure the application raises deliberately derives from :class:`ItineraError`,
which carries the machine-readable ``code`` and the HTTP status the API layer should
surface. Keeping that mapping on the exception - rather than in the route handlers -
means a new failure mode is translated correctly without touching the API layer.
"""

from __future__ import annotations

from typing import Any


class ItineraError(Exception):
    """Base class for all application errors.

    Attributes:
        code: Stable, machine-readable identifier used as the RFC 7807 ``type`` suffix.
        http_status: Status code the API layer returns for this failure.
        detail: Human-readable explanation safe to expose to API clients.
        context: Structured metadata for logs; never serialised into responses.
    """

    code: str = "internal_error"
    http_status: int = 500

    def __init__(self, detail: str, **context: Any) -> None:
        """Store the client-facing detail and any structured log context."""
        super().__init__(detail)
        self.detail = detail
        self.context = context

    def __str__(self) -> str:
        """Return the human-readable detail message."""
        return self.detail


class ConfigurationError(ItineraError):
    """Raised when configuration is missing, malformed or internally inconsistent."""

    code = "configuration_error"
    http_status = 500


class MissingCredentialsError(ConfigurationError):
    """Raised when an operation needs an API key that has not been configured."""

    code = "missing_credentials"
    http_status = 503


class ExternalServiceError(ItineraError):
    """Base class for failures originating in a third-party service."""

    code = "external_service_error"
    http_status = 502


class ProviderError(ExternalServiceError):
    """A single upstream provider failed; other providers in the chain may still work."""

    code = "provider_error"


class RateLimitedError(ExternalServiceError):
    """An upstream provider rejected the call with a rate-limit response.

    Attributes:
        retry_after_seconds: Upstream-advertised cooldown, when the response supplied one.
    """

    code = "rate_limited"
    http_status = 429

    def __init__(
        self, detail: str, retry_after_seconds: float | None = None, **context: Any
    ) -> None:
        """Record the failure alongside the upstream cooldown hint, when given."""
        super().__init__(detail, **context)
        self.retry_after_seconds = retry_after_seconds


class CircuitOpenError(ExternalServiceError):
    """A provider is in cooldown after repeated failures and is being skipped."""

    code = "circuit_open"
    http_status = 503


class AllProvidersFailedError(ExternalServiceError):
    """Every provider in the failover chain was exhausted without success."""

    code = "all_providers_failed"
    http_status = 503


class ToolExecutionError(ItineraError):
    """A tool invoked by an agent failed in a way the agent cannot recover from."""

    code = "tool_execution_error"
    http_status = 502


class PlanNotFoundError(ItineraError):
    """No plan exists for the supplied identifier."""

    code = "plan_not_found"
    http_status = 404


class InvalidPlanStateError(ItineraError):
    """The requested transition is not legal from the plan's current status.

    Used for the two conflict cases the API contract calls out: submitting a review
    when the workflow is not paused for one, and requesting the final plan before it
    has been approved. Both return 409 - the resource exists, its state is wrong.
    """

    code = "invalid_plan_state"
    http_status = 409

    def __init__(self, detail: str, current_status: str | None = None, **context: Any) -> None:
        """Record the failure alongside the status that made the transition illegal."""
        super().__init__(detail, current_status=current_status, **context)
        self.current_status = current_status


class WorkflowError(ItineraError):
    """The orchestrator could not complete a workflow step."""

    code = "workflow_error"
    http_status = 500


class PlanValidationError(ItineraError):
    """A travel request or a modification patch failed domain validation."""

    code = "plan_validation_error"
    http_status = 422
