"""Application configuration.

All tunables - model identifiers, provider ordering, cache TTLs, budget ratios,
timeouts and thresholds - are declared here and sourced from the environment, so
behaviour can be changed without touching code. Values are validated once at
start-up; an inconsistent configuration fails fast rather than surfacing later as a
confusing runtime error.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, SecretStr, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from itinera.core.exceptions import ConfigurationError, MissingCredentialsError

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_RATIO_TOLERANCE = 1e-6


class Environment(str, Enum):
    """Deployment environment."""

    LOCAL = "local"
    STAGING = "staging"
    PRODUCTION = "production"


class LogFormat(str, Enum):
    """Log rendering style."""

    CONSOLE = "console"
    JSON = "json"


class LLMProviderName(str, Enum):
    """Supported chat-model providers, in the order they may appear in the chain."""

    GROQ = "groq"
    GEMINI = "gemini"
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    FAKE = "fake"


class SearchProviderName(str, Enum):
    """Supported web-search backends."""

    SERPER = "serper"
    EXA = "exa"


Ratio = Annotated[float, Field(ge=0.0, le=1.0)]
PositiveInt = Annotated[int, Field(gt=0)]


class Settings(BaseSettings):
    """Typed view over the process environment.

    Instantiated once per process through :func:`get_settings`.
    """

    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: Environment = Environment.LOCAL
    log_level: str = "INFO"
    log_format: LogFormat = LogFormat.CONSOLE
    api_host: str = "127.0.0.1"
    api_port: PositiveInt = 8000

    llm_provider_chain: str = "openrouter,groq,gemini"

    groq_api_key: SecretStr | None = None
    groq_api_key_alt: SecretStr | None = None
    groq_models: str = "qwen/qwen3.6-27b,openai/gpt-oss-120b,openai/gpt-oss-20b"

    google_api_key: SecretStr | None = None
    google_api_key_alt: SecretStr | None = None
    gemini_models: str = "gemini-3.5-flash-lite,gemini-3.7-flash"

    openrouter_api_key: SecretStr | None = None
    openrouter_api_key_alt: SecretStr | None = None
    openrouter_models: str = (
        "nvidia/nemotron-3-super-120b-a12b:free,z-ai/glm-5.2:free,minimax/minimax-m3:free"
    )
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    ollama_base_url: str = "http://localhost:11434"
    ollama_models: str = "qwen3:4b"
    ollama_num_ctx: PositiveInt = 16_384

    llm_temperature: Annotated[float, Field(ge=0.0, le=2.0)] = 0.2
    llm_max_tokens: PositiveInt = 8192
    llm_timeout_seconds: Annotated[float, Field(gt=0)] = 90.0
    llm_rate_limit_rpm: PositiveInt = 25
    llm_extraction_attempts: PositiveInt = 3

    search_provider: SearchProviderName = SearchProviderName.SERPER
    serper_api_key: SecretStr | None = None
    serper_base_url: str = "https://google.serper.dev"
    exa_api_key: SecretStr | None = None
    exa_base_url: str = "https://api.exa.ai"
    search_results_per_query: PositiveInt = 6
    search_max_queries_per_run: PositiveInt = 6

    open_meteo_geocoding_url: str = "https://geocoding-api.open-meteo.com/v1/search"
    open_meteo_forecast_url: str = "https://api.open-meteo.com/v1/forecast"
    open_meteo_archive_url: str = "https://archive-api.open-meteo.com/v1/archive"
    forecast_horizon_days: PositiveInt = 16
    climate_normal_years: PositiveInt = 5
    frankfurter_base_url: str = "https://api.frankfurter.dev/v1"

    max_revisions: Annotated[int, Field(ge=0)] = 3
    agent_max_tool_iterations: PositiveInt = 16
    max_trip_days: PositiveInt = 30
    max_travelers: PositiveInt = 20
    prompt_version: str = "v1"

    day_min_scheduled_hours: Annotated[float, Field(ge=0)] = 3.0
    day_max_scheduled_hours: Annotated[float, Field(gt=0)] = 9.0
    day_max_neighbourhoods: PositiveInt = 3

    budget_ratio_lodging: Ratio = 0.40
    budget_ratio_food: Ratio = 0.25
    budget_ratio_activities: Ratio = 0.20
    budget_ratio_transport: Ratio = 0.10
    budget_ratio_contingency: Ratio = 0.05
    budget_emphasis_shift: Ratio = 0.05

    database_url: str = "sqlite+aiosqlite:///./data/itinera.db"
    checkpoint_db_path: Path = Path("./data/checkpoints.db")

    cache_enabled: bool = True
    cache_dir: Path = Path("./data/cache")
    cache_memory_maxsize: PositiveInt = 512
    cache_ttl_search_seconds: Annotated[int, Field(ge=0)] = 86_400
    cache_ttl_forecast_seconds: Annotated[int, Field(ge=0)] = 10_800
    cache_ttl_climate_seconds: Annotated[int, Field(ge=0)] = 2_592_000
    cache_ttl_fx_seconds: Annotated[int, Field(ge=0)] = 43_200
    cache_ttl_llm_seconds: Annotated[int, Field(ge=0)] = 0

    ssl_use_system_trust_store: bool = True

    http_timeout_seconds: Annotated[float, Field(gt=0)] = 20.0
    http_max_retries: Annotated[int, Field(ge=0)] = 3
    http_backoff_base_seconds: Annotated[float, Field(gt=0)] = 0.5
    http_backoff_max_seconds: Annotated[float, Field(gt=0)] = 8.0
    circuit_breaker_failure_threshold: PositiveInt = 3
    circuit_breaker_cooldown_seconds: Annotated[float, Field(gt=0)] = 60.0


    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        normalised = value.strip().upper()
        if normalised not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}, got {value!r}")
        return normalised

    @model_validator(mode="after")
    def _validate_budget_ratios(self) -> Settings:
        total = sum(self.budget_ratios.values())
        if abs(total - 1.0) > _RATIO_TOLERANCE:
            raise ValueError(
                f"BUDGET_RATIO_* values must sum to 1.0, got {total:.4f}. "
                "Adjust the ratios so the full budget is allocated."
            )
        return self

    @model_validator(mode="after")
    def _validate_provider_chain(self) -> Settings:
        chain = self.provider_chain
        if not chain:
            raise ValueError("LLM_PROVIDER_CHAIN must name at least one provider")
        if len(set(chain)) != len(chain):
            raise ValueError(
                f"LLM_PROVIDER_CHAIN contains duplicates: {self.llm_provider_chain!r}"
            )
        return self

    @model_validator(mode="after")
    def _validate_pacing_bounds(self) -> Settings:
        if self.day_max_scheduled_hours < self.day_min_scheduled_hours:
            raise ValueError(
                "DAY_MAX_SCHEDULED_HOURS must be greater than or equal to "
                "DAY_MIN_SCHEDULED_HOURS"
            )
        return self

    @model_validator(mode="after")
    def _validate_backoff_bounds(self) -> Settings:
        if self.http_backoff_max_seconds < self.http_backoff_base_seconds:
            raise ValueError(
                "HTTP_BACKOFF_MAX_SECONDS must be greater than or equal to "
                "HTTP_BACKOFF_BASE_SECONDS"
            )
        return self


    @computed_field  # type: ignore[prop-decorator]
    @property
    def provider_chain(self) -> tuple[LLMProviderName, ...]:
        """Ordered chat-model providers, parsed from the comma-separated env value.

        Raises:
            ConfigurationError: If the value names an unsupported provider.
        """
        names = [
            part.strip().lower() for part in self.llm_provider_chain.split(",") if part.strip()
        ]
        try:
            return tuple(LLMProviderName(name) for name in names)
        except ValueError as exc:
            supported = ", ".join(p.value for p in LLMProviderName)
            raise ConfigurationError(
                f"LLM_PROVIDER_CHAIN names an unsupported provider: {exc}. "
                f"Supported: {supported}"
            ) from exc

    @property
    def budget_ratios(self) -> dict[str, float]:
        """Budget split by category, guaranteed by validation to sum to 1.0."""
        return {
            "lodging": self.budget_ratio_lodging,
            "food": self.budget_ratio_food,
            "activities": self.budget_ratio_activities,
            "transport": self.budget_ratio_transport,
            "contingency": self.budget_ratio_contingency,
        }

    @property
    def is_production(self) -> bool:
        """Whether the process is running in the production environment."""
        return self.app_env is Environment.PRODUCTION

    def llm_api_keys(self, provider: LLMProviderName) -> tuple[SecretStr, ...]:
        """Return every usable key for ``provider``, primary first.

        Blank and unset entries are dropped, and duplicates are removed so pasting the
        same key into both slots does not silently double the chain's length without
        adding any quota.
        """
        configured = {
            LLMProviderName.GROQ: (self.groq_api_key, self.groq_api_key_alt),
            LLMProviderName.GEMINI: (self.google_api_key, self.google_api_key_alt),
            LLMProviderName.OPENROUTER: (self.openrouter_api_key, self.openrouter_api_key_alt),
            LLMProviderName.OLLAMA: (SecretStr("local"),),
            LLMProviderName.FAKE: (SecretStr("offline"),),
        }[provider]

        keys: list[SecretStr] = []
        seen: set[str] = set()
        for key in configured:
            if key is None:
                continue
            value = key.get_secret_value().strip()
            if value and value not in seen:
                seen.add(value)
                keys.append(key)
        return tuple(keys)

    def llm_models(self, provider: LLMProviderName) -> tuple[str, ...]:
        """Return the configured model identifiers for ``provider``, in order."""
        raw = {
            LLMProviderName.GROQ: self.groq_models,
            LLMProviderName.GEMINI: self.gemini_models,
            LLMProviderName.OPENROUTER: self.openrouter_models,
            LLMProviderName.OLLAMA: self.ollama_models,
            LLMProviderName.FAKE: "fake",
        }[provider]

        models: list[str] = []
        for part in raw.split(","):
            name = part.strip()
            if name and name not in models:
                models.append(name)
        return tuple(models)

    def require_search_api_key(self) -> str:
        """Return the active search provider's API key.

        Resolved lazily rather than at start-up so the service boots, serves ``/health``
        and runs its test suite without credentials; only a live research call fails.

        Raises:
            MissingCredentialsError: If the selected provider has no key configured.
        """
        key = (
            self.serper_api_key
            if self.search_provider is SearchProviderName.SERPER
            else self.exa_api_key
        )
        if key is None or not key.get_secret_value().strip():
            env_var = f"{self.search_provider.value.upper()}_API_KEY"
            raise MissingCredentialsError(
                f"Web search is configured to use {self.search_provider.value!r} but "
                f"{env_var} is not set. Add it to .env and restart.",
                provider=self.search_provider.value,
            )
        return key.get_secret_value()

    def resolve_path(self, path: Path) -> Path:
        """Resolve ``path`` against the project root when it is relative."""
        return path if path.is_absolute() else (_PROJECT_ROOT / path).resolve()

    def ensure_runtime_directories(self) -> None:
        """Create the directories the process writes to, if they do not yet exist."""
        targets = (
            self.resolve_path(self.cache_dir),
            self.resolve_path(self.checkpoint_db_path).parent,
        )
        for path in targets:
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance.

    Cached so the ``.env`` file is parsed and validated exactly once. Tests that need
    a different configuration should call ``get_settings.cache_clear()`` after
    patching the environment.
    """
    return Settings()
