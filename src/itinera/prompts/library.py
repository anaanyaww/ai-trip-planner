"""Versioned prompt loading.

Prompts are the part of an LLM system most likely to change and hardest to review
when they change. Keeping them in files under a version directory means a prompt edit
shows up in a diff as a prompt edit, and means two versions can coexist while one is
being evaluated against the other - neither of which is true of strings embedded in
agent code.

Templates are Jinja, so the context an agent passes is substituted by name rather than
by positional formatting, and a template referencing something the caller did not
supply fails loudly at render time instead of silently emitting an empty string.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateNotFound

from itinera.core.config import Settings
from itinera.core.exceptions import ConfigurationError
from itinera.core.logging import get_logger

logger = get_logger(__name__)

_TEMPLATE_ROOT = Path(__file__).parent / "templates"
_TEMPLATE_SUFFIX = ".md.j2"


class PromptLibrary:
    """Renders the prompt templates belonging to one prompt version."""

    def __init__(self, version: str) -> None:
        """Bind the library to a version directory.

        Args:
            version: Directory name under ``prompts/templates``, from configuration.

        Raises:
            ConfigurationError: If no directory exists for that version.
        """
        self._version = version
        self._directory = _TEMPLATE_ROOT / version
        if not self._directory.is_dir():
            available = sorted(path.name for path in _TEMPLATE_ROOT.iterdir() if path.is_dir())
            raise ConfigurationError(
                f"PROMPT_VERSION={version!r} has no template directory. "
                f"Available versions: {', '.join(available) or 'none'}."
            )

        self._environment = Environment(
            loader=FileSystemLoader(self._directory),
            undefined=StrictUndefined,
            autoescape=False,  # noqa: S701
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=False,
        )

    @property
    def version(self) -> str:
        """The prompt version this library serves."""
        return self._version

    def render(self, name: str, **context: Any) -> str:
        """Render a template to a prompt string.

        Args:
            name: Template name without its extension, for example ``"research_system"``.
            **context: Values substituted into the template.

        Returns:
            The rendered prompt, stripped of surrounding whitespace.

        Raises:
            ConfigurationError: If the template does not exist in this version.
        """
        try:
            template = self._environment.get_template(f"{name}{_TEMPLATE_SUFFIX}")
        except TemplateNotFound as exc:
            raise ConfigurationError(
                f"Prompt {name!r} does not exist in version {self._version!r}"
            ) from exc

        return template.render(**context).strip()

    def names(self) -> list[str]:
        """Return every prompt available in this version, for diagnostics."""
        return sorted(
            path.name.removesuffix(_TEMPLATE_SUFFIX)
            for path in self._directory.glob(f"*{_TEMPLATE_SUFFIX}")
        )


@lru_cache(maxsize=4)
def _library_for(version: str) -> PromptLibrary:
    """Return the cached library for ``version``.

    Cached because building the Jinja environment reads the directory, and agents ask
    for prompts on every turn.
    """
    library = PromptLibrary(version)
    logger.info("prompts.loaded", version=version, prompts=library.names())
    return library


def get_prompt_library(settings: Settings) -> PromptLibrary:
    """Return the prompt library for the configured version."""
    return _library_for(settings.prompt_version)
