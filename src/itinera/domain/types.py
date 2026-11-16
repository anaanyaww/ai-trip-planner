"""Shared field types for the domain models.

``SourceUrl`` exists because the two things a URL field has to do here pull in opposite
directions. It has to be *validated* - a search result whose URL is missing or malformed
must be rejected at the boundary rather than carried into a plan - and it has to be
*serialisable*, because every domain object ends up inside a LangGraph checkpoint.

``pydantic.HttpUrl`` gives the first and breaks the second: it validates well, but stores
a ``pydantic_core.Url`` object that the checkpointer's msgpack encoder cannot encode, so
a research brief carrying even one real source fails to checkpoint. That failure only
appears once a live search returns something, which is exactly the wrong time to find it.

This type runs the same validation and then stores the normalised string, so the field is
still rejected when it is not a URL and is still an ordinary ``str`` afterwards.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BeforeValidator, HttpUrl, TypeAdapter

_HTTP_URL = TypeAdapter(HttpUrl)


def _validate_url(value: Any) -> Any:
    """Validate ``value`` as an HTTP URL and return its normalised string form.

    Non-string values are passed through untouched so pydantic reports the type error
    itself rather than having it surface as a confusing URL-parsing failure.
    """
    if not isinstance(value, str):
        return value
    return str(_HTTP_URL.validate_python(value))


SourceUrl = Annotated[str, BeforeValidator(_validate_url)]
"""An HTTP URL, validated on the way in and stored as a string."""
