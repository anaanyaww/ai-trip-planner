"""Python 3.10 compatibility for LangGraph interrupts.

``interrupt()`` reads the active runnable config from a context variable. LangGraph
populates that variable by starting each async node inside a copied context, which
needs ``asyncio.create_task(..., context=...)`` - a parameter added in Python 3.11. On
3.10 the variable is simply never set, and calling ``interrupt`` from an async node
fails with "Called get_config outside of a runnable context".

The workflow's whole design rests on that interrupt, and the project targets 3.10, so
the config is set here instead. LangGraph passes it to any node whose signature asks
for it, which makes this a supported hand-off rather than a reach into internals.

This module can be deleted once the project's floor moves to Python 3.11.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager

from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import var_child_runnable_config

_NATIVE_CONTEXT_VERSION = (3, 11)


@contextmanager
def runnable_context(config: RunnableConfig) -> Iterator[None]:
    """Make ``config`` visible to ``interrupt()`` for the duration of the block.

    A no-op on Python 3.11 and later, where LangGraph has already established the
    context itself and setting it again would be redundant.

    Args:
        config: The config LangGraph passed to the node.
    """
    if sys.version_info >= _NATIVE_CONTEXT_VERSION:
        yield
        return

    token = var_child_runnable_config.set(config)
    try:
        yield
    finally:
        var_child_runnable_config.reset(token)
