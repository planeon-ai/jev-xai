"""Run a coroutine from sync code, and refuse to do so inside a running loop."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

import anyio

from jev_xai.errors import JevXaiUsageError

T = TypeVar("T")


def run_sync(fn: Callable[..., Awaitable[T]], /, *args: object, **kwargs: object) -> T:
    """Run ``fn`` to completion, unless an event loop is already running."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:

        async def _call() -> T:
            return await fn(*args, **kwargs)

        return anyio.run(_call)
    raise JevXaiUsageError(
        "jev-xai sync API was called from a running event loop. "
        "Await the async method instead (for example `await client.predict(x)`)."
    )
