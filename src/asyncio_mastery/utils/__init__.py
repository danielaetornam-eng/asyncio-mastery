"""
Utils - Everyday async utilities: decorators, helpers, and logging.

This package collects the "glue" code that almost every real asyncio
program ends up needing, but which is not part of the language. Importing
from ``asyncio_mastery.utils`` re-exports everything below, so you can
write ``from asyncio_mastery.utils import async_retry`` directly.

what is INSIDE (AND WHEN TO USE IT)
----------------------------------
* :mod:`~asyncio_mastery.utils.decorators` — wrappers that add behaviors
  to your async functions with one ``@decorator`` line. Use these when
  you need retries with backoff, timeouts, caching, debounce/throttle,
  one-time initialization, or fire-and-forget background tasks — WITHOUT
  rewriting the function's core logic.

* :mod:`~asyncio_mastery.utils.helpers` — small building-block classes
  and functions. Use these when you want an in-memory cache with TTL,
  a rate limiter, a convenient ``async for`` wrapper over plain
  iterables, bounded-concurrency ``gather``, or a way to run async code
  from sync code.

* :mod:`~asyncio_mastery.utils.logging` — async-friendly, structured
  logging. Use this in ANY multi-task program: it attaches per-task
  context (request/user/trace ids via ``contextvars``) to every log
  line automatically, can emit JSON for log aggregators, and can
  offload blocking writes to a thread so the event loop never stalls.

TIP FOR LEARNERS: browse decorators first — they are the most immediately
useful — then helpers, then logging once you build a multi-task server.

Example:
    >>> from asyncio_mastery.utils import async_retry, get_logger
    >>> log = get_logger(__name__)
    >>> @async_retry(max_attempts=3)
    ... async def fetch(): ...
"""

from asyncio_mastery.utils.decorators import (
    async_cached,
    async_retry,
    async_singleton,
    async_timeout,
    background_task,
    debounce,
    throttle,
)
from asyncio_mastery.utils.helpers import (
    AsyncCache,
    AsyncIterator,
    AsyncRateLimiter,
    create_task_group,
    gather_with_concurrency,
    run_async,
)
from asyncio_mastery.utils.logging import AsyncLogger, StructuredLogger, get_logger

__all__ = [
    "AsyncCache",
    "AsyncRateLimiter",
    "AsyncIterator",
    "AsyncLogger",
    "StructuredLogger",
    "async_cached",
    "async_retry",
    "async_singleton",
    "async_timeout",
    "background_task",
    "create_task_group",
    "debounce",
    "gather_with_concurrency",
    "get_logger",
    "run_async",
    "throttle",
]
