"""
Advanced - Advanced asyncio concepts for serious async programmers.

This subpackage contains advanced asyncio modules. You should know basic
``async def``/``await``, ``asyncio.gather``, and tasks before you use these
modules. Each module is self-contained. You can run a module as a demo with
``python -m asyncio_mastery.advanced.<module>``. The modules follow a
suggested learning order.

MODULES
-------
cancellation
    Stop tasks gracefully: ``task.cancel()``, ``CancelledError`` handling,
    cancellation tokens, and protecting critical sections with
    ``asyncio.shield``. Start here because timeouts use cancellation.

timeouts
    Limit how long operations run. Python 3.11+ ``asyncio.timeout`` /
    ``asyncio.timeout_at`` context managers, the unified builtin
    ``TimeoutError``, plus reusable ``Timeout`` and ``TimeoutManager``
    helpers. Think of them as kitchen timers for your coroutines.

context_vars
    ``contextvars.ContextVar`` - the async-safe replacement for global and
    thread-local variables. Carry request IDs, user IDs, and trace context
    automatically across awaits and tasks.

async_generators
    ``async def`` + ``yield`` = lazy async streams that produce values
    over time. Includes ``async_range``, ``async_map``, and ``async_filter``
    building blocks.

run_in_executor
    Run blocking code in thread pools and CPU-bound work in process pools
    via ``loop.run_in_executor`` without freezing the event loop.

LEARNING ORDER
--------------
1. cancellation
2. timeouts (timeouts ARE scheduled cancellations)
3. context_vars
4. async_generators
5. run_in_executor

Every module defines a ``demonstrate_*()`` coroutine with an
``if __name__ == "__main__":`` block. Run the modules, watch the output,
then read the fully-commented source code.
"""

from asyncio_mastery.advanced.async_generators import (
    AsyncGenerator,
    async_filter,
    async_map,
    async_range,
    demonstrate_async_generators,
)
from asyncio_mastery.advanced.cancellation import (
    CancellationToken,
    cancel_on,
    demonstrate_cancellation,
    shield,
)
from asyncio_mastery.advanced.context_vars import (
    ContextVarManager,
    demonstrate_context_vars,
    request_id_var,
    user_id_var,
)
from asyncio_mastery.advanced.run_in_executor import (
    ProcessPool,
    ThreadPool,
    demonstrate_executors,
    run_in_executor,
    run_in_process,
    run_in_thread,
)
from asyncio_mastery.advanced.timeouts import (
    Timeout,
    demonstrate_timeouts,
    run_with_timeout,
    timeout_after,
)

__all__ = [
    "AsyncGenerator",
    "CancellationToken",
    "ContextVarManager",
    "ProcessPool",
    "ThreadPool",
    "Timeout",
    "async_filter",
    "async_map",
    "async_range",
    "cancel_on",
    "demonstrate_async_generators",
    "demonstrate_cancellation",
    "demonstrate_context_vars",
    "demonstrate_executors",
    "demonstrate_timeouts",
    "request_id_var",
    "run_in_executor",
    "run_in_process",
    "run_in_thread",
    "run_with_timeout",
    "shield",
    "timeout_after",
    "user_id_var",
]
