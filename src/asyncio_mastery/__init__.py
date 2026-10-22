"""
Asyncio Mastery - Learn Python's asyncio from first steps to production.

WELCOME, ABSOLUTE BEGINNER
--------------------------
``asyncio_mastery`` is a teaching library: every module is fully
documented working code that demonstrates a piece of Python's ``asyncio``
(the standard library's framework for *concurrent* programming with
``async``/``await``). Read the modules in order, run the demos under
``if __name__ == "__main__":`` blocks, and tweak them.

WHY ASYNCIO MATTERS
-------------------
Programs spend most of their time WAITING — for networks, disks,
databases (this is called I/O, input/output). The naive solutions are:
do one thing at a time (slow), or use threads (heavy: each thread costs
memory and context-switch overhead, and forces you to guard shared data
with locks). Asyncio offers a third way: ONE thread runs MANY tasks,
and whenever a task hits ``await`` (an I/O wait), the *event loop* —
the scheduler at asyncio's core — instantly switches to another ready
task. Hundreds of concurrent connections on a single thread, no thread
locks needed. that is the whole magic.

RECOMMENDED LEARNING PATH
-------------------------
1. :mod:`~asyncio_mastery.basics` — coroutines, tasks, futures, and the
   event loop itself. Start here; everything else builds on these.
2. :mod:`~asyncio_mastery.synchronization` — locks, semaphores, events,
   conditions, barriers: how tasks coordinate without stepping on each
   other.
3. :mod:`~asyncio_mastery.streams` — TCP/UDP/SSL networking with async
   streams (clients and servers).
4. :mod:`~asyncio_mastery.subprocess` — launching and talking to other
   programs asynchronously.
5. :mod:`~asyncio_mastery.advanced` — context variables, async
   generators, cancellation, shielding, timeouts.
6. :mod:`~asyncio_mastery.patterns` — production building blocks:
   worker/connection pools, rate limiters, retry policies, circuit
   breakers, pub/sub, batching.
7. :mod:`~asyncio_mastery.testing` — how to test async code reliably.
8. :mod:`~asyncio_mastery.integrations` — bridging asyncio with
   threads, processes, HTTP clients, and databases.
9. :mod:`~asyncio_mastery.utils` — everyday utilities: decorators
   (retry/timeout/cache), small helpers, and structured logging.

FURTHER READING IN THIS REPO
----------------------------
* ``docs/concepts.md`` — the theory behind each concept, in plain words.
* ``docs/patterns.md`` — when and why to reach for each pattern.

This package provides:
- Basic asyncio primitives (coroutines, tasks, futures, event loop)
- Synchronization primitives (locks, semaphores, events, conditions, barriers)
- Stream-based networking (TCP, UDP, SSL)
- Subprocess management
- Advanced concepts (context vars, async generators, cancellation)
- Production patterns (pools, rate limiting, retry, circuit breakers)
- Testing utilities
- Integrations (threading, multiprocessing, HTTP, databases)
"""

__version__ = "1.0.0"
__author__ = "Asyncio Mastery Team"
__all__ = [
    "advanced",
    "basics",
    "integrations",
    "patterns",
    "streams",
    "subprocess",
    "synchronization",
    "testing",
    "utils",
]

# Re-export key public APIs for convenience
from asyncio_mastery.advanced import (  # noqa: F401
    cancel_on,
    run_with_timeout,
    shield,
)
from asyncio_mastery.basics import (  # noqa: F401
    demonstrate_coroutines,
    demonstrate_event_loop,
    demonstrate_futures,
    demonstrate_tasks,
)
from asyncio_mastery.patterns import (  # noqa: F401
    AsyncContextManager,
    Batcher,
    CircuitBreaker,
    ConnectionPool,
    PubSub,
    RateLimiter,
    RetryPolicy,
    TaskGroup,
    WorkerPool,
    async_context_manager,
)
from asyncio_mastery.synchronization import (  # noqa: F401
    AsyncBarrier,
    AsyncCondition,
    AsyncEvent,
    AsyncLock,
    AsyncSemaphore,
)

__all__.extend(
    [
        "AsyncBarrier",
        "AsyncCondition",
        "AsyncContextManager",
        "AsyncEvent",
        "AsyncLock",
        "AsyncSemaphore",
        "Batcher",
        "CircuitBreaker",
        "ConnectionPool",
        "PubSub",
        "RateLimiter",
        "RetryPolicy",
        "TaskGroup",
        "WorkerPool",
        "async_context_manager",
        "cancel_on",
        "demonstrate_coroutines",
        "demonstrate_event_loop",
        "demonstrate_futures",
        "demonstrate_tasks",
        "run_with_timeout",
        "shield",
    ]
)
