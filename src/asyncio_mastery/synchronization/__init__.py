"""
Synchronization Primitives - Coordinating concurrent async tasks.

PURPOSE
-------
asyncio runs all tasks on one thread. Beginners often assume synchronization is unnecessary. This assumption is wrong. Every await is a point where a task pauses. The event loop switches to another task. Two tasks can interleave in the middle of a read-modify-write sequence. They can corrupt shared state silently. This bug is a race condition. The outcome depends on unpredictable timing. Synchronization primitives prevent race conditions. They let you control when tasks proceed.

PRIMITIVES
----------
Each primitive answers a different question:

- AsyncLock: Only one task at a time. Bathroom key analogy. Mutual exclusion for a critical section. Code touches shared data. Use for protecting counters, dicts, and invariants.

- AsyncSemaphore / BoundedSemaphore: Up to N tasks at once. Parking lot with N spaces. Limit concurrency against a scarce resource. Database connections. API rate limits. SemaphorePool builds a full borrow-return resource pool on top.

- AsyncEvent: Wait until X happens. Starting gun at a race. One task signals. All waiters wake. Events carry no count and no data. Just has it happened yet? EventNotifier organizes them by name.

- AsyncCondition: Wait until this state is true. Bakery queue analogy. Lock plus wait-slash-notify combined. Waiters sleep until a predicate holds. Example: len(buffer) greater-than 0. The producer-slash-consumer workhorse.

- AsyncBarrier: Nobody proceeds until all N arrive. Friends meeting outside a restaurant. Synchronizes phased parallel work.

MINIMAL EXAMPLE
---------------
::

    import asyncio
    from asyncio_mastery.synchronization import AsyncLock, AsyncEvent

    lock = AsyncLock()
    done = AsyncEvent()
    counter = 0

    async def worker():
        global counter
        async with lock:
            current = counter
            await asyncio.sleep(0)
            counter = current + 1
        done.set()

DECISION GUIDE
--------------
- Guarding shared data? -> AsyncLock
- Limiting how many run at once? -> AsyncSemaphore or SemaphorePool
- One task telling others go? -> AsyncEvent
- Waiting for a complex state? -> AsyncCondition
- Everyone must meet per phase? -> AsyncBarrier
"""

from asyncio_mastery.synchronization.barriers import (
    AsyncBarrier,
    BarrierExample,
    demonstrate_barriers,
)
from asyncio_mastery.synchronization.conditions import (
    AsyncCondition,
    ConditionExample,
    demonstrate_conditions,
)
from asyncio_mastery.synchronization.events import AsyncEvent, EventNotifier, demonstrate_events
from asyncio_mastery.synchronization.locks import AsyncLock, LockExample, demonstrate_locks
from asyncio_mastery.synchronization.semaphores import (
    AsyncSemaphore,
    BoundedSemaphore,
    SemaphorePool,
    demonstrate_semaphores,
)

# Public API: importing from this package gives you every primitive plus
# runnable demos (call e.g. asyncio.run(demonstrate_locks()) to see one).
__all__ = [
    "AsyncBarrier",
    "AsyncCondition",
    "AsyncEvent",
    "AsyncLock",
    "AsyncSemaphore",
    "BarrierExample",
    "BoundedSemaphore",
    "ConditionExample",
    "EventNotifier",
    "LockExample",
    "SemaphorePool",
    "demonstrate_barriers",
    "demonstrate_conditions",
    "demonstrate_events",
    "demonstrate_locks",
    "demonstrate_semaphores",
]
