"""
Semaphores - Controlling access to limited resources.

WHAT IS A SEMAPHORE
-------------------
A semaphore is a counter-based gatekeeper. Where a Lock allows exactly one task through, a Semaphore allows up to N tasks through at once. Internally:

- It starts with a counter set to N (the number of permits).
- acquire decrements the counter. If the counter is 0, the task waits (suspended at an await) until a permit becomes available.
- release increments the counter, possibly waking one waiting task.

Think of each permit as a token. Tasks need a token to proceed, and hand it back when finished.

ANALOGY: THE PARKING LOT
------------------------
A parking lot with N spaces. Each car (task) that arrives:

- If a space is free (counter greater-than 0), the car drives in and the spaces-free sign ticks down by one.
- If the lot is full (counter equals 0), cars queue at the entrance until someone leaves and frees a space.

A Lock is just the special case of a 1-space parking lot. A Semaphore generalizes it to N spaces.

WHY DO I NEED THIS IN ASYNCIO
-----------------------------
Even though asyncio is single-threaded, tasks can be in flight simultaneously. Hundreds of tasks may be sitting on await for the same database, API, or file resource at once. Real resources have limits. A database might allow only 20 connections. An API might rate-limit you to 10 concurrent requests. Without a semaphore, all your tasks would pile onto the resource at once and overload it. With a semaphore, at most N are using it at any moment and the rest wait their turns politely.

Because every await yields control to the event loop, a semaphore is the standard way to say only N of my coroutines may be past this point at once.

TERMS DEFINED
-------------
- Permit: one unit of capacity. Acquiring takes one. Releasing returns one.
- Bounded semaphore: a semaphore that refuses to be released more times than it was acquired. Catches extra release bugs.
- Resource pool: a fixed set of reusable expensive objects (connections, threads, browser tabs) that tasks borrow and return.

MINIMAL EXAMPLE
---------------
::

    import asyncio
    from asyncio_mastery.synchronization import AsyncSemaphore

    sem = AsyncSemaphore(3)   # parking lot with 3 spaces

    async def work(i):
        async with sem:               # grab a parking space
            print(f"task {i} started")
            await asyncio.sleep(1)    # at most 3 of these run at once
            print(f"task {i} done")

    asyncio.run(asyncio.gather(*[work(i) for i in range(10)]))

Run this file directly to see semaphore patterns in action:

    python -m asyncio_mastery.synchronization.semaphores
"""

import asyncio
import time
from asyncio import BoundedSemaphore as StdBoundedSemaphore
from asyncio import Semaphore
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Generic, TypeVar

T = TypeVar("T")


class AsyncSemaphore:
    """An asyncio semaphore with monitoring for learning and debugging.

    Wraps asyncio.Semaphore and adds a name plus counters of how many times it has been acquired and released. So you can inspect its state (for example sem.value) while experimenting.

    Example:
        ::

            sem = AsyncSemaphore(2, name="downloads")

            async with sem:
                # At most 2 tasks execute this block concurrently.
                await download(url)
    """

    def __init__(self, value: int = 1, *, name: str | None = None) -> None:
        """Create a semaphore with value permits (parking spaces).

        Args:
            value: Number of tasks allowed through simultaneously. 1 behaves exactly like a lock. Must be greater-than-or-equal-to 0. 0 means every acquirer waits until someone releases. Occasionally useful as a gate that starts closed.
            name: Optional human-readable name for debugging and logging.

        Raises:
            ValueError: If value is negative.
        """
        if value < 0:
            raise ValueError("Semaphore value must be >= 0")
        self._semaphore = Semaphore(value)
        self._initial_value = value
        self._name = name or "semaphore"
        self._acquire_count = 0
        self._release_count = 0

    @property
    def name(self) -> str:
        """The human-readable name of this semaphore."""
        return self._name

    @property
    def value(self) -> int:
        """Approximate number of free permits (parking spaces) right now.

        Computed from our bookkeeping: initial permits, plus releases, minus acquires. Approximate because another task could acquire the instant after you read it. Treat it as a debugging aid, not a decision input.
        """
        return self._initial_value + self._release_count - self._acquire_count

    async def acquire(self) -> bool:
        """Take one permit, waiting if none are free.

        If the lot is full, this coroutine suspends at the await and the event loop runs other tasks. We are woken when someone releases.

        Returns:
            True once a permit has been acquired.
        """
        # The await inside here is the yield point where we wait in line at the parking-lot entrance if all spaces are taken.
        await self._semaphore.acquire()
        self._acquire_count += 1
        return True

    def release(self) -> None:
        """Return one permit, allowing one waiting task through.

        A plain asyncio.Semaphore lets you release more times than you acquired (growing the counter). Use BoundedSemaphore to make that a loud error instead.
        """
        self._semaphore.release()
        self._release_count += 1

    async def __aenter__(self) -> "AsyncSemaphore":
        """Enter async with. Acquire one permit."""
        await self.acquire()
        return self

    async def __aexit__(self, *args: object) -> None:
        """Exit async with. Release the permit even if the body raised."""
        self.release()

    def locked(self) -> bool:
        """True if no permits are currently available (the lot is full).

        Returns:
            bool: value less-than-or-equal-to 0.
        """
        return self.value <= 0


class BoundedSemaphore:
    """A semaphore that raises if you release more times than you acquired.

    A plain semaphore counter can grow beyond its initial value if you call release too many times. Usually a bug (like a parking lot reporting 15 free spaces when it only has 10). A bounded semaphore caps the counter at its initial value and raises ValueError on over-release, catching the bug immediately.

    Rule of thumb: use BoundedSemaphore unless you specifically need extra releases (rare).
    """

    def __init__(self, value: int = 1, *, name: str | None = None) -> None:
        """Create a bounded semaphore with value permits.

        Args:
            value: Maximum number of permits. Releasing beyond this raises.
            name: Optional human-readable name for debugging and logging.

        Raises:
            ValueError: If value is negative.
        """
        if value < 0:
            raise ValueError("Semaphore value must be >= 0")
        self._semaphore = StdBoundedSemaphore(value)
        self._name = name or "bounded_semaphore"

    async def acquire(self) -> bool:
        """Take one permit, waiting (suspended) if the lot is full.

        Returns:
            True once a permit has been acquired.
        """
        await self._semaphore.acquire()
        return True

    def release(self) -> None:
        """Return one permit.

        Raises:
            ValueError: If the semaphore is already at its maximum. You released a permit you never acquired.
        """
        self._semaphore.release()

    async def __aenter__(self) -> "BoundedSemaphore":
        """Enter async with. Acquire one permit."""
        await self.acquire()
        return self

    async def __aexit__(self, *args: object) -> None:
        """Exit async with. Release the permit even on exceptions."""
        self.release()


class SemaphorePool(Generic[T]):
    """A pool of reusable resources (connections, handles) capped by a semaphore.

    Classic use case: database connections. Opening a connection is expensive, and the database only allows a limited number. So you keep a small pool and lend them out:

    - acquire_resource hands you a resource, creating one if the pool is empty but capacity remains, or waiting if all are checked out.
    - release_resource returns it to the pool for reuse.

    The semaphore limits total live resources. The internal lock protects the _resources/_in_use bookkeeping from interleaving at awaits.

    Type Parameters:
        T: The kind of resource being pooled (for example a connection object).

    Example:
        ::

            pool = SemaphorePool[str](max_size=2, name="db")
            async with pool.resource() as conn:
                run_query(conn)   # at most 2 tasks hold a conn at once
    """

    def __init__(self, max_size: int, *, name: str = "pool") -> None:
        """Create a pool holding at most max_size resources.

        Args:
            max_size: Maximum resources ever checked out at once.
            name: Name for the underlying semaphore (debugging/logging).
        """
        self._semaphore = AsyncSemaphore(max_size, name=name)
        self._resources: list[T] = []
        self._in_use: set[T] = set()
        self._lock = asyncio.Lock()

    async def acquire_resource(self) -> T:
        """Borrow a resource, waiting if the pool is exhausted.

        Waits on the semaphore for capacity, then either reuses an idle resource from the pool or creates a fresh one via _create_resource.

        Returns:
            A resource of type T now exclusively yours until you call release_resource.
        """
        # First grab a permit: if max_size resources are already checked out, we park here until someone returns theirs.
        await self._semaphore.acquire()
        # The lock protects the list/set mutation below from interleaving with another task acquire/release at the await inside _create_resource.
        async with self._lock:
            if self._resources:
                resource = self._resources.pop()  # reuse an idle one
            else:
                resource = await self._create_resource()
            self._in_use.add(resource)
            return resource

    async def release_resource(self, resource: T) -> None:
        """Return a borrowed resource to the pool for reuse.

        Args:
            resource: The object previously returned by acquire_resource. Unknown resources are ignored for the bookkeeping but still release a semaphore permit.
        """
        async with self._lock:
            if resource in self._in_use:
                self._in_use.remove(resource)
                self._resources.append(resource)  # back on the shelf
            # Freeing a permit may wake a task waiting in acquire_resource.
            self._semaphore.release()

    async def _create_resource(self) -> T:
        """Create one new resource. Override in a subclass.

        By default returns a bare object() placeholder. Real pools override this to open a DB connection, HTTP session, etc.

        Returns:
            A new resource instance of type T.
        """
        return object()  # type: ignore[return-value]

    @asynccontextmanager
    async def resource(self) -> AsyncIterator[T]:
        """Borrow a resource as an async context manager (recommended).

        Guarantees the resource is returned to the pool even if the body raises an exception.

        Yields:
            The borrowed resource of type T.

        Example:
            ::

                async with pool.resource() as conn:
                    await conn.execute(...)
        """
        resource = await self.acquire_resource()
        try:
            yield resource
        finally:
            # finally = the resource always goes back, even on exceptions.
            await self.release_resource(resource)

    @property
    def available(self) -> int:
        """How many more resources can be checked out without waiting."""
        return self._semaphore.value

    @property
    def in_use(self) -> int:
        """How many resources are currently checked out."""
        return len(self._in_use)


async def demonstrate_semaphores() -> None:
    """Run a narrated tour of common semaphore patterns.

    Demonstrates:
      1. A basic semaphore letting 3 of 5 workers run at once.
      2. A bounded semaphore (over-release protection).
      3. A resource pool emulating limited DB connections.
      4. Rate limiting: 20 API calls, at most 10 concurrent.

    Returns:
        None. Prints demonstration output to stdout.
    """
    print("=== Basic Semaphore ===")
    sem = AsyncSemaphore(3, name="basic")

    async def worker(id: int) -> None:
        # At most 3 workers can be inside this block at any moment. The others are parked at the await inside acquire.
        async with sem:
            print(f"Worker {id} acquired (available: {sem.value})")
            await asyncio.sleep(0.05)
            print(f"Worker {id} releasing (available: {sem.value})")

    await asyncio.gather(*[worker(i) for i in range(5)])

    print("\n=== Bounded Semaphore ===")
    bounded = BoundedSemaphore(2, name="bounded")

    async def bounded_worker(id: int) -> None:
        async with bounded:
            print(f"Bounded worker {id} acquired")
            await asyncio.sleep(0.01)

    await asyncio.gather(*[bounded_worker(i) for i in range(4)])

    print("\n=== Semaphore Pool ===")
    # Only 2 connections can be checked out at once, even though 5 queries want one.
    pool = SemaphorePool[str](max_size=2, name="db-connections")

    class DBConnection:
        def __init__(self, id: int) -> None:
            self.id = id

        def query(self, sql: str) -> str:
            return f"Conn {self.id}: {sql}"

    async def use_db(query: str) -> str:
        # resource() borrows a connection and ALWAYS returns it, even if the query explodes. So a failing task cannot leak permits.
        async with pool.resource() as conn:
            await asyncio.sleep(0.01)
            return conn.query(query)

    results = await asyncio.gather(*[use_db(f"SELECT {i}") for i in range(5)])
    for r in results:
        print(f"  {r}")

    print("\n=== Rate Limiting with Semaphore ===")
    # 10 permits => at most 10 HTTP requests in flight concurrently.
    rate_limiter = AsyncSemaphore(10, name="rate-limit")

    async def api_call(endpoint: str) -> str:
        async with rate_limiter:
            await asyncio.sleep(0.01)  # Simulate API call
            return f"Response from {endpoint}"

    # Simulate a burst of requests: 20 calls run in two waves of ~10.
    start = time.perf_counter()
    await asyncio.gather(*[api_call(f"/api/{i}") for i in range(20)])
    elapsed = time.perf_counter() - start
    print(f"20 requests with limit 10 took: {elapsed:.3f}s")


if __name__ == "__main__":
    asyncio.run(demonstrate_semaphores())
