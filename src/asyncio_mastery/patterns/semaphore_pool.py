"""
Semaphore Pool - A named collection of semaphores for resource management.

THE PROBLEM
-----------
A single :class:`asyncio.Semaphore` limits access to *one* resource
(e.g., "at most 5 database queries at once"). But real applications
juggle *several* limited resources at once — a database pool, an
external API, a cache, a disk — each with its own capacity. Creating
and tracking a separate semaphore variable for each is repetitive and
error-prone. A semaphore pool gives you a central registry: ask for a
semaphore by *name* and the pool creates it on demand.

REAL-WORLD ANALOGY
------------------
A **gym with limited equipment**: there are only 5 treadmills, 3 squat
racks, and 10 stationary bikes. Each equipment type is a named
semaphore; you must "acquire" a station before using it and "release"
it when done, so nobody hoards equipment and everyone waits fairly.

JARGON
------
* **Semaphore**: a counter used to limit concurrency. Acquiring
  decrements the counter; when it hits 0, further acquirers wait.
  Releasing increments it and wakes a waiter.
* **Deadlock**: two tasks each holding a resource the other needs,
  waiting forever on each other. :class:`MultiSemaphore` avoids this
  by always acquiring in sorted name order.

WHEN TO USE IT
--------------
* Limiting several resource types with one registry
  (``pool.create("db", 5)``, ``pool.create("api", 10)``).
* Operations that need *several* resources at once, safely
  (``async with pool.acquire_many("db", "api")``).
* Bulk quota-style operations with :class:`ResourceGuard` (e.g., "this
  job needs 2 database slots and 1 API slot").

Minimal example
---------------
>>> import asyncio
>>> async def main() -> None:
...     pool = SemaphorePool()
...     pool.create("database", 2)          # at most 2 concurrent users
...     async with pool.acquire("database"):
...         print("using the database")
>>> asyncio.run(main())
"""

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from asyncio_mastery.synchronization import AsyncSemaphore


@dataclass
class SemaphoreConfig:
    """Plain data holder describing one named semaphore.

    Attributes:
        value: Maximum number of concurrent holders (the semaphore's
            initial counter).
        name: Human-readable label used to look the semaphore up.
    """

    value: int = 1
    name: str | None = None


class SemaphorePool:
    """A registry of named semaphores, created on demand.

    The pool maps a string name (e.g., ``"database"``) to an
    :class:`AsyncSemaphore`. Names make intent obvious at call sites
    and let many distant parts of a program share the same limits.

    Example:
        >>> pool = SemaphorePool()
        >>> pool.create("api", 10)
        >>> async with pool.acquire("api"):
        ...     await call_api()
    """

    def __init__(self) -> None:
        self._semaphores: dict[str, AsyncSemaphore] = {}
        self._configs: dict[str, SemaphoreConfig] = {}
        self._lock = asyncio.Lock()

    def create(self, name: str, value: int = 1) -> AsyncSemaphore:
        """Create the named semaphore if missing, then return it.

        Calling ``create`` twice with the same name is safe: the
        existing semaphore is returned and the ``value`` argument of
        the second call is ignored.

        Args:
            name: Unique label for this resource (e.g., ``"cache"``).
            value: Max concurrent holders. Only used on first creation.

        Returns:
            AsyncSemaphore: The existing or newly created semaphore.
        """
        if name not in self._semaphores:
            self._semaphores[name] = AsyncSemaphore(value, name=name)
            self._configs[name] = SemaphoreConfig(value=value, name=name)
        return self._semaphores[name]

    def get(self, name: str) -> AsyncSemaphore | None:
        """Look up a semaphore without creating it.

        Args:
            name: The label previously passed to :meth:`create`.

        Returns:
            AsyncSemaphore | None: The semaphore, or None if the pool
            has never heard of this name.
        """
        return self._semaphores.get(name)

    def remove(self, name: str) -> bool:
        """Remove a semaphore from the pool.

        WARNING: tasks currently holding the removed semaphore still
        work (they hold a direct reference), but new lookups get None.

        Args:
            name: Label of the semaphore to drop.

        Returns:
            bool: True if it existed and was removed; False otherwise.
        """
        if name in self._semaphores:
            del self._semaphores[name]
            del self._configs[name]
            return True
        return False

    @asynccontextmanager
    async def acquire(self, name: str, value: int = 1) -> AsyncIterator[AsyncSemaphore]:
        """Acquire the named semaphore for the duration of a block.

        Guarantees ``release`` runs even if the body raises — writing
        ``async with pool.acquire("db"):`` means you can never forget
        to release, which is the #1 semaphore bug.

        Args:
            name: Semaphore label (created with ``value`` if missing).
            value: Capacity to use if the semaphore must be created.

        Yields:
            AsyncSemaphore: The semaphore itself (handy for inspecting
            its remaining value).
        """
        sem = self.create(name, value)
        await sem.acquire()
        try:
            yield sem
        finally:
            sem.release()

    def acquire_many(self, *names: str) -> "MultiSemaphore":
        """Acquire several semaphores atomically as one ``async with``.

        Args:
            *names: Labels of all semaphores needed at once.

        Returns:
            MultiSemaphore: An async context manager that acquires all
            named semaphores (in sorted order, to avoid deadlock) and
            releases them in reverse.
        """
        return MultiSemaphore(self, names)

    def stats(self) -> dict[str, dict[str, Any]]:
        """Snapshot of every semaphore's current state.

        Returns:
            dict: ``{name: {"value": ..., "initial_value": ...,
            "available": ...}}`` — useful for dashboards and debugging
            "who is holding all the slots?".
        """
        return {
            name: {"value": sem.value, "initial_value": config.value, "available": sem.value}
            for name, (sem, config) in self._get_all().items()
        }

    def _get_all(self) -> dict[str, tuple[AsyncSemaphore, SemaphoreConfig]]:
        return {name: (self._semaphores[name], self._configs[name]) for name in self._semaphores}


class MultiSemaphore:
    """Acquire multiple semaphores atomically via ``async with``.

    Acquiring several semaphores naively (task A grabs "db" then waits
    for "api", task B grabs "api" then waits for "db") deadlocks. This
    class prevents that by always acquiring in **sorted name order** —
    a classic technique from operating-systems textbooks. Releases
    happen in reverse order.

    Example:
        >>> async with pool.acquire_many("database", "api"):
        ...     await do_something_needing_both()
    """

    def __init__(self, pool: SemaphorePool, names: tuple[str, ...]) -> None:
        """Create the multi-acquire guard.

        Args:
            pool: The pool the named semaphores come from.
            names: The semaphore labels to acquire together.
        """
        self._pool = pool
        self._names = names
        self._acquired: list[AsyncSemaphore] = []

    async def __aenter__(self) -> "MultiSemaphore":
        # Acquire in order to avoid deadlock
        for name in sorted(self._names):
            sem = self._pool.create(name)
            await sem.acquire()
            self._acquired.append(sem)
        return self

    async def __aexit__(self, *args: Any) -> None:
        for sem in reversed(self._acquired):
            sem.release()


class ResourceGuard:
    """Acquire *quantities* of named resources and track what you hold.

    Unlike :class:`MultiSemaphore` (one slot per name), a guard lets
    an operation claim *several* units of the same resource at once —
    e.g., ``guard.use(database=2, api=1)`` — and remembers how many it
    holds so :meth:`release` gives back exactly what was taken.

    Example:
        >>> guard = ResourceGuard(pool)
        >>> async with guard.use(database=2):
        ...     await heavy_two_slot_job()
    """

    def __init__(self, pool: SemaphorePool) -> None:
        """Create a guard bound to a pool.

        Args:
            pool: The :class:`SemaphorePool` providing the semaphores.
        """
        self._pool = pool
        self._resources: dict[str, int] = defaultdict(int)

    async def acquire(self, **resources: int) -> None:
        """Acquire ``count`` units of each named resource.

        Suspends until every requested unit is available. Acquisition
        happens in sorted name order (deadlock avoidance).

        Args:
            **resources: Pairs like ``database=2, api=1`` meaning
                "claim 2 database slots and 1 api slot".
        """
        for name, count in sorted(resources.items()):
            for _ in range(count):
                sem = self._pool.create(name)
                await sem.acquire()
                self._resources[name] += 1

    async def release(self, **resources: int) -> None:
        """Release ``count`` units of each named resource.

        Only units this guard actually holds (tracked internally) are
        released, so over-releasing cannot corrupt the semaphore.

        Args:
            **resources: Same shape as :meth:`acquire`.
        """
        for name, count in resources.items():
            for _ in range(count):
                if self._resources[name] > 0:
                    sem = self._pool.get(name)
                    if sem:
                        sem.release()
                        self._resources[name] -= 1

    @asynccontextmanager
    async def use(self, **resources: int) -> Any:
        """Acquire the resources for a block, then auto-release.

        Args:
            **resources: Pairs like ``database=2, api=1``.

        Yields:
            None: The block simply runs while the resources are held.
        """
        await self.acquire(**resources)
        try:
            yield
        finally:
            await self.release(**resources)


async def demonstrate_semaphore_pool() -> None:
    """Run a self-contained demo of the pool, multi-semaphore, and guard.

    Run ``python -m asyncio_mastery.patterns.semaphore_pool`` to see it.
    """
    print("=== Semaphore Pool Demo ===")

    pool = SemaphorePool()

    # Create semaphores for different resources
    pool.create("database", 5)
    pool.create("api", 10)
    pool.create("cache", 20)

    # Use individual semaphores
    print("\n--- Individual Semaphores ---")

    async def use_db() -> str:
        async with pool.acquire("database"):
            await asyncio.sleep(0.01)
            return "db done"

    results = await asyncio.gather(*[use_db() for _ in range(8)])
    print(f"DB results: {len(results)} completed")

    # Multi-semaphore
    print("\n--- Multi-Semaphore ---")

    async def use_multiple() -> str:
        async with pool.acquire_many("database", "api"):
            await asyncio.sleep(0.01)
            return "multi done"

    results = await asyncio.gather(*[use_multiple() for _ in range(3)])
    print(f"Multi results: {results}")

    # Resource guard
    print("\n--- Resource Guard ---")
    guard = ResourceGuard(pool)

    async def complex_operation() -> str:
        async with guard.use(database=2, api=1):
            await asyncio.sleep(0.01)
            return "complex done"

    results = await asyncio.gather(*[complex_operation() for _ in range(3)])
    print(f"Complex results: {results}")

    print(f"\nPool stats: {pool.stats()}")


if __name__ == "__main__":
    asyncio.run(demonstrate_semaphore_pool())
