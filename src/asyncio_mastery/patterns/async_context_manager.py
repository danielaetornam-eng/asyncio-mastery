"""
Async Context Managers - Clean resource setup/teardown for asyncio.

THE PROBLEM
-----------
Many resources (network connections, files, temp workers) need an
*async* setup step (``await connect()``) and a matching *async* cleanup
step (``await close()``). Plain ``with`` blocks cannot await, so until
Python 3.5 there was no tidy way to say "use this resource, then
definitely clean it up". The ``async with`` statement fills that gap —
and this module gives you reusable building blocks for writing your own.

REAL-WORLD ANALOGY
------------------
**Borrowing a library book**: checking the book out (async setup) and
returning it (async cleanup) bracket your reading. Even if you abandon
the book halfway (an exception!), the return slot guarantees the book
goes back — the library never leaks books.

JARGON
------
* **Async context manager**: any object with ``__aenter__`` (runs at
  ``async with`` entry, must return the resource) and ``__aexit__``
  (runs at exit, even on exceptions, must clean up).
* **Lifecycle**: the states a resource passes through — created,
  initializing, ready, closing, closed (or error). Tracking the state
  catches misuse like re-entering an already-closed manager.

what is IN HERE
--------------
* :class:`AsyncContextManager` — an abstract base class that handles
  state tracking so you only write ``_initialize`` and ``_close``.
* :class:`AsyncResourceManager` — opens/closes *several* resources as
  a unit (like an app's startup/shutdown).
* :func:`async_context_manager` — turn two plain functions into a
  context manager without writing a class.
* :class:`AsyncLazyResource` — initialize an expensive resource only
  on first use.

Minimal example
---------------
>>> import asyncio
>>> class Connection(AsyncContextManager[str]):
...     async def _initialize(self) -> str:
...         return "connected!"
...     async def _close(self) -> None:
...         print("closed")
>>> async def main() -> None:
...     async with Connection() as conn:
...         print(conn)
>>> asyncio.run(main())
"""

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, TypeVar, cast

T = TypeVar("T")


class LifecycleState(Enum):
    """The states a managed resource moves through.

    CREATED -> INITIALIZING -> READY -> CLOSING -> CLOSED, with ERROR
    as a possible detour from INITIALIZING or CLOSING.
    """

    CREATED = "created"
    INITIALIZING = "initializing"
    READY = "ready"
    CLOSING = "closing"
    CLOSED = "closed"
    ERROR = "error"


@dataclass
class ResourceInfo:
    """Diagnostic snapshot of a managed resource.

    Attributes:
        name: Human-readable label.
        state: Current :class:`LifecycleState`.
        created_at: Monotonic timestamp of creation.
        initialized_at: Monotonic timestamp when setup finished
            (None until then).
        closed_at: Monotonic timestamp when cleanup finished.
        error: The exception that moved the resource to ERROR, if any.
    """

    name: str
    state: LifecycleState = LifecycleState.CREATED
    created_at: float = field(default_factory=time.monotonic)
    initialized_at: float | None = None
    closed_at: float | None = None
    error: Exception | None = None


class AsyncContextManager(ABC, Generic[T]):
    """Base class doing the boring lifecycle bookkeeping for you.

    Subclass it and implement only two coroutines:

    * ``_initialize`` — acquire/create the resource, return it.
    * ``_close`` — release it (must be safe to call once).

    The base class handles state transitions, timestamps, error capture,
    and misuse detection (entering twice raises ``RuntimeError``).

    Type parameter ``T`` is the resource type handed to the ``async
    with`` body.
    """

    def __init__(self, name: str | None = None) -> None:
        """Create the manager.

        Args:
            name: Label for diagnostics; defaults to the class name.
        """
        self._name = name or self.__class__.__name__
        self._info = ResourceInfo(self._name)
        self._resource: T | None = None
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        """Human-readable label of this manager (used in diagnostics).

        Returns:
            The name passed to ``__init__``, or the class name.
        """
        return self._name

    @property
    def info(self) -> ResourceInfo:
        """Live diagnostic snapshot: current state, timestamps, error.

        Returns:
            The :class:`ResourceInfo` that this manager keeps updated
            through every lifecycle transition.
        """
        return self._info

    @property
    def resource(self) -> T | None:
        """The underlying resource, once initialized.

        Returns:
            The object created by ``_initialize``, or None before
            entering (or after closing).
        """
        return self._resource

    @property
    def is_ready(self) -> bool:
        """True only while the resource is usable (state READY).

        Returns:
            False before ``_initialize`` finishes and after closing
            starts — handy for guards and assertions.
        """
        return self._info.state == LifecycleState.READY

    @abstractmethod
    async def _initialize(self) -> T:
        """Acquire/create the resource (subclasses must implement).

        Returns:
            T: The resource handed to the ``async with ... as x`` body.
        """
        pass

    @abstractmethod
    async def _close(self) -> None:
        """Release the resource (subclasses must implement).

        Called once during ``__aexit__``, even if the body raised.
        """
        pass

    async def __aenter__(self) -> T:
        """Run ``_initialize`` and hand the resource to the body.

        Returns:
            T: The initialized resource (state becomes READY).

        Raises:
            RuntimeError: If this manager was already entered before.
            Exception: Whatever ``_initialize`` raised (state becomes
            ERROR and the error is recorded in ``info``).
        """
        if self._info.state != LifecycleState.CREATED:
            raise RuntimeError(f"Context manager already used (state: {self._info.state})")

        self._info.state = LifecycleState.INITIALIZING
        try:
            self._resource = await self._initialize()
            self._info.state = LifecycleState.READY
            self._info.initialized_at = time.monotonic()
            return self._resource
        except Exception as e:
            self._info.state = LifecycleState.ERROR
            self._info.error = e
            raise

    async def __aexit__(
        self, exc_type: type | None, exc_val: Exception | None, exc_tb: object
    ) -> bool:
        """Run ``_close`` on block exit, whatever happened in the body.

        Args:
            exc_type: Exception class if the body raised, else None.
            exc_val: The exception instance, if any.
            exc_tb: The traceback object, if any.

        Returns:
            bool: Always False — we never *suppress* exceptions from
            the body; they keep propagating after cleanup.
        """
        if self._info.state not in (LifecycleState.READY, LifecycleState.ERROR):
            return False

        self._info.state = LifecycleState.CLOSING
        try:
            await self._close()
        except Exception as e:
            self._info.error = e
            self._info.state = LifecycleState.ERROR
            if exc_val is None:
                raise
        else:
            self._info.state = LifecycleState.CLOSED
            self._info.closed_at = time.monotonic()
        return False


class AsyncResourceManager(Generic[T]):
    """Open and close SEVERAL async resources as one unit.

    Think of it as your application's power switch: on startup you flip
    it ON and the database, cache, and message queue all connect; on
    shutdown you flip it OFF and everything disconnects cleanly — in
    REVERSE order, like unplugging the monitor before the PC.

    The key convenience is :meth:`lifespan`, an ``async with`` block
    that brackets "everything is running".

    Example:
        >>> manager = AsyncResourceManager()
        >>> manager.add("db", DatabaseConnection("db"))
        >>> manager.add("cache", CacheConnection("cache"))
        >>> async with manager.lifespan() as resources:
        ...     print(resources["db"])  # all initialized
        >>> # all closed here
    """

    def __init__(self) -> None:
        """Create an empty manager; register resources with :meth:`add`."""
        self._resources: dict[str, AsyncContextManager[T]] = {}
        self._lock = asyncio.Lock()

    def add(self, name: str, manager: AsyncContextManager[T]) -> None:
        """Register a resource manager under a lookup name.

        Args:
            name: Unique label, e.g. ``"database"``. Used by :meth:`get`
                and as the key in the dict returned by
                :meth:`initialize_all`.
            manager: The :class:`AsyncContextManager` that owns setup and
                teardown of this resource.

        Raises:
            ValueError: If ``name`` is already registered.
        """
        if name in self._resources:
            raise ValueError(f"Resource '{name}' already exists")
        self._resources[name] = manager

    async def initialize_all(self) -> dict[str, T]:
        """Enter every registered resource manager, in registration order.

        Returns:
            A dict mapping each name to its initialized resource (the
            same value its manager's ``__aenter__`` returned).
        """
        results: dict[str, T] = {}
        for name, manager in self._resources.items():
            results[name] = await manager.__aenter__()
        return results

    async def close_all(self) -> None:
        """Exit every manager in REVERSE registration order.

        Reverse order matters because resources often depend on each
        other (tear down the cache client before its connection pool).

        Individual close failures are printed and swallowed so one bad
        resource can't prevent the others from closing.
        """
        for name in reversed(list(self._resources.keys())):
            manager = self._resources[name]
            try:
                await manager.__aexit__(None, None, None)
            except Exception as e:
                print(f"Error closing {name}: {e}")

    def get(self, name: str) -> AsyncContextManager[T] | None:
        """Look up a registered manager by name.

        Args:
            name: The label passed to :meth:`add`.

        Returns:
            The manager, or None if no resource has that name.
        """
        return self._resources.get(name)

    @asynccontextmanager
    async def lifespan(self) -> AsyncIterator[dict[str, T]]:
        """One ``async with`` that opens everything and closes everything.

        ``@asynccontextmanager`` (from ``contextlib``) turns a generator
        into a context manager: code before ``yield`` runs at entry,
        ``finally`` after it runs at exit — guaranteed.

        Yields:
            dict[str, T]: Mapping of resource names to initialized
            resources, valid inside the ``async with`` body only.

        Example:
            >>> async with manager.lifespan() as resources:
            ...     db = resources["database"]
        """
        resources = await self.initialize_all()
        try:
            yield resources
        finally:
            # finally runs even if the body raised — cleanup is certain.
            await self.close_all()


@asynccontextmanager
async def async_context_manager(
    init_func: Callable[[], Awaitable[T]], close_func: Callable[[T], Awaitable[None]]
) -> AsyncIterator[T]:
    """Build a context manager from two plain async functions.

    The lightweight alternative to subclassing :class:`AsyncContextManager`:
    supply a setup coroutine and a teardown coroutine and get a ready-made
    ``async with`` target. ``close_func`` always runs, even if the body
    raises, because the ``yield`` sits inside ``try/finally``.

    Args:
        init_func: Coroutine function that acquires the resource and
            returns it. Called on entry.
        close_func: Coroutine function that releases the resource.
            Receives the resource produced by ``init_func``. Called on
            exit, guaranteed.

    Yields:
        T: The resource, for use in the ``async with`` body.

    Example:
        >>> async with async_context_manager(connect, disconnect) as conn:
        ...     await conn.query("SELECT 1")
    """
    resource = await init_func()
    try:
        yield resource
    finally:
        await close_func(resource)


class AsyncLazyResource(Generic[T]):
    """Create an expensive resource only when it is first USED.

    Real-world analogy: a hotel kettle is only plugged in when a guest
    wants tea — you don't wire 300 rooms' kettles at check-in. *Lazy
    initialization* defers the setup cost (and the failure risk) until
    someone actually awaits :meth:`get`. A lock ensures that even if
    100 tasks call ``get()`` simultaneously, the factory runs ONCE.

    Example:
        >>> lazy = AsyncLazyResource(lambda: connect_db(), name="db")
        >>> db = await lazy.get()   # connect_db() runs here
        >>> db = await lazy.get()   # cached; no second connect
        >>> await lazy.close()
    """

    def __init__(self, factory: Callable[[], Awaitable[T]], *, name: str | None = None) -> None:
        """Create a lazy wrapper around an async factory.

        Args:
            factory: Coroutine function that creates the resource when
                first needed, e.g. ``lambda: create_connection()``.
            name: Optional label for diagnostics.
        """
        self._factory = factory
        self._name = name or "lazy-resource"
        self._resource: T | None = None
        self._initialized = False
        self._lock = asyncio.Lock()

    async def get(self) -> T:
        """Return the resource, initializing it on first call only.

        The lock is held across both the check and the ``await`` of the
        factory, so concurrent tasks queue up and the factory is
        guaranteed to run exactly once (this check-then-await-inside-lock
        pattern is the classic double-checked initialization).

        Returns:
            The initialized resource; the SAME object on every call
            until :meth:`close` is called.
        """
        async with self._lock:
            if not self._initialized:
                self._resource = await self._factory()
                self._initialized = True
            return cast("T", self._resource)

    async def close(self) -> None:
        """Release the resource if it was ever initialized.

        Safe to call even if :meth:`get` was never called (it's a no-op
        then). The resource is closed via its ``close()`` method if it
        has one, else via ``__aexit__`` if it's a context manager; after
        closing, the wrapper forgets it, so a later :meth:`get` will
        create a fresh one.
        """
        async with self._lock:
            if self._initialized and self._resource:
                # duck-typing: accept anything that looks closeable
                if hasattr(self._resource, "close"):
                    await self._resource.close()
                elif hasattr(self._resource, "__aexit__"):
                    await self._resource.__aexit__(None, None, None)
                self._initialized = False
                self._resource = None


async def demonstrate_async_context_manager() -> None:
    """Runnable tour of every tool in this module.

    Demonstrates, in order: a custom :class:`AsyncContextManager`
    subclass (fake database), an :class:`AsyncResourceManager` grouping
    two resources via ``lifespan()``, a context manager built from bare
    functions with :func:`async_context_manager`, and an
    :class:`AsyncLazyResource` initializing on first access only.

    Run it directly::

        python -m asyncio_mastery.patterns.async_context_manager
    """
    print("=== Async Context Manager Demo ===")

    # Custom context manager
    class DatabaseConnection(AsyncContextManager[str]):
        async def _initialize(self) -> str:
            print("  Connecting to database...")
            await asyncio.sleep(0.05)
            return "db-connection-123"

        async def _close(self) -> None:
            print("  Closing database connection")
            await asyncio.sleep(0.01)

    print("\n--- Custom Context Manager ---")
    async with DatabaseConnection("main-db") as conn:
        print(f"  Using: {conn}")

    print("  Connection closed")

    # Resource manager
    print("\n--- Resource Manager ---")
    manager: AsyncResourceManager[str] = AsyncResourceManager()

    class CacheConnection(AsyncContextManager[str]):
        async def _initialize(self) -> str:
            print("  Connecting to cache...")
            await asyncio.sleep(0.02)
            return "cache-conn-456"

        async def _close(self) -> None:
            print("  Closing cache connection")

    manager.add("database", DatabaseConnection("db"))
    manager.add("cache", CacheConnection("cache"))

    async with manager.lifespan() as resources:
        print(f"  Resources: {list(resources.keys())}")
        print(f"  DB: {resources['database']}")
        print(f"  Cache: {resources['cache']}")

    print("  All closed")

    # Function-based context manager
    print("\n--- Function-based ---")

    async def create_connection() -> str:
        print("  Creating connection...")
        await asyncio.sleep(0.01)
        return "func-conn"

    async def close_connection(conn: str) -> None:
        print(f"  Closing {conn}")

    async with async_context_manager(create_connection, close_connection) as conn:
        print(f"  Using: {conn}")

    # Lazy resource
    print("\n--- Lazy Resource ---")

    async def create_lazy_value() -> str:
        await asyncio.sleep(0.01)
        return "lazy-value"

    lazy = AsyncLazyResource(create_lazy_value, name="lazy")

    # First access initializes
    val1 = await lazy.get()
    print(f"  First get: {val1}")

    # Subsequent accesses return cached
    val2 = await lazy.get()
    print(f"  Second get: {val2}")

    await lazy.close()
    print("  Closed")


if __name__ == "__main__":
    asyncio.run(demonstrate_async_context_manager())
