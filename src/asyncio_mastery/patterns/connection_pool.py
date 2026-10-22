"""
Connection Pool - Generic async connection pool.

Opening a network connection is expensive. The TCP handshake, TLS
negotiation, and authentication can take hundreds of milliseconds.
Opening one connection per operation makes everything slow. Opening
hundreds of connections at once can overwhelm the server. You want a
fixed set of warm connections reused across many operations.

Real-world analogy: A taxi rank at the airport. A few taxis
(connections) wait idling. You (a task) take one, ride (do your query),
and hand it back so the next traveler can use it immediately. The rank
keeps a minimum number waiting and never allows more than a maximum on
the road. An inspector (the health-check loop) periodically makes sure
idling taxis still start.

Why async primitives are needed:

* The pool shelf is an asyncio.Queue. The await queue.get() call
  suspends a task until a connection is returned. No polling is needed.
* An asyncio.Lock makes the acquire and release bookkeeping atomic.
  Without it, two coroutines could both grab the same connection or
  double-count statistics.
* A background health-check task prunes dead or expired connections
  and tops the pool back up to min_size.

When to use it:

* Database drivers, HTTP clients, message-bus connections. Use it for
  anything expensive to create and safe to reuse.
* When you need to cap concurrency to a downstream resource at exactly
  max_size.

Minimal example:

>>> import asyncio
>>> async def main() -> None:
...     pool = ConnectionPool(DemoConnectionFactory(), PoolConfig(min_size=1, max_size=3))
...     await pool.start()
...     async with pool.connection() as conn:
...         print("using", conn)
...     await pool.close()
>>> asyncio.run(main()
"""

import asyncio
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass
class PoolConfig:
    """Knobs controlling pool size and connection lifetime.

    Attributes:
        min_size: Connections created at start and maintained by the
            health-check loop so a surge does not pay the creation cost.
        max_size: Hard cap on total connections (idle + in use). This
            is your concurrency ceiling for the downstream resource.
        acquire_timeout: Seconds a caller waits for a free connection
            before failing with TimeoutError (pool exhausted).
        idle_timeout: Connections older than this (seconds) are treated
            as stale and replaced. Servers often kill long-idle sockets.
        max_uses: If set, a connection is retired after this many
            checkouts. Protects against server-side reuse limits or
            memory leaks in buggy drivers. None means unlimited reuse.
        health_check_interval: Seconds between health-check sweeps.
    """

    min_size: int = 2
    max_size: int = 10
    acquire_timeout: float = 30.0
    idle_timeout: float = 300.0
    max_uses: int | None = None
    health_check_interval: float = 30.0


@dataclass
class PoolStats:
    """Observability for the pool.

    Attributes:
        created: Connections ever created.
        acquired: Total checkouts.
        released: Total check-ins.
        closed: Connections ever closed or retired.
        failed: Bad releases (for example, releasing a connection the
            pool does not think is in use). This indicates a usage bug.
        current_size: Connections currently alive (idle + in use).
        available: Idle connections sitting in the queue.
        in_use: Connections currently checked out.
    """

    created: int = 0
    acquired: int = 0
    released: int = 0
    closed: int = 0
    failed: int = 0
    current_size: int = 0
    available: int = 0
    in_use: int = 0


class PooledConnection(Generic[T]):
    """Wrapper for pooled connection.

    Decorates a raw connection with the metadata the pool needs to make
    lifetime decisions: when it was created, how many times it has been
    used, and whether it is still healthy.

    Type Parameters:
        T: The underlying connection type (driver-specific).

    Attributes:
        _connection: The raw connection object.
        _pool: Back-reference to the owning pool (for close method).
        _created_at: time.monotonic() at creation.
        _uses: How many times this connection has been checked out.
        _healthy: False once a use raised an exception. The connection
            may be mid-error on the wire and must not be reused.
    """

    def __init__(self, connection: T, pool: "ConnectionPool[T]", created_at: float) -> None:
        """Wrap a raw connection.

        Args:
            connection: The driver-specific connection object.
            pool: The pool that owns this wrapper.
            created_at: time.monotonic() timestamp of creation.
        """
        self._connection = connection
        self._pool = pool
        self._created_at = created_at
        self._uses = 0
        self._healthy = True

    @property
    def connection(self) -> T:
        """The underlying raw connection.

        Returns:
            T: The driver-specific connection to actually use.
        """
        return self._connection

    @property
    def uses(self) -> int:
        """Number of checkouts so far.

        Returns:
            int: Times this connection has been acquired.
        """
        return self._uses

    @property
    def age(self) -> float:
        """Seconds since this connection was created.

        Returns:
            float: time.monotonic() minus created_at.
        """
        return time.monotonic() - self._created_at

    @property
    def healthy(self) -> bool:
        """Whether the connection is considered usable.

        Returns:
            bool: False once mark_unhealthy has been called.
        """
        return self._healthy

    def mark_unhealthy(self) -> None:
        """Flag this connection for retirement, not reuse.

        Typically called automatically when a with pool.connection()
        block raises. A connection that errored mid-operation may be in
        a broken state, so the pool will close it instead of recycling.
        """
        self._healthy = False

    async def close(self) -> None:
        """Close through the owning pool (keeps stats consistent)."""
        await self._pool._close_connection(self)


class ConnectionFactory(ABC, Generic[T]):
    """Factory for creating connections.

    The pool never speaks driver-specific code. It delegates create,
    close, and health-check to your implementation of this interface.

    Example:
        >>> class PgFactory(ConnectionFactory[Connection]):
        ...     async def create(self) -> Connection: ...
        ...     async def close(self, c: Connection) -> None: ...
        ...     async def health_check(self, c: Connection) -> bool: ...
    """

    @abstractmethod
    async def create(self) -> T:
        """Create new connection.

        Returns:
            T: A fresh, ready-to-use connection.

        Raises:
            Exception: Whatever the driver raises when connect fails.
        """
        pass

    @abstractmethod
    async def close(self, connection: T) -> None:
        """Close connection.

        Args:
            connection: The connection to shut down cleanly.
        """
        pass

    @abstractmethod
    async def health_check(self, connection: T) -> bool:
        """Check connection health.

        Args:
            connection: An idle connection to test.

        Returns:
            bool: True if usable; False means the pool closes and
                replaces it.
        """
        pass


class ConnectionPool(Generic[T]):
    """Generic async connection pool.

    Idle connections are kept in a bounded asyncio.Queue (capacity
    max_size). Busy ones are kept in the _in_use set. Every
    acquire/release path mutates shared state. Hence a single lock
    serializes that bookkeeping across all coroutines.

    Type Parameters:
        T: The underlying connection type.

    Example:
        >>> pool = ConnectionPool(factory, PoolConfig(min_size=2, max_size=10))
        >>> await pool.start()
        >>> async with pool.connection() as raw_conn:
        ...     await raw_conn.execute("SELECT 1")
        >>> await pool.close()
    """

    def __init__(self, factory: ConnectionFactory[T], config: PoolConfig | None = None) -> None:
        """Create the pool (connections are not opened until start).

        Args:
            factory: Driver-specific create/close/health-check hooks.
            config: Sizing and lifetime knobs; sensible defaults if None.
        """
        self._factory = factory
        self._config = config or PoolConfig()
        self._pool: asyncio.Queue[PooledConnection[T]] = asyncio.Queue(
            maxsize=self._config.max_size
        )
        self._in_use: set[PooledConnection[T]] = set()
        self._lock = asyncio.Lock()
        self._stats = PoolStats()
        self._health_check_task: asyncio.Task | None = None
        self._closed = False

    async def start(self) -> None:
        """Initialize pool with minimum connections.

        Pre-warms min_size connections and launches the periodic
        health-check task.
        """
        for _ in range(self._config.min_size):
            await self._create_connection()
        self._health_check_task = asyncio.create_task(self._health_check_loop())

    async def close(self) -> None:
        """Close all connections and stop pool.

        Cancels the health checker, closes every idle connection, then
        gathers and closes anything still checked out (returning it
        would be pointless mid-shutdown).
        """
        self._closed = True
        if self._health_check_task:
            self._health_check_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._health_check_task

        # Close available connections
        while not self._pool.empty():
            conn = self._pool.get_nowait()
            await self._close_connection(conn)

        # Wait for in-use connections
        if self._in_use:
            await asyncio.gather(*[conn.close() for conn in self._in_use], return_exceptions=True)

    async def _create_connection(self) -> PooledConnection[T]:
        """Create and add connection to pool.

        Returns:
            PooledConnection: The freshly wrapped connection (already
                enqueued as available).
        """
        conn = await self._factory.create()
        pooled = PooledConnection(conn, self, time.monotonic())
        await self._pool.put(pooled)
        self._stats.created += 1
        self._stats.current_size += 1
        self._stats.available += 1
        return pooled

    async def _close_connection(self, pooled: PooledConnection[T]) -> None:
        """Close a pooled connection.

        Adjusts the accounting (current_size / in_use / available)
        regardless of whether the driver close succeeds. The finally
        block guarantees stats never drift even on driver errors.

        Args:
            pooled: The wrapper to close.
        """
        try:
            await self._factory.close(pooled.connection)
        finally:
            self._stats.closed += 1
            self._stats.current_size -= 1
            if pooled in self._in_use:
                self._in_use.remove(pooled)
                self._stats.in_use -= 1
            else:
                self._stats.available -= 1

    async def acquire(self) -> PooledConnection[T]:
        """Acquire a connection from pool.

        Order of preference:
        (1) Reuse an idle-but-valid connection.
        (2) Create a new one if under max_size.
        (3) Wait up to acquire_timeout for somebody to release theirs.

        A checkout also validates on the way out. Unhealthy, over-used
        (max_uses), or too-old (idle_timeout) connections are discarded
        and replaced.

        Returns:
            PooledConnection: A checked-out wrapper; return it with
                release.

        Raises:
            RuntimeError: If the pool is closed.
            TimeoutError: If no connection freed up within
                acquire_timeout seconds (pool exhausted).
        """
        if self._closed:
            raise RuntimeError("Pool is closed")

        async with self._lock:
            # Try to get from pool
            try:
                pooled = self._pool.get_nowait()
                if (
                    not pooled.healthy
                    or (self._config.max_uses and pooled.uses >= self._config.max_uses)
                    or pooled.age > self._config.idle_timeout
                ):
                    # Connection stale, create new
                    await self._close_connection(pooled)
                    raise asyncio.QueueEmpty
            except asyncio.QueueEmpty:
                # Create new if under max
                if self._stats.current_size < self._config.max_size:
                    pooled = await self._create_connection()
                else:
                    # Wait for available
                    try:
                        pooled = await asyncio.wait_for(
                            self._pool.get(), timeout=self._config.acquire_timeout
                        )
                    except TimeoutError:
                        raise TimeoutError("Connection pool exhausted")

            pooled._uses += 1
            self._in_use.add(pooled)
            self._stats.acquired += 1
            self._stats.available -= 1
            self._stats.in_use += 1
            return pooled

    async def release(self, pooled: PooledConnection[T]) -> None:
        """Return connection to pool.

        Healthy connections go back on the idle shelf. Unhealthy ones
        (or a closed pool) trigger closure. Releasing something the pool
        does not have checked out is a usage bug. We bump failed so the
        mistake is visible in stats.

        Args:
            pooled: The wrapper previously returned by acquire.
        """
        async with self._lock:
            if pooled in self._in_use:
                self._in_use.remove(pooled)
                self._stats.in_use -= 1
                self._stats.released += 1

                if pooled.healthy and not self._closed:
                    await self._pool.put(pooled)
                    self._stats.available += 1
                else:
                    await self._close_connection(pooled)
            else:
                self._stats.failed += 1

    @asynccontextmanager
    async def connection(self) -> Any:
        """Context manager for connection.

        The recommended way to use the pool: acquire, hand you the raw
        connection, and guarantee release. If your block raises, the
        connection is marked unhealthy (it may be protocol-corrupted)
        and the pool closes rather than recycles it.

        Yields:
            T: The raw connection.

        Example:
            >>> async with pool.connection() as conn:
            ...     rows = await conn.fetch("SELECT ...")
        """
        pooled = await self.acquire()
        try:
            yield pooled.connection
        except Exception:
            # An exception mid-operation may leave the wire protocol in
            # an unknown state. Mark unhealthy so release() closes it
            # instead of poisoning future borrowers with a broken conn.
            pooled.mark_unhealthy()
            raise
        finally:
            await self.release(pooled)

    async def _health_check_loop(self) -> None:
        """Periodic health check.

        Every health_check_interval seconds: drain the idle queue, test
        each connection via the factory, retire stale or broken ones,
        and top the pool back up to min_size. This keeps the taxi rank
        full of working taxis instead of dead ones.
        """
        while not self._closed:
            await asyncio.sleep(self._config.health_check_interval)
            if self._closed:
                break

            # Check available connections
            stale = []
            while not self._pool.empty():
                pooled = self._pool.get_nowait()
                if (
                    not await self._factory.health_check(pooled.connection)
                    or pooled.age > self._config.idle_timeout
                ):
                    stale.append(pooled)
                else:
                    await self._pool.put(pooled)

            for pooled in stale:
                await self._close_connection(pooled)

            # make sure minimum size
            while self._stats.current_size < self._config.min_size and not self._closed:
                await self._create_connection()

    @property
    def stats(self) -> PoolStats:
        """The live stats object (all counters).

        Returns:
            PoolStats: Shared mutable stats. Inspect after bursts of
                activity to understand pool behavior.
        """
        return self._stats


# Example factory for demo
class DemoConnectionFactory(ConnectionFactory[str]):
    """Demo connection factory.

    Pretends a plain string is a connection. This is enough to exercise
    the pool machinery without a real database in a teaching context.
    """

    def __init__(self) -> None:
        """Start the fake-connection counter at zero."""
        self._counter = 0

    async def create(self) -> str:
        """Fake-create: returns a numbered string like "conn-3".

        Returns:
            str: A pretend connection handle.
        """
        self._counter += 1
        await asyncio.sleep(0.01)
        return f"conn-{self._counter}"

    async def close(self, connection: str) -> None:
        """Fake-close: tiny sleep to simulate teardown cost.

        Args:
            connection: The fake connection string.
        """
        await asyncio.sleep(0.001)

    async def health_check(self, connection: str) -> bool:
        """Fake health check: everything is always healthy here.

        Args:
            connection: The fake connection string.

        Returns:
            bool: Always True in this demo.
        """
        await asyncio.sleep(0.001)
        return True


async def demonstrate_connection_pool() -> None:
    """Run a self-contained demo of the connection pool.

    Watch the stats: 10 workers never use more than max_size=5
    connections concurrently. That is pooling in action.
    """
    print("=== Connection Pool Demo ===")

    factory = DemoConnectionFactory()
    config = PoolConfig(min_size=2, max_size=5, acquire_timeout=1.0)
    pool = ConnectionPool(factory, config)

    await pool.start()
    print(f"Initial stats: {pool.stats}")

    # Acquire connections
    async def use_connection(id: int) -> str:
        async with pool.connection() as conn:
            await asyncio.sleep(0.05)
            return f"Worker {id} used {conn}"

    results = await asyncio.gather(*[use_connection(i) for i in range(10)])
    for r in results:
        print(f"  {r}")

    print(f"Final stats: {pool.stats}")
    await pool.close()
    print(f"After close: {pool.stats}")


if __name__ == "__main__":
    asyncio.run(demonstrate_connection_pool())
