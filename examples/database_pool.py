"""
Database Connection Pool

Production-ready async database pool with health checks and monitoring.
"""

import asyncio
import time
import uuid
from collections import defaultdict
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any

try:
    import asyncpg

    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    asyncpg = None

try:
    import aiomysql

    AIOMYSQL_AVAILABLE = True
except ImportError:
    AIOMYSQL_AVAILABLE = False
    aiomysql = None

try:
    import aiosqlite

    AIOSQLITE_AVAILABLE = True
except ImportError:
    AIOSQLITE_AVAILABLE = False
    aiosqlite = None


@dataclass
class DatabaseConfig:
    """Database connection configuration."""

    host: str = "localhost"
    port: int = 5432
    database: str = ""
    user: str = ""
    password: str = ""
    ssl: bool = False
    ssl_ca: str | None = None
    ssl_cert: str | None = None
    ssl_key: str | None = None

    # Connection options
    command_timeout: float = 60.0
    statement_cache_size: int = 100
    max_cacheable_statement_size: int = 1024 * 1024

    def to_asyncpg_kwargs(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "password": self.password,
            "ssl": self.ssl,
            "command_timeout": self.command_timeout,
        }

    def to_aiomysql_kwargs(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "db": self.database,
            "user": self.user,
            "password": self.password,
            "ssl": self.ssl,
        }

    def to_aiosqlite_kwargs(self) -> dict[str, Any]:
        return {"database": self.database or ":memory:"}


@dataclass
class PoolConfig:
    """Connection pool configuration."""

    min_size: int = 2
    max_size: int = 10
    acquire_timeout: float = 30.0
    idle_timeout: float = 300.0
    max_queries: int = 50000
    max_inactive_connection_lifetime: float = 300.0
    setup: Callable[["Connection"], Awaitable[None]] | None = None
    init: Callable[["Connection"], Awaitable[None]] | None = None


@dataclass
class QueryStats:
    """Query execution statistics."""

    query: str
    count: int = 0
    total_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0
    errors: int = 0
    last_executed: float | None = None

    @property
    def avg_time(self) -> float:
        return self.total_time / self.count if self.count > 0 else 0


class Connection:
    """Wrapper for database connection with monitoring."""

    def __init__(
        self, raw_connection: Any, pool: "DatabasePool", *, connection_id: str | None = None
    ) -> None:
        self._conn = raw_connection
        self._pool = pool
        self._connection_id = connection_id or str(uuid.uuid4())[:8]
        self._created_at = time.time()
        self._last_used = time.time()
        self._query_count = 0
        self._in_transaction = False
        self._closed = False

    @property
    def connection_id(self) -> str:
        return self._connection_id

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def in_transaction(self) -> bool:
        return self._in_transaction

    @property
    def age(self) -> float:
        return time.time() - self._created_at

    @property
    def idle_time(self) -> float:
        return time.time() - self._last_used

    def _mark_used(self) -> None:
        self._last_used = time.time()
        self._query_count += 1

    async def execute(self, query: str, *args, **kwargs) -> str:
        """Execute a query."""
        self._mark_used()
        start = time.time()

        try:
            if hasattr(self._conn, "execute"):
                result = await self._conn.execute(query, *args, **kwargs)
            else:
                async with self._conn.cursor() as cur:
                    await cur.execute(query, args)
                    result = cur.rowcount

            await self._pool._record_query(query, time.time() - start, success=True)
            return result
        except Exception as _e:
            await self._pool._record_query(query, time.time() - start, success=False)
            raise

    async def fetch(self, query: str, *args, **kwargs) -> list[Any]:
        """Fetch multiple rows."""
        self._mark_used()
        start = time.time()

        try:
            if hasattr(self._conn, "fetch"):
                result = await self._conn.fetch(query, *args, **kwargs)
            else:
                async with self._conn.cursor() as cur:
                    await cur.execute(query, args)
                    result = await cur.fetchall()

            await self._pool._record_query(query, time.time() - start, success=True)
            return result
        except Exception as _e:
            await self._pool._record_query(query, time.time() - start, success=False)
            raise

    async def fetchrow(self, query: str, *args, **kwargs) -> Any | None:
        """Fetch a single row."""
        self._mark_used()
        start = time.time()

        try:
            if hasattr(self._conn, "fetchrow"):
                result = await self._conn.fetchrow(query, *args, **kwargs)
            else:
                async with self._conn.cursor() as cur:
                    await cur.execute(query, args)
                    result = await cur.fetchone()

            await self._pool._record_query(query, time.time() - start, success=True)
            return result
        except Exception as _e:
            await self._pool._record_query(query, time.time() - start, success=False)
            raise

    async def fetchval(self, query: str, *args, **kwargs) -> Any:
        """Fetch a single value."""
        self._mark_used()
        start = time.time()

        try:
            if hasattr(self._conn, "fetchval"):
                result = await self._conn.fetchval(query, *args, **kwargs)
            else:
                async with self._conn.cursor() as cur:
                    await cur.execute(query, args)
                    row = await cur.fetchone()
                    result = row[0] if row else None

            await self._pool._record_query(query, time.time() - start, success=True)
            return result
        except Exception as _e:
            await self._pool._record_query(query, time.time() - start, success=False)
            raise

    async def close(self) -> None:
        """Close the connection."""
        if not self._closed:
            self._closed = True
            await self._conn.close()


class Transaction:
    """Database transaction wrapper."""

    def __init__(self, connection: Connection) -> None:
        self._connection = connection
        self._transaction = None
        self._completed = False

    async def __aenter__(self) -> "Transaction":
        if hasattr(self._connection._conn, "transaction"):
            self._transaction = self._connection._conn.transaction()
            await self._transaction.start()
        else:
            await self._connection.execute("BEGIN")

        self._connection._in_transaction = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._completed:
            return

        try:
            if exc_type:
                await self.rollback()
            else:
                await self.commit()
        except Exception:
            await self.rollback()
            raise

    async def commit(self) -> None:
        if self._transaction:
            await self._transaction.commit()
        else:
            await self._connection.execute("COMMIT")

        self._completed = True
        self._connection._in_transaction = False

    async def rollback(self) -> None:
        if self._transaction:
            await self._transaction.rollback()
        else:
            await self._connection.execute("ROLLBACK")

        self._completed = True
        self._connection._in_transaction = False


class DatabasePool:
    """Async database connection pool."""

    def __init__(
        self,
        db_config: DatabaseConfig,
        pool_config: PoolConfig | None = None,
        *,
        driver: str = "asyncpg",
    ) -> None:
        self._db_config = db_config
        self._pool_config = pool_config or PoolConfig()
        self._driver = driver
        self._pool: Any = None
        self._running = False

        # Stats
        self._stats = {
            "connections_created": 0,
            "connections_closed": 0,
            "queries_executed": 0,
            "query_errors": 0,
            "total_query_time": 0.0,
            "acquire_count": 0,
            "acquire_timeouts": 0,
        }
        self._query_stats: dict[str, QueryStats] = defaultdict(QueryStats)
        self._lock = asyncio.Lock()

        # Health check
        self._health_check_query = "SELECT 1"
        self._health_check_interval = 30.0
        self._health_check_task: asyncio.Task | None = None

    @property
    def running(self) -> bool:
        return self._running

    @property
    def driver(self) -> str:
        return self._driver

    async def start(self) -> None:
        """Start the connection pool."""
        if self._running:
            return

        if self._driver == "asyncpg":
            if not ASYNCPG_AVAILABLE:
                raise RuntimeError("asyncpg is required. Install with: pip install asyncpg")
            await self._start_asyncpg()
        elif self._driver == "aiomysql":
            if not AIOMYSQL_AVAILABLE:
                raise RuntimeError("aiomysql is required. Install with: pip install aiomysql")
            await self._start_aiomysql()
        elif self._driver == "aiosqlite":
            if not AIOSQLITE_AVAILABLE:
                raise RuntimeError("aiosqlite is required. Install with: pip install aiosqlite")
            await self._start_aiosqlite()
        else:
            raise ValueError(f"Unknown driver: {self._driver}")

        self._running = True

        # Start health check
        self._health_check_task = asyncio.create_task(self._health_check_loop())

    async def _start_asyncpg(self) -> None:
        kwargs = self._db_config.to_asyncpg_kwargs()
        kwargs.update(
            {
                "min_size": self._pool_config.min_size,
                "max_size": self._pool_config.max_size,
                "timeout": self._pool_config.acquire_timeout,
                "command_timeout": self._pool_config.command_timeout,
            }
        )

        if self._pool_config.setup:
            kwargs["setup"] = self._pool_config.setup
        if self._pool_config.init:
            kwargs["init"] = self._pool_config.init

        self._pool = await asyncpg.create_pool(**kwargs)

    async def _start_aiomysql(self) -> None:
        kwargs = self._db_config.to_aiomysql_kwargs()
        kwargs.update(
            {
                "minsize": self._pool_config.min_size,
                "maxsize": self._pool_config.max_size,
                "connect_timeout": self._pool_config.acquire_timeout,
            }
        )

        self._pool = await aiomysql.create_pool(**kwargs)

    async def _start_aiosqlite(self) -> None:
        kwargs = self._db_config.to_aiosqlite_kwargs()

        # SQLite doesn't have native pooling, create simple wrapper
        class SQLitePool:
            def __init__(self, config) -> None:
                self._config = config
                self._conn = None

            async def acquire(self):
                if self._conn is None:
                    self._conn = await aiosqlite.connect(**self._config)
                    self._conn.row_factory = aiosqlite.Row
                return self._conn

            async def release(self, conn) -> None:
                pass  # SQLite uses single connection

            def close(self):
                if self._conn:
                    return self._conn.close()

            async def __aenter__(self):
                return await self.acquire()

            async def __aexit__(self, *args):
                pass

        self._pool = SQLitePool(kwargs)

    async def stop(self) -> None:
        """Stop the connection pool."""
        self._running = False

        if self._health_check_task:
            self._health_check_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._health_check_task

        if self._pool:
            if hasattr(self._pool, "close"):
                await self._pool.close()
            elif hasattr(self._pool, "close"):
                self._pool.close()

            self._pool = None

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[Connection, None]:
        """Acquire a connection from the pool."""
        if not self._running:
            raise RuntimeError("Pool not started")

        _start = time.time()
        raw_conn = None

        try:
            async with asyncio.timeout(self._pool_config.acquire_timeout):
                raw_conn = await self._pool.acquire()

            conn = Connection(raw_conn, self)

            async with self._lock:
                self._stats["acquire_count"] += 1

            yield conn

        except TimeoutError:
            async with self._lock:
                self._stats["acquire_timeouts"] += 1
            raise TimeoutError("Connection acquire timeout")
        finally:
            if raw_conn:
                await self._pool.release(raw_conn)

    async def execute(self, query: str, *args, **kwargs) -> str:
        """Execute a query directly."""
        async with self.acquire() as conn:
            return await conn.execute(query, *args, **kwargs)

    async def fetch(self, query: str, *args, **kwargs) -> list[Any]:
        """Fetch multiple rows directly."""
        async with self.acquire() as conn:
            return await conn.fetch(query, *args, **kwargs)

    async def fetchrow(self, query: str, *args, **kwargs) -> Any | None:
        """Fetch a single row directly."""
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args, **kwargs)

    async def fetchval(self, query: str, *args, **kwargs) -> Any:
        """Fetch a single value directly."""
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args, **kwargs)

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[Transaction, None]:
        """Create a transaction."""
        async with self.acquire() as conn:
            tx = Transaction(conn)
            async with tx:
                yield tx

    async def _record_query(self, query: str, elapsed: float, success: bool) -> None:
        """Record query statistics."""
        stats = self._query_stats[query]
        stats.query = query
        stats.count += 1
        stats.total_time += elapsed
        stats.min_time = min(stats.min_time, elapsed)
        stats.max_time = max(stats.max_time, elapsed)
        stats.last_executed = time.time()

        if not success:
            stats.errors += 1
            async with self._lock:
                self._stats["query_errors"] += 1
        else:
            async with self._lock:
                self._stats["queries_executed"] += 1
                self._stats["total_query_time"] += elapsed

    async def _health_check_loop(self) -> None:
        """Periodic health check."""
        while self._running:
            await asyncio.sleep(self._health_check_interval)

            if not self._running:
                break

            try:
                async with self.acquire() as conn:
                    await conn.fetchval(self._health_check_query)
            except Exception:
                pass  # Health check failed

    def get_stats(self) -> dict[str, Any]:
        return {
            **self._stats.copy(),
            "driver": self._driver,
            "running": self._running,
            "pool_size": self._pool.get_size() if hasattr(self._pool, "get_size") else "N/A",
            "pool_free": self._pool.get_idle_size()
            if hasattr(self._pool, "get_idle_size")
            else "N/A",
        }

    def get_query_stats(self) -> list[QueryStats]:
        return list(self._query_stats.values())

    def get_slow_queries(self, threshold: float = 1.0) -> list[QueryStats]:
        return [s for s in self._query_stats.values() if s.avg_time > threshold]

    async def __aenter__(self) -> "DatabasePool":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()


def create_db_pool(
    host: str = "localhost",
    port: int = 5432,
    database: str = "",
    user: str = "",
    password: str = "",
    driver: str = "asyncpg",
    min_size: int = 2,
    max_size: int = 10,
    **kwargs,
) -> DatabasePool:
    """Factory for creating database pool."""
    db_config = DatabaseConfig(
        host=host, port=port, database=database, user=user, password=password, **kwargs
    )
    pool_config = PoolConfig(min_size=min_size, max_size=max_size)
    return DatabasePool(db_config, pool_config, driver=driver)


# Example usage
async def example_usage() -> None:
    """Example database pool usage."""
    pool = create_db_pool(
        host="localhost",
        port=5432,
        database="myapp",
        user="postgres",
        password="secret",
        min_size=5,
        max_size=20,
    )

    async with pool:
        # Simple query
        _users = await pool.fetch("SELECT * FROM users WHERE active = $1", True)

        # Transaction
        async with pool.transaction() as tx:
            await tx._connection.execute(
                "INSERT INTO users (name, email) VALUES ($1, $2)", "John Doe", "john@example.com"
            )

        # Connection-level operations
        async with pool.acquire() as conn:
            _user = await conn.fetchrow("SELECT * FROM users WHERE id = $1", 1)

    # Print stats
    print(f"Total queries: {pool.get_stats()['queries_executed']}")
    slow = pool.get_slow_queries(0.1)
    for q in slow:
        print(f"Slow query: {q.query} (avg: {q.avg_time:.3f}s)")
