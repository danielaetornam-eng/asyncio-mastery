"""
Database Integration - Async database access with connection pooling.

WHAT IS THIS, FOR A BEGINNER?
-----------------------------
Databases answer queries over a network (or disk), which is slow -- exactly
the kind of "waiting" asyncio is good at hiding. Normal (blocking) database
drivers freeze your program until each query finishes. Async drivers like
``aiosqlite`` (SQLite) and ``asyncpg`` (PostgreSQL) let other tasks run
while a query is in flight, so a web server can keep serving other users
while one user's query runs.

Two ideas matter most here:

1. *Connection pool.* Opening a database connection is expensive. A pool
   keeps several connections open and lends them out; each coroutine
   "acquires" one, uses it, and releases it back. That way 1000 concurrent
   queries can share, say, 10 connections.
2. *Transaction.* A transaction groups several statements into an
   all-or-nothing unit: either every statement succeeds (COMMIT) or none of
   them takes effect (ROLLBACK). This module's :class:`Transaction` is an
   async context manager that commits on success and rolls back on error --
  no forgotten ``conn.rollback()`` in an ``except`` block.

JARGON
------
- dialect: which database family we are talking to (``sqlite`` vs
  ``postgresql``); chosen automatically from the URL prefix.
- execute / fetch / fetchrow / fetchval: the four classic query shapes --
  run a statement with no rows back; get all rows; get one row; get a
  single value (e.g. ``COUNT(*)``).
- ABC / abstractmethod: a "blueprint" class (:class:`DatabaseConnection`)
  defining methods that concrete connections MUST implement, so the rest of
  the code works identically for SQLite and PostgreSQL.
- graceful degradation: like the aiohttp module, this file imports fine
  without the optional drivers and raises a clear error only if used.

MINIMAL EXAMPLE
---------------
>>> pool = DatabasePool(DatabaseConfig(url="sqlite://app.db"))
>>> await pool.connect()
>>> await pool.execute("CREATE TABLE t (id INTEGER)")
>>> rows = await pool.fetch("SELECT * FROM t")
>>> await pool.close()
"""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

try:
    import aiosqlite

    AIOSQLITE_AVAILABLE = True
except ImportError:
    # Optional dependency; the module still imports without it.
    AIOSQLITE_AVAILABLE = False
    aiosqlite = None

try:
    import asyncpg

    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    asyncpg = None


@dataclass
class DatabaseConfig:
    """Settings for a :class:`DatabasePool`.

    Attributes:
        url: Connection string, e.g. ``"sqlite://data.db"`` or
            ``"postgresql://user:pass@host/db"``. The prefix picks the dialect.
        min_size: Connections opened eagerly when the pool starts.
        max_size: Maximum connections the pool will ever hold.
        timeout: Max seconds a single operation may take.
    """

    url: str
    min_size: int = 2
    max_size: int = 10
    timeout: float = 30.0


class DatabaseConnection(ABC):
    """Abstract base class defining the uniform async database API.

    An *abstract* class cannot be used directly; it exists so that
    :class:`SQLiteConnection` and :class:`PostgresConnection` share one
    documented interface. Code written against this interface works with
    either database without changes.
    """

    @abstractmethod
    async def execute(self, query: str, *args: Any) -> Any:
        """Run a statement that returns no rows (INSERT/UPDATE/CREATE...).

        Args:
            query: SQL string with placeholders for parameters.
            *args: Values to substitute into the placeholders (safe against
                SQL injection -- never build SQL by string concatenation!).

        Returns:
            Driver-specific statement result (e.g. a cursor).
        """
        pass

    @abstractmethod
    async def fetch(self, query: str, *args: Any) -> list[Any]:
        """Run a query and return ALL matching rows as a list."""
        pass

    @abstractmethod
    async def fetchrow(self, query: str, *args: Any) -> Any | None:
        """Run a query and return the FIRST matching row, or None."""
        pass

    @abstractmethod
    async def fetchval(self, query: str, *args: Any) -> Any:
        """Run a query and return a single value (first column of first row)."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the underlying connection and release its resources."""
        pass


class SQLiteConnection(DatabaseConnection):
    """Concrete :class:`DatabaseConnection` wrapping an aiosqlite connection.

    Args:
        db: An open ``aiosqlite.Connection`` (usually borrowed from a pool).
    """

    def __init__(self, db: "aiosqlite.Connection") -> None:
        self._db = db

    async def execute(self, query: str, *args: Any) -> Any:
        # SQLite requires explicit commits for writes; Postgres's asyncpg
        # autocommits, which is why the implementations differ.
        cursor = await self._db.execute(query, args)
        await self._db.commit()
        return cursor

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        cursor = await self._db.execute(query, args)
        return await cursor.fetchall()

    async def fetchrow(self, query: str, *args: Any) -> Any | None:
        cursor = await self._db.execute(query, args)
        return await cursor.fetchone()

    async def fetchval(self, query: str, *args: Any) -> Any:
        cursor = await self._db.execute(query, args)
        row = await cursor.fetchone()
        return row[0] if row else None

    async def close(self) -> None:
        await self._db.close()


class PostgresConnection(DatabaseConnection):
    """Concrete :class:`DatabaseConnection` wrapping an asyncpg connection.

    Args:
        conn: An open ``asyncpg.Connection`` (usually borrowed from a pool).
    """

    def __init__(self, conn: "asyncpg.Connection") -> None:
        self._conn = conn

    async def execute(self, query: str, *args: Any) -> Any:
        return await self._conn.execute(query, *args)

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        return await self._conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args: Any) -> Any | None:
        return await self._conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args: Any) -> Any:
        return await self._conn.fetchval(query, *args)

    async def close(self) -> None:
        await self._conn.close()


class DatabasePool:
    """A pool of database connections shared by many coroutines.

    Create once at app startup with :meth:`connect`, use throughout, and
    :meth:`close` at shutdown. Either borrow a connection explicitly with
    ``async with pool.connection() as conn:`` or use the convenience
    ``execute``/``fetch``/``fetchrow``/``fetchval`` methods, which borrow and
    release a connection per call.

    Args:
        config: A :class:`DatabaseConfig`; its ``url`` prefix selects SQLite
            (``sqlite://``) or PostgreSQL (anything else).

    Raises:
        RuntimeError: If the driver for the chosen dialect is not installed.

    Example:
        >>> pool = DatabasePool(DatabaseConfig(url="sqlite://app.db"))
        >>> await pool.connect()
        >>> n = await pool.fetchval("SELECT COUNT(*) FROM users")
    """

    def __init__(self, config: DatabaseConfig) -> None:
        self._config = config
        self._pool: Any = None
        # Choose the dialect from the URL scheme prefix.
        self._dialect = "sqlite" if config.url.startswith("sqlite") else "postgresql"

    async def connect(self) -> None:
        """Open the underlying driver pool according to the config.

        Raises:
            RuntimeError: If aiosqlite / asyncpg is not installed.
        """
        if self._dialect == "sqlite":
            if not AIOSQLITE_AVAILABLE:
                raise RuntimeError("aiosqlite not installed")
            self._pool = await aiosqlite.create_pool(
                self._config.url.replace("sqlite://", ""),
                minsize=self._config.min_size,
                maxsize=self._config.max_size,
            )
        elif self._dialect == "postgresql":
            if not ASYNCPG_AVAILABLE:
                raise RuntimeError("asyncpg not installed")
            self._pool = await asyncpg.create_pool(
                self._config.url, min_size=self._config.min_size, max_size=self._config.max_size
            )

    async def close(self) -> None:
        """Close every connection in the pool."""
        if self._pool:
            await self._pool.close()

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[DatabaseConnection]:
        """Borrow one connection from the pool, releasing it on exit.

        Yields:
            A :class:`DatabaseConnection` (SQLite or Postgres flavor).

        Example:
            >>> async with pool.connection() as conn:
            ...     await conn.execute("INSERT INTO t VALUES (?)", 1)
        """
        if self._dialect == "sqlite":
            async with self._pool.acquire() as db:
                yield SQLiteConnection(db)
        elif self._dialect == "postgresql":
            async with self._pool.acquire() as conn:
                yield PostgresConnection(conn)

    async def execute(self, query: str, *args: Any) -> Any:
        """Run a no-rows statement using a pooled connection."""
        async with self.connection() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        """Return all rows for a query using a pooled connection."""
        async with self.connection() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args: Any) -> Any | None:
        """Return the first row for a query using a pooled connection."""
        async with self.connection() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args: Any) -> Any:
        """Return a single value for a query using a pooled connection."""
        async with self.connection() as conn:
            return await conn.fetchval(query, *args)


class Transaction:
    """An async context manager giving you all-or-nothing SQL transactions.

    On entering the ``async with`` block a ``BEGIN`` is issued; when the
    block exits, statements are COMMITted if everything went well, or
    ROLLBACKed (undone) if an exception escaped. You can also commit or
    roll back early by hand.

    Args:
        conn: The :class:`DatabaseConnection` the transaction runs on. Note
            that ALL statements in the transaction must use this same
            connection.

    Example:
        >>> async with pool.connection() as conn, Transaction(conn):
        ...     await conn.execute("UPDATE accounts SET ...")
        ...     await conn.execute("UPDATE accounts SET ...")
        ...     # both applied together, or neither if an exception occurs
    """

    def __init__(self, conn: DatabaseConnection) -> None:
        self._conn = conn
        self._active = False

    async def __aenter__(self) -> "Transaction":
        await self._conn.execute("BEGIN")
        self._active = True
        return self

    async def __aexit__(
        self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any
    ) -> bool:
        # exc_type is set when the block raised: undo everything then.
        if self._active:
            if exc_type:
                await self._conn.execute("ROLLBACK")
            else:
                await self._conn.execute("COMMIT")
            self._active = False
        # False = never swallow the exception; let it propagate to the caller.
        return False

    async def commit(self) -> None:
        """Commit early (make changes permanent) and end the transaction."""
        if self._active:
            await self._conn.execute("COMMIT")
            self._active = False

    async def rollback(self) -> None:
        """Roll back early (discard changes) and end the transaction."""
        if self._active:
            await self._conn.execute("ROLLBACK")
            self._active = False


async def demonstrate_database() -> None:
    """Narrated demo: create a SQLite DB, insert, query, and use a transaction.

    Uses a temporary file database so nothing persists on your machine.
    Skips gracefully if aiosqlite is not installed.
    """
    if not AIOSQLITE_AVAILABLE:
        print("aiosqlite not installed, skipping demo")
        return

    print("=== Database Integration Demo ===")

    import os
    import tempfile

    # Create temp database file; deleted in `finally` below.
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    config = DatabaseConfig(url=f"sqlite://{db_path}")
    pool = DatabasePool(config)

    try:
        await pool.connect()

        # Create table
        await pool.execute("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE
            )
        """)

        # Insert data. Note the `?` placeholders: values are passed
        # separately, which is the safe way to put data into SQL.
        print("\n--- Insert ---")
        for i in range(5):
            await pool.execute(
                "INSERT INTO users (name, email) VALUES (?, ?)", f"User {i}", f"user{i}@example.com"
            )

        # Query data
        print("\n--- Query ---")
        users = await pool.fetch("SELECT * FROM users")
        for user in users:
            print(f"  {user}")

        # Transaction: two inserts succeed or fail together.
        print("\n--- Transaction ---")
        async with pool.connection() as conn, Transaction(conn):
            await conn.execute(
                "INSERT INTO users (name, email) VALUES (?, ?)", "Tx User", "tx@example.com"
            )
            await conn.execute(
                "INSERT INTO users (name, email) VALUES (?, ?)", "Tx User 2", "tx2@example.com"
            )

        # Verify transaction
        count = await pool.fetchval("SELECT COUNT(*) FROM users")
        print(f"Total users: {count}")

    finally:
        # Always release pool and delete the temp file, even after errors.
        await pool.close()
        os.unlink(db_path)


if __name__ == "__main__":
    asyncio.run(demonstrate_database())
