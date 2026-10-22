"""
Resource Management

Async resource lifecycle management and pooling.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from typing import Any, Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class AsyncResource(ABC, Generic[T]):
    """Base class for async resources with lifecycle management."""

    def __init__(self, *, name: str | None = None) -> None:
        self._name = name or f"{self.__class__.__name__}-{id(self)}"
        self._state = "created"  # created, initializing, ready, closing, closed, failed
        self._lock = asyncio.Lock()
        self._created_at = time.monotonic()
        self._initialized_at: float | None = None
        self._closed_at: float | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_ready(self) -> bool:
        return self._state == "ready"

    @property
    def is_closed(self) -> bool:
        return self._state in ("closed", "failed")

    async def initialize(self) -> None:
        """Initialize the resource."""
        async with self._lock:
            if self._state != "created":
                raise RuntimeError(f"Resource already {self._state}")

            self._state = "initializing"

            try:
                await self._do_initialize()
                self._state = "ready"
                self._initialized_at = time.monotonic()
            except Exception as _e:
                self._state = "failed"
                raise

    async def close(self) -> None:
        """Close the resource."""
        async with self._lock:
            if self._state in ("closed", "closing"):
                return

            if self._state == "created":
                self._state = "closed"
                self._closed_at = time.monotonic()
                return

            self._state = "closing"

        try:
            await self._do_close()
        finally:
            async with self._lock:
                self._state = "closed"
                self._closed_at = time.monotonic()

    @abstractmethod
    async def _do_initialize(self) -> None:
        """Implement actual initialization."""
        pass

    @abstractmethod
    async def _do_close(self) -> None:
        """Implement actual cleanup."""
        pass

    async def __aenter__(self) -> "AsyncResource[T]":
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    def get_stats(self) -> dict[str, Any]:
        return {
            "name": self._name,
            "state": self._state,
            "created_at": self._created_at,
            "initialized_at": self._initialized_at,
            "closed_at": self._closed_at,
            "uptime": (
                (self._closed_at or time.monotonic()) - self._initialized_at
                if self._initialized_at
                else 0
            ),
        }


class ResourcePool(Generic[T]):
    """Pool of reusable async resources."""

    def __init__(
        self,
        factory: Callable[[], AsyncResource[T]],
        min_size: int = 0,
        max_size: int = 10,
        *,
        name: str | None = None,
        acquire_timeout: float = 30.0,
        idle_timeout: float = 300.0,
        validate_on_acquire: bool = True,
    ) -> None:
        if min_size < 0 or max_size < min_size:
            raise ValueError("Invalid pool size")

        self._factory = factory
        self._min_size = min_size
        self._max_size = max_size
        self._name = name or f"ResourcePool-{id(self)}"
        self._acquire_timeout = acquire_timeout
        self._idle_timeout = idle_timeout
        self._validate_on_acquire = validate_on_acquire

        self._available: deque[AsyncResource[T]] = deque()
        self._in_use: set[AsyncResource[T]] = set()
        self._lock = asyncio.Lock()
        self._created_count = 0
        self._stats = {
            "acquired": 0,
            "released": 0,
            "created": 0,
            "destroyed": 0,
            "validation_failures": 0,
            "timeouts": 0,
        }
        self._cleanup_task: asyncio.Task | None = None
        self._running = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def size(self) -> int:
        return len(self._available) + len(self._in_use)

    @property
    def available_count(self) -> int:
        return len(self._available)

    @property
    def in_use_count(self) -> int:
        return len(self._in_use)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            **self._stats.copy(),
            "size": self.size,
            "available": self.available_count,
            "in_use": self.in_use_count,
        }

    async def start(self) -> None:
        """Start the pool and create minimum resources."""
        if self._running:
            return

        self._running = True

        # Create minimum resources
        for _ in range(self._min_size):
            await self._create_resource()

        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        """Stop the pool and close all resources."""
        self._running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._cleanup_task

        # Close all resources
        async with self._lock:
            all_resources = list(self._available) + list(self._in_use)
            self._available.clear()
            self._in_use.clear()

        for resource in all_resources:
            try:
                await resource.close()
                self._stats["destroyed"] += 1
            except Exception:
                pass

    async def _create_resource(self) -> AsyncResource[T]:
        """Create and initialize a new resource."""
        resource = self._factory()
        await resource.initialize()

        async with self._lock:
            self._available.append(resource)
            self._created_count += 1
            self._stats["created"] += 1

        return resource

    async def acquire(self, timeout: float | None = None) -> AsyncResource[T]:
        """Acquire a resource from the pool."""
        effective_timeout = timeout or self._acquire_timeout
        start = time.monotonic()

        while True:
            async with self._lock:
                # Try to get available resource
                while self._available:
                    resource = self._available.popleft()

                    if self._validate_on_acquire:
                        # Validate resource is still healthy
                        try:
                            if hasattr(resource, "validate"):
                                await resource.validate()
                        except Exception:
                            await resource.close()
                            self._stats["destroyed"] += 1
                            self._stats["validation_failures"] += 1
                            continue

                    self._in_use.add(resource)
                    self._stats["acquired"] += 1
                    return resource

                # Create new resource if under max
                if self.size < self._max_size:
                    resource = await self._create_resource()
                    self._in_use.add(resource)
                    self._stats["acquired"] += 1
                    return resource

            # Wait for resource
            if effective_timeout and (time.monotonic() - start) >= effective_timeout:
                self._stats["timeouts"] += 1
                raise TimeoutError(
                    f"Pool '{self._name}' acquire timeout after {effective_timeout}s"
                )

            await asyncio.sleep(0.01)

    async def release(self, resource: AsyncResource[T]) -> None:
        """Release a resource back to the pool."""
        async with self._lock:
            if resource not in self._in_use:
                return

            self._in_use.remove(resource)

            if resource.is_closed:
                self._stats["destroyed"] += 1
            else:
                self._available.append(resource)

            self._stats["released"] += 1

    @asynccontextmanager
    async def get(self, timeout: float | None = None) -> AsyncGenerator[AsyncResource[T], None]:
        """Context manager for acquiring and releasing a resource."""
        resource = await self.acquire(timeout)
        try:
            yield resource
        finally:
            await self.release(resource)

    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of idle resources."""
        while self._running:
            await asyncio.sleep(60)  # Check every minute

            if not self._running:
                break

            await self._cleanup_idle()

    async def _cleanup_idle(self) -> None:
        """Remove idle resources exceeding min_size."""
        async with self._lock:
            now = time.monotonic()
            to_remove = []

            for resource in self._available:
                if self.size > self._min_size:
                    # Check if resource has been idle too long
                    if hasattr(resource, "_last_used"):
                        idle_time = now - resource._last_used
                        if idle_time > self._idle_timeout:
                            to_remove.append(resource)
                    else:
                        to_remove.append(resource)

            for resource in to_remove:
                self._available.remove(resource)
                try:
                    await resource.close()
                    self._stats["destroyed"] += 1
                except Exception:
                    pass

    async def resize(self, min_size: int, max_size: int) -> None:
        """Resize the pool."""
        if min_size < 0 or max_size < min_size:
            raise ValueError("Invalid pool size")

        async with self._lock:
            self._min_size = min_size
            self._max_size = max_size

            # Create resources if needed
            while self.size < min_size:
                await self._create_resource()

            # Remove excess if needed
            while self.size > max_size and self._available:
                resource = self._available.popleft()
                try:
                    await resource.close()
                    self._stats["destroyed"] += 1
                except Exception:
                    pass


class ResourceManager:
    """Manage multiple named resource pools."""

    def __init__(self) -> None:
        self._pools: dict[str, ResourcePool] = {}
        self._lock = asyncio.Lock()

    def create_pool(
        self,
        name: str,
        factory: Callable[[], AsyncResource[T]],
        min_size: int = 0,
        max_size: int = 10,
        **kwargs,
    ) -> ResourcePool[T]:
        if name in self._pools:
            raise ValueError(f"Pool '{name}' already exists")

        pool = ResourcePool(factory, min_size, max_size, name=name, **kwargs)
        self._pools[name] = pool
        return pool

    def get_pool(self, name: str) -> ResourcePool | None:
        return self._pools.get(name)

    def remove_pool(self, name: str) -> bool:
        if name in self._pools:
            del self._pools[name]
            return True
        return False

    async def start_all(self) -> None:
        for pool in self._pools.values():
            await pool.start()

    async def stop_all(self) -> None:
        for pool in self._pools.values():
            await pool.stop()

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        return {name: pool.stats for name, pool in self._pools.items()}


class ConnectionPool(ResourcePool):
    """Specialized pool for connection-like resources."""

    def __init__(
        self,
        connect_func: Callable[[], Awaitable[T]],
        close_func: Callable[[T], Awaitable[None]],
        min_size: int = 0,
        max_size: int = 10,
        *,
        name: str | None = None,
        **kwargs,
    ) -> None:
        class ConnectionResource(AsyncResource[T]):
            def __init__(self, pool_name: str) -> None:
                super().__init__(name=f"{pool_name}-conn")
                self._connect_func = connect_func
                self._close_func = close_func
                self._connection: T | None = None
                self._last_used = time.monotonic()

            async def _do_initialize(self) -> None:
                self._connection = await self._connect_func()

            async def _do_close(self) -> None:
                if self._connection:
                    await self._close_func(self._connection)
                    self._connection = None

            async def validate(self) -> bool:
                if self._connection and hasattr(self._connection, "is_closed"):
                    return not self._connection.is_closed()
                return True

            def get_connection(self) -> T:
                self._last_used = time.monotonic()
                return self._connection

        def factory() -> ConnectionResource:
            return ConnectionResource(name or "pool")

        super().__init__(factory, min_size, max_size, name=name, **kwargs)
        self._connect_func = connect_func
        self._close_func = close_func

    @asynccontextmanager
    async def connection(self, timeout: float | None = None) -> AsyncGenerator[T, None]:
        """Get a connection from the pool."""
        async with self.get(timeout) as resource:
            if hasattr(resource, "get_connection"):
                yield resource.get_connection()
            else:
                yield resource  # type: ignore


@asynccontextmanager
async def managed_resource(resource: AsyncResource[T]) -> AsyncGenerator[T, None]:
    """Context manager for resource lifecycle."""
    await resource.initialize()
    try:
        yield resource
    finally:
        await resource.close()


class LifecycleManager:
    """Manage lifecycle of multiple resources."""

    def __init__(self) -> None:
        self._resources: dict[str, AsyncResource] = {}
        self._start_order: list[str] = []
        self._lock = asyncio.Lock()

    def register(self, name: str, resource: AsyncResource) -> None:
        if name in self._resources:
            raise ValueError(f"Resource '{name}' already registered")
        self._resources[name] = resource
        self._start_order.append(name)

    def unregister(self, name: str) -> bool:
        if name in self._resources:
            del self._resources[name]
            self._start_order.remove(name)
            return True
        return False

    async def start_all(self) -> None:
        for name in self._start_order:
            resource = self._resources[name]
            if resource.state == "created":
                await resource.initialize()

    async def stop_all(self) -> None:
        # Stop in reverse order
        for name in reversed(self._start_order):
            resource = self._resources.get(name)
            if resource and not resource.is_closed:
                with suppress(Exception):
                    await resource.close()

    async def restart(self, name: str) -> bool:
        resource = self._resources.get(name)
        if not resource:
            return False

        if not resource.is_closed:
            await resource.close()

        await resource.initialize()
        return True

    def get_resource(self, name: str) -> AsyncResource | None:
        return self._resources.get(name)

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        return {name: r.get_stats() for name, r in self._resources.items()}


_global_resource_manager = ResourceManager()


def get_resource_manager() -> ResourceManager:
    return _global_resource_manager
