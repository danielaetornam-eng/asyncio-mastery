"""
Bulkhead Pattern

Isolation patterns for preventing cascade failures.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class BulkheadRejected(Exception):
    """Raised when bulkhead rejects a request."""

    def __init__(self, message: str, bulkhead_name: str, available: int = 0) -> None:
        super().__init__(message)
        self.bulkhead_name = bulkhead_name
        self.available = available


@dataclass
class BulkheadConfig:
    """Configuration for bulkhead."""

    max_concurrent: int = 10
    max_queue_size: int = 0  # 0 = unbounded
    timeout: float | None = None
    name: str | None = None


@dataclass
class BulkheadStats:
    """Bulkhead statistics."""

    name: str
    max_concurrent: int
    current_concurrent: int = 0
    queued_requests: int = 0
    total_requests: int = 0
    completed_requests: int = 0
    rejected_requests: int = 0
    total_wait_time: float = 0.0
    total_execution_time: float = 0.0
    peak_concurrent: int = 0


class Bulkhead(Generic[T]):
    """Semaphore-based bulkhead for limiting concurrency."""

    def __init__(self, config: BulkheadConfig | None = None) -> None:
        self._config = config or BulkheadConfig()
        self._semaphore = asyncio.Semaphore(self._config.max_concurrent)
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=self._config.max_queue_size)
        self._name = self._config.name or f"Bulkhead-{id(self)}"
        self._stats = BulkheadStats(name=self._name, max_concurrent=self._config.max_concurrent)
        self._lock = asyncio.Lock()
        self._active_tasks: set = set()

    @property
    def name(self) -> str:
        return self._name

    @property
    def max_concurrent(self) -> int:
        return self._config.max_concurrent

    @property
    def available(self) -> int:
        return self._config.max_concurrent - self._stats.current_concurrent

    @property
    def stats(self) -> BulkheadStats:
        return self._stats

    async def acquire(self, timeout: float | None = None) -> bool:
        """Acquire a slot in the bulkhead."""
        effective_timeout = timeout or self._config.timeout

        async with self._lock:
            self._stats.total_requests += 1
            self._stats.queued_requests += 1

        start_wait = time.monotonic()

        try:
            if effective_timeout is not None:
                await asyncio.wait_for(self._semaphore.acquire(), timeout=effective_timeout)
            else:
                await self._semaphore.acquire()
        except TimeoutError:
            async with self._lock:
                self._stats.queued_requests -= 1
                self._stats.rejected_requests += 1
            raise BulkheadRejected(
                f"Bulkhead '{self._name}' timeout after {effective_timeout}s",
                self._name,
                self.available,
            )

        wait_time = time.monotonic() - start_wait

        async with self._lock:
            self._stats.queued_requests -= 1
            self._stats.current_concurrent += 1
            self._stats.peak_concurrent = max(
                self._stats.peak_concurrent, self._stats.current_concurrent
            )
            self._stats.total_wait_time += wait_time

        return True

    async def release(self) -> None:
        """Release a slot in the bulkhead."""
        self._semaphore.release()

        async with self._lock:
            self._stats.current_concurrent = max(0, self._stats.current_concurrent - 1)
            self._stats.completed_requests += 1

    @asynccontextmanager
    async def execute(self, timeout: float | None = None):
        """Context manager for executing within bulkhead."""
        await self.acquire(timeout)
        start_exec = time.monotonic()
        try:
            yield
        finally:
            exec_time = time.monotonic() - start_exec
            async with self._lock:
                self._stats.total_execution_time += exec_time
            self.release()

    async def run(self, coro: Awaitable[T], timeout: float | None = None) -> T:
        """Run a coroutine within the bulkhead."""
        async with self.execute(timeout):
            return await coro

    async def submit(self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        """Submit a function to run within the bulkhead."""
        return await self.run(func(*args, **kwargs), timeout=self._config.timeout)

    def get_metrics(self) -> dict[str, Any]:
        return {
            "name": self._name,
            "max_concurrent": self._config.max_concurrent,
            "current_concurrent": self._stats.current_concurrent,
            "available": self.available,
            "queued": self._stats.queued_requests,
            "total_requests": self._stats.total_requests,
            "completed": self._stats.completed_requests,
            "rejected": self._stats.rejected_requests,
            "rejection_rate": self._stats.rejected_requests / max(1, self._stats.total_requests),
            "peak_concurrent": self._stats.peak_concurrent,
            "avg_wait_time": self._stats.total_wait_time / max(1, self._stats.total_requests),
            "avg_execution_time": (
                self._stats.total_execution_time / max(1, self._stats.completed_requests)
            ),
        }


class SemaphoreBulkhead(Bulkhead):
    """Alias for semaphore-based bulkhead."""

    pass


class ThreadPoolBulkhead(Generic[T]):
    """Thread pool based bulkhead for CPU-bound work."""

    def __init__(
        self,
        max_workers: int = 4,
        max_queue_size: int = 0,
        *,
        name: str | None = None,
        thread_name_prefix: str = "bulkhead",
    ) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix=thread_name_prefix
        )
        self._max_workers = max_workers
        self._max_queue_size = max_queue_size
        self._name = name or f"ThreadPoolBulkhead-{id(self)}"
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._stats = {"submitted": 0, "completed": 0, "failed": 0, "rejected": 0, "active": 0}
        self._lock = asyncio.Lock()
        self._worker_tasks: list[asyncio.Task] = []
        self._running = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def max_workers(self) -> int:
        return self._max_workers

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        for i in range(self._max_workers):
            task = asyncio.create_task(self._worker(f"worker-{i}"))
            self._worker_tasks.append(task)

    async def stop(self, wait: bool = True) -> None:
        self._running = False
        for _ in range(self._max_workers):
            await self._queue.put(None)

        if wait:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()

    async def _worker(self, name: str) -> None:
        while self._running:
            try:
                item = await self._queue.get()
                if item is None:
                    break

                func, args, kwargs, future = item
                async with self._lock:
                    self._stats["active"] += 1

                try:
                    result = await asyncio.get_running_loop().run_in_executor(
                        self._executor, lambda: func(*args, **kwargs)
                    )
                    future.set_result(result)
                    async with self._lock:
                        self._stats["completed"] += 1
                except Exception as e:
                    future.set_exception(e)
                    async with self._lock:
                        self._stats["failed"] += 1
                finally:
                    async with self._lock:
                        self._stats["active"] -= 1

            except asyncio.CancelledError:
                break
            except Exception:
                pass

    async def submit(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        if not self._running:
            await self.start()

        if self._max_queue_size > 0 and self._queue.qsize() >= self._max_queue_size:
            async with self._lock:
                self._stats["rejected"] += 1
            raise BulkheadRejected(f"Thread pool bulkhead '{self._name}' queue full", self._name)

        loop = asyncio.get_running_loop()
        future = loop.create_future()

        await self._queue.put((func, args, kwargs, future))

        async with self._lock:
            self._stats["submitted"] += 1

        return await future

    def get_stats(self) -> dict[str, Any]:
        return {
            "name": self._name,
            "max_workers": self._max_workers,
            "active": self._stats["active"],
            "queue_size": self._queue.qsize(),
            "submitted": self._stats["submitted"],
            "completed": self._stats["completed"],
            "failed": self._stats["failed"],
            "rejected": self._stats["rejected"],
        }


class AsyncBulkhead(Generic[T]):
    """Async bulkhead with queue and priority support."""

    def __init__(
        self,
        max_concurrent: int,
        max_queue_size: int = 0,
        *,
        name: str | None = None,
        priority_levels: int = 1,
    ) -> None:
        if max_concurrent <= 0:
            raise ValueError("max_concurrent must be > 0")
        if priority_levels <= 0:
            raise ValueError("priority_levels must be > 0")

        self._max_concurrent = max_concurrent
        self._max_queue_size = max_queue_size
        self._priority_levels = priority_levels
        self._name = name or f"AsyncBulkhead-{id(self)}"

        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._queues: list[asyncio.Queue] = [
            asyncio.Queue(maxsize=max_queue_size // priority_levels if max_queue_size else 0)
            for _ in range(priority_levels)
        ]

        self._stats = {
            "total": 0,
            "completed": 0,
            "rejected": 0,
            "by_priority": [0] * priority_levels,
        }
        self._lock = asyncio.Lock()
        self._running = False
        self._dispatcher_task: asyncio.Task | None = None
        self._active_count = 0

    @property
    def name(self) -> str:
        return self._name

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._dispatcher_task = asyncio.create_task(self._dispatch_loop())

    async def stop(self) -> None:
        self._running = False
        for q in self._queues:
            await q.put(None)
        if self._dispatcher_task:
            await self._dispatcher_task

    async def _dispatch_loop(self) -> None:
        while self._running:
            # Find highest priority non-empty queue
            for priority in range(self._priority_levels):
                queue = self._queues[priority]
                try:
                    item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    continue

                if item is None:
                    return

                coro, future = item
                asyncio.create_task(self._execute(coro, future, priority))
                break
            else:
                # All queues empty, wait for next item
                await asyncio.sleep(0.001)

    async def _execute(self, coro: Awaitable[T], future: asyncio.Future, priority: int) -> None:
        await self._semaphore.acquire()

        async with self._lock:
            self._active_count += 1
            self._stats["by_priority"][priority] += 1

        try:
            result = await coro
            future.set_result(result)
            async with self._lock:
                self._stats["completed"] += 1
        except Exception as e:
            future.set_exception(e)
        finally:
            self._semaphore.release()
            async with self._lock:
                self._active_count -= 1

    async def submit(self, coro: Awaitable[T], priority: int = 0) -> T:
        if not self._running:
            await self.start()

        if priority < 0 or priority >= self._priority_levels:
            raise ValueError(f"Invalid priority: {priority}")

        queue = self._queues[priority]

        if self._max_queue_size > 0 and queue.qsize() >= queue.maxsize:
            async with self._lock:
                self._stats["rejected"] += 1
            raise BulkheadRejected(
                f"Bulkhead '{self._name}' priority {priority} queue full", self._name
            )

        loop = asyncio.get_running_loop()
        future = loop.create_future()

        await queue.put((coro, future))

        async with self._lock:
            self._stats["total"] += 1

        return await future

    def get_stats(self) -> dict[str, Any]:
        return {
            "name": self._name,
            "max_concurrent": self._max_concurrent,
            "active": self._active_count,
            "available": self._max_concurrent - self._active_count,
            "priority_levels": self._priority_levels,
            "queue_sizes": [q.qsize() for q in self._queues],
            **self._stats,
        }


class BulkheadRegistry:
    """Registry for managing multiple bulkheads."""

    def __init__(self) -> None:
        self._bulkheads: dict[str, Bulkhead] = {}
        self._lock = asyncio.Lock()

    def get_or_create(self, name: str, config: BulkheadConfig | None = None) -> Bulkhead:
        if name not in self._bulkheads:
            cfg = config or BulkheadConfig(name=name)
            self._bulkheads[name] = Bulkhead(cfg)
        return self._bulkheads[name]

    def get(self, name: str) -> Bulkhead | None:
        return self._bulkheads.get(name)

    def remove(self, name: str) -> bool:
        if name in self._bulkheads:
            del self._bulkheads[name]
            return True
        return False

    def all_bulkheads(self) -> list[Bulkhead]:
        return list(self._bulkheads.values())

    def get_all_metrics(self) -> dict[str, dict[str, Any]]:
        return {name: bh.get_metrics() for name, bh in self._bulkheads.items()}


_global_bulkhead_registry = BulkheadRegistry()


def get_bulkhead_registry() -> BulkheadRegistry:
    return _global_bulkhead_registry


def bulkhead(
    max_concurrent: int = 10, max_queue_size: int = 0, timeout: float | None = None, **kwargs: Any
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator for bulkhead isolation."""
    config = BulkheadConfig(
        max_concurrent=max_concurrent, max_queue_size=max_queue_size, timeout=timeout, **kwargs
    )
    bh = Bulkhead(config)

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await bh.run(func(*args, **kwargs), timeout=timeout)

        return wrapper

    return decorator


from functools import wraps
