"""
Nursery Pattern

Structured concurrency nursery for managing task lifecycles.
"""

import asyncio
import contextvars
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass
class NurseryTask(Generic[T]):
    """Task managed by nursery."""

    task_id: str
    name: str
    coro: Coroutine[Any, Any, T]
    task: asyncio.Task | None = None
    result: T | None = None
    exception: BaseException | None = None
    cancelled: bool = False
    started_at: float = field(
        default_factory=lambda: (
            asyncio.get_event_loop().time() if asyncio.get_event_loop() else time.time()
        )
    )
    completed_at: float | None = None
    parent_id: str | None = None


class Nursery(Generic[T]):
    """Nursery for structured concurrency - manages a group of tasks."""

    def __init__(
        self,
        *,
        name: str | None = None,
        max_concurrent: int | None = None,
        timeout: float | None = None,
        error_handler: Callable[[str, BaseException], Awaitable[None]] | None = None,
    ) -> None:
        self._name = name or f"Nursery-{id(self)}"
        self._max_concurrent = max_concurrent
        self._timeout = timeout
        self._error_handler = error_handler

        self._tasks: dict[str, NurseryTask] = {}
        self._running = False
        self._closed = False
        self._semaphore: asyncio.Semaphore | None = None
        self._lock = asyncio.Lock()
        self._parent_nursery: Nursery | None = None
        self._child_nurseries: set[Nursery] = set()

        if max_concurrent:
            self._semaphore = asyncio.Semaphore(max_concurrent)

    @property
    def name(self) -> str:
        return self._name

    @property
    def running(self) -> bool:
        return self._running

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def task_count(self) -> int:
        return len(self._tasks)

    @property
    def active_count(self) -> int:
        return sum(1 for t in self._tasks.values() if t.task and not t.task.done())

    def start(self, coro: Coroutine[Any, Any, T], *, name: str | None = None) -> asyncio.Task[T]:
        """Start a task in the nursery."""
        if self._closed:
            raise RuntimeError("Nursery is closed")

        if not self._running:
            raise RuntimeError("Nursery not running")

        task_id = str(uuid.uuid4())[:8]
        task_name = name or getattr(coro, "__qualname__", f"task-{task_id}")

        nursery_task = NurseryTask(
            task_id=task_id, name=task_name, coro=coro, parent_id=self._current_task_id()
        )

        async def wrapped():
            try:
                if self._semaphore:
                    async with self._semaphore:
                        result = await coro
                else:
                    result = await coro

                nursery_task.result = result
                return result
            except asyncio.CancelledError:
                nursery_task.cancelled = True
                raise
            except BaseException as e:
                nursery_task.exception = e
                if self._error_handler:
                    with suppress(Exception):
                        await self._error_handler(task_id, e)
                raise
            finally:
                nursery_task.completed_at = asyncio.get_event_loop().time()

        task = asyncio.create_task(wrapped(), name=f"{self._name}-{task_name}")
        nursery_task.task = task

        self._tasks[task_id] = nursery_task

        # Add done callback to track completion
        def on_done(t: asyncio.Task) -> None:
            pass  # State tracked in nursery_task

        task.add_done_callback(on_done)
        return task

    def start_soon(
        self, coro: Coroutine[Any, Any, T], *, name: str | None = None
    ) -> asyncio.Task[T]:
        """Alias for start."""
        return self.start(coro, name=name)

    def _current_task_id(self) -> str | None:
        """Get current task ID if running in nursery."""
        current = asyncio.current_task()
        if current:
            for task in self._tasks.values():
                if task.task is current:
                    return task.task_id
        return None

    async def wait_all(self) -> list[NurseryTask[T]]:
        """Wait for all tasks to complete."""
        if not self._tasks:
            return []

        pending = [t.task for t in self._tasks.values() if t.task and not t.task.done()]

        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        return list(self._tasks.values())

    async def wait_any(self) -> NurseryTask[T]:
        """Wait for any task to complete."""
        pending = [t.task for t in self._tasks.values() if t.task and not t.task.done()]

        if not pending:
            raise RuntimeError("No pending tasks")

        done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        task = done.pop()

        for nt in self._tasks.values():
            if nt.task is task:
                return nt

        raise RuntimeError("Task not found")

    def cancel_all(self, msg: str | None = None) -> int:
        """Cancel all tasks in nursery."""
        count = 0
        for nt in self._tasks.values():
            if nt.task and not nt.task.done():
                nt.task.cancel(msg)
                count += 1
        return count

    def get_task(self, task_id: str) -> NurseryTask[T] | None:
        """Get task by ID."""
        return self._tasks.get(task_id)

    def get_results(self) -> list[NurseryTask[T]]:
        """Get all task results."""
        return list(self._tasks.values())

    def get_exceptions(self) -> list[BaseException]:
        """Get all exceptions."""
        return [t.exception for t in self._tasks.values() if t.exception]

    def create_nursery(
        self,
        *,
        name: str | None = None,
        max_concurrent: int | None = None,
        timeout: float | None = None,
        error_handler: Callable[[str, BaseException], Awaitable[None]] | None = None,
    ) -> "Nursery[T]":
        """Create a child nursery."""
        child = Nursery(
            name=name or f"{self._name}-child",
            max_concurrent=max_concurrent,
            timeout=timeout,
            error_handler=error_handler,
        )
        child._parent_nursery = self
        self._child_nurseries.add(child)
        return child

    async def __aenter__(self) -> "Nursery[T]":
        self._running = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        # Wait for all tasks
        if self._tasks:
            await self.wait_all()

        # Close child nurseries
        for child in self._child_nurseries:
            await child.__aexit__(exc_type, exc_val, exc_tb)

        self._closed = True
        self._running = False
        return False


@asynccontextmanager
async def open_nursery(
    *,
    name: str | None = None,
    max_concurrent: int | None = None,
    timeout: float | None = None,
    error_handler: Callable[[str, BaseException], Awaitable[None]] | None = None,
) -> AsyncGenerator[Nursery[T], None]:
    """Open a new nursery context."""
    nursery = Nursery(
        name=name, max_concurrent=max_concurrent, timeout=timeout, error_handler=error_handler
    )

    async with nursery:
        yield nursery


class MultiNursery(Generic[T]):
    """Manage multiple named nurseries."""

    def __init__(self) -> None:
        self._nurseries: dict[str, Nursery] = {}
        self._lock = asyncio.Lock()

    def create_nursery(self, name: str, **kwargs) -> Nursery:
        """Create a named nursery."""
        if name in self._nurseries:
            raise ValueError(f"Nursery '{name}' already exists")

        nursery = Nursery(name=name, **kwargs)
        self._nurseries[name] = nursery
        return nursery

    def get_nursery(self, name: str) -> Nursery | None:
        return self._nurseries.get(name)

    def remove_nursery(self, name: str) -> bool:
        if name in self._nurseries:
            del self._nurseries[name]
            return True
        return False

    async def start_all(self) -> None:
        for nursery in self._nurseries.values():
            await nursery.__aenter__()

    async def close_all(self) -> None:
        for nursery in self._nurseries.values():
            await nursery.__aexit__(None, None, None)

    async def wait_all(self) -> dict[str, list[NurseryTask]]:
        results = {}
        for name, nursery in self._nurseries.items():
            results[name] = await nursery.wait_all()
        return results

    def get_all_tasks(self) -> list[NurseryTask]:
        all_tasks = []
        for nursery in self._nurseries.values():
            all_tasks.extend(nursery.get_results())
        return all_tasks


# Context variable for current nursery
_current_nursery: contextvars.ContextVar[Nursery | None] = contextvars.ContextVar(
    "current_nursery", default=None
)


def get_current_nursery() -> Nursery | None:
    """Get current nursery context."""
    return _current_nursery.get()


@asynccontextmanager
async def nursery_context(nursery: Nursery) -> AsyncGenerator[Nursery, None]:
    """Set nursery as current context."""
    token = _current_nursery.set(nursery)
    try:
        yield nursery
    finally:
        _current_nursery.reset(token)


class NurseryScope:
    """Scope for managing nursery lifetime."""

    def __init__(self) -> None:
        self._nurseries: list[Nursery] = []

    def add(self, nursery: Nursery) -> None:
        self._nurseries.append(nursery)

    async def close_all(self) -> None:
        for nursery in reversed(self._nurseries):
            if not nursery.closed:
                await nursery.__aexit__(None, None, None)
        self._nurseries.clear()

    async def wait_all(self) -> list[NurseryTask]:
        all_results = []
        for nursery in self._nurseries:
            results = await nursery.wait_all()
            all_results.extend(results)
        return all_results
