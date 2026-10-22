"""
Task Groups

Structured concurrency with task groups and error handling.
"""

import asyncio
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class TaskGroupErrorPolicy(Enum):
    """Policy for handling errors in task group."""

    CANCEL_ALL = "cancel_all"  # Cancel all tasks on first error
    CONTINUE = "continue"  # Continue other tasks, collect errors
    RETURN_FIRST = "return_first"  # Return first result (success or error)


@dataclass
class TaskGroupConfig:
    """Configuration for task group."""

    name: str | None = None
    error_policy: TaskGroupErrorPolicy = TaskGroupErrorPolicy.CANCEL_ALL
    max_concurrent: int | None = None
    timeout: float | None = None
    return_exceptions: bool = False


@dataclass
class TaskResult(Generic[T]):
    """Result of a task in the group."""

    task_id: str
    name: str
    success: bool
    result: T | None = None
    exception: BaseException | None = None
    started_at: float = field(
        default_factory=lambda: (
            asyncio.get_event_loop().time() if asyncio.get_event_loop() else time.time()
        )
    )
    completed_at: float | None = None

    @property
    def duration(self) -> float:
        if self.completed_at:
            return self.completed_at - self.started_at
        return 0.0


class TaskGroup(Generic[T]):
    """Structured task group with error handling and lifecycle management."""

    def __init__(
        self,
        config: TaskGroupConfig | None = None,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._config = config or TaskGroupConfig()
        self._loop = loop or asyncio.get_running_loop()
        self._name = self._config.name or f"TaskGroup-{id(self)}"

        self._tasks: dict[str, asyncio.Task] = {}
        self._results: dict[str, TaskResult] = {}
        self._semaphore: asyncio.Semaphore | None = None
        self._running = False
        self._completed = False
        self._error: BaseException | None = None
        self._lock = asyncio.Lock()
        self._on_error_callbacks: list[Callable[[str, BaseException], Awaitable[None]]] = []
        self._on_complete_callbacks: list[Callable[[str, TaskResult], Awaitable[None]]] = []

        if self._config.max_concurrent:
            self._semaphore = asyncio.Semaphore(self._config.max_concurrent)

    @property
    def name(self) -> str:
        return self._name

    @property
    def config(self) -> TaskGroupConfig:
        return self._config

    @property
    def running(self) -> bool:
        return self._running

    @property
    def completed(self) -> bool:
        return self._completed

    @property
    def error(self) -> BaseException | None:
        return self._error

    @property
    def task_count(self) -> int:
        return len(self._tasks)

    @property
    def completed_count(self) -> int:
        return len([r for r in self._results.values() if r.completed_at is not None])

    def create_task(
        self, coro: Coroutine[Any, Any, T], *, name: str | None = None
    ) -> asyncio.Task[T]:
        """Create and track a task in this group."""
        if not self._running:
            raise RuntimeError("Task group not running")

        task_id = str(uuid.uuid4())[:8]
        task_name = name or getattr(coro, "__qualname__", f"task-{task_id}")

        async def wrapped():
            result_obj = TaskResult(task_id=task_id, name=task_name, success=False)
            self._results[task_id] = result_obj

            try:
                if self._semaphore:
                    async with self._semaphore:
                        result = await coro
                else:
                    result = await coro

                result_obj.success = True
                result_obj.result = result
                return result
            except BaseException as e:
                result_obj.exception = e
                raise
            finally:
                result_obj.completed_at = self._loop.time()
                # Fire completion callbacks
                for cb in self._on_complete_callbacks:
                    with suppress(Exception):
                        await cb(task_id, result_obj)

        task = self._loop.create_task(wrapped(), name=f"{self._name}-{task_name}")
        self._tasks[task_id] = task

        # Add error callback
        def on_done(t: asyncio.Task) -> None:
            if t.cancelled():
                return
            exc = t.exception()
            if exc:
                self._handle_task_error(task_id, exc)

        task.add_done_callback(on_done)
        return task

    def _handle_task_error(self, task_id: str, exception: BaseException) -> None:
        """Handle task error according to policy."""
        self._error = exception

        # Fire error callbacks
        for cb in self._on_error_callbacks:
            with suppress(Exception):
                self._loop.create_task(cb(task_id, exception))

        if self._config.error_policy == TaskGroupErrorPolicy.CANCEL_ALL:
            self.cancel_all(str(exception))

    def cancel_all(self, msg: str | None = None) -> int:
        """Cancel all tasks in the group."""
        count = 0
        for task in self._tasks.values():
            if not task.done():
                task.cancel(msg)
                count += 1
        return count

    async def wait_all(self) -> list[TaskResult[T]]:
        """Wait for all tasks to complete."""
        if not self._tasks:
            return []

        # Wait for all tasks
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._completed = True

        return list(self._results.values())

    async def wait_any(self) -> TaskResult[T]:
        """Wait for any task to complete."""
        if not self._tasks:
            raise RuntimeError("No tasks in group")

        done, _pending = await asyncio.wait(
            self._tasks.values(), return_when=asyncio.FIRST_COMPLETED
        )

        task = done.pop()
        task_id = next(k for k, v in self._tasks.items() if v is task)
        return self._results[task_id]

    def add_error_callback(self, callback: Callable[[str, BaseException], Awaitable[None]]) -> None:
        """Add callback for task errors."""
        self._on_error_callbacks.append(callback)

    def add_complete_callback(self, callback: Callable[[str, TaskResult], Awaitable[None]]) -> None:
        """Add callback for task completion."""
        self._on_complete_callbacks.append(callback)

    def get_result(self, task_id: str) -> TaskResult[T] | None:
        """Get result for a specific task."""
        return self._results.get(task_id)

    def get_all_results(self) -> list[TaskResult[T]]:
        """Get all results."""
        return list(self._results.values())

    def get_exceptions(self) -> list[BaseException]:
        """Get all exceptions."""
        return [r.exception for r in self._results.values() if r.exception]

    async def __aenter__(self) -> "TaskGroup[T]":
        self._running = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self._tasks:
            await self.wait_all()
        self._completed = True
        return False  # Don't suppress exceptions

    def stats(self) -> dict[str, Any]:
        successful = sum(1 for r in self._results.values() if r.success)
        failed = sum(1 for r in self._results.values() if not r.success)

        return {
            "name": self._name,
            "total_tasks": len(self._results),
            "completed": self.completed_count,
            "successful": successful,
            "failed": failed,
            "running": self._running,
            "completed_all": self._completed,
        }


def create_task_group(
    config: TaskGroupConfig | None = None, *, loop: asyncio.AbstractEventLoop | None = None
) -> TaskGroup:
    """Create a new task group."""
    return TaskGroup(config, loop=loop)


@asynccontextmanager
async def task_group(config: TaskGroupConfig | None = None) -> AsyncGenerator[TaskGroup[T], None]:
    """Context manager for task group."""
    group = create_task_group(config)
    async with group:
        yield group


async def gather_with_concurrency(
    *coros: Coroutine[Any, Any, T], max_concurrent: int, return_exceptions: bool = False
) -> list[T | BaseException]:
    """Gather with concurrency limit."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited(coro: Coroutine[Any, Any, T]) -> T | BaseException:
        async with semaphore:
            try:
                return await coro
            except BaseException as e:
                if return_exceptions:
                    return e
                raise

    return await asyncio.gather(*[limited(c) for c in coros], return_exceptions=return_exceptions)


class DynamicTaskGroup(TaskGroup[T]):
    """Task group that can add tasks dynamically."""

    def __init__(self, config: TaskGroupConfig | None = None, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self._task_factory: Callable[[], Coroutine[Any, Any, T]] | None = None

    def set_task_factory(self, factory: Callable[[], Coroutine[Any, Any, T]]) -> None:
        """Set a factory for creating new tasks."""
        self._task_factory = factory

    async def spawn(self, *args, **kwargs) -> asyncio.Task[T]:
        """Spawn a new task from factory."""
        if self._task_factory is None:
            raise RuntimeError("No task factory set")
        return self.create_task(self._task_factory(), *args, **kwargs)


class TaskGroupManager:
    """Manage multiple task groups."""

    def __init__(self) -> None:
        self._groups: dict[str, TaskGroup] = {}
        self._lock = asyncio.Lock()

    def create_group(self, name: str, config: TaskGroupConfig | None = None) -> TaskGroup:
        if name in self._groups:
            raise ValueError(f"Task group '{name}' already exists")

        cfg = config or TaskGroupConfig(name=name)
        group = TaskGroup(cfg)
        self._groups[name] = group
        return group

    def get_group(self, name: str) -> TaskGroup | None:
        return self._groups.get(name)

    def remove_group(self, name: str) -> bool:
        if name in self._groups:
            del self._groups[name]
            return True
        return False

    async def wait_all_groups(self) -> dict[str, list[TaskResult]]:
        results = {}
        for name, group in self._groups.items():
            results[name] = await group.wait_all()
        return results

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        return {name: group.stats() for name, group in self._groups.items()}


_global_task_group_manager = TaskGroupManager()


def get_task_group_manager() -> TaskGroupManager:
    return _global_task_group_manager
