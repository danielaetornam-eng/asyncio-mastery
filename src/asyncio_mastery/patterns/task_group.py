"""
Task Group - Structured concurrency for managing groups of tasks.

THE PROBLEM
-----------
When you fire off many tasks yourself with ``asyncio.create_task()``, it is
easy to lose track of them: a task might crash silently, you might forget to
``await`` one, or you might exit a function while background tasks are still
running. These "orphan" tasks leak resources and make programs very hard to
reason about.

REAL-WORLD ANALOGY
------------------
Think of a TaskGroup like a **team meet**. The coach (you) gathers the
players (tasks), gives everyone their job, and does not let anyone leave the
meet until *every* player has finished their drill. If one player gets
hurt (raises an exception), the coach can stop the whole drill and attend
to the problem — nobody wanders off unsupervised.

WHY ASYNC PRIMITIVES ARE NEEDED
-------------------------------
A task group relies on the event loop's ability to run many coroutines
concurrently. Structured concurrency means the *lifetime* of child tasks is
bound to the scope of the ``async with`` block: entering the block creates
tasks, and leaving it (normally or via an exception) guarantees every task
is awaited or cancelled. This eliminates the classic bug of "fire and
forget" tasks.

WHEN TO USE IT
--------------
* You need to run a fixed set of tasks concurrently and wait for all of them.
* You want automatic cleanup: if one task fails, cancel the rest.
* You want to aggregate results and exceptions from many tasks.

NOTE
----
Python 3.11+ ships a built-in :class:`asyncio.TaskGroup`. This module
provides an enhanced, teaching-oriented version that also collects
results and exceptions for you to inspect afterwards.

Minimal example
---------------
>>> import asyncio
>>> async def main() -> None:
...     async with create_task_group(name="demo") as group:
...         group.create_task(asyncio.sleep(0.1, result="done"))
...     print(group.result.results)  # ['done']
>>> asyncio.run(main())
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass
class TaskGroupResult:
    """Aggregated outcome of a finished task group.

    After the ``async with`` block exits, you can inspect this object to
    see exactly what succeeded, what failed, and whether the group was
    cancelled.

    Attributes:
        results: Return values from every task that completed successfully.
        exceptions: The exception raised by each task that failed. The
            order matches the order in which tasks finished, not the order
            in which they were created.
        cancelled: True if the group was cancelled early (for example,
            because the surrounding ``async with`` body raised and the
            remaining tasks had to be torn down).
    """

    results: list[Any] = field(default_factory=list)
    exceptions: list[BaseException] = field(default_factory=list)
    cancelled: bool = False

    @property
    def success(self) -> bool:
        """True only when no task raised and the group was never cancelled.

        Returns:
            bool: False if any exception was recorded or the group was
            cancelled; True otherwise.
        """
        return len(self.exceptions) == 0 and not self.cancelled


class TaskGroup:
    """Enhanced task group with structured concurrency.

    Tasks created inside the group's ``async with`` block are guaranteed
    not to outlive the block. When the block exits, all remaining tasks
    are awaited (or cancelled first if the block raised).

    Attributes:
        _name: Human-readable name, useful in logs and task names.
        _tasks: Set of currently tracked :class:`asyncio.Task` objects.
        _results: Collected return values of tasks that completed.
        _exceptions: Collected exceptions from tasks that failed.
        _cancelled: Flag flipped when the group tears down due to an error.
        _entered: Guard flag; tasks may only be created inside the block.

    Example:
        >>> async with create_task_group(name="batch") as group:
        ...     group.create_task(fetch("a"))
        ...     group.create_task(fetch("b"))
        >>> assert group.result.success
    """

    def __init__(self, *, name: str | None = None) -> None:
        """Create a new (not yet entered) task group.

        Args:
            name: Optional label used in logs and as the prefix for task
                names. Defaults to "task-group".
        """
        self._name = name or "task-group"
        self._tasks: set[asyncio.Task] = set()
        self._results: list[Any] = []
        self._exceptions: list[BaseException] = []
        self._cancelled = False
        self._entered = False

    def create_task(
        self, coro: asyncio.coroutines.Coroutine[Any, Any, T], *, name: str | None = None
    ) -> asyncio.Task[T]:
        """Schedule a coroutine as a task owned by this group.

        Args:
            coro: The coroutine to run concurrently.
            name: Optional task name (shows up in :func:`asyncio.all_tasks`
                and debugging tools).

        Returns:
            The created :class:`asyncio.Task`, in case you want to await or
            cancel it individually while it is still running.

        Raises:
            RuntimeError: If you try to add a task *before* entering the
                ``async with`` block (``_entered`` is False) or after the
                group has been cancelled. This guard prevents orphan tasks.
        """
        if not self._entered:
            raise RuntimeError("TaskGroup not entered")
        if self._cancelled:
            raise RuntimeError("TaskGroup cancelled")

        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        # When the task finishes, remove it from the tracking set so the
        # group doesn't hold references to dead tasks forever.
        task.add_done_callback(self._tasks.discard)
        # Also record the outcome (result or exception) for later inspection.
        task.add_done_callback(self._on_task_done)
        return task

    def _on_task_done(self, task: asyncio.Task) -> None:
        """Internal callback: capture a finished task's outcome.

        Calling ``task.result()`` re-raises the task's exception, so the
        try/except below is how we *extract* what happened without crashing.

        Args:
            task: The task that just completed.
        """
        try:
            result = task.result()
            self._results.append(result)
        except asyncio.CancelledError:
            pass  # Expected during cancellation
        except BaseException as e:
            self._exceptions.append(e)

    async def __aenter__(self) -> "TaskGroup":
        """Enter the group — enables :meth:`create_task`.

        Returns:
            This TaskGroup instance, bound by ``async with ... as group``.
        """
        self._entered = True
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> bool:
        """Exit the group: cancel on error, then wait for every task.

        This is the heart of structured concurrency: the block cannot be
        exited until all child tasks have finished one way or another.

        Args:
            exc_type: Type of the exception raised in the block, if any.
            exc_val: The exception instance raised in the block, if any.
            exc_tb: Traceback of the exception, if any.

        Returns:
            bool: True suppresses a block-level exception when *no* child
            task failed; False lets exceptions propagate.
        """
        self._entered = False

        # Cancel remaining tasks if exception occurred
        if exc_val is not None:
            self._cancelled = True
            for task in self._tasks:
                task.cancel()

        # Wait for all tasks. return_exceptions=True is essential: without
        # it, the first failing task would make gather() raise and we'd
        # abandon the others (breaking the "nobody leaves early" guarantee).
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        # Return True to suppress exception if all tasks completed
        return len(self._exceptions) == 0

    @property
    def result(self) -> TaskGroupResult:
        """Snapshot of everything that happened inside the group.

        Returns:
            TaskGroupResult: Aggregated results, exceptions, and the
            cancelled flag (typically inspected after the ``async with``).
        """
        return TaskGroupResult(
            results=self._results, exceptions=self._exceptions, cancelled=self._cancelled
        )

    @property
    def tasks(self) -> set[asyncio.Task]:
        """A copy of the still-pending tasks in the group.

        Returns:
            set: Currently tracked tasks (a copy, so callers cannot mutate
            the group's internal set).
        """
        return self._tasks.copy()


@asynccontextmanager
async def create_task_group(*, name: str | None = None) -> AsyncIterator[TaskGroup]:
    """Create task group context manager.

    Convenience wrapper so you can write ``async with create_task_group()``
    instead of constructing and entering a :class:`TaskGroup` manually.

    Args:
        name: Optional label for the group.

    Yields:
        TaskGroup: The entered group, ready to accept tasks via
        :meth:`TaskGroup.create_task`.

    Example:
        >>> async with create_task_group(name="fetchers") as group:
        ...     group.create_task(fetch("https://example.com"))
    """
    group = TaskGroup(name=name)
    async with group:
        yield group


async def demonstrate_task_group() -> None:
    """Run a self-contained demo exploring the task group's behavior.

    Covers: basic usage, exception collection, nested groups, and manual
    management. Run ``python -m asyncio_mastery.patterns.task_group`` to
    see the output.
    """
    print("=== Task Group Demo ===")

    # Basic usage
    print("\n--- Basic Task Group ---")
    async with create_task_group(name="basic") as group:

        async def worker(id: int) -> str:
            await asyncio.sleep(0.05 * (3 - id))
            return f"worker-{id}"

        for i in range(3):
            group.create_task(worker(i), name=f"worker-{i}")

    print(f"Results: {group.result.results}")
    print(f"Exceptions: {group.result.exceptions}")

    # Exception handling
    print("\n--- Exception Handling ---")

    async def failing_worker(id: int) -> str:
        if id == 1:
            raise ValueError(f"Worker {id} failed")
        await asyncio.sleep(0.01)
        return f"worker-{id}"

    async with create_task_group(name="exceptions") as group:
        for i in range(3):
            group.create_task(failing_worker(i))

    result = group.result
    print(f"Results: {result.results}")
    print(f"Exceptions: {[type(e).__name__ for e in result.exceptions]}")
    print(f"Cancelled: {result.cancelled}")

    # Nested task groups
    print("\n--- Nested Groups ---")
    async with create_task_group(name="outer") as outer:
        async with create_task_group(name="inner") as inner:
            for i in range(2):
                inner.create_task(asyncio.sleep(0.01, result=f"inner-{i}"))

        for i in range(2):
            outer.create_task(asyncio.sleep(0.01, result=f"outer-{i}"))

    print(f"Outer results: {outer.result.results}")

    # Manual task management
    print("\n--- Manual Management ---")
    group = TaskGroup(name="manual")

    async with group:
        _task1 = group.create_task(asyncio.sleep(0.1, result="task1"))
        _task2 = group.create_task(asyncio.sleep(0.05, result="task2"))

        # Can access tasks before completion
        print(f"Pending tasks: {len(group.tasks)}")

    print(f"Completed: {group.result.results}")


if __name__ == "__main__":
    asyncio.run(demonstrate_task_group())
