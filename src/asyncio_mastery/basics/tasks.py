"""
Task Management

Advanced task creation, management, and coordination utilities.
"""

import asyncio
import contextvars
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class TaskStatus(Enum):
    """Task execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class TaskInfo:
    """Information about a task."""

    task: asyncio.Task
    name: str
    created_at: float
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    exception: BaseException | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Task(Generic[T]):
    """Enhanced Task wrapper with additional utilities."""

    def __init__(
        self,
        coro: Coroutine[Any, Any, T],
        *,
        name: str | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        context: contextvars.Context | None = None,
    ) -> None:
        self._coro = coro
        self._name = name or getattr(coro, "__qualname__", "Task")
        self._loop = loop or asyncio.get_running_loop()
        self._context = context or contextvars.copy_context()
        self._task: asyncio.Task | None = None
        self._callbacks: list[Callable[[asyncio.Task], None]] = []
        self._info = TaskInfo(
            task=None,  # Will be set after creation
            name=self._name,
            created_at=self._loop.time(),
            status=TaskStatus.PENDING,
        )

    def __await__(self):
        return self.create().__await__()

    @property
    def name(self) -> str:
        return self._name

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        return self._loop

    @property
    def task(self) -> asyncio.Task | None:
        return self._task

    @property
    def info(self) -> TaskInfo:
        return self._info

    def create(self) -> asyncio.Task[T]:
        """Create the underlying asyncio Task."""
        if self._task is None:
            self._task = self._loop.create_task(
                self._run_with_context(), name=self._name, context=self._context
            )
            self._info.task = self._task
            self._info.status = TaskStatus.RUNNING
            for callback in self._callbacks:
                self._task.add_done_callback(callback)
        return self._task

    async def _run_with_context(self) -> T:
        self._info.status = TaskStatus.RUNNING
        try:
            result = await self._coro
            self._info.status = TaskStatus.COMPLETED
            self._info.result = result
            return result
        except asyncio.CancelledError:
            self._info.status = TaskStatus.CANCELLED
            raise
        except BaseException as e:
            self._info.status = TaskStatus.FAILED
            self._info.exception = e
            raise

    def add_done_callback(self, callback: Callable[[asyncio.Task], None]) -> None:
        """Add a callback to be called when task completes."""
        self._callbacks.append(callback)
        if self._task:
            self._task.add_done_callback(callback)

    def remove_done_callback(self, callback: Callable[[asyncio.Task], None]) -> int:
        """Remove a done callback."""
        if self._task:
            return self._task.remove_done_callback(callback)
        try:
            self._callbacks.remove(callback)
            return 1
        except ValueError:
            return 0

    def cancel(self, msg: str | None = None) -> bool:
        """Cancel the task."""
        if self._task:
            return self._task.cancel(msg)
        return False

    def cancelled(self) -> bool:
        """Check if task was cancelled."""
        return self._task.cancelled() if self._task else False

    def done(self) -> bool:
        """Check if task is done."""
        return self._task.done() if self._task else False

    def result(self) -> T:
        """Get task result."""
        if self._task:
            return self._task.result()
        raise RuntimeError("Task not created yet")

    def exception(self) -> BaseException | None:
        """Get task exception."""
        return self._task.exception() if self._task else None

    async def __aenter__(self) -> "Task[T]":
        self.create()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._task and not self._task.done():
            await self._task


def create_task(
    coro: Coroutine[Any, Any, T],
    *,
    name: str | None = None,
    loop: asyncio.AbstractEventLoop | None = None,
    context: contextvars.Context | None = None,
) -> asyncio.Task[T]:
    """Create a task with enhanced options."""
    if loop is None:
        loop = asyncio.get_running_loop()
    task = loop.create_task(coro, name=name, context=context)
    return task


class TaskGroup:
    """Manage a group of related tasks with structured concurrency."""

    def __init__(
        self,
        *,
        name: str | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        cancel_on_error: bool = True,
    ) -> None:
        self._name = name or "TaskGroup"
        self._loop = loop or asyncio.get_running_loop()
        self._cancel_on_error = cancel_on_error
        self._tasks: set[asyncio.Task] = set()
        self._completed: list[asyncio.Task] = []
        self._failed: list[asyncio.Task] = []
        self._entered = False
        self._exiting = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def tasks(self) -> set[asyncio.Task]:
        return self._tasks

    @property
    def completed_tasks(self) -> list[asyncio.Task]:
        return self._completed

    @property
    def failed_tasks(self) -> list[asyncio.Task]:
        return self._failed

    def create_task(
        self, coro: Coroutine[Any, Any, T], *, name: str | None = None
    ) -> asyncio.Task[T]:
        """Create and track a task in this group."""
        task = self._loop.create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._task_done)
        return task

    def _task_done(self, task: asyncio.Task) -> None:
        """Callback when a task completes."""
        self._tasks.discard(task)
        if task.cancelled():
            pass
        elif task.exception():
            self._failed.append(task)
            if self._cancel_on_error and not self._exiting:
                self.cancel_all()
        else:
            self._completed.append(task)

    def cancel_all(self, msg: str | None = None) -> int:
        """Cancel all tasks in the group."""
        count = 0
        for task in self._tasks:
            if task.cancel(msg):
                count += 1
        return count

    async def wait_all(self, return_exceptions: bool = False) -> list[Any]:
        """Wait for all tasks to complete."""
        if not self._tasks:
            return []
        results = await asyncio.gather(*self._tasks, return_exceptions=return_exceptions)
        return list(results)

    async def __aenter__(self) -> "TaskGroup":
        self._entered = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        self._exiting = True
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._entered = False
        return False


def create_task_group(
    *,
    name: str | None = None,
    loop: asyncio.AbstractEventLoop | None = None,
    cancel_on_error: bool = True,
) -> TaskGroup:
    """Create a new task group."""
    return TaskGroup(name=name, loop=loop, cancel_on_error=cancel_on_error)


async def gather(
    *coros: Coroutine[Any, Any, T],
    return_exceptions: bool = False,
    max_concurrent: int | None = None,
) -> list[T | BaseException]:
    """Enhanced gather with concurrency limit."""
    if max_concurrent is None:
        return await asyncio.gather(*coros, return_exceptions=return_exceptions)

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


async def wait(
    futures: list[Awaitable[T]],
    *,
    timeout: float | None = None,
    return_when: str = asyncio.ALL_COMPLETED,
) -> tuple[set[asyncio.Task], set[asyncio.Task]]:
    """Wait for futures with enhanced options."""
    tasks = [asyncio.ensure_future(f) for f in futures]
    done, pending = await asyncio.wait(tasks, timeout=timeout, return_when=return_when)
    return done, pending


async def wait_for(awaitable: Awaitable[T], timeout: float) -> T:
    """Wait for a single awaitable with timeout."""
    return await asyncio.wait_for(awaitable, timeout=timeout)


async def demonstrate_tasks() -> None:
    """🎓 Task management tutorial for asyncio beginners.

    This function builds on the coroutine foundation and teaches the next
    critical layer: ``asyncio.Task``. While coroutines are the units of
    async execution, Tasks provide scheduling, cancellation, and result
    management on the event loop.

    TASK vs COROUTINE:
    ------------------
    - A **coroutine** is a pausable function defined with ``async def``.
    - A **Task** is ``asyncio.Task`` that wraps a coroutine and schedules
      it on the event loop. You create one with ``asyncio.create_task(coro)``.

    Key differences:
    * Tasks can be cancelled via ``task.cancel()``
    * Tasks have ``.result()`` and ``.exception()`` methods
    * Tasks appear in ``asyncio.all_tasks()``
    * Tasks have names for debugging (default: ``"Task-<N>"``)
    * Tasks support done callbacks via ``add_done_callback()``

    COOPERATIVE CANCELLATION:
    ------------------------
    Task cancellation in asyncio is *cooperative*, not preemptive.
    When you ``task.cancel()``, an ``asyncio.CancelledError`` is thrown
    into the task at the next ``await`` point. The task can catch this
    and clean up, or re-raise it to terminate. This is safer than
    preemptive threading cancellation which can leave resources locked.

    WHAT YOU'LL LEARN:
    ------------------
    1. **``asyncio.create_task()``** - How to schedule a coroutine as a Task
    2. **Named tasks** - Identifying tasks in stack traces and debugging
    3. **Task cancellation** - The cooperative way to stop a task
    4. **Exception handling** - How exceptions propagate from tasks
    5. **``asyncio.gather()``** - Running multiple tasks concurrently
    6. **``asyncio.wait()``** - More flexible waiting with
       ``FIRST_COMPLETED``/``ALL_COMPLETED``
    7. **``asyncio.as_completed()``** - Iterating over results as they finish
    8. **Custom TaskManager** - Tracking and managing your own tasks

    RUNNABLE EXAMPLE:
    -----------------
    The function demonstrates all key concepts with live ``print()`` output,
    including:
    * Creating and naming tasks with ``asyncio.create_task()``
    * Waiting for completion with ``await task``
    * Handling cancellation gracefully (try/except ``CancelledError``)
    * Using ``asyncio.gather()`` for parallel execution
    * Collecting results as they complete with ``as_completed``
    * A custom TaskManager for tracking multiple tasks simultaneously

    SEE ALSO:
    ---------
    - ``demonstrate_coroutines()`` - the prerequisite foundation
    - ``asyncio.gather()`` - documented Python standard library function
    - ``asyncio.wait()`` - lower-level waiting API
    - ``asyncio.as_completed()`` - iterate results as they finish
    - `Real Python: Python Concurrency <https://realpython.com/asyncio-python/#task-cancellation>`_
    """
    print("=== Task Basics ===")
    await create_task_example()

    print("\n=== Named Tasks ===")
    await task_with_name()

    print("\n=== Task Cancellation ===")
    await task_cancellation()

    print("\n=== Exception Handling ===")
    await task_exception_handling()

    print("\n=== Result Handling ===")
    await task_result_handling()

    print("\n=== Gather ===")
    await gather_example()

    print("\n=== Wait ===")
    await wait_example()

    print("\n=== As Completed ===")
    await as_completed_example()

    print("\n=== TaskManager ===")
    manager = TaskManager()

    async def worker(name: str) -> str:
        await asyncio.sleep(0.05)
        return name

    for i in range(5):
        manager.create_task(worker(f"worker-{i}"))

    results = await manager.wait_all()
    print(f"Manager results: {results}")


async def create_task_example() -> None:
    """Basic task creation and execution."""
    async def worker(name: str, delay: float) -> str:
        await asyncio.sleep(delay)
        return f"{name} done"

    # Create tasks - they start running immediately
    task1 = asyncio.create_task(worker("Task-1", 0.1))
    task2 = asyncio.create_task(worker("Task-2", 0.05))

    # Wait for both
    result1 = await task1
    result2 = await task2
    print(f"Results: {result1}, {result2}")


async def task_with_name() -> None:
    """Tasks can have names for debugging."""
    async def named_worker() -> str:
        await asyncio.sleep(0.01)
        current = asyncio.current_task()
        return f"Running as: {current.get_name() if current else 'unknown'}"

    task = asyncio.create_task(named_worker(), name="MyWorker")
    print(f"Task name: {task.get_name()}")
    result = await task
    print(result)


async def task_cancellation() -> None:
    """Demonstrates task cancellation."""
    async def long_running() -> str:
        try:
            for i in range(10):
                print(f"Working... {i}")
                await asyncio.sleep(0.1)
            return "Completed"
        except asyncio.CancelledError:
            print("Task was cancelled!")
            raise

    task = asyncio.create_task(long_running())
    await asyncio.sleep(0.25)  # Let it run a bit
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        print("Cancellation caught")


async def task_exception_handling() -> None:
    """Handling exceptions in tasks."""
    async def failing_task() -> str:
        await asyncio.sleep(0.01)
        raise RuntimeError("Task failed!")

    async def successful_task() -> str:
        await asyncio.sleep(0.01)
        return "Success"

    # Exception in task
    task = asyncio.create_task(failing_task())
    try:
        await task
    except RuntimeError as e:
        print(f"Caught: {e}")

    # Exception with gather
    results = await asyncio.gather(
        successful_task(), failing_task(), successful_task(), return_exceptions=True
    )
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Task {i} failed: {result}")
        else:
            print(f"Task {i} succeeded: {result}")


async def task_result_handling() -> None:
    """Various ways to get task results."""
    async def compute(value: int) -> int:
        await asyncio.sleep(0.01)
        return value * 2

    task = asyncio.create_task(compute(21))

    # Method 1: await directly
    result = await task
    print(f"Direct await: {result}")

    # Method 2: task.result() (only after done)
    task2 = asyncio.create_task(compute(10))
    await task2
    print(f"task.result(): {task2.result()}")

    # Method 3: add_done_callback
    results = []

    def callback(t: asyncio.Task) -> None:
        results.append(t.result())

    task3 = asyncio.create_task(compute(5))
    task3.add_done_callback(callback)
    await task3
    print(f"Callback result: {results}")


async def gather_example() -> None:
    """asyncio.gather - run multiple awaitables concurrently."""
    async def fetch(id: int) -> dict:
        await asyncio.sleep(0.01 * id)
        return {"id": id, "data": f"item-{id}"}

    # Basic gather
    results = await asyncio.gather(*[fetch(i) for i in range(5)])
    print(f"Gather results: {results}")

    # gather with return_exceptions
    async def maybe_fail(id: int) -> str:
        if id == 2:
            raise ValueError(f"Failed {id}")
        await asyncio.sleep(0.01)
        return f"ok-{id}"

    results = await asyncio.gather(*[maybe_fail(i) for i in range(5)], return_exceptions=True)
    print(f"With exceptions: {results}")


async def wait_example() -> None:
    """asyncio.wait - more control over waiting."""
    async def worker(id: int, delay: float) -> str:
        await asyncio.sleep(delay)
        return f"worker-{id}"

    tasks = [
        asyncio.create_task(worker(1, 0.1)),
        asyncio.create_task(worker(2, 0.05)),
        asyncio.create_task(worker(3, 0.15)),
    ]

    # Wait for first to complete
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    print(f"First completed: {[t.result() for t in done]}")

    # Wait for all
    done, pending = await asyncio.wait(pending, return_when=asyncio.ALL_COMPLETED)
    print(f"All completed: {[t.result() for t in done]}")


async def as_completed_example() -> None:
    """asyncio.as_completed - iterate as tasks complete."""
    async def fetch(id: int) -> str:
        await asyncio.sleep(0.1 * (5 - id))  # Reverse order completion
        return f"result-{id}"

    tasks = [fetch(i) for i in range(5)]

    print("Results as completed:")
    async for coro in asyncio.as_completed(tasks):
        result = await coro
        print(f"  {result}")


def shield(awaitable: Awaitable[T]) -> Awaitable[T]:
    """Shield an awaitable from cancellation."""
    return asyncio.shield(awaitable)


class TaskPool:
    """Pool of reusable tasks for high-throughput scenarios."""

    def __init__(self, max_workers: int, *, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._max_workers = max_workers
        self._loop = loop or asyncio.get_running_loop()
        self._semaphore = asyncio.Semaphore(max_workers)
        self._active_tasks: set[asyncio.Task] = set()
        self._queue: asyncio.Queue = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._closed = False

    async def start(self) -> None:
        """Start worker tasks."""
        for i in range(self._max_workers):
            worker = self._loop.create_task(self._worker(f"worker-{i}"))
            self._workers.append(worker)

    async def _worker(self, name: str) -> None:
        while not self._closed:
            try:
                coro, future = await self._queue.get()
                if coro is None:
                    break
                async with self._semaphore:
                    try:
                        result = await coro
                        if not future.done():
                            future.set_result(result)
                    except asyncio.CancelledError:
                        if not future.done():
                            future.cancel()
                        raise
                    except BaseException as e:
                        if not future.done():
                            future.set_exception(e)
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    async def submit(self, coro: Coroutine[Any, Any, T]) -> T:
        """Submit a coroutine to the pool."""
        if self._closed:
            raise RuntimeError("Task pool is closed")
        future = self._loop.create_future()
        await self._queue.put((coro, future))
        return await future

    async def close(self) -> None:
        """Close the pool gracefully."""
        self._closed = True
        for _ in self._workers:
            await self._queue.put((None, None))
        await asyncio.gather(*self._workers, return_exceptions=True)

    async def __aenter__(self) -> "TaskPool":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()


@asynccontextmanager
async def task_scope(*, name: str | None = None, cancel_on_error: bool = True):
    """Context manager for structured task scope."""
    group = create_task_group(name=name, cancel_on_error=cancel_on_error)
    try:
        yield group
    finally:
        if group.tasks:
            await asyncio.gather(*group.tasks, return_exceptions=True)


async def run_in_background(coro: Coroutine[Any, Any, T]) -> asyncio.Task[T]:
    """Run coroutine in background without awaiting."""
    return asyncio.create_task(coro)


async def run_periodic(
    coro: Callable[[], Coroutine[Any, Any, Any]],
    interval: float,
    *,
    name: str | None = None,
    loop: asyncio.AbstractEventLoop | None = None,
) -> asyncio.Task:
    """Run a coroutine periodically."""
    loop = loop or asyncio.get_running_loop()

    async def periodic_runner() -> None:
        while True:
            try:
                await coro()
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
            await asyncio.sleep(interval)

    return loop.create_task(periodic_runner(), name=name or "periodic")


class TaskManager:
    """Manage background tasks with lifecycle control."""

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    def create_task(self, coro: Coroutine[Any, Any, Any], *, name: str | None = None) -> asyncio.Task:
        """Create and start a background task."""
        task = asyncio.create_task(coro, name=name)
        key = name or str(id(task))
        self._tasks[key] = task
        task.add_done_callback(lambda _task: self._tasks.pop(key, None))
        return task

    async def start(self, name: str, coro: Coroutine[Any, Any, Any]) -> asyncio.Task:
        """Start a named background task."""
        async with self._lock:
            if name in self._tasks:
                raise ValueError(f"Task '{name}' already running")
            task = asyncio.create_task(coro, name=name)
            self._tasks[name] = task
            task.add_done_callback(lambda _task: self._tasks.pop(name, None))
            return task

    async def stop(self, name: str, timeout: float | None = None) -> bool:
        """Stop a background task."""
        async with self._lock:
            task = self._tasks.get(name)
            if not task:
                return False
            task.cancel()
            with suppress(TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=timeout)
            return True

    async def stop_all(self, timeout: float | None = None) -> list[str]:
        """Stop all background tasks."""
        names = list(self._tasks.keys())
        for name in names:
            await self.stop(name, timeout)
        return names

    async def wait_all(self, timeout: float | None = None) -> list[Any]:
        """Wait for all tracked tasks to complete.

        A cancelled task contributes ``None`` and a failed task contributes its
        exception, mirroring ``asyncio.gather(return_exceptions=True)``.
        """
        results: list[Any] = []
        for task in list(self._tasks.values()):
            try:
                results.append(await task)
            except asyncio.CancelledError:
                if task.cancelled():
                    results.append(None)
                else:
                    raise
            except Exception as e:
                results.append(e)
        return results

    def get_task(self, name: str) -> asyncio.Task | None:
        """Get a background task by name."""
        return self._tasks.get(name)

    def list_tasks(self) -> list[str]:
        """List all running background task names."""
        return list(self._tasks.keys())
