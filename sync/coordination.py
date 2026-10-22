"""
Advanced Coordination Primitives

High-level synchronization patterns for complex async workflows.
"""

import asyncio
import time
import uuid
from asyncio import QueueEmpty
from collections import defaultdict
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar, Union

T = TypeVar("T")
R = TypeVar("R")


class WaitGroup:
    """WaitGroup for coordinating multiple goroutines (Go-style)."""

    def __init__(self) -> None:
        self._count = 0
        self._event = asyncio.Event()
        self._event.set()
        self._lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()

    def add(self, delta: int = 1) -> None:
        """Add delta to the wait group counter."""
        if delta < 0:
            raise ValueError("delta must be positive")
        self._count += delta
        self._event.clear()

    def done(self) -> None:
        """Decrement the wait group counter."""
        self._count -= 1
        if self._count <= 0:
            self._count = 0
            self._event.set()

    async def wait(self) -> None:
        """Wait until the counter reaches zero."""
        await self._event.wait()

    async def wait_for(self, timeout: float) -> bool:
        """Wait with timeout. Returns True if completed, False if timed out."""
        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout)
            return True
        except TimeoutError:
            return False

    @asynccontextmanager
    async def worker(self) -> AsyncGenerator[None, None]:
        """Context manager for a worker."""
        self.add(1)
        try:
            yield
        finally:
            self.done()

    def create_task(self, coro: Coroutine[Any, Any, T]) -> asyncio.Task[T]:
        """Create a task tracked by this wait group."""
        self.add(1)
        task = asyncio.create_task(coro)
        task.add_done_callback(lambda _: self.done())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def wait_all(self, timeout: float | None = None) -> list[Any]:
        """Wait for all tasks and return results."""
        if not self._tasks:
            return []

        if timeout is not None:
            done, pending = await asyncio.wait(
                self._tasks, timeout=timeout, return_when=asyncio.ALL_COMPLETED
            )
            for task in pending:
                task.cancel()
        else:
            done = await asyncio.gather(*self._tasks, return_exceptions=True)
            return list(done)

        results = []
        for task in done:
            try:
                results.append(task.result())
            except Exception as e:
                results.append(e)
        return results


class CountDownLatch:
    """CountDownLatch - waits for a count to reach zero."""

    def __init__(self, count: int) -> None:
        if count < 0:
            raise ValueError("count must be >= 0")
        self._count = count
        self._event = asyncio.Event()
        if count == 0:
            self._event.set()

    @property
    def count(self) -> int:
        return self._count

    def count_down(self) -> None:
        """Decrement the count."""
        if self._count > 0:
            self._count -= 1
            if self._count == 0:
                self._event.set()

    async def wait(self) -> None:
        """Wait until count reaches zero."""
        await self._event.wait()

    async def wait_for(self, timeout: float) -> bool:
        """Wait with timeout."""
        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout)
            return True
        except TimeoutError:
            return False


class CyclicBarrier:
    """CyclicBarrier - reusable barrier for a fixed number of parties."""

    def __init__(self, parties: int, action: Callable[[], Any] | None = None) -> None:
        if parties <= 0:
            raise ValueError("parties must be > 0")
        self._parties = parties
        self._action = action
        self._count = parties
        self._generation = 0
        self._lock = asyncio.Lock()
        self._event = asyncio.Event()

    @property
    def parties(self) -> int:
        return self._parties

    @property
    def number_waiting(self) -> int:
        return self._parties - self._count

    async def wait(self) -> int:
        """Wait for all parties to arrive. Returns arrival index."""
        async with self._lock:
            gen = self._generation
            self._count -= 1
            index = self._parties - self._count

            if self._count == 0:
                # Last party - execute action and reset
                if self._action:
                    try:
                        if asyncio.iscoroutinefunction(self._action):
                            await self._action()
                        else:
                            self._action()
                    except Exception:
                        pass

                self._count = self._parties
                self._generation += 1
                self._event.set()
                self._event.clear()
                return index

            # Not last party - wait
            while gen == self._generation:
                await self._event.wait()

            return index

    async def reset(self) -> None:
        """Reset the barrier."""
        async with self._lock:
            self._count = self._parties
            self._generation += 1
            self._event.set()
            self._event.clear()

    def is_broken(self) -> bool:
        """Check if barrier is broken (not applicable in this implementation)."""
        return False


class Phaser:
    """Phaser - flexible barrier with dynamic party registration."""

    def __init__(self, parties: int = 0) -> None:
        self._phase = 0
        self._registered = parties
        self._arrived = 0
        self._unarrived = parties
        self._lock = asyncio.Lock()
        self._events: dict[int, asyncio.Event] = defaultdict(asyncio.Event)
        self._terminated = False

    @property
    def phase(self) -> int:
        return self._phase

    @property
    def registered_parties(self) -> int:
        return self._registered

    @property
    def arrived_parties(self) -> int:
        return self._arrived

    @property
    def unarrived_parties(self) -> int:
        return self._unarrived

    def register(self, parties: int = 1) -> int:
        """Register new parties."""
        if self._terminated:
            raise RuntimeError("Phaser is terminated")
        if parties < 0:
            raise ValueError("parties must be >= 0")

        self._registered += parties
        self._unarrived += parties
        return self._phase

    def arrive(self, parties: int = 1) -> int:
        """Arrive without waiting."""
        if parties < 0:
            raise ValueError("parties must be >= 0")
        if parties > self._unarrived:
            raise ValueError("Too many arrivals")

        self._arrived += parties
        self._unarrived -= parties

        if self._unarrived == 0:
            self._advance_phase()

        return self._phase

    def arrive_and_deregister(self) -> int:
        """Arrive and deregister."""
        self.arrive(1)
        self._registered -= 1
        return self._phase

    async def await_advance(self, phase: int) -> int:
        """Wait for phase to advance."""
        if phase < self._phase:
            return self._phase

        event = self._events[phase]
        await event.wait()
        return self._phase

    async def arrive_and_await_advance(self, parties: int = 1) -> int:
        """Arrive and wait for phase to advance."""
        self.arrive(parties)
        return await self.await_advance(self._phase)

    def _advance_phase(self) -> None:
        self._phase += 1
        self._arrived = 0
        self._unarrived = self._registered

        # Notify waiters
        old_event = self._events.get(self._phase - 1)
        if old_event:
            old_event.set()

        self._events[self._phase] = asyncio.Event()

    def force_termination(self) -> None:
        """Force termination of the phaser."""
        self._terminated = True
        for event in self._events.values():
            event.set()

    def is_terminated(self) -> bool:
        return self._terminated


class FutureGroup(Generic[T]):
    """Group of futures with collective operations."""

    def __init__(self, *futures: Awaitable[T]) -> None:
        self._futures: list[asyncio.Future] = [asyncio.ensure_future(f) for f in futures]
        self._results: dict[int, Any] = {}
        self._exceptions: dict[int, BaseException] = {}
        self._completed = asyncio.Event()
        self._callbacks: list[Callable[[FutureGroup], None]] = []

        for i, fut in enumerate(self._futures):
            fut.add_done_callback(lambda f, idx=i: self._on_done(idx, f))

    def _on_done(self, index: int, future: asyncio.Future) -> None:
        try:
            self._results[index] = future.result()
        except BaseException as e:
            self._exceptions[index] = e

        if len(self._results) + len(self._exceptions) == len(self._futures):
            self._completed.set()
            for cb in self._callbacks:
                with suppress(Exception):
                    cb(self)

    def add(self, future: Awaitable[T]) -> int:
        """Add a future to the group."""
        fut = asyncio.ensure_future(future)
        index = len(self._futures)
        self._futures.append(fut)
        fut.add_done_callback(lambda f, idx=index: self._on_done(idx, f))

        if len(self._results) + len(self._exceptions) == len(self._futures) - 1:
            # Was already complete
            self._completed.set()
        return index

    async def wait_all(self, timeout: float | None = None) -> list[T | BaseException]:
        """Wait for all futures to complete."""
        if timeout is not None:
            await asyncio.wait_for(self._completed.wait(), timeout=timeout)
        else:
            await self._completed.wait()

        results = []
        for i in range(len(self._futures)):
            if i in self._exceptions:
                results.append(self._exceptions[i])
            else:
                results.append(self._results[i])
        return results

    async def wait_any(self, timeout: float | None = None) -> tuple[int, T | BaseException]:
        """Wait for any future to complete."""
        done, _pending = await asyncio.wait(
            self._futures, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
        )

        if not done:
            raise TimeoutError()

        task = done.pop()
        index = self._futures.index(task)

        if index in self._exceptions:
            return index, self._exceptions[index]
        return index, self._results[index]

    def add_done_callback(self, callback: Callable[["FutureGroup"], None]) -> None:
        self._callbacks.append(callback)

    @property
    def done(self) -> bool:
        return self._completed.is_set()

    @property
    def completed_count(self) -> int:
        return len(self._results) + len(self._exceptions)

    @property
    def total_count(self) -> int:
        return len(self._futures)


class CompletionService(Generic[T, R]):
    """Service that separates submission from completion."""

    def __init__(
        self,
        executor: Callable[[T], Coroutine[Any, Any, R]] | None = None,
        max_workers: int | None = None,
    ) -> None:
        self._executor = executor
        self._max_workers = max_workers
        self._queue: asyncio.Queue = asyncio.Queue()
        self._results: asyncio.Queue = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._submitted = 0
        self._completed = 0
        self._closed = False
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start worker tasks."""
        if self._executor is None:
            raise ValueError("Executor function required")

        workers = self._max_workers or 1
        for i in range(workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self._workers.append(worker)

    async def _worker(self, name: str) -> None:
        while True:
            try:
                item = await self._queue.get()
                if item is None:
                    break

                task_id, value = item
                try:
                    result = await self._executor(value)
                    await self._results.put((task_id, result, None))
                except BaseException as e:
                    await self._results.put((task_id, None, e))
                finally:
                    self._completed += 1
                    self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    async def submit(self, value: T) -> int:
        """Submit a task for execution."""
        if self._closed:
            raise RuntimeError("CompletionService is closed")

        task_id = self._submitted
        self._submitted += 1
        await self._queue.put((task_id, value))
        return task_id

    async def take(self, timeout: float | None = None) -> tuple[int, R]:
        """Take the next completed result."""
        try:
            if timeout is not None:
                task_id, result, exc = await asyncio.wait_for(self._results.get(), timeout=timeout)
            else:
                task_id, result, exc = await self._results.get()

            if exc:
                raise exc
            return task_id, result
        except TimeoutError:
            raise QueueEmpty("No completed tasks available")

    async def poll(self) -> tuple[int, R] | None:
        """Poll for a completed result without blocking."""
        try:
            task_id, result, exc = self._results.get_nowait()
            if exc:
                raise exc
            return task_id, result
        except asyncio.QueueEmpty:
            return None

    async def close(self) -> None:
        """Close the service."""
        self._closed = True
        for _ in self._workers:
            await self._queue.put(None)
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    @property
    def submitted_count(self) -> int:
        return self._submitted

    @property
    def completed_count(self) -> int:
        return self._completed

    @property
    def pending_count(self) -> int:
        return self._submitted - self._completed

    async def __aenter__(self) -> "CompletionService":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()


class AsyncSemaphore:
    """Async semaphore with async context manager support."""

    def __init__(self, value: int = 1) -> None:
        if value < 0:
            raise ValueError("value must be >= 0")
        self._value = value
        self._waiters: list[asyncio.Future] = []
        self._lock = asyncio.Lock()

    @property
    def value(self) -> int:
        return self._value

    async def acquire(self) -> None:
        async with self._lock:
            if self._value > 0:
                self._value -= 1
                return

            waiter = asyncio.get_running_loop().create_future()
            self._waiters.append(waiter)

        try:
            await waiter
        except:
            async with self._lock:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)
            raise

    async def release(self) -> None:
        async with self._lock:
            if self._waiters:
                waiter = self._waiters.pop(0)
                if not waiter.done():
                    waiter.set_result(None)
            else:
                self._value += 1

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.release()


class AsyncLock:
    """Simple async lock implementation."""

    def __init__(self) -> None:
        self._locked = False
        self._waiters: list[asyncio.Future] = []
        self._lock = asyncio.Lock()

    @property
    def locked(self) -> bool:
        return self._locked

    async def acquire(self) -> bool:
        async with self._lock:
            if not self._locked:
                self._locked = True
                return True

            waiter = asyncio.get_running_loop().create_future()
            self._waiters.append(waiter)

        try:
            await waiter
            return True
        except:
            async with self._lock:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)
            raise

    async def release(self) -> None:
        async with self._lock:
            if self._waiters:
                waiter = self._waiters.pop(0)
                if not waiter.done():
                    waiter.set_result(None)
            else:
                self._locked = False

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.release()


class AsyncEvent:
    """Simple async event implementation."""

    def __init__(self) -> None:
        self._set = False
        self._waiters: list[asyncio.Future] = []
        self._lock = asyncio.Lock()

    @property
    def is_set(self) -> bool:
        return self._set

    async def set(self) -> None:
        async with self._lock:
            self._set = True
            for waiter in self._waiters:
                if not waiter.done():
                    waiter.set_result(True)
            self._waiters.clear()

    def clear(self) -> None:
        self._set = False

    async def wait(self) -> bool:
        async with self._lock:
            if self._set:
                return True

            waiter = asyncio.get_running_loop().create_future()
            self._waiters.append(waiter)

        try:
            return await waiter
        except:
            async with self._lock:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)
            raise

    async def __aenter__(self):
        await self.wait()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@dataclass
class TaskResult(Generic[T]):
    """Result of a task execution."""

    task_id: str
    success: bool
    result: T | None = None
    exception: BaseException | None = None
    start_time: float = field(default_factory=time.monotonic)
    end_time: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.monotonic() - self.start_time


class TaskExecutor(Generic[T, R]):
    """Execute tasks with controlled concurrency."""

    def __init__(
        self,
        max_concurrent: int,
        func: Callable[[T], Coroutine[Any, Any, R]],
        *,
        name: str | None = None,
    ) -> None:
        self._max_concurrent = max_concurrent
        self._func = func
        self._name = name or f"TaskExecutor-{id(self)}"
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._results: list[TaskResult[R]] = []
        self._lock = asyncio.Lock()
        self._running = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def running_count(self) -> int:
        return self._running

    @property
    def completed_count(self) -> int:
        return len(self._results)

    async def execute(self, item: T) -> R:
        """Execute a single item."""
        async with self._semaphore:
            async with self._lock:
                self._running += 1

            task_id = str(uuid.uuid4())
            _start = time.monotonic()

            try:
                result = await self._func(item)
                task_result = TaskResult(
                    task_id=task_id, success=True, result=result, end_time=time.monotonic()
                )
                return result
            except BaseException as e:
                task_result = TaskResult(
                    task_id=task_id, success=False, exception=e, end_time=time.monotonic()
                )
                raise
            finally:
                async with self._lock:
                    self._running -= 1
                    self._results.append(task_result)

    async def execute_batch(self, items: list[T]) -> list[TaskResult[R]]:
        """Execute a batch of items."""
        tasks = [self.execute(item) for item in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        task_results = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                task_results.append(
                    TaskResult(
                        task_id=str(uuid.uuid4()),
                        success=False,
                        exception=result,
                        end_time=time.monotonic(),
                    )
                )
            else:
                # Result already captured in execute
                pass

        return self._results[-len(items) :] if items else []

    def get_results(self) -> list[TaskResult[R]]:
        return self._results.copy()

    def clear_results(self) -> None:
        self._results.clear()


async def run_with_semaphore(semaphore: asyncio.Semaphore, coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine with a semaphore."""
    async with semaphore:
        return await coro


async def limited_gather(*coros: Coroutine[Any, Any, T], max_concurrent: int) -> list[T]:
    """Gather with concurrency limit."""
    semaphore = asyncio.Semaphore(max_concurrent)
    return await asyncio.gather(*[run_with_semaphore(semaphore, c) for c in coros])


class BarrierGroup:
    """Group of barriers for multi-phase synchronization."""

    def __init__(self, phases: int, parties: int) -> None:
        self._phases = phases
        self._parties = parties
        self._barriers = [CyclicBarrier(parties) for _ in range(phases)]
        self._current_phase = 0

    async def wait(self, phase: int | None = None) -> int:
        """Wait at a specific phase."""
        if phase is None:
            phase = self._current_phase
        if phase >= self._phases:
            raise ValueError(f"Phase {phase} out of range")
        return await self._barriers[phase].wait()

    async def advance(self) -> None:
        """Advance to next phase."""
        if self._current_phase < self._phases - 1:
            self._current_phase += 1
        else:
            raise RuntimeError("Already at final phase")

    def reset(self) -> None:
        """Reset all barriers."""
        for barrier in self._barriers:
            asyncio.create_task(barrier.reset())
        self._current_phase = 0


@asynccontextmanager
async def synchronized(lock: Union[asyncio.Lock, "AsyncLock"]):
    """Context manager for any lock-like object."""
    await lock.acquire()
    try:
        yield
    finally:
        lock.release()
