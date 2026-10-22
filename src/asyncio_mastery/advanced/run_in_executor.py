"""
Run in Executor - Bridge sync and async code.

================================================================================
WHY DOES THIS MODULE EXIST?
================================================================================

asyncio runs on ONE thread, so any blocking call (``time.sleep``, ``requests.get``,
heavy computation) freezes the ENTIRE event loop -- no other task can run.
``run_in_executor`` is the bridge between the async world and the sync world:
it hands a blocking function to a pool of worker threads (or processes) and
gives you back an awaitable result, so the event loop stays responsive.

MENTAL MODEL
------------
``run_in_executor`` is like delegating a heavy chore to a worker next door.
You (the event loop) shout the chore over the fence, keep doing your own
work, and the worker shouts back when done. ``await`` is you listening for
that shout.

THREADS vs PROCESSES -- WHICH POOL?
-----------------------------------
- **ThreadPoolExecutor**: for *I/O-bound* blocking work (file reads, network
  calls via sync libraries). Threads share memory; Python's GIL is released
  during I/O so they truly overlap.
- **ProcessPoolExecutor**: for *CPU-bound* work (number crunching, image
  processing). Separate processes each have their own Python interpreter,
  sidestepping the GIL. Arguments must be picklable!

WHEN YOU NEED THIS
------------------
- Using a sync-only library (``requests``, old DB drivers) inside async code.
- Speeding up CPU-heavy calculations across multiple cores.

MINIMAL EXAMPLE
---------------
>>> import asyncio, time
>>> def blocking(): time.sleep(1); return "done"
>>> async def main():
...     loop = asyncio.get_running_loop()
...     result = await loop.run_in_executor(None, blocking)  # None = default pool
...     print(result)

Run the full demo with: ``python -m asyncio_mastery.advanced.run_in_executor``
"""

import asyncio
import concurrent.futures
import time
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass
class ExecutorStats:
    """Simple counters describing how a pool has been used.

    Attributes:
        submitted: Jobs handed to the pool.
        completed: Jobs that finished successfully.
        failed: Jobs that raised an exception.
        total_time: Seconds spent inside ``run_in_executor`` awaits.
    """

    submitted: int = 0
    completed: int = 0
    failed: int = 0
    total_time: float = 0.0


class ThreadPool:
    """A thread pool with async-friendly submit and usage statistics.

    Wraps ``concurrent.futures.ThreadPoolExecutor`` so a *sync* function can
    be ``await``ed from async code. Use this for I/O-bound blocking work.

    Example:
        >>> async with ThreadPool(max_workers=4) as pool:
        ...     result = await pool.submit(requests.get, "https://example.com")

    Args:
        max_workers: Number of worker threads (None = Python's default,
            roughly ``min(32, cpu_count + 4)``).
        name: Prefix for worker thread names (helps when debugging).
    """

    def __init__(self, max_workers: int | None = None, *, name: str = "thread-pool") -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=name)
        self._stats = ExecutorStats()
        self._loop = asyncio.get_event_loop()

    async def submit(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Run ``func(*args, **kwargs)`` on a worker thread and await the result.

        Args:
            func: A *synchronous* (possibly blocking) callable.
            *args/**kwargs: Forwarded to ``func``.

        Returns:
            Whatever ``func`` returns.

        Raises:
            Exception: Any exception raised by ``func`` is re-raised here at
                the await point -- the bridge carries errors across too.
        """
        self._stats.submitted += 1
        start = time.monotonic()
        try:
            # THE BRIDGE: run_in_executor schedules the sync lambda on a
            # worker thread and returns an awaitable future. The event loop
            # stays free to run other tasks while the thread blocks.
            result = await self._loop.run_in_executor(self._executor, lambda: func(*args, **kwargs))
            self._stats.completed += 1
            return result
        except Exception:
            self._stats.failed += 1
            raise
        finally:
            self._stats.total_time += time.monotonic() - start

    def submit_sync(
        self, func: Callable[..., T], *args: Any, **kwargs: Any
    ) -> concurrent.futures.Future[T]:
        """Submit WITHOUT awaiting; returns a concurrent.futures.Future.

        Useful when you want raw fire-and-forget semantics or plan to use
        blocking ``future.result()`` yourself.
        """
        return self._executor.submit(func, *args, **kwargs)

    async def map(self, func: Callable[..., T], *iterables: Any) -> AsyncIterator[T]:
        """Async generator yielding ``func(*args)`` results as they complete.

        Unlike built-in ``map``, results arrive in COMPLETION order (fastest
        first), thanks to ``asyncio.as_completed``.

        Yields:
            Results of applying ``func`` to each zipped argument tuple.
        """
        futures = [self._executor.submit(func, *args) for args in zip(*iterables)]
        for fut in asyncio.as_completed(futures):
            # wrap_future adapts a concurrent.futures.Future into an
            # asyncio-awaitable -- another sync/async bridge.
            yield await asyncio.wrap_future(fut)

    def shutdown(self, wait: bool = True) -> None:
        """Stop the pool.

        Args:
            wait: If True, block until in-flight jobs finish; if False,
                return immediately (queued-but-unstarted jobs are dropped).
        """
        self._executor.shutdown(wait=wait)

    @property
    def stats(self) -> ExecutorStats:
        """Usage statistics for this pool."""
        return self._stats

    async def __aenter__(self) -> "ThreadPool":
        return self

    async def __aexit__(self, *args: Any) -> None:
        self.shutdown()


class ProcessPool:
    """A process pool for CPU-bound work, with async submit and stats.

    Each worker is a separate Python process, so the GIL doesn't limit
    parallelism. The trade-off: the function and its arguments must be
    PICKLABLE (top-level functions work; lambdas/closures generally don't),
    and starting processes is slower than starting threads.

    Args:
        max_workers: Number of worker processes (None defaults to cpu_count).
        name: Label (kept for API symmetry with ThreadPool).
    """

    def __init__(self, max_workers: int | None = None, *, name: str = "process-pool") -> None:
        self._executor = ProcessPoolExecutor(max_workers=max_workers)
        self._stats = ExecutorStats()
        self._loop = asyncio.get_event_loop()

    async def submit(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Run ``func`` in a worker process and await its (pickled) result.

        Raises:
            Exception: Exceptions are serialized across the process boundary
                and re-raised here.
        """
        self._stats.submitted += 1
        start = time.monotonic()
        try:
            result = await self._loop.run_in_executor(self._executor, lambda: func(*args, **kwargs))
            self._stats.completed += 1
            return result
        except Exception:
            self._stats.failed += 1
            raise
        finally:
            self._stats.total_time += time.monotonic() - start

    def shutdown(self, wait: bool = True) -> None:
        """Stop the pool; ``wait=True`` blocks until workers finish."""
        self._executor.shutdown(wait=wait)

    @property
    def stats(self) -> ExecutorStats:
        """Usage statistics for this pool."""
        return self._stats

    async def __aenter__(self) -> "ProcessPool":
        return self

    async def __aexit__(self, *args: Any) -> None:
        self.shutdown()


# Convenience functions
async def run_in_executor(
    func: Callable[..., T],
    *args: Any,
    executor: ThreadPoolExecutor | ProcessPoolExecutor | None = None,
    **kwargs: Any,
) -> T:
    """Run a sync function in an executor without owning a pool object.

    Args:
        func: A synchronous callable.
        *args/**kwargs: Forwarded to ``func``.
        executor: Which pool to use; None means the event loop's DEFAULT
            thread pool (created on demand).

    Returns:
        ``func``'s return value.

    Example:
        >>> result = await run_in_executor(time.sleep, 1)
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, lambda: func(*args, **kwargs))


async def run_in_thread(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Shortcut: run ``func`` in the default thread pool.

    Ideal for blocking I/O (sync HTTP/DB libraries). New code may prefer the
    built-in ``asyncio.to_thread``, which does the same thing.
    """
    return await run_in_executor(func, *args, **kwargs)


async def run_in_process(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Shortcut: run ``func`` in a fresh process pool.

    A new :class:`ProcessPoolExecutor` is created and shut down around the
    call -- convenient for one-off jobs, but for many jobs create ONE pool
    with :func:`process_pool` and reuse it (process startup is expensive).
    """
    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor() as executor:
        return await loop.run_in_executor(executor, lambda: func(*args, **kwargs))


@asynccontextmanager
async def thread_pool(max_workers: int | None = None) -> AsyncIterator[ThreadPool]:
    """Async context manager yielding a ThreadPool that is shut down on exit.

    Example:
        >>> async with thread_pool(4) as pool:
        ...     await pool.submit(blocking_fn)
    """
    pool = ThreadPool(max_workers)
    try:
        yield pool
    finally:
        # finally guarantees the pool is shut down even if the block raised.
        pool.shutdown()


@asynccontextmanager
async def process_pool(max_workers: int | None = None) -> AsyncIterator[ProcessPool]:
    """Async context manager yielding a ProcessPool, shut down on exit."""
    pool = ProcessPool(max_workers)
    try:
        yield pool
    finally:
        pool.shutdown()


# Example blocking functions
def blocking_io(duration: float = 0.1) -> str:
    """Simulate blocking I/O by sleeping on a worker thread.

    ``time.sleep`` BLOCKS -- if called directly in the event loop it would
    freeze everything; that is exactly what executors protect you from.

    Args:
        duration: Seconds to block.

    Returns:
        A description of what "happened".
    """
    time.sleep(duration)
    return f"IO done in {duration}s"


def cpu_bound(n: int = 100000) -> int:
    """Simulate CPU-bound work (sum of squares).

    Args:
        n: Count up to this number.

    Returns:
        ``sum(i*i for i in range(n))``.
    """
    return sum(i * i for i in range(n))


async def demonstrate_executors() -> None:
    """Narrated tour: thread pool for I/O, process pool for CPU, mixing
    sync/async work, and pool.map.

    Run with: ``python -m asyncio_mastery.advanced.run_in_executor``.
    """
    print("=== Executors Demo ===")

    # Thread pool for I/O
    print("\n--- Thread Pool (I/O) ---")
    async with thread_pool(max_workers=4) as pool:
        # Submit multiple tasks
        tasks = [pool.submit(blocking_io, 0.1) for _ in range(8)]
        results = await asyncio.gather(*tasks)
        for r in results:
            print(f"  {r}")

        print(f"Stats: {pool.stats.submitted} submitted, {pool.stats.completed} completed")

    # Process pool for CPU
    print("\n--- Process Pool (CPU) ---")
    async with process_pool(max_workers=2) as pool:
        tasks = [pool.submit(cpu_bound, 50000) for _ in range(4)]
        results = await asyncio.gather(*tasks)
        for r in results:
            print(f"  Sum: {r}")

        print(f"Stats: {pool.stats.submitted} submitted, {pool.stats.completed} completed")

    # Mix sync and async
    print("\n--- Mixed Sync/Async ---")

    async def async_work() -> str:
        await asyncio.sleep(0.1)
        return "async done"

    async def mixed() -> None:
        # Run blocking in thread, async concurrently
        io_task = run_in_thread(blocking_io, 0.2)
        async_task = async_work()
        io_result, async_result = await asyncio.gather(io_task, async_task)
        print(f"  IO: {io_result}")
        print(f"  Async: {async_result}")

    await mixed()

    # Run sync function from async
    print("\n--- run_in_executor ---")
    result = await run_in_executor(cpu_bound, 20000)
    print(f"CPU result: {result}")

    # Map over iterable
    print("\n--- Map ---")
    async with thread_pool(max_workers=2) as pool:
        async for result in pool.map(lambda x: x * 2, range(5)):
            print(f"  {result}")


if __name__ == "__main__":
    asyncio.run(demonstrate_executors())
