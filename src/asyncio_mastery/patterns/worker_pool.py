"""
Worker Pool - Managed pool of worker tasks.

THE PROBLEM
-----------
You have a stream of jobs arriving over time (emails to send, images to
resize, files to parse), and you want a fixed number of workers chewing
through them — not thousands of uncontrolled tasks, and not one slow
sequential loop either.

REAL-WORLD ANALOGY
------------------
A **call center**: callers (jobs) wait in a phone queue, and a fixed
team of operators (workers) answers them one by one. No matter how many
people call, only N conversations happen at once. Hiring more operators
increases throughput; shrinking the queue applies **backpressure**
(callers get a busy signal / have to wait in line).

JARGON
------
* **Backpressure**: when submission outpaces processing, the bounded
  queue makes producers wait (suspension, not blocking) until a worker
  frees up capacity.

WHY ASYNC PRIMITIVES ARE NEEDED
-------------------------------
``asyncio.Queue`` mediates between submitters and workers: workers
``await queue.get()`` and sleep efficiently when idle; submitters
``await queue.put()`` and sleep efficiently when full. The event loop
fairly interleaves everything with zero busy-waiting.

WHEN TO USE IT
--------------
* You need bounded concurrency over an open-ended stream of work.
* Jobs should be tracked with status/timing metadata.
* You want a ``map``-style helper: submit a list, get results back,
  with only N items processed concurrently.

Minimal example
---------------
>>> import asyncio
>>> async def work(n: int) -> str:
...     return f"done-{n}"
>>> async def main() -> None:
...     async with worker_pool(work, max_workers=2) as pool:
...         print(await pool.map([1, 2, 3]))
>>> asyncio.run(main())
"""

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class JobStatus(Enum):
    """Lifecycle states a Job moves through.

    A job is born PENDING, becomes RUNNING when :meth:`Job.execute`
    starts, and ends in exactly one terminal state: COMPLETED, FAILED,
    or CANCELLED.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job(Generic[T, R]):
    """A unit of work plus its own tracking metadata.

    A Job bundles *what to run* (``func``, ``args``, ``kwargs``) with
    *what happened* (``status``, ``result``/``error``, timestamps). This
    lets you submit work and inspect its outcome later — essential when
    the worker that runs it is a different task from the one that
    submitted it.

    Type Parameters:
        T: The input type accepted by the worker function.
        R: The return type produced by the worker function.

    Attributes:
        func: The async function to execute.
        args: Positional arguments for ``func``.
        kwargs: Keyword arguments for ``func``.
        job_id: Short unique id (first 8 chars of a UUID), for logging.
        status: Current :class:`JobStatus` (PENDING at birth).
        result: The return value (set after COMPLETED).
        error: The exception raised (set after FAILED).
        created_at: ``time.monotonic()`` timestamp of creation.
        started_at: Timestamp when execution began (None until then).
        completed_at: Timestamp when execution ended (None until then).
    """

    func: Callable[[T], Awaitable[R]]
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    job_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: JobStatus = JobStatus.PENDING
    result: R | None = None
    error: Exception | None = None
    created_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    completed_at: float | None = None

    async def execute(self) -> R:
        """Execute the job.

        Records the RUNNING status and start time, calls the wrapped
        function, then records the terminal state and end time.

        Returns:
            R: Whatever the wrapped worker function returns.

        Raises:
            Exception: Re-raises whatever the worker function raised
            (the exception is also stored on :attr:`error` first).
        """
        self.status = JobStatus.RUNNING
        self.started_at = time.monotonic()
        try:
            self.result = await self.func(*self.args, **self.kwargs)
            self.status = JobStatus.COMPLETED
            return self.result
        except Exception as e:
            self.error = e
            self.status = JobStatus.FAILED
            raise
        finally:
            self.completed_at = time.monotonic()

    @property
    def duration(self) -> float | None:
        """Wall time of the job, if it has finished.

        Returns:
            float | None: Seconds between start and end, or None if the
            job has not both started and completed yet.
        """
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None


class WorkerPool(Generic[T, R]):
    """Pool of workers processing jobs.

    Submission goes through a bounded internal queue; a fixed set of
    worker tasks drains it. This bounds concurrency (no matter how much
    you submit, only ``max_workers`` jobs run at once).

    Type Parameters:
        T: Job input type. R: Job result type.

    Example:
        >>> pool = WorkerPool(process, max_workers=4)
        >>> await pool.start()
        >>> job = await pool.submit(42)
        >>> result = await job.execute()  # wait for this specific job
    """

    def __init__(
        self,
        worker_func: Callable[[T], Awaitable[R]],
        *,
        max_workers: int = 4,
        max_queue_size: int = 100,
        name: str = "worker-pool",
    ) -> None:
        """Configure the pool (workers are not started yet).

        Args:
            worker_func: Async function applied to each job's arguments.
            max_workers: How many jobs may run concurrently.
            max_queue_size: Queue capacity — once full, :meth:`submit`
                suspends (backpressure!) until a slot frees up.
            name: Prefix used when naming worker tasks.
        """
        self._worker_func = worker_func
        self._max_workers = max_workers
        self._max_queue_size = max_queue_size
        self._name = name
        self._queue: asyncio.Queue[Job[T, R]] = asyncio.Queue(maxsize=max_queue_size)
        self._workers: list[asyncio.Task] = []
        self._running = False
        self._stats = {"submitted": 0, "completed": 0, "failed": 0, "cancelled": 0}

    async def start(self) -> None:
        """Start worker tasks.

        Idempotent: a second call is a no-op. Spawns ``max_workers``
        background tasks that loop forever pulling jobs from the queue.
        """
        if self._running:
            return
        self._running = True
        for i in range(self._max_workers):
            worker = asyncio.create_task(self._worker_loop(i), name=f"{self._name}-worker-{i}")
            self._workers.append(worker)

    async def stop(self, timeout: float = 30.0) -> None:
        """Stop all workers.

        Cancels every worker task and waits (up to ``timeout``) for them
        to finish cancellation. Jobs still in the queue are abandoned.

        Args:
            timeout: Max seconds to wait for workers to wind down before
                giving up (``asyncio.wait_for`` raises TimeoutError).

        Raises:
            asyncio.TimeoutError: If workers do not stop within ``timeout``.
        """
        self._running = False
        # Cancel workers
        for worker in self._workers:
            worker.cancel()
        # Wait for completion
        if self._workers:
            await asyncio.wait_for(
                asyncio.gather(*self._workers, return_exceptions=True), timeout=timeout
            )
        self._workers.clear()

    async def submit(self, *args: Any, **kwargs: Any) -> Job[T, R]:
        """Submit job to pool.

        Wraps the call into a :class:`Job` and enqueues it. Suspends if
        the queue is full — that is backpressure protecting the workers.

        Args:
            *args: Positional arguments forwarded to the worker function.
            **kwargs: Keyword arguments forwarded to the worker function.

        Returns:
            Job: The tracked Job object — keep it to await
            ``job.execute()`` or inspect ``job.status`` later.
        """
        job = Job(self._worker_func, args, kwargs)
        await self._queue.put(job)
        self._stats["submitted"] += 1
        return job

    async def submit_and_wait(self, *args: Any, **kwargs: Any) -> R:
        """Submit job and wait for result.

        Args:
            *args: Positional arguments for the worker function.
            **kwargs: Keyword arguments for the worker function.

        Returns:
            R: The worker function's return value.

        Raises:
            Exception: Whatever the worker function raised.
        """
        job = await self.submit(*args, **kwargs)
        return await job.execute()

    async def map(self, items: list[T]) -> list[R]:
        """Process multiple items.

        Like the built-in ``map``, but bounded by the pool: all items are
        submitted, then their results are gathered. Failed items are
        silently dropped from the output list.

        Args:
            items: Inputs, each passed as the single argument to the
                worker function.

        Returns:
            list[R]: Results of successful jobs only (failures filtered
            out — check ``stats["failed"]`` for the failure count).
        """
        jobs = [await self.submit(item) for item in items]
        results = await asyncio.gather(*[job.execute() for job in jobs], return_exceptions=True)
        return [r for r in results if not isinstance(r, Exception)]

    async def _worker_loop(self, worker_id: int) -> None:
        """Worker main loop.

        Workers use a 1-second ``wait_for`` timeout around
        ``queue.get()``. Without it, a worker on an empty queue would
        sleep forever and ignore ``task.cancel()`` promptly — actually
        cancel() does interrupt await; the timeout mainly lets the loop
        re-evaluate ``self._running`` so `stop()` exits cleanly even
        without cancellation arriving mid-get.

        Args:
            worker_id: Numeric identifier of this worker (used in
                debugging/names).
        """
        while self._running:
            try:
                job = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except TimeoutError:
                continue

            try:
                await job.execute()
                self._stats["completed"] += 1
            except asyncio.CancelledError:
                # State transition: the job was running when cancellation
                # arrived, so mark it CANCELLED and re-raise to let the
                # worker task itself actually cancel.
                job.status = JobStatus.CANCELLED
                self._stats["cancelled"] += 1
                raise
            except Exception:
                self._stats["failed"] += 1

    @property
    def stats(self) -> dict[str, int]:
        """Cumulative counters for the pool.

        Returns:
            dict: Copy of {'submitted', 'completed', 'failed', 'cancelled'}
            counters.
        """
        return self._stats.copy()

    @property
    def queue_size(self) -> int:
        """Jobs currently waiting in the queue.

        Returns:
            int: Submitted but not yet started jobs.
        """
        return self._queue.qsize()

    @property
    def active_workers(self) -> int:
        """Number of worker tasks currently running.

        Returns:
            int: Worker tasks not yet stopped (not the number *busy*).
        """
        return len(self._workers)


@asynccontextmanager
async def worker_pool(
    worker_func: Callable[[T], Awaitable[R]], *, max_workers: int = 4, max_queue_size: int = 100
) -> AsyncIterator[WorkerPool[T, R]]:
    """Context manager for worker pool.

    Starts the pool on entry, stops it (cancelling workers) on exit —
    you will never leak worker tasks.

    Args:
        worker_func: Async function applied to each job.
        max_workers: Maximum concurrent jobs.
        max_queue_size: Queue capacity (backpressure threshold).

    Yields:
        WorkerPool: A started pool ready for :meth:`WorkerPool.submit`.

    Example:
        >>> async with worker_pool(process, max_workers=3) as pool:
        ...     results = await pool.map(range(10))
    """
    pool = WorkerPool(worker_func, max_workers=max_workers, max_queue_size=max_queue_size)
    await pool.start()
    try:
        yield pool
    finally:
        await pool.stop()


async def demonstrate_worker_pool() -> None:
    """Run a self-contained demo of the worker pool.

    Covers: submitting jobs via the context manager and the ``map``
    helper. Run ``python -m asyncio_mastery.patterns.worker_pool``.
    """
    print("=== Worker Pool Demo ===")

    async def process_item(item: int) -> str:
        await asyncio.sleep(0.05)
        return f"processed-{item}"

    # Using context manager
    print("\n--- Context Manager ---")
    async with worker_pool(process_item, max_workers=3) as pool:
        # Submit jobs
        jobs = [await pool.submit(i) for i in range(10)]
        print(f"Submitted: {len(jobs)}, Queue: {pool.queue_size}")

        # Wait for completion
        results = await asyncio.gather(*[job.execute() for job in jobs])
        for r in results:
            print(f"  {r}")

        print(f"Stats: {pool.stats}")

    # Map operation
    print("\n--- Map ---")
    async with worker_pool(process_item, max_workers=2) as pool:
        results = await pool.map(list(range(5)))
        print(f"Results: {results}")


if __name__ == "__main__":
    asyncio.run(demonstrate_worker_pool())
