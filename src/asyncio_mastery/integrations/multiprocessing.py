"""
Multiprocessing Integration - Run CPU-bound work in separate processes.

WHAT IS THIS, FOR A BEGINNER?
-----------------------------
Asyncio is fantastic for *I/O-bound* work (network, disk): while one task
waits, others run. But Python's *GIL* (Global Interpreter Lock) means one
Python process executes only one line of Python bytecode at a time. So
*CPU-bound* work -- heavy math, image processing, parsing big files --
blocks the entire event loop, no matter how async your code is.

The escape hatch is **multiprocessing**: spawn entirely separate Python
processes, each with its own GIL, so real parallel computation happens on
all your CPU cores. asyncio cannot await a process directly, so this module
bridges the two worlds using an *executor*: a pool of worker processes that
the event loop can submit jobs to and await results from.

JARGON
------
- executor: an object (here ``ProcessPoolExecutor``) that manages a pool of
  worker processes. ``loop.run_in_executor`` is asyncio's standard way to
  await work happening outside the event loop.
- pickling: how Python ships function arguments/results between processes.
  Consequently, only *top-level* functions and picklable data can cross the
  process boundary -- lambdas and closures work here only because they are
  wrapped into the executor call carefully (and real code should prefer
  top-level functions).
- Queue (multiprocessing.Queue): a pipe for sending Python objects safely
  from a child process back to its parent.

WHEN TO USE WHICH TOOL
----------------------
- Waiting for network/disk? -> stay in asyncio.
- Heavy pure-Python computation? -> :class:`ProcessBridge` /
  :func:`run_in_process` / :class:`ProcessPool`.
- You need a whole async sub-program in another process? -> :class:`AsyncProcess`.

MINIMAL EXAMPLE
---------------
>>> def heavy(n): return sum(i*i for i in range(n))  # must be top-level
>>> result = await run_in_process(heavy, 1_000_000)
"""

import asyncio
import multiprocessing as mp
from collections.abc import Awaitable, Callable
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from multiprocessing import Pool, Process, Queue
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass
class ProcessResult(Generic[T]):
    """Outcome of some work executed in another process.

    Attributes:
        value: The returned value on success (None until finished).
        error: The exception on failure (None on success).
        completed: True once the work finished, successfully or not.
    """

    value: T | None = None
    error: Exception | None = None
    completed: bool = False


class ProcessBridge:
    """Simple bridge: await CPU-bound functions in a process pool.

    Wraps :class:`concurrent.futures.ProcessPoolExecutor` and asyncio's
    ``run_in_executor`` so that ``await bridge.run(f, *args)`` executes
    ``f`` in a child process and returns its result without freezing the
    event loop.

    Args:
        max_workers: Number of child processes. None = one per CPU core.

    Example:
        >>> bridge = ProcessBridge(max_workers=4)
        >>> results = await asyncio.gather(*[bridge.run(f, i) for i in data])
        >>> bridge.shutdown()
    """

    def __init__(self, max_workers: int | None = None) -> None:
        self._executor = ProcessPoolExecutor(max_workers=max_workers)
        self._loop = asyncio.get_event_loop()

    async def run(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Run ``func(*args, **kwargs)`` in a worker process and await it.

        Args:
            func: A *top-level, picklable* function.
            *args, **kwargs: Arguments forwarded to ``func``.

        Returns:
            Whatever ``func`` returned.
        """
        # The inner lambda bundles args for the executor; the partial call
        # itself (not just `func`) is what gets pickled across.
        return await self._loop.run_in_executor(self._executor, lambda: func(*args, **kwargs))

    def submit(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> Any:
        """Submit work and get back a ``concurrent.futures.Future`` immediately.

        Unlike :meth:`run`, this is non-async: you get a Future handle you
        can poll or wait on yourself.
        """
        return self._executor.submit(func, *args, **kwargs)

    def map(self, func: Callable[..., T], *iterables: Any) -> Any:
        """Apply ``func`` to every item of the iterables, in parallel.

        Returns an iterator of results in input order (blocking style).
        """
        return self._executor.map(func, *iterables)

    def shutdown(self, wait: bool = True) -> None:
        """Shut the pool down; with ``wait=True`` block until workers finish."""
        self._executor.shutdown(wait=wait)

    @asynccontextmanager
    async def managed(self) -> Any:
        """Async context manager that shuts the pool down on exit.

        Yields:
            This bridge.

        Example:
            >>> async with bridge.managed() as b:
            ...     await b.run(heavy, n)
        """
        try:
            yield self
        finally:
            self.shutdown()


class AsyncProcess(Process):
    """A ``multiprocessing.Process`` whose target is an async function.

    A normal ``Process`` runs a synchronous function in the child. This
    subclass creates a fresh event loop *inside the child process* and runs
    your coroutine there. Because processes can't share return values
    directly, the result (or exception) is ferried back through a
    ``multiprocessing.Queue`` and retrieved with :meth:`result`.

    Args:
        target: An async function to run in the child process.
        *args, **kwargs: Forwarded to ``target``.

    Example:
        >>> p = AsyncProcess(my_coro, 1, 2)
        >>> p.start()
        >>> print(p.result(timeout=10))
    """

    def __init__(self, target: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._target = target
        self._args = args
        self._kwargs = kwargs
        # Cross-process channel used to send ("success", value) or
        # ("error", exception) back to the parent process.
        self._result_queue: Queue = Queue()

    def run(self) -> None:
        """Process entry point: build an event loop and run the coroutine.

        Called automatically in the CHILD process by ``Process.start()``.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def wrapper() -> None:
            try:
                result = await self._target(*self._args, **self._kwargs)
                self._result_queue.put(("success", result))
            except Exception as e:
                self._result_queue.put(("error", e))
            finally:
                loop.close()

        loop.run_until_complete(wrapper())

    def result(self, timeout: float | None = None) -> Any:
        """Wait (in the parent) for the child and return its result.

        Args:
            timeout: Max seconds to wait for the child to finish.

        Returns:
            The value the coroutine returned.

        Raises:
            Exception: Whatever the child coroutine raised (re-raised here).
            TimeoutError: If no result was available within ``timeout``.
        """
        self.join(timeout)
        if not self._result_queue.empty():
            status, value = self._result_queue.get()
            if status == "error":
                raise value
            return value
        raise TimeoutError("Process result not ready")


async def run_in_process(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """One-shot helper: run ``func`` in a fresh process pool and await it.

    Convenient for single calls; for repeated work create a
    :class:`ProcessBridge` once and reuse it, since creating a pool has cost.

    Args:
        func: A top-level picklable function.
        *args, **kwargs: Arguments forwarded to ``func``.

    Returns:
        ``func``'s return value.
    """
    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor() as executor:
        return await loop.run_in_executor(executor, lambda: func(*args, **kwargs))


class ProcessPool:
    """Async-friendly wrapper around :class:`multiprocessing.Pool`.

    `multiprocessing.Pool` predates asyncio and its async methods need a
    callback API; this class pushes those blocking calls into a thread via
    ``run_in_executor`` so you can simply ``await pool.map(...)``.

    Args:
        max_workers: Number of child processes. None = one per CPU core.

    Example:
        >>> async with ProcessPool(4).managed() as pool:
        ...     results = await pool.map(heavy, work_items)
    """

    def __init__(self, max_workers: int | None = None) -> None:
        self._pool = Pool(processes=max_workers)
        self._loop = asyncio.get_event_loop()

    async def apply(self, func: Callable[..., T], args: tuple = (), kwds: dict | None = None) -> T:
        """Run one function call in the pool and await its result.

        Args:
            func: Top-level picklable function.
            args: Positional arguments tuple.
            kwds: Keyword arguments dict.

        Returns:
            ``func``'s return value.
        """
        return await self._loop.run_in_executor(None, self._pool.apply, func, args, kwds or {})

    async def map(self, func: Callable[..., T], iterable: Any, chunksize: int = 1) -> list[T]:
        """Parallel map: apply ``func`` to each item, awaiting all results.

        Args:
            func: Top-level picklable function taking one argument.
            iterable: Items to process.
            chunksize: How many items are handed to a worker at a time;
                larger values reduce inter-process overhead for many small
                items.

        Returns:
            Results in the same order as the input.
        """
        return await self._loop.run_in_executor(None, self._pool.map, func, iterable, chunksize)

    async def starmap(self, func: Callable[..., T], iterable: Any, chunksize: int = 1) -> list[T]:
        """Like :meth:`map`, but each item is a tuple unpacked as arguments.

        ``pool.starmap(f, [(1, 2), (3, 4)])`` computes ``f(1, 2), f(3, 4)``.
        """
        return await self._loop.run_in_executor(None, self._pool.starmap, func, iterable, chunksize)

    def close(self) -> None:
        """Prevent new tasks; running tasks finish normally."""
        self._pool.close()

    def terminate(self) -> None:
        """Immediately kill worker processes (use only in emergencies)."""
        self._pool.terminate()

    def join(self) -> None:
        """Block until all worker processes have exited."""
        self._pool.join()

    @asynccontextmanager
    async def managed(self) -> Any:
        """Async context manager that closes and joins the pool on exit."""
        try:
            yield self
        finally:
            self.close()
            self.join()


async def demonstrate_multiprocessing() -> None:
    """Narrated demo of every tool in this module.

    Note the CPU-bound ``cpu_task`` is defined *inside* this function purely
    for compactness of the demo; in real multiprocessing code, worker
    functions should be defined at module top level so they can be pickled.
    """
    print("=== Multiprocessing Integration Demo ===")

    # CPU-bound function (sum of squares) -- would block the event loop
    # if run directly.
    def cpu_task(n: int) -> int:
        return sum(i * i for i in range(n))

    # Process bridge
    print("\n--- Process Bridge ---")
    bridge = ProcessBridge(max_workers=4)

    tasks = [bridge.run(cpu_task, 50000) for _ in range(8)]
    results = await asyncio.gather(*tasks)
    for r in results[:3]:
        print(f"  Result: {r}")
    print(f"  ... and {len(results) - 3} more")

    bridge.shutdown()

    # AsyncProcess
    print("\n--- AsyncProcess ---")

    async def async_cpu(name: str, n: int) -> str:
        await asyncio.sleep(0.01)  # Simulate some async work
        return f"{name}: {sum(i * i for i in range(n))}"

    process = AsyncProcess(async_cpu, "Worker-1", 20000)
    process.start()
    result = process.result(timeout=5.0)
    print(f"  {result}")

    # run_in_process
    print("\n--- run_in_process ---")
    result = await run_in_process(cpu_task, 30000)
    print(f"  Result: {result}")

    # ProcessPool
    print("\n--- ProcessPool ---")
    async with ProcessPool(max_workers=2) as pool:
        results = await pool.map(cpu_task, [10000, 20000, 30000, 40000])
        for r in results:
            print(f"  {r}")


if __name__ == "__main__":
    # On Windows, child processes are spawned by re-importing the main
    # module; freeze_support() (and the __main__ guard) prevents a fork-bomb
    # of recursive spawns. Always keep both in multiprocessing scripts.
    mp.freeze_support()
    asyncio.run(demonstrate_multiprocessing())
