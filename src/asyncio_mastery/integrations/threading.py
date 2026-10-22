"""
Threading Integration - Bridge the async world and the thread world.

WHAT IS THIS, FOR A BEGINNER?
-----------------------------
asyncio runs many tasks on ONE thread by pausing tasks at every ``await``
and resuming others. that is perfect for code that mostly *waits*. But you
will constantly meet libraries that are *blocking* -- old database drivers,
``time.sleep``, file-format parsers, most third-party SDKs. Calling one of
those directly freezes the whole event loop (every task stalls).

The fix: run the blocking call on a separate *thread*, and have asyncio
*await* its completion. Threads share one Python process and its memory,
but only one thread executes Python bytecode at a time (the GIL) -- still,
while a blocking call is stuck inside C code (sleeping, reading a socket),
other threads and the event loop DO get to run. So threads are great for
blocking I/O; use processes (see the multiprocessing module) for heavy CPU
work.

JARGON
------
- executor: a pool of worker threads (``ThreadPoolExecutor``). asyncio's
  ``loop.run_in_executor`` is THE standard bridge from async to threads.
- thread-safe: safe to use from multiple threads at once. Ordinary Python
  objects often are not; the primitives here (``queue.Queue``,
  ``threading.Event``) are ``threading``-module objects designed to be.
- race condition: a bug where two threads/tasks touch shared data at the
  same time and corrupt it. Built-in thread-safe primitives avoid this.

what is INSIDE
-------------
- :class:`ThreadBridge` and :func:`run_in_thread`: await blocking functions.
- :class:`AsyncThread`: run a coroutine inside a background thread.
- :class:`AsyncQueue`: pass items between async code and threads safely.
- :class:`ThreadSafeEvent`: a thread can signal async code "it happened".

MINIMAL EXAMPLE
---------------
>>> def blocking(): time.sleep(1); return "done"
>>> result = await run_in_thread(blocking)   # event loop stays responsive
"""

import asyncio
import threading
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from queue import Queue
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass
class ThreadResult(Generic[T]):
    """Outcome of work executed on a thread.

    Attributes:
        value: The returned value on success.
        error: The exception on failure.
        completed: True once the work finished either way.
    """

    value: T | None = None
    error: Exception | None = None
    completed: bool = False


class ThreadBridge:
    """Await blocking (synchronous) functions without stalling the event loop.

    Wraps a :class:`concurrent.futures.ThreadPoolExecutor`. The core line of
    the whole module is in :meth:`run`: ``run_in_executor`` schedules the
    call on a worker thread and returns an asyncio Future you can ``await``.

    Args:
        max_workers: Size of the thread pool. None = Python's default
            (min(32, cpu_count + 4)).

    Example:
        >>> bridge = ThreadBridge(max_workers=4)
        >>> out = await bridge.run(requests.get, "https://example.com")
        >>> bridge.shutdown()
    """

    def __init__(self, max_workers: int | None = None) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._loop = asyncio.get_event_loop()

    async def run(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Run ``func(*args, **kwargs)`` on a worker thread and await result.

        Args:
            func: A normal (blocking) function.
            *args, **kwargs: Forwarded to ``func``.

        Returns:
            ``func``'s return value; exceptions are re-raised at the await.
        """
        return await self._loop.run_in_executor(self._executor, lambda: func(*args, **kwargs))

    def run_sync(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> Any:
        """Submit work and get a ``concurrent.futures.Future`` immediately.

        Non-async: you manage waiting on the returned Future yourself.
        """
        return self._executor.submit(func, *args, **kwargs)

    def submit(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> Any:
        """Alias of :meth:`run_sync`: submit work, get a Future back."""
        return self._executor.submit(func, *args, **kwargs)

    def map(self, func: Callable[..., T], *iterables: Any) -> Any:
        """Apply ``func`` over iterables across the thread pool (blocking API)."""
        return self._executor.map(func, *iterables)

    def shutdown(self, wait: bool = True) -> None:
        """Stop accepting new work; with ``wait=True`` block until threads finish."""
        self._executor.shutdown(wait=wait)

    @asynccontextmanager
    async def managed(self) -> Any:
        """Async context manager that shuts the pool down on exit.

        Example:
            >>> async with ThreadBridge().managed() as b:
            ...     await b.run(blocking_fn)
        """
        try:
            yield self
        finally:
            self.shutdown()


class AsyncThread(threading.Thread):
    """A background thread that runs a coroutine on its own event loop.

    The mirror image of :class:`ThreadBridge`: instead of running sync code
    from async code, this runs *async* code from a (sync) thread. The thread
    creates its own event loop, runs your coroutine to completion, and
    stores the outcome in a :class:`ThreadResult` you retrieve via
    :meth:`result`.

    Args:
        target: The async function to execute in the new thread.
        *args, **kwargs: Forwarded to ``target``.

    Example:
        >>> t = AsyncThread(fetch_all, urls)
        >>> t.start()
        >>> data = t.result(timeout=10)
    """

    def __init__(self, target: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._target = target
        self._args = args
        self._kwargs = kwargs
        self._result: ThreadResult[Any] = ThreadResult()
        self._loop: asyncio.AbstractEventLoop | None = None

    def run(self) -> None:
        """Thread entry point: create a loop and run the coroutine.

        Called automatically by ``Thread.start()`` on the new thread. Each
        thread needs its OWN event loop -- loops are not thread-safe.
        """
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        async def wrapper() -> None:
            try:
                self._result.value = await self._target(*self._args, **self._kwargs)
            except Exception as e:
                self._result.error = e
            finally:
                self._result.completed = True

        try:
            self._loop.run_until_complete(wrapper())
        finally:
            self._loop.close()

    def result(self, timeout: float | None = None) -> Any:
        """Wait for the thread to finish and return the coroutine's value.

        Args:
            timeout: Max seconds to wait.

        Returns:
            The coroutine's return value.

        Raises:
            Exception: Whatever the coroutine raised is re-raised here.
        """
        self.join(timeout)
        if self._result.error:
            raise self._result.error
        return self._result.value


class AsyncQueue:
    """Queue that both async code and threads can use safely.

    Built on :class:`queue.Queue`, which is *thread-safe* by design. The
    async :meth:`put`/:meth:`get` run the (possibly blocking) queue
    operations in a thread via ``run_in_executor``, so waiting for an item
    never stalls the event loop; the ``*_nowait`` variants never block at
    all. Use it as the mailbox between a background thread and your
    coroutines.

    Args:
        maxsize: Max items before putters block; 0 = unbounded.

    Example:
        >>> q = AsyncQueue()
        >>> await q.put("hello")     # from async code
        >>> q.put_nowait("hi")       # from a thread
        >>> item = await q.get()
    """

    def __init__(self, maxsize: int = 0) -> None:
        self._queue: Queue = Queue(maxsize=maxsize)
        self._loop = asyncio.get_event_loop()

    async def put(self, item: Any) -> None:
        """Put an item, blocking-asynchronously if the queue is full.

        Args:
            item: Any picklable Python object.
        """
        await self._loop.run_in_executor(None, self._queue.put, item)

    async def get(self) -> Any:
        """Take the next item, waiting asynchronously if the queue is empty.

        Returns:
            The oldest queued item.
        """
        return await self._loop.run_in_executor(None, self._queue.get)

    def put_nowait(self, item: Any) -> None:
        """Put without blocking; raises ``queue.Full`` if full. Thread-safe."""
        self._queue.put_nowait(item)

    def get_nowait(self) -> Any:
        """Get without blocking; raises ``queue.Empty`` if empty. Thread-safe."""
        return self._queue.get_nowait()

    def qsize(self) -> int:
        """Approximate number of queued items."""
        return self._queue.qsize()


async def run_in_thread(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """One-shot helper: run a blocking function on a thread and await it.

    Uses asyncio's default executor (no explicit pool to manage), so it is
    the quickest way to "async-ify" one blocking call.

    Args:
        func: A normal blocking function.
        *args, **kwargs: Forwarded to ``func``.

    Returns:
        ``func``'s return value.

    Example:
        >>> text = await run_in_thread(path.read_text)
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


class ThreadSafeEvent:
    """One-way signal usable from BOTH threads and async code.

    Wraps :class:`threading.Event` (a boolean flag with waiters). A thread
    calls :meth:`set`; async code ``await``s :meth:`wait` and wakes the
    instant the flag is set -- without freezing the event loop, because the
    blocking wait happens on a worker thread.

    Example:
        >>> ev = ThreadSafeEvent()
        >>> threading.Thread(target=lambda: (time.sleep(1), ev.set())).start()
        >>> await ev.wait()   # resumes after ~1 second
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._loop = asyncio.get_event_loop()

    def set(self) -> None:
        """Set the flag, waking every waiter. Safe from any thread."""
        self._event.set()

    def clear(self) -> None:
        """Reset the flag so future waiters block until the next set()."""
        self._event.clear()

    def is_set(self) -> bool:
        """True if the flag is currently set."""
        return self._event.is_set()

    async def wait(self) -> None:
        """Wait for the flag WITHOUT blocking the event loop."""
        await self._loop.run_in_executor(None, self._event.wait)

    def wait_sync(self, timeout: float | None = None) -> bool:
        """Synchronous wait with optional timeout; True if set in time.

        For use from threads; async code should use :meth:`wait` instead.
        """
        return self._event.wait(timeout)


async def demonstrate_threading() -> None:
    """Narrated demo: threads running blocking work, cross-world queues/events."""
    print("=== Threading Integration Demo ===")

    # Thread bridge: 8 blocking sleeps run concurrently on 4 worker threads,
    # and the event loop stays free the whole time.
    print("\n--- Thread Bridge ---")
    bridge = ThreadBridge(max_workers=4)

    def blocking_io(task_id: int) -> str:
        time.sleep(0.1)  # Pretends to be a blocking file/network read.
        return f"task-{task_id}-done"

    tasks = [bridge.run(blocking_io, i) for i in range(8)]
    results = await asyncio.gather(*tasks)
    for r in results:
        print(f"  {r}")

    bridge.shutdown()

    # AsyncThread
    print("\n--- AsyncThread ---")

    async def async_work(name: str) -> str:
        await asyncio.sleep(0.1)
        return f"{name} complete"

    thread = AsyncThread(async_work, "Worker-1")
    thread.start()
    result = thread.result(timeout=1.0)
    print(f"  Result: {result}")

    # AsyncQueue
    print("\n--- AsyncQueue ---")
    queue = AsyncQueue()

    async def producer() -> None:
        for i in range(5):
            await queue.put(f"item-{i}")
            await asyncio.sleep(0.02)

    async def consumer() -> None:
        for _ in range(5):
            item = await queue.get()
            print(f"  Consumed: {item}")

    await asyncio.gather(producer(), consumer())

    # ThreadSafeEvent
    print("\n--- ThreadSafeEvent ---")
    event = ThreadSafeEvent()

    def setter() -> None:
        time.sleep(0.1)
        event.set()
        print("  Event set from thread")

    thread2 = threading.Thread(target=setter)
    thread2.start()
    await event.wait()
    print("  Event received in async")
    thread2.join()

    # run_in_thread
    print("\n--- run_in_thread ---")

    def cpu_intensive(n: int) -> int:
        return sum(i * i for i in range(n))

    result = await run_in_thread(cpu_intensive, 100000)
    print(f"  CPU result: {result}")


if __name__ == "__main__":
    asyncio.run(demonstrate_threading())
