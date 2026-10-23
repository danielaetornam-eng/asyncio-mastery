"""
Helpers - Everyday async utilities that asyncio does not give you built-in.

WHAT IS THIS MODULE, FOR A BEGINNER?
------------------------------------
The asyncio standard library gives you powerful primitives (``gather``,
``Semaphore``, ``Queue``), but common *combinations* of them are left as
an exercise. This module collects those exercises, solved: concurrency
caps, task supervision, caching with expiry, rate limiting, and helpers
for *async iterables*.

REAL-WORLD ANALOGY
------------------
Think of these as kitchen equipment, not ingredients. ``asyncio`` hands
you flour and water; this module hands you the mixer and the oven
timer. You could build each from scratch — semaphore + gather, token
counting by hand — but you should not have to.

JARGON
------
* **semaphore**: a counter that limits how many coroutines may hold a
  resource at once; the extra ones wait in line.
* **TTL (time-to-live)**: how long a cached value stays valid before it
  is treated as stale and deleted.
* **token bucket**: a rate-limiting scheme where the bucket fills with
  ``rate`` tokens per second up to a ``burst`` maximum; each operation
  spends tokens, so bursts are allowed but the average rate is capped.
* **async iterable / async iterator**: objects you can loop over with
  ``async for`` — each "next item" is itself awaited (used for streams,
  database cursors, paginated APIs).
* **backpressure**: slowing producers when consumers cannot keep up;
  a semaphore or a bounded queue provides it.

WHEN TO USE EACH PIECE
----------------------
* :func:`gather_with_concurrency` — you have 10,000 URLs to fetch but
  must not open 10,000 connections at once.
* :func:`run_async` — you need to call async code from synchronous code.
* :class:`TaskGroup` / :func:`create_task_group` — spawn tasks and be
  sure they are all awaited before your function returns.
* :class:`AsyncCache` — memoize expensive async lookups with expiry.
* :class:`AsyncRateLimiter` — respect an API's "max N requests/second".
* :func:`wait_for_all` / :func:`wait_for_first` — "all must finish" vs
  "first answer wins; cancel the rest".
* :func:`async_map` / :func:`async_filter` / :func:`async_reduce` — the
  familiar ``map``/``filter``/``reduce`` trio, but for ``async for``
  sources and ``await``-using functions.

MINIMAL EXAMPLE
---------------
>>> import asyncio
>>> async def fetch(i: int) -> int:
...     await asyncio.sleep(0.01)
...     return i
>>> async def main() -> None:
...     # At most 3 fetches run at the same moment, not 10.
...     results = await gather_with_concurrency(
...         3, *[fetch(i) for i in range(10)]
...     )
...     print(results)
>>> asyncio.run(main())
"""

import asyncio
import time
from collections.abc import AsyncIterable, Awaitable, Callable, Coroutine
from collections.abc import AsyncIterator as ABCAsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Generic, TypeVar, cast

T = TypeVar("T")
U = TypeVar("U")


async def gather_with_concurrency(
    limit: int, *tasks: Awaitable[T], return_exceptions: bool = False
) -> list[T | BaseException]:
    """Run many awaitables, but only ``limit`` of them at a time.

    Plain ``asyncio.gather(*tasks)`` starts EVERYTHING simultaneously —
    fine for 3 tasks, disastrous for 10,000 network requests. This wraps
    each awaitable in a semaphore so at most ``limit`` run at once; the
    rest wait for a slot. Results come back in the SAME ORDER as the
    input, just like ``gather``.

    Args:
        limit: Maximum number of awaitables running concurrently. Think
            "connection budget" — e.g. 10 for HTTP fetching.
        *tasks: The awaitables to run (coroutines, tasks, futures).
        return_exceptions: If False (default), the first failure cancels
            the rest and raises. If True, failures appear as exception
            objects in the results list.

    Returns:
        A list of results (or exceptions, if ``return_exceptions=True``),
        ordered to match ``tasks``.

    Example:
        >>> results = await gather_with_concurrency(
        ...     5, *[fetch(url) for url in urls]
        ... )
    """
    semaphore = asyncio.Semaphore(limit)

    async def bounded(task: Awaitable[T]) -> T:
        # acquire a semaphore slot BEFORE awaiting the task body
        async with semaphore:
            return await task

    return await asyncio.gather(*[bounded(t) for t in tasks], return_exceptions=return_exceptions)


def run_async(coro: Awaitable[T]) -> T:
    """Run a coroutine to completion from synchronous code.

    Async functions can only be awaited inside async functions — this is
    the bridge for calling one from plain (sync) code. It handles two
    situations: no event loop is running yet (create one and drive the
    coroutine on it), or a loop is ALREADY running (e.g. inside Jupyter),
    in which case the coroutine is submitted thread-safely and we block
    until it finishes.

    Args:
        coro: Any awaitable — usually a coroutine like ``my_func()``.

    Returns:
        Whatever the coroutine returns.

    Note:
        In async-heavy apps prefer ``asyncio.run()`` at the top level;
        this helper is for mixed sync/async boundaries.

    Example:
        >>> result = run_async(fetch_data())
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # no loop in this thread yet — build one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        # loop already owned by this thread/context: submit and block
        return asyncio.run_coroutine_threadsafe(
            cast("Coroutine[Any, Any, T]", coro), loop
        ).result()
    else:
        return loop.run_until_complete(cast("Coroutine[Any, Any, T]", coro))


@asynccontextmanager
async def create_task_group() -> ABCAsyncIterator["TaskGroup"]:
    """Shortcut: ``async with create_task_group() as g: g.create_task(...)``.

    Equivalent to ``async with TaskGroup() as g:`` — exists for symmetry
    with the other factory helpers. On exit, ALL tasks spawned inside the
    block are awaited, so nothing is left running by accident.

    Yields:
        TaskGroup: The group; call ``.create_task(coro)`` inside the
        block.

    Example:
        >>> async with create_task_group() as g:
        ...     g.create_task(job(1))
        ...     g.create_task(job(2))
        >>> # both jobs finished here
    """
    group = TaskGroup()
    async with group:
        yield group


class TaskGroup:
    """Supervise a batch of tasks so none outlives its scope.

    A *task group* (structured concurrency) solves the "fire and forget"
    leak: tasks created with bare ``asyncio.create_task`` can be silently
    abandoned if you forget them. Here, the ``async with`` block waits
    for every spawned task — exceptions included — before your code
    moves on. (Python 3.11+ has ``asyncio.TaskGroup`` built in; this
    class is the learning-friendly version.)

    Example:
        >>> async with TaskGroup() as tg:
        ...     task = tg.create_task(worker(1), name="w1")
    """

    def __init__(self) -> None:
        """Create an empty group; tasks join via :meth:`create_task`."""
        self._tasks: set[asyncio.Task[Any]] = set()

    def create_task(self, coro: Awaitable[T], *, name: str | None = None) -> asyncio.Task[T]:
        """Schedule a coroutine as a task tracked by this group.

        Args:
            coro: The coroutine to run concurrently.
            name: Optional label; shows up in tracebacks and
                ``asyncio.all_tasks()`` output — makes debugging easier.

        Returns:
            The created ``asyncio.Task``; it is removed from the group's
            registry automatically when it completes (via a done
            callback), so the set only ever holds LIVE tasks.
        """
        task: asyncio.Task[T] = asyncio.create_task(
            cast("Coroutine[Any, Any, T]", coro), name=name
        )
        self._tasks.add(task)
        # discard() will not raise if the task was already removed
        task.add_done_callback(self._tasks.discard)
        return task

    async def __aenter__(self) -> "TaskGroup":
        """Enter the ``async with`` block; returns the group itself."""
        return self

    async def __aexit__(self, *args: Any) -> bool:
        """Wait for every still-running task, swallowing their exceptions.

        ``return_exceptions=True`` means one failing task does not abort
        the wait for its siblings. Returning False means exceptions from
        the BLOCK body (if any) still propagate normally after cleanup.
        """
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        return False


class AsyncIterator(Generic[T]):
    """Wrap any async iterable so you can manually pull items one at a time.

    Normally you write ``async for item in source:`` and the loop drives
    the iteration. Sometimes you want explicit control — "give me the
    next item NOW, I will decide when to ask again" — like hitting 'next'
    on a streaming playlist instead of letting it autoplay. That needs
    the iterator protocol (``__aiter__``/``__anext__``), which this
    class exposes around a plain async iterable.

    Example:
        >>> it = AsyncIterator(paginated_api("users"))
        >>> it.__aiter__()
        >>> first_page = await it.__anext__()
    """

    def __init__(self, iterable: AsyncIterable[T]) -> None:
        """Wrap an async iterable (anything with ``__aiter__``).

        Args:
            iterable: The source, e.g. an async generator.
        """
        self._iterable = iterable
        self._iterator: ABCAsyncIterator[T] | None = None

    def __aiter__(self) -> "AsyncIterator[T]":
        """Prepare the underlying iterator (called by ``async for``).

        Returns:
            This same object — it IS its own iterator.
        """
        self._iterator = self._iterable.__aiter__()
        return self

    async def __anext__(self) -> T:
        """Await and return the next item from the source.

        Returns:
            The next item.

        Raises:
            StopAsyncIteration: If ``__aiter__`` was never called, or
            (from the source) when items are exhausted.
        """
        if self._iterator is None:
            raise StopAsyncIteration
        return await self._iterator.__anext__()


async def async_map(
    func: Callable[[T], Awaitable[U]], iterable: AsyncIterable[T]
) -> ABCAsyncIterator[U]:
    """Apply an async function to every item of an async source.

    The async version of Python's built-in ``map`` — but because each
    call must be awaited, results are produced lazily, one per
    ``__anext__``. NOTE: this runs sequentially; for parallel mapping
    use :func:`gather_with_concurrency` instead.

    Args:
        func: Coroutine function applied to each item.
        iterable: Any async iterable (async generator, stream, ...).

    Yields:
        ``func(item)`` results, in source order.

    Example:
        >>> async for page_html in async_map(fetch, url_stream):
        ...     parse(page_html)
    """
    async for item in iterable:
        yield await func(item)


async def async_filter(
    predicate: Callable[[T], Awaitable[bool]], iterable: AsyncIterable[T]
) -> ABCAsyncIterator[T]:
    """Keep only items an async test says yes to.

    Like built-in ``filter``, but the test itself is a coroutine (e.g.
    "does this user exist in the DB?"). Items failing the test are
    skipped; the rest pass through in order.

    Args:
        predicate: Coroutine function returning True to keep an item.
        iterable: Any async iterable.

    Yields:
        Only the items for which ``predicate`` returned True.

    Example:
        >>> async for active_user in async_filter(is_active, users):
        ...     notify(active_user)
    """
    async for item in iterable:
        if await predicate(item):
            yield item


async def async_reduce(
    func: Callable[[U, T], Awaitable[U]], iterable: AsyncIterable[T], initial: U
) -> U:
    """Fold an async source down to ONE value (async version of reduce).

    Each step combines the running accumulator with the next item:
    ``acc = await func(acc, item)``. Classic uses: summing streamed
    numbers, merging partial results.

    Args:
        func: Coroutine function ``(accumulator, item) -> new_acc``.
        iterable: Any async iterable.
        initial: Starting value of the accumulator.

    Returns:
        The final accumulator after the last item.

    Example:
        >>> total = await async_reduce(
        ...     lambda acc, x: asyncio.sleep(0, acc + x), numbers, 0
        ... )
    """
    result = initial
    async for item in iterable:
        result = await func(result, item)
    return result


class AsyncCache(Generic[T]):
    """A small in-memory cache that is safe for concurrent async tasks.

    A cache remembers previous results so you do not recompute or
    re-fetch them. This one adds two behaviors on top of a dict:

    * **TTL expiry** — entries older than ``ttl`` seconds are treated
      as missing (fresh milk vs. expired milk in the fridge).
    * **Max size with oldest-first eviction** — when full, the stalest
      entry is dropped to make room.

    An ``asyncio.Lock`` guards every operation so interleaved tasks can
    never see a half-written entry.

    Example:
        >>> cache = AsyncCache[str](maxsize=100, ttl=60.0)
        >>> value = await cache.get_or_set("user:1", lambda: load_user(1))
    """

    def __init__(self, maxsize: int = 128, ttl: float | None = None) -> None:
        """Create an empty cache.

        Args:
            maxsize: Maximum entries; oldest is evicted when full.
            ttl: Seconds an entry stays fresh. None = never expires.
        """
        self._maxsize = maxsize
        self._ttl = ttl
        # values stored as (insertion_timestamp, value) tuples — the
        # timestamp powers BOTH ttl checks and oldest-first eviction
        self._cache: dict[str, tuple[float, T]] = {}
        self._lock = asyncio.Lock()

    def _make_key(self, *args: Any, **kwargs: Any) -> str:
        """Build a string key from arbitrary call arguments.

        Useful when caching by "function + arguments", e.g.
        ``cache._make_key("get_user", user_id=3)``.
        """
        return f"{args}:{kwargs}"

    async def get(self, key: str) -> T | None:
        """Fetch a cached value, respecting TTL.

        Args:
            key: The lookup key (see :meth:`set`).

        Returns:
            The cached value, or None if absent or expired. Note a cache
            storing literal None is indistinguishable from a miss.
        """
        async with self._lock:
            if key in self._cache:
                timestamp, value = self._cache[key]
                # monotonic() is a clock immune to wall-clock changes
                if self._ttl is None or time.monotonic() - timestamp < self._ttl:
                    return value
                # expired: evict eagerly rather than leaving garbage
                del self._cache[key]
        return None

    async def set(self, key: str, value: T) -> None:
        """Store a value, evicting the oldest entry if the cache is full.

        Args:
            key: Lookup key for later :meth:`get` calls.
            value: The value to cache.
        """
        async with self._lock:
            if len(self._cache) >= self._maxsize:
                # entry with the smallest timestamp = oldest
                oldest = min(self._cache.items(), key=lambda x: x[1][0])
                del self._cache[oldest[0]]
            self._cache[key] = (time.monotonic(), value)

    async def delete(self, key: str) -> bool:
        """Remove one entry.

        Args:
            key: The key to remove.

        Returns:
            True if the key existed (and was removed), False otherwise.
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
        return False

    async def clear(self) -> None:
        """Empty the entire cache — all keys, expired or not."""
        async with self._lock:
            self._cache.clear()

    async def get_or_set(self, key: str, factory: Callable[[], Awaitable[T]]) -> T:
        """Return the cached value, or compute-and-cache it on a miss.

        The most common cache pattern ("cache-aside"): try the cache,
        and only call the expensive ``factory`` when the answer isn't
        there.

        Args:
            key: Lookup key.
            factory: Coroutine function producing the value on a miss,
                e.g. ``lambda: fetch_from_db(user_id)``.

        Returns:
            The cached or freshly computed value.
        """
        value = await self.get(key)
        if value is not None:
            return value
        value = await factory()
        await self.set(key, value)
        return value


class AsyncRateLimiter:
    """Cap how fast operations happen, using the *token bucket* algorithm.

    Analogy: a bucket that refills with ``rate`` tokens per second, up
    to a maximum of ``burst``. Every operation must grab a token before
    running; when the bucket is empty, :meth:`acquire` waits for tokens
    to trickle in. The "burst" headroom means short traffic spikes pass
    instantly while the AVERAGE rate stays capped — exactly how the
    GitHub or Twitter APIs throttle clients.

    Example:
        >>> limiter = AsyncRateLimiter(rate=10.0, burst=3)
        >>> for url in urls:
        ...     await limiter.acquire()      # at most 10/sec on average
        ...     await call_api(url)
    """

    def __init__(self, rate: float, burst: int | None = None) -> None:
        """Create a limiter with a full starting bucket.

        Args:
            rate: Tokens refilled per second — the sustained limit.
            burst: Bucket capacity — the max tokens that can accumulate
                (allows short bursts). Defaults to ``int(rate * 2)``.
        """
        self._rate = rate
        self._burst = burst or int(rate * 2)
        self._tokens = float(self._burst)  # start full: first burst is free
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> None:
        """Wait until ``tokens`` are available, then spend them.

        Sleeps EXACTLY as long as needed for the bucket to refill enough
        tokens — computed as deficit / rate — then tries again. The lock
        ensures two tasks can't spend the same tokens.

        Args:
            tokens: How many tokens this operation costs (default 1).
        """
        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                # time until the bucket holds enough tokens
                wait = (tokens - self._tokens) / self._rate
                await asyncio.sleep(wait)

    def try_acquire(self, tokens: int = 1) -> bool:
        """Spend tokens ONLY if available right now — never sleeps.

        Useful for "drop or degrade" strategies: if you can't do a full
        job within the rate limit, skip it instead of queueing.

        Args:
            tokens: Cost of the operation.

        Returns:
            True if tokens were spent (go ahead), False if the bucket
            is too empty (skip or defer).
        """
        self._refill()
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False

    def _refill(self) -> None:
        """Top up tokens based on elapsed real time (never above burst).

        Tokens accumulate lazily — they're computed when needed from
        ``elapsed * rate`` rather than by a background timer.
        """
        now = time.monotonic()
        elapsed = now - self._last_update
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
        self._last_update = now

    @property
    def available(self) -> float:
        """Current token count after refilling (mostly for debugging/tests).

        Returns:
            Tokens currently in the bucket, capped at ``burst``.
        """
        self._refill()
        return self._tokens


async def wait_for_all(*awaitables: Awaitable[T], timeout: float | None = None) -> list[T]:
    """Await EVERY awaitable, optionally with a hard deadline.

    A convenience wrapper: ``gather`` for the "all must succeed" case,
    plus ``asyncio.wait_for`` timeout support. If the deadline passes,
    remaining work is cancelled and ``TimeoutError`` is raised.

    Args:
        *awaitables: The work to run concurrently.
        timeout: Max total seconds to wait. None = no limit.

    Returns:
        list[T]: Results in the same order as the inputs.

    Raises:
        TimeoutError: If ``timeout`` elapses before all finish.

    Example:
        >>> a, b, c = await wait_for_all(x(), y(), z(), timeout=5.0)
    """
    if timeout:
        return await asyncio.wait_for(asyncio.gather(*awaitables), timeout=timeout)
    return await asyncio.gather(*awaitables)


async def wait_for_first(*awaitables: Awaitable[T], timeout: float | None = None) -> T:
    """Return the FIRST awaitable to finish; cancel the losers.

    The "race" pattern: query three mirror servers, take whichever
    answers first, cancel the other two requests to free resources.

    Args:
        *awaitables: The competing operations.
        timeout: Max seconds to wait for ANY of them.

    Returns:
        The result of the first completor.

    Raises:
        TimeoutError: If nothing finished within ``timeout``.
        Exception: Whatever the first finisher raised, if it failed.

    Example:
        >>> fastest = await wait_for_first(query_a(), query_b())
    """
    tasks = [asyncio.ensure_future(a) for a in awaitables]
    done, pending = await asyncio.wait(
        tasks, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
    )
    if not done:
        raise TimeoutError("No awaitable completed in time")
    # cancel the losers so they do not keep consuming resources in vain
    for task in pending:
        task.cancel()
    return done.pop().result()


# Example usage
if __name__ == "__main__":

    async def demo() -> None:
        print("Testing helpers...")

        # gather_with_concurrency
        async def task(i: int) -> int:
            await asyncio.sleep(0.05)
            return i * 2

        results = await gather_with_concurrency(3, *[task(i) for i in range(10)])
        print(f"Concurrency limited: {results}")

        # AsyncCache
        cache: AsyncCache[int] = AsyncCache(ttl=1.0)
        call_count = 0

        async def expensive(x: int) -> int:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return x * 2

        r1 = await cache.get_or_set("key1", lambda: expensive(5))
        r2 = await cache.get_or_set("key1", lambda: expensive(5))
        print(f"Cache: {r1}, {r2}, calls: {call_count}")

        # AsyncRateLimiter
        limiter = AsyncRateLimiter(10, burst=3)
        start = time.monotonic()
        for _ in range(5):
            await limiter.acquire()
        elapsed = time.monotonic() - start
        print(f"Rate limited 5 requests in {elapsed:.3f}s")

    asyncio.run(demo())
