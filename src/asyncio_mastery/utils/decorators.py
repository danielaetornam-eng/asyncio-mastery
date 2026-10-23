"""
Decorators - Reusable wrappers that add behaviors to async functions.

WHAT IS A DECORATOR, FOR A BEGINNER?
------------------------------------
A *decorator* is a function that takes a function and returns a new,
"enhanced" one. You apply it with the ``@`` syntax::

    @async_retry(max_attempts=3)
    async def fetch(): ...

Now every call to ``fetch()`` automatically gets retry behavior -- without
changing the code inside ``fetch`` itself. Decorators keep cross-cutting
concerns (retrying, timing out, caching, rate limiting) in ONE place
instead of copy-pasting them into every function.

Notice most decorators here are *decorator factories*: ``async_retry(...)``
is called with settings and RETURNS the decorator, which then wraps your
function. That is why the definitions have three nested levels.

what is INSIDE
-------------
- :func:`async_retry` -- re-run a flaky operation with exponential backoff
  (each failure waits longer before trying again).
- :func:`async_timeout` -- fail a coroutine if it runs too long.
- :func:`async_cached` -- remember results so repeat calls are instant,
  with an optional *TTL* (time-to-live) after which entries expire.
- :func:`async_singleton` -- guarantee only ONE execution ever happens, no
  matter how many tasks call concurrently (e.g. app initialization).
- :func:`debounce` -- fire only after calls STOP coming for ``wait``
  seconds (like a search box that triggers when you stop typing).
- :func:`throttle` -- allow at most one real call per ``interval`` seconds.
- :func:`background_task` -- each call spawns an ``asyncio.Task`` instead
  of awaiting inline.

JARGON
------
- jitter: adding a random fuzz to retry delays so that many clients do not
  all retry at the exact same moment (a "thundering herd").
- functools.wraps: copies the original function's name/docstring onto the
  wrapper so debugging and introspection stay sensible.
- ParamSpec/TypeVar: typing machinery letting decorators preserve the exact
  signature of the wrapped function; safe to ignore at first.

MINIMAL EXAMPLE
---------------
>>> @async_retry(max_attempts=3, base_delay=0.1)
... async def flaky_call(): ...
>>> result = await flaky_call()
"""

import asyncio
import functools
import random
import time
from collections.abc import Awaitable, Callable, Coroutine, Hashable
from typing import Any, ParamSpec, TypeVar, cast

P = ParamSpec("P")
T = TypeVar("T")


class RetryConfig:
    """Settings bundle for :func:`async_retry`.

    Attributes:
        max_attempts: Total tries (1 = no retry, just a single attempt).
        base_delay: Seconds to wait after the FIRST failure.
        max_delay: Cap on the delay so backoff does not grow forever.
        exponential_base: Delay multiplier per attempt (2.0 = doubling:
            1s, 2s, 4s, ...). Longer waits give a struggling service time
            to recover.
        jitter: If True, randomize each delay by +/-50% to avoid many
            retries landing simultaneously.
        exceptions: Tuple of exception types that trigger a retry. Any other
            exception type propagates immediately.
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        exceptions: tuple[type[Exception], ...] = (Exception,),
    ) -> None:
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.exceptions = exceptions


def async_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Decorator: retry an async function on failure, with backoff.

    Each failure waits ``base_delay * exponential_base**attempt`` seconds
    (capped at ``max_delay``, optionally jittered) before the next attempt.
    If every attempt fails, the last exception is re-raised.

    Args:
        max_attempts: How many total attempts before giving up.
        base_delay: Initial delay in seconds after the first failure.
        max_delay: Maximum delay between attempts.
        exponential_base: Growth factor of the delay per attempt.
        jitter: Whether to randomize delays (+/-50%).
        exceptions: Which exception types are considered retryable.

    Returns:
        A decorator you apply to an async function.

    Example:
        >>> @async_retry(max_attempts=5, base_delay=0.5)
        ... async def call_api(): ...
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        exceptions=exceptions,
    )

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Exception | None = None
            for attempt in range(config.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except config.exceptions as e:
                    last_exception = e
                    # Don't sleep after the final attempt -- just give up.
                    if attempt < config.max_attempts - 1:
                        delay = min(
                            config.base_delay * (config.exponential_base**attempt), config.max_delay
                        )
                        if config.jitter:
                            # Randomize: prevents synchronized retry storms.
                            delay *= random.uniform(0.5, 1.5)
                        await asyncio.sleep(delay)
            assert last_exception is not None
            raise last_exception

        return wrapper

    return decorator


def async_timeout(
    seconds: float,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Decorator: cancel the function if it runs longer than ``seconds``.

    Uses :func:`asyncio.wait_for` under the hood. Essential for protecting
    your app from calls that hang forever.

    Args:
        seconds: Maximum runtime allowed per call.

    Returns:
        A decorator.

    Raises:
        asyncio.TimeoutError: From the wrapped function when it exceeds the
            limit (``TimeoutError`` on Python 3.11+).

    Example:
        >>> @async_timeout(2.0)
        ... async def slow_query(): ...
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)

        return wrapper

    return decorator


def async_cached(
    maxsize: int = 128, ttl: float | None = None
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Decorator: cache results of an async function.

    A *cache* stores results keyed by the call arguments; a repeated call
    with the same arguments returns the stored value instantly instead of
    recomputing. Entries expire after ``ttl`` seconds (time-to-live) if set.
    When the cache is full, the OLDEST entry is evicted to make room.

    Args:
        maxsize: Maximum number of cached results.
        ttl: Seconds a cached result stays valid; None = never expires.

    Returns:
        A decorator. The wrapped function gains two extra attributes:
        ``cache_clear()`` (empty the cache) and ``cache_info()`` returning
        ``{"size": ..., "maxsize": ...}``.

    Example:
        >>> @async_cached(maxsize=64, ttl=30)
        ... async def get_user(user_id): ...
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        # Maps argument-key -> (timestamp, result).
        cache: dict[Hashable, tuple[float, T]] = {}

        def make_key(args: tuple[object, ...], kwargs: dict[str, object]) -> Hashable:
            # Sorting kwargs makes f(a=1, b=2) and f(b=2, a=1) share a key.
            return (args, tuple(sorted(kwargs.items())))

        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            key = make_key(args, kwargs)
            now = time.monotonic()

            if key in cache:
                cached_time, value = cache[key]
                if ttl is None or now - cached_time < ttl:
                    return value  # Fresh cache hit: skip the real work.
                del cache[key]  # Expired entry: drop it.

            value = await func(*args, **kwargs)

            if len(cache) >= maxsize:
                # Remove oldest entry (the one with the smallest timestamp).
                oldest = min(cache.items(), key=lambda x: x[1][0])
                del cache[oldest[0]]

            cache[key] = (now, value)
            return value

        def cache_clear() -> None:
            """Empty the entire cache."""
            cache.clear()

        def cache_info() -> dict[str, int]:
            """Return current and maximum cache sizes."""
            return {"size": len(cache), "maxsize": maxsize}

        wrapper_any = cast("Any", wrapper)
        wrapper_any.cache_clear = cache_clear
        wrapper_any.cache_info = cache_info
        return wrapper

    return decorator


def async_singleton(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
    """Decorator: ensure the async function's body runs AT MOST once, ever.

    All callers -- even concurrent ones -- receive the same single result.
    An :class:`asyncio.Lock` guarantees that only one task enters the
    function; every later (or concurrent) caller gets the stored result.
    Perfect for one-time async initialization like "connect the database
    pool".

    Args:
        func: The async function to make single-execution.

    Returns:
        The wrapped function.

    Example:
        >>> @async_singleton
        ... async def initialize_app(): ...
    """
    lock = asyncio.Lock()
    result: T | None = None
    completed = False

    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        nonlocal result, completed
        # The lock makes the check-and-run atomic across concurrent tasks.
        async with lock:
            if completed:
                return cast("T", result)
            value = await func(*args, **kwargs)
            result = value
            completed = True
            return value

    return wrapper


def debounce(
    wait: float,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T | None]]]:
    """Decorator: run the function only after calls PAUSE for ``wait`` seconds.

    *Debounce* means: every new call resets a timer; the function executes
    only when no new call has arrived for ``wait`` seconds. Classic use:
    reacting to user typing -- you do not search on every keystroke, you
    search when the user stops typing.

    Note the wrapped function returns ``None`` immediately; the real call
    happens later in the background. A ``cancel()`` attribute is attached to
    cancel any pending execution.

    Args:
        wait: Quiet period in seconds before the deferred call fires.

    Returns:
        A decorator.

    Example:
        >>> @debounce(0.3)
        ... async def search(text): ...
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T | None]]:
        timer: asyncio.TimerHandle | None = None
        pending_args: tuple[tuple[object, ...], dict[str, object]] | None = None
        pending_task: asyncio.Task[Any] | None = None

        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T | None:
            nonlocal timer, pending_args, pending_task
            # Always overwrite: only the LATEST call's arguments are used.
            pending_args = (cast("tuple[object, ...]", args), cast("dict[str, object]", kwargs))

            def execute() -> None:
                nonlocal pending_args, pending_task
                if pending_args:
                    args_, kwargs_ = pending_args
                    pending_task = asyncio.create_task(
                        cast(
                            "Coroutine[Any, Any, T]",
                            func(*cast("Any", args_), **cast("Any", kwargs_)),
                        )
                    )
                    pending_args = None

            if timer:
                timer.cancel()  # A new call arrived: reset the countdown.
            timer = asyncio.get_event_loop().call_later(wait, execute)
            return None

        def cancel() -> None:
            """Cancel any pending scheduled or running execution."""
            if timer:
                timer.cancel()
            if pending_task:
                pending_task.cancel()

        wrapper_any = cast("Any", wrapper)
        wrapper_any.cancel = cancel
        return wrapper

    return decorator


def throttle(
    interval: float,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T | None]]]:
    """Decorator: allow at most one real call per ``interval`` seconds.

    *Throttle* is debounce's cousin: the FIRST call in each interval goes
    through immediately; extra calls inside the same interval are dropped
    (return ``None``). Use it to respect API rate limits or to cap UI
    updates.

    Args:
        interval: Minimum seconds between real executions.

    Returns:
        A decorator. Wrapped calls return the real result, or ``None`` if
        throttled.

    Example:
        >>> @throttle(1.0)
        ... async def notify(msg): ...
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T | None]]:
        last_called = 0.0
        lock = asyncio.Lock()

        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T | None:
            nonlocal last_called
            # The lock prevents two concurrent calls from both passing the
            # time check at once.
            async with lock:
                now = time.monotonic()
                if now - last_called >= interval:
                    last_called = now
                    return await func(*args, **kwargs)
                return None

        return wrapper

    return decorator


def background_task(
    name: str | None = None,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, asyncio.Task[T]]]:
    """Decorator: every call starts the coroutine as a background Task.

    Instead of awaiting the function inline, calling it schedules it on the
    event loop via :func:`asyncio.create_task` and returns the Task
    immediately. Await the returned task later if you need the result.

    Args:
        name: Optional task name (nice for debugging via repr(task));
            defaults to the function's name.

    Returns:
        A decorator.

    Example:
        >>> @background_task()
        ... async def cleanup(): ...
        >>> task = cleanup()       # starts now, in the background
        >>> await task             # collect result when needed
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, asyncio.Task[T]]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> asyncio.Task[T]:
            # NOTE: not `async def` on purpose -- create_task itself is sync.
            return asyncio.create_task(cast("Coroutine[Any, Any, T]", func(*args, **kwargs)), name=name or func.__name__)

        return wrapper

    return decorator


# Example usage
if __name__ == "__main__":
    import asyncio

    class FlakyState:
        """Holds the failure counter for the demo's fake-flaky coroutine.

        A small class (instead of a global) keeps the demo state tidy.
        """

        def __init__(self) -> None:
            self.call_count = 0

    flaky_state = FlakyState()

    @async_retry(max_attempts=3, base_delay=0.05)
    async def flaky() -> str:
        """Fails twice, then succeeds -- to show retry in action."""
        flaky_state.call_count += 1
        if flaky_state.call_count < 3:
            raise ConnectionError("Flaky")
        return "success"

    @async_cached(ttl=1.0)
    async def cached(x: int) -> int:
        await asyncio.sleep(0.01)
        return x * 2

    @debounce(0.1)
    async def debounced(msg: str) -> None:
        print(f"Debounced: {msg}")

    @throttle(0.1)
    async def throttled(msg: str) -> None:
        print(f"Throttled: {msg}")

    @background_task()
    async def background(msg: str) -> str:
        await asyncio.sleep(0.05)
        return f"Background: {msg}"

    async def test() -> None:
        """Exercise every decorator and print what happens."""
        print("Testing decorators...")

        # Retry
        result = await flaky()
        print(f"Retry result: {result}")

        # Cache: second call is instant (served from memory).
        r1 = await cached(5)
        r2 = await cached(5)
        print(f"Cache: {r1}, {r2}")

        # Debounce: only "second" prints, after the pause.
        await debounced("first")
        await debounced("second")
        await asyncio.sleep(0.15)

        # Throttle: "2" is dropped; "3" fires after the interval.
        await throttled("1")
        await throttled("2")
        await asyncio.sleep(0.15)
        await throttled("3")

        # Background
        task = background("test")
        result = await task
        print(f"Background result: {result}")

    asyncio.run(test())
