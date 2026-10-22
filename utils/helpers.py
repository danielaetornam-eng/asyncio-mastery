"""
Helper Functions

Common async helper functions and utilities.
"""

import asyncio
import functools
import threading
import time
import weakref
from collections.abc import Awaitable, Callable, Coroutine
from functools import wraps
from typing import Any, Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run coroutine in current or new event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(coro)


def run_sync(func: Callable[..., T], *args, **kwargs) -> T:
    """Run sync function in thread pool."""
    loop = asyncio.get_running_loop()
    return loop.run_in_executor(None, functools.partial(func, *args, **kwargs))


def ensure_async(func: Callable[..., T]) -> Callable[..., Coroutine[Any, Any, T]]:
    """Ensure a function is async."""
    if asyncio.iscoroutinefunction(func):
        return func

    @wraps(func)
    async def async_func(*args, **kwargs) -> T:
        return func(*args, **kwargs)

    return async_func


def async_property(func: Callable[..., T]) -> property:
    """Create async property."""

    @wraps(func)
    async def wrapper(self):
        return await func(self)

    return property(wrapper)


def async_cached_property(func: Callable[..., T]) -> property:
    """Create async cached property."""
    cache_name = f"_cached_{func.__name__}"

    @wraps(func)
    async def wrapper(self):
        if not hasattr(self, cache_name):
            setattr(self, cache_name, await func(self))
        return getattr(self, cache_name)

    return property(wrapper)


class Singleton(type):
    """Singleton metaclass."""

    _instances: dict[type, Any] = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


def singleton(cls: type[T]) -> type[T]:
    """Decorator to make class a singleton."""
    instances: dict[type, T] = {}
    lock = threading.Lock()

    @wraps(cls)
    def wrapper(*args, **kwargs) -> T:
        with lock:
            if cls not in instances:
                instances[cls] = cls(*args, **kwargs)
            return instances[cls]

    return wrapper


class WeakRefDict(dict):
    """Dictionary with weak references to values."""

    def __setitem__(self, key: Any, value: Any) -> None:
        if hasattr(value, "__weakref__"):
            super().__setitem__(key, weakref.ref(value))
        else:
            super().__setitem__(key, value)

    def __getitem__(self, key: Any) -> Any:
        ref = super().__getitem__(key)
        if isinstance(ref, weakref.ref):
            obj = ref()
            if obj is None:
                raise KeyError(key)
            return obj
        return ref


async def gather_with_concurrency(
    *coros: Coroutine[Any, Any, T], max_concurrent: int, return_exceptions: bool = False
) -> list[T | BaseException]:
    """Gather coroutines with concurrency limit."""
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


async def wait_for_any(*coros: Coroutine[Any, Any, T], timeout: float | None = None) -> T:
    """Wait for any coroutine to complete."""
    tasks = [asyncio.create_task(c) for c in coros]

    done, pending = await asyncio.wait(tasks, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)

    for task in pending:
        task.cancel()

    if not done:
        raise TimeoutError()

    return done.pop().result()


async def wait_for_all(
    *coros: Coroutine[Any, Any, T], timeout: float | None = None, return_exceptions: bool = False
) -> list[T | BaseException]:
    """Wait for all coroutines to complete."""
    tasks = [asyncio.create_task(c) for c in coros]

    if timeout:
        _done, pending = await asyncio.wait(
            tasks, timeout=timeout, return_when=asyncio.ALL_COMPLETED
        )

        for task in pending:
            task.cancel()

        if return_exceptions:
            return [
                t.result() if t.done() and not t.cancelled() else asyncio.CancelledError()
                for t in tasks
            ]

    return await asyncio.gather(*tasks, return_exceptions=return_exceptions)


class AsyncLazy(Generic[T]):
    """Lazy async initialization."""

    def __init__(self, factory: Callable[[], Awaitable[T]]) -> None:
        self._factory = factory
        self._value: T | None = None
        self._initialized = False
        self._lock = asyncio.Lock()

    async def get(self) -> T:
        if self._initialized:
            return self._value

        async with self._lock:
            if self._initialized:
                return self._value

            self._value = await self._factory()
            self._initialized = True
            return self._value

    def reset(self) -> None:
        self._value = None
        self._initialized = False


class AsyncMemoize(Generic[T, R]):
    """Memoize async function results."""

    def __init__(self, func: Callable[..., Awaitable[R]], maxsize: int = 128) -> None:
        self._func = func
        self._maxsize = maxsize
        self._cache: dict[Any, R] = {}
        self._lock = asyncio.Lock()

    def __call__(self, *args, **kwargs) -> Awaitable[R]:
        return self.get(*args, **kwargs)

    async def get(self, *args, **kwargs) -> R:
        key = (args, frozenset(kwargs.items()))

        async with self._lock:
            if key in self._cache:
                return self._cache[key]

        result = await self._func(*args, **kwargs)

        async with self._lock:
            if len(self._cache) >= self._maxsize:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = result

        return result

    def clear(self) -> None:
        self._cache.clear()


def memoize_async(maxsize: int = 128) -> Callable:
    """Decorator for memoizing async functions."""

    def decorator(func: Callable[..., Awaitable[R]]) -> Callable[..., Awaitable[R]]:
        memo = AsyncMemoize(func, maxsize)

        @wraps(func)
        async def wrapper(*args, **kwargs) -> R:
            return await memo.get(*args, **kwargs)

        return wrapper

    return decorator


class AsyncRetry:
    """Async retry with exponential backoff."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        exceptions: tuple = (Exception,),
    ) -> None:
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.exceptions = exceptions

    def __call__(self, func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(self.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except self.exceptions as e:
                    last_exception = e
                    if attempt < self.max_attempts - 1:
                        delay = min(
                            self.base_delay * (self.exponential_base**attempt), self.max_delay
                        )
                        if self.jitter:
                            import random

                            delay *= 0.5 + random.random()
                        await asyncio.sleep(delay)

            raise last_exception

        return wrapper


def retry_async(
    max_attempts: int = 3, base_delay: float = 1.0, max_delay: float = 60.0, **kwargs
) -> Callable:
    """Decorator for async retry."""
    return AsyncRetry(max_attempts, base_delay, max_delay, **kwargs)


class AsyncTimeout:
    """Async timeout context manager."""

    def __init__(self, timeout: float, default: T | None = None) -> None:
        self._timeout = timeout
        self._default = default

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False

    async def run(self, coro: Coroutine[Any, Any, T]) -> T:
        try:
            return await asyncio.wait_for(coro, timeout=self._timeout)
        except TimeoutError:
            if self._default is not None:
                return self._default
            raise


def timeout_async(seconds: float) -> Callable:
    """Decorator for async timeout."""

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)

        return wrapper

    return decorator


def rate_limited_async(max_calls: int, period: float) -> Callable:
    """Decorator for rate limiting async functions."""
    import time
    from collections import deque

    calls = deque()
    lock = asyncio.Lock()

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            async with lock:
                now = time.monotonic()
                # Remove old calls
                while calls and calls[0] <= now - period:
                    calls.popleft()

                if len(calls) >= max_calls:
                    # Wait until we can make a call
                    wait_time = calls[0] + period - now
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                        # Retry after wait
                        return await wrapper(*args, **kwargs)

                calls.append(now)

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def measured_async(metrics: dict[str, Any] | None = None) -> Callable:
    """Decorator to measure async function performance."""
    import time

    if metrics is None:
        metrics = {}

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            start = time.monotonic()
            try:
                result = await func(*args, **kwargs)
                elapsed = time.monotonic() - start

                metrics[func.__name__] = metrics.get(func.__name__, {})
                metrics[func.__name__]["calls"] = metrics[func.__name__].get("calls", 0) + 1
                metrics[func.__name__]["total_time"] = (
                    metrics[func.__name__].get("total_time", 0) + elapsed
                )
                metrics[func.__name__]["avg_time"] = (
                    metrics[func.__name__]["total_time"] / metrics[func.__name__]["calls"]
                )

                return result
            except Exception as e:
                elapsed = time.monotonic() - start
                metrics[func.__name__] = metrics.get(func.__name__, {})
                metrics[func.__name__]["errors"] = metrics[func.__name__].get("errors", 0) + 1
                metrics[func.__name__]["last_error"] = str(e)
                raise

        return wrapper

    return decorator


def logged_async(logger: Callable[[str], None] | None = None, level: str = "INFO") -> Callable:
    """Decorator to log async function calls."""
    if logger is None:
        logger = print

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            logger(f"[{level}] Calling {func.__name__} with args={args}, kwargs={kwargs}")
            start = time.monotonic()

            try:
                result = await func(*args, **kwargs)
                elapsed = time.monotonic() - start
                logger(f"[{level}] {func.__name__} completed in {elapsed:.3f}s")
                return result
            except Exception as e:
                elapsed = time.monotonic() - start
                logger(f"[{level}] {func.__name__} failed after {elapsed:.3f}s: {e}")
                raise

        return wrapper

    return decorator


def compose(*funcs: Callable) -> Callable:
    """Compose functions right to left."""

    def composed(*args, **kwargs):
        result = funcs[-1](*args, **kwargs)
        for func in reversed(funcs[:-1]):
            result = func(result)
        return result

    return composed


def pipe(*funcs: Callable) -> Callable:
    """Pipe functions left to right."""

    def piped(*args, **kwargs):
        result = funcs[0](*args, **kwargs)
        for func in funcs[1:]:
            result = func(result)
        return result

    return piped


def curry(func: Callable) -> Callable:
    """Curry a function."""

    @wraps(func)
    def curried(*args, **kwargs):
        if len(args) + len(kwargs) >= func.__code__.co_argcount:
            return func(*args, **kwargs)

        @wraps(func)
        def partial_func(*more_args, **more_kwargs):
            return func(*(args + more_args), **{**kwargs, **more_kwargs})

        return partial_func

    return curried


def partial(func: Callable, *args, **kwargs) -> Callable:
    """Create partial function."""

    @wraps(func)
    def wrapper(*more_args, **more_kwargs):
        return func(*(args + more_args), **{**kwargs, **more_kwargs})

    return wrapper


def identity(x: T) -> T:
    """Identity function."""
    return x


def constant(value: T) -> Callable[..., T]:
    """Constant function."""
    return lambda *args, **kwargs: value


def flip(func: Callable) -> Callable:
    """Flip first two arguments of a function."""

    @wraps(func)
    def flipped(*args, **kwargs):
        if len(args) >= 2:
            return func(args[1], args[0], *args[2:], **kwargs)
        return func(*args, **kwargs)

    return flipped
