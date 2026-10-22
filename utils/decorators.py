"""
Decorators

Async decorators for common patterns.
"""

import asyncio
import functools
import random
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from functools import wraps
from typing import Any, TypeVar

T = TypeVar("T")
R = TypeVar("R")


@asynccontextmanager
async def asynccontextmanager(func: Callable) -> AsyncGenerator[Any, None]:
    """Async context manager decorator."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        gen = func(*args, **kwargs)
        try:
            value = await gen.__anext__()
        except StopAsyncIteration:
            raise RuntimeError("async generator didn't yield")

        try:
            yield value
        except Exception as e:
            with suppress(StopAsyncIteration):
                await gen.athrow(e)
            raise
        else:
            with suppress(StopAsyncIteration):
                await gen.__anext__()

    return wrapper


def async_cached(
    maxsize: int = 128, ttl: float | None = None, key_func: Callable | None = None
) -> Callable:
    """Async caching decorator."""
    cache: dict[str, tuple] = {}  # key -> (value, timestamp)
    lock = asyncio.Lock()

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = f"{func.__module__}.{func.__name__}:{args}:{frozenset(kwargs.items())}"

            now = time.monotonic()

            async with lock:
                if key in cache:
                    value, timestamp = cache[key]
                    if ttl is None or (now - timestamp) < ttl:
                        return value

            result = await func(*args, **kwargs)

            async with lock:
                if len(cache) >= maxsize:
                    # Remove oldest
                    oldest_key = min(cache.keys(), key=lambda k: cache[k][1])
                    del cache[oldest_key]
                cache[key] = (result, now)

            return result

        wrapper.cache = cache
        wrapper.clear_cache = cache.clear
        return wrapper

    return decorator


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: tuple = (Exception,),
) -> Callable:
    """Retry decorator with exponential backoff."""

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        delay = min(base_delay * (exponential_base**attempt), max_delay)
                        if jitter:
                            delay *= 0.5 + random.random()
                        await asyncio.sleep(delay)

            raise last_exception

        return wrapper

    return decorator


def timeout(seconds: float) -> Callable:
    """Timeout decorator for async functions."""

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)

        return wrapper

    return decorator


def rate_limited(max_calls: int, period: float) -> Callable:
    """Rate limit decorator for async functions."""
    calls: list[float] = []
    lock = asyncio.Lock()

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            async with lock:
                now = time.monotonic()
                # Remove old calls
                while calls and calls[0] <= now - period:
                    calls.pop(0)

                if len(calls) >= max_calls:
                    wait_time = calls[0] + period - now
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                        # Retry after wait
                        return await wrapper(*args, **kwargs)

                calls.append(now)

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def measured(metrics: dict[str, Any] | None = None) -> Callable:
    """Decorator to measure function performance."""
    if metrics is None:
        metrics = {}

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            start = time.monotonic()
            try:
                result = await func(*args, **kwargs)
                elapsed = time.monotonic() - start

                if func.__name__ not in metrics:
                    metrics[func.__name__] = {"calls": 0, "total_time": 0.0, "errors": 0}

                metrics[func.__name__]["calls"] += 1
                metrics[func.__name__]["total_time"] += elapsed
                metrics[func.__name__]["avg_time"] = (
                    metrics[func.__name__]["total_time"] / metrics[func.__name__]["calls"]
                )

                return result
            except Exception as e:
                elapsed = time.monotonic() - start
                if func.__name__ not in metrics:
                    metrics[func.__name__] = {"calls": 0, "total_time": 0.0, "errors": 0}
                metrics[func.__name__]["errors"] += 1
                metrics[func.__name__]["last_error"] = str(e)
                raise

        return wrapper

    return decorator


def logged(
    logger: Callable[[str], None] | None = None,
    level: str = "INFO",
    log_args: bool = True,
    log_result: bool = False,
) -> Callable:
    """Decorator to log function calls."""
    if logger is None:
        logger = print

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            if log_args:
                logger(f"[{level}] {func.__name__} called with args={args}, kwargs={kwargs}")
            else:
                logger(f"[{level}] {func.__name__} called")

            start = time.monotonic()

            try:
                result = await func(*args, **kwargs)
                elapsed = time.monotonic() - start

                if log_result:
                    logger(f"[{level}] {func.__name__} returned {result} in {elapsed:.3f}s")
                else:
                    logger(f"[{level}] {func.__name__} completed in {elapsed:.3f}s")

                return result
            except Exception as e:
                elapsed = time.monotonic() - start
                logger(f"[{level}] {func.__name__} failed after {elapsed:.3f}s: {e}")
                raise

        return wrapper

    return decorator


def singleton_async(cls: type) -> type:
    """Make class a singleton with async initialization."""
    instances: dict[type, Any] = {}
    locks: dict[type, asyncio.Lock] = {}

    original_new = cls.__new__

    @wraps(cls.__new__)
    async def async_new(cls, *args, **kwargs):
        if cls not in instances:
            if cls not in locks:
                locks[cls] = asyncio.Lock()

            async with locks[cls]:
                if cls not in instances:
                    instance = original_new(cls)
                    if asyncio.iscoroutinefunction(cls.__init__):
                        await cls.__init__(instance, *args, **kwargs)
                    else:
                        cls.__init__(instance, *args, **kwargs)
                    instances[cls] = instance
        return instances[cls]

    cls.__new__ = staticmethod(async_new)
    return cls


def deprecated_async(reason: str = "", version: str = "", alternative: str = "") -> Callable:
    """Mark async function as deprecated."""

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            import warnings

            msg = f"{func.__name__} is deprecated"
            if reason:
                msg += f": {reason}"
            if version:
                msg += f" (since v{version})"
            if alternative:
                msg += f". Use {alternative} instead."

            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return await func(*args, **kwargs)

        return wrapper

    return decorator


def validate_args_async(
    validator: Callable[..., bool], error_msg: str = "Validation failed"
) -> Callable:
    """Validate async function arguments."""

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            if not validator(*args, **kwargs):
                raise ValueError(error_msg)
            return await func(*args, **kwargs)

        return wrapper

    return decorator


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


def circuit_breaker_async(
    failure_threshold: int = 5, recovery_timeout: float = 30.0, expected_exception: type = Exception
) -> Callable:
    """Circuit breaker for async functions."""
    state = {"closed": True, "failures": 0, "last_failure": 0}
    lock = asyncio.Lock()

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            async with lock:
                if not state["closed"]:
                    if time.monotonic() - state["last_failure"] > recovery_timeout:
                        state["closed"] = True
                        state["failures"] = 0
                    else:
                        raise Exception("Circuit breaker open")

            try:
                result = await func(*args, **kwargs)

                async with lock:
                    state["failures"] = 0
                return result
            except expected_exception:
                async with lock:
                    state["failures"] += 1
                    state["last_failure"] = time.monotonic()
                    if state["failures"] >= failure_threshold:
                        state["closed"] = False
                raise

        return wrapper

    return decorator


def bulkhead_async(max_concurrent: int = 10, timeout: float | None = None) -> Callable:
    """Bulkhead pattern for async functions."""
    semaphore = asyncio.Semaphore(max_concurrent)

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            async with semaphore:
                if timeout:
                    return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
                return await func(*args, **kwargs)

        return wrapper

    return decorator


def fallback_async(
    fallback_value: Any = None,
    fallback_func: Callable | None = None,
    exceptions: tuple = (Exception,),
) -> Callable:
    """Fallback for async functions."""

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            try:
                return await func(*args, **kwargs)
            except exceptions:
                if fallback_func:
                    return await fallback_func(*args, **kwargs)
                return fallback_value

        return wrapper

    return decorator


def hedge_async(hedge_after: float = 0.1, max_hedges: int = 2) -> Callable:
    """Request hedging for async functions."""

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            # Start primary
            primary = asyncio.create_task(func(*args, **kwargs))

            try:
                return await asyncio.wait_for(primary, timeout=hedge_after)
            except TimeoutError:
                pass

            # Start hedges
            hedges = [asyncio.create_task(func(*args, **kwargs)) for _ in range(max_hedges)]

            # Wait for any
            done, pending = await asyncio.wait(
                [primary, *hedges], return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel pending
            for task in pending:
                task.cancel()

            return done.pop().result()

        return wrapper

    return decorator


def idempotent_async(key_func: Callable | None = None, ttl: float | None = None) -> Callable:
    """Make async function idempotent."""
    results: dict[str, Any] = {}
    lock = asyncio.Lock()

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = f"{func.__module__}.{func.__name__}:{args}:{frozenset(kwargs.items())}"

            async with lock:
                if key in results:
                    return results[key]

            result = await func(*args, **kwargs)

            async with lock:
                results[key] = result

            return result

        wrapper.results = results
        return wrapper

    return decorator


# Decorator composition
def compose_decorators(*decorators: Callable) -> Callable:
    """Compose multiple decorators."""

    def decorator(func):
        for dec in reversed(decorators):
            func = dec(func)
        return func

    return decorator


def conditional_decorator(condition: Callable[..., bool], decorator: Callable) -> Callable:
    """Apply decorator conditionally."""

    def wrapper(func):
        if condition(func):
            return decorator(func)
        return func

    return wrapper
