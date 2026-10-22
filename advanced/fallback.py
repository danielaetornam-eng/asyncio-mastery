"""
Fallback Patterns

Graceful degradation and fallback mechanisms.
"""

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class FallbackConfig:
    """Configuration for fallback behavior."""

    exceptions: tuple[type, ...] = (Exception,)
    fallback_value: Any | None = None
    fallback_func: Callable[..., Awaitable[T]] | None = None
    on_fallback: Callable[[Exception], Awaitable[None]] | None = None
    max_fallback_attempts: int = 1


@dataclass
class FallbackStats:
    """Fallback statistics."""

    total_calls: int = 0
    primary_success: int = 0
    fallback_used: int = 0
    fallback_failed: int = 0
    exceptions_caught: dict[str, int] = field(default_factory=dict)


class Fallback(Generic[T]):
    """Fallback mechanism for async operations."""

    def __init__(
        self,
        primary: Callable[..., Awaitable[T]],
        fallback: T | Callable[..., Awaitable[T]],
        config: FallbackConfig | None = None,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._config = config or FallbackConfig()
        self._stats = FallbackStats()

    @property
    def stats(self) -> FallbackStats:
        return self._stats

    async def execute(self, *args, **kwargs) -> T:
        """Execute primary with fallback."""
        self._stats.total_calls += 1

        try:
            result = await self._primary(*args, **kwargs)
            self._stats.primary_success += 1
            return result
        except self._config.exceptions as e:
            exc_name = type(e).__name__
            self._stats.exceptions_caught[exc_name] = (
                self._stats.exceptions_caught.get(exc_name, 0) + 1
            )

            if self._config.on_fallback:
                await self._config.on_fallback(e)

            return await self._execute_fallback(e, *args, **kwargs)

    async def _execute_fallback(self, exception: Exception, *args, **kwargs) -> T:
        self._stats.fallback_used += 1

        try:
            if callable(self._fallback):
                result = await self._fallback(*args, **kwargs)
            else:
                result = self._fallback

            return result
        except Exception as _e:
            self._stats.fallback_failed += 1
            raise


class ChainedFallback(Generic[T]):
    """Chain of fallbacks - tries each in order."""

    def __init__(
        self, functions: list[Callable[..., Awaitable[T]]], config: FallbackConfig | None = None
    ) -> None:
        if not functions:
            raise ValueError("At least one function required")

        self._functions = functions
        self._config = config or FallbackConfig()
        self._stats = FallbackStats()

    @property
    def stats(self) -> FallbackStats:
        return self._stats

    async def execute(self, *args, **kwargs) -> T:
        last_exception = None

        for i, func in enumerate(self._functions):
            self._stats.total_calls += 1

            try:
                result = await func(*args, **kwargs)
                if i == 0:
                    self._stats.primary_success += 1
                else:
                    self._stats.fallback_used += 1
                return result
            except self._config.exceptions as e:
                last_exception = e
                exc_name = type(e).__name__
                self._stats.exceptions_caught[exc_name] = (
                    self._stats.exceptions_caught.get(exc_name, 0) + 1
                )

                if self._config.on_fallback:
                    await self._config.on_fallback(e)

                continue

        self._stats.fallback_failed += 1
        raise last_exception


class FallbackPolicy(Generic[T]):
    """Policy-based fallback with conditions."""

    def __init__(
        self,
        primary: Callable[..., Awaitable[T]],
        fallbacks: list[tuple[Callable[[Exception], bool], Callable[..., Awaitable[T]]]],
        *,
        default: T | None = None,
        on_fallback: Callable[[Exception, int], Awaitable[None]] | None = None,
    ) -> None:
        self._primary = primary
        self._fallbacks = fallbacks
        self._default = default
        self._on_fallback = on_fallback
        self._stats = FallbackStats()

    @property
    def stats(self) -> FallbackStats:
        return self._stats

    async def execute(self, *args, **kwargs) -> T:
        self._stats.total_calls += 1

        try:
            result = await self._primary(*args, **kwargs)
            self._stats.primary_success += 1
            return result
        except Exception as e:
            exc_name = type(e).__name__
            self._stats.exceptions_caught[exc_name] = (
                self._stats.exceptions_caught.get(exc_name, 0) + 1
            )

            for i, (condition, fallback) in enumerate(self._fallbacks):
                if condition(e):
                    self._stats.fallback_used += 1

                    if self._on_fallback:
                        await self._on_fallback(e, i)

                    try:
                        return await fallback(*args, **kwargs)
                    except Exception:
                        self._stats.fallback_failed += 1
                        continue

            if self._default is not None:
                return self._default

            raise


@asynccontextmanager
async def fallback_context(
    fallback_value: T,
    exceptions: tuple[type, ...] = (Exception,),
    on_fallback: Callable[[Exception], Awaitable[None]] | None = None,
):
    """Context manager for fallback."""
    try:
        yield
    except exceptions as e:
        if on_fallback:
            await on_fallback(e)
        # Can't actually return fallback_value from context manager
        # This is more for cleanup/notification
        raise


def fallback(
    fallback_value: T | None = None,
    fallback_func: Callable[..., Awaitable[T]] | None = None,
    exceptions: tuple[type, ...] = (Exception,),
    on_fallback: Callable[[Exception], Awaitable[None]] | None = None,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator for fallback."""

    if fallback_value is None and fallback_func is None:
        raise ValueError("Either fallback_value or fallback_func must be provided")

    config = FallbackConfig(
        exceptions=exceptions,
        fallback_value=fallback_value,
        fallback_func=fallback_func,
        on_fallback=on_fallback,
    )

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        fallback_handler = Fallback(func, fallback_value or fallback_func, config)

        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await fallback_handler.execute(*args, **kwargs)

        return wrapper

    return decorator


class ConditionalFallback(Generic[T]):
    """Fallback that triggers based on result conditions."""

    def __init__(
        self,
        primary: Callable[..., Awaitable[T]],
        fallback: T | Callable[..., Awaitable[T]],
        condition: Callable[[T], bool],
        exceptions: tuple[type, ...] = (Exception,),
        on_fallback: Callable[[Exception], Awaitable[None]] | None = None,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._condition = condition
        self._exceptions = exceptions
        self._on_fallback = on_fallback
        self._stats = FallbackStats()

    @property
    def stats(self) -> FallbackStats:
        return self._stats

    async def execute(self, *args, **kwargs) -> T:
        self._stats.total_calls += 1

        try:
            result = await self._primary(*args, **kwargs)

            if self._condition(result):
                self._stats.primary_success += 1
                return result

            # Result doesn't meet condition, use fallback
            if self._on_fallback:
                await self._on_fallback(Exception("Condition not met"))

            self._stats.fallback_used += 1

            if callable(self._fallback):
                return await self._fallback(*args, **kwargs)
            return self._fallback

        except self._exceptions as e:
            if self._on_fallback:
                await self._on_fallback(e)

            self._stats.exceptions_caught[type(e).__name__] = (
                self._stats.exceptions_caught.get(type(e).__name__, 0) + 1
            )
            self._stats.fallback_used += 1

            if callable(self._fallback):
                return await self._fallback(*args, **kwargs)
            return self._fallback


class RetryWithFallback(Generic[T]):
    """Retry primary, then fallback."""

    def __init__(
        self,
        primary: Callable[..., Awaitable[T]],
        fallback: T | Callable[..., Awaitable[T]],
        max_retries: int = 2,
        base_delay: float = 1.0,
        exceptions: tuple[type, ...] = (Exception,),
        on_retry: Callable[[Exception, int], Awaitable[None]] | None = None,
        on_fallback: Callable[[Exception], Awaitable[None]] | None = None,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._exceptions = exceptions
        self._on_retry = on_retry
        self._on_fallback = on_fallback
        self._stats = FallbackStats()

    @property
    def stats(self) -> FallbackStats:
        return self._stats

    async def execute(self, *args, **kwargs) -> T:
        self._stats.total_calls += 1
        last_exception = None

        for attempt in range(self._max_retries + 1):
            try:
                result = await self._primary(*args, **kwargs)
                self._stats.primary_success += 1
                return result
            except self._exceptions as e:
                last_exception = e
                exc_name = type(e).__name__
                self._stats.exceptions_caught[exc_name] = (
                    self._stats.exceptions_caught.get(exc_name, 0) + 1
                )

                if attempt < self._max_retries:
                    if self._on_retry:
                        await self._on_retry(e, attempt)
                    delay = self._base_delay * (2**attempt)
                    await asyncio.sleep(delay)
                else:
                    # All retries exhausted, use fallback
                    if self._on_fallback:
                        await self._on_fallback(e)

                    self._stats.fallback_used += 1

                    if callable(self._fallback):
                        return await self._fallback(*args, **kwargs)
                    return self._fallback

        self._stats.fallback_failed += 1
        raise last_exception


def create_fallback_chain(
    *functions: Callable[..., Awaitable[T]], exceptions: tuple[type, ...] = (Exception,)
) -> ChainedFallback[T]:
    """Create a chained fallback from functions."""
    return ChainedFallback(list(functions), FallbackConfig(exceptions=exceptions))


def fallback_value(value: T) -> Callable[[], Awaitable[T]]:
    """Create a fallback function that returns a fixed value."""

    async def _fallback() -> T:
        return value

    return _fallback


def fallback_exception(exc: Exception) -> Callable[[], Awaitable[None]]:
    """Create a fallback function that raises an exception."""

    async def _fallback() -> None:
        raise exc

    return _fallback
