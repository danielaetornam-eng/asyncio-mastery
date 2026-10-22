"""
Retry Policies

Flexible retry mechanisms with various backoff strategies.
"""

import asyncio
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class RetryStrategy(Enum):
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    FIBONACCI = "fibonacci"
    LINEAR = "linear"
    CUSTOM = "custom"


@dataclass
class RetryConfig:
    """Configuration for retry policy."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_factor: float = 0.1
    retry_exceptions: tuple[type[Exception], ...] = (Exception,)
    stop_exceptions: tuple[type[Exception], ...] = ()
    should_retry: Callable[[Exception], bool] | None = None
    on_retry: Callable[[Exception, int], Awaitable[None]] | None = None


class RetryPolicy(ABC):
    """Abstract base class for retry policies."""

    @abstractmethod
    def get_delay(self, attempt: int) -> float:
        """Get delay for given attempt number (0-indexed)."""
        pass

    @abstractmethod
    def should_retry(self, attempt: int, exception: Exception) -> bool:
        """Determine if should retry."""
        pass


class FixedDelay(RetryPolicy):
    """Fixed delay between retries."""

    def __init__(
        self,
        delay: float = 1.0,
        max_attempts: int = 3,
        retry_exceptions: tuple[type[Exception], ...] = (Exception,),
    ) -> None:
        self._delay = delay
        self._max_attempts = max_attempts
        self._retry_exceptions = retry_exceptions

    def get_delay(self, attempt: int) -> float:
        return self._delay

    def should_retry(self, attempt: int, exception: Exception) -> bool:
        if attempt >= self._max_attempts - 1:
            return False
        return isinstance(exception, self._retry_exceptions)


class ExponentialBackoff(RetryPolicy):
    """Exponential backoff with optional jitter."""

    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        max_attempts: int = 3,
        jitter: bool = True,
        jitter_factor: float = 0.1,
        retry_exceptions: tuple[type[Exception], ...] = (Exception,),
    ) -> None:
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._exponential_base = exponential_base
        self._max_attempts = max_attempts
        self._jitter = jitter
        self._jitter_factor = jitter_factor
        self._retry_exceptions = retry_exceptions

    def get_delay(self, attempt: int) -> float:
        delay = min(self._base_delay * (self._exponential_base**attempt), self._max_delay)

        if self._jitter:
            # Add jitter: delay * (1 ± jitter_factor)
            jitter_range = delay * self._jitter_factor
            delay += random.uniform(-jitter_range, jitter_range)
            delay = max(0, delay)

        return delay

    def should_retry(self, attempt: int, exception: Exception) -> bool:
        if attempt >= self._max_attempts - 1:
            return False
        return isinstance(exception, self._retry_exceptions)


class FibonacciBackoff(RetryPolicy):
    """Fibonacci sequence backoff."""

    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        max_attempts: int = 3,
        jitter: bool = True,
        jitter_factor: float = 0.1,
        retry_exceptions: tuple[type[Exception], ...] = (Exception,),
    ) -> None:
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._max_attempts = max_attempts
        self._jitter = jitter
        self._jitter_factor = jitter_factor
        self._retry_exceptions = retry_exceptions
        self._fib_cache = {0: 0, 1: 1}

    def _fib(self, n: int) -> int:
        if n not in self._fib_cache:
            self._fib_cache[n] = self._fib(n - 1) + self._fib(n - 2)
        return self._fib_cache[n]

    def get_delay(self, attempt: int) -> float:
        fib_num = self._fib(attempt + 1)
        delay = min(self._base_delay * fib_num, self._max_delay)

        if self._jitter:
            jitter_range = delay * self._jitter_factor
            delay += random.uniform(-jitter_range, jitter_range)
            delay = max(0, delay)

        return delay

    def should_retry(self, attempt: int, exception: Exception) -> bool:
        if attempt >= self._max_attempts - 1:
            return False
        return isinstance(exception, self._retry_exceptions)


class LinearBackoff(RetryPolicy):
    """Linear backoff."""

    def __init__(
        self,
        base_delay: float = 1.0,
        increment: float = 1.0,
        max_delay: float = 60.0,
        max_attempts: int = 3,
        jitter: bool = True,
        jitter_factor: float = 0.1,
        retry_exceptions: tuple[type[Exception], ...] = (Exception,),
    ) -> None:
        self._base_delay = base_delay
        self._increment = increment
        self._max_delay = max_delay
        self._max_attempts = max_attempts
        self._jitter = jitter
        self._jitter_factor = jitter_factor
        self._retry_exceptions = retry_exceptions

    def get_delay(self, attempt: int) -> float:
        delay = min(self._base_delay + self._increment * attempt, self._max_delay)

        if self._jitter:
            jitter_range = delay * self._jitter_factor
            delay += random.uniform(-jitter_range, jitter_range)
            delay = max(0, delay)

        return delay

    def should_retry(self, attempt: int, exception: Exception) -> bool:
        if attempt >= self._max_attempts - 1:
            return False
        return isinstance(exception, self._retry_exceptions)


class CustomRetryPolicy(RetryPolicy):
    """Custom retry policy with user-defined delay function."""

    def __init__(
        self,
        delay_func: Callable[[int], float],
        max_attempts: int = 3,
        retry_exceptions: tuple[type[Exception], ...] = (Exception,),
        should_retry_func: Callable[[int, Exception], bool] | None = None,
    ) -> None:
        self._delay_func = delay_func
        self._max_attempts = max_attempts
        self._retry_exceptions = retry_exceptions
        self._should_retry_func = should_retry_func

    def get_delay(self, attempt: int) -> float:
        return max(0, self._delay_func(attempt))

    def should_retry(self, attempt: int, exception: Exception) -> bool:
        if attempt >= self._max_attempts - 1:
            return False
        if not isinstance(exception, self._retry_exceptions):
            return False
        if self._should_retry_func:
            return self._should_retry_func(attempt, exception)
        return True


class CompositeRetryPolicy(RetryPolicy):
    """Composite policy that tries multiple strategies."""

    def __init__(self, policies: list[RetryPolicy]) -> None:
        self._policies = policies

    def get_delay(self, attempt: int) -> float:
        # Use the maximum delay from all policies
        return max(p.get_delay(attempt) for p in self._policies)

    def should_retry(self, attempt: int, exception: Exception) -> bool:
        # Retry if any policy says to retry
        return any(p.should_retry(attempt, exception) for p in self._policies)


@asynccontextmanager
async def retry_context(
    policy: RetryPolicy, *, on_retry: Callable[[Exception, int], Awaitable[None]] | None = None
):
    """Context manager for retry logic."""
    attempt = 0

    while True:
        try:
            yield attempt
            return
        except Exception as _e:
            if not policy.should_retry(attempt, _e):
                raise

            if on_retry:
                await on_retry(_e, attempt)

            delay = policy.get_delay(attempt)
            if delay > 0:
                await asyncio.sleep(delay)

            attempt += 1


async def retry_async(
    func: Callable[..., Awaitable[T]],
    *args,
    policy: RetryPolicy | None = None,
    config: RetryConfig | None = None,
    **kwargs,
) -> T:
    """Execute async function with retry policy."""
    if policy is None:
        if config is None:
            config = RetryConfig()
        policy = ExponentialBackoff(
            base_delay=config.base_delay,
            max_delay=config.max_delay,
            exponential_base=config.exponential_base,
            max_attempts=config.max_attempts,
            jitter=config.jitter,
            jitter_factor=config.jitter_factor,
            retry_exceptions=config.retry_exceptions,
        )

    attempt = 0
    while True:
        try:
            return await func(*args, **kwargs)
        except Exception as _e:
            # Check stop exceptions
            if config and isinstance(_e, config.stop_exceptions):
                raise

            # Check custom should_retry
            if config and config.should_retry and not config.should_retry(_e):
                raise

            if not policy.should_retry(attempt, _e):
                raise

            if config and config.on_retry:
                await config.on_retry(_e, attempt)

            delay = policy.get_delay(attempt)
            if delay > 0:
                await asyncio.sleep(delay)

            attempt += 1


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    jitter_factor: float = 0.1,
    retry_exceptions: tuple[type[Exception], ...] = (Exception,),
    stop_exceptions: tuple[type[Exception], ...] = (),
    should_retry: Callable[[Exception], bool] | None = None,
    on_retry: Callable[[Exception, int], Awaitable[None]] | None = None,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator for retry with exponential backoff."""

    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        jitter_factor=jitter_factor,
        retry_exceptions=retry_exceptions,
        stop_exceptions=stop_exceptions,
        should_retry=should_retry,
        on_retry=on_retry,
    )

    policy = ExponentialBackoff(
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        max_attempts=max_attempts,
        jitter=jitter,
        jitter_factor=jitter_factor,
        retry_exceptions=retry_exceptions,
    )

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await retry_async(func, *args, policy=policy, config=config, **kwargs)

        return wrapper

    return decorator


class RetryStats:
    """Statistics for retry operations."""

    def __init__(self) -> None:
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.total_retries = 0
        self.total_delay = 0.0
        self.exceptions: dict[str, int] = {}

    def record_call(
        self, success: bool, retries: int, delay: float, exception: Exception | None = None
    ) -> None:
        self.total_calls += 1
        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
        self.total_retries += retries
        self.total_delay += delay
        if exception:
            exc_name = type(exception).__name__
            self.exceptions[exc_name] = self.exceptions.get(exc_name, 0) + 1

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": self.successful_calls / max(1, self.total_calls),
            "total_retries": self.total_retries,
            "avg_retries_per_call": self.total_retries / max(1, self.total_calls),
            "total_delay": self.total_delay,
            "avg_delay_per_call": self.total_delay / max(1, self.total_calls),
            "exceptions": self.exceptions.copy(),
        }


class RetryingClient(Generic[T]):
    """Client wrapper with automatic retries."""

    def __init__(
        self,
        client: T,
        policy: RetryPolicy,
        *,
        methods: list[str] | None = None,
        excluded_methods: list[str] | None = None,
    ) -> None:
        self._client = client
        self._policy = policy
        self._methods = set(methods) if methods else None
        self._excluded_methods = set(excluded_methods) if excluded_methods else set()
        self._stats = RetryStats()

    def __getattr__(self, name: str) -> object:
        attr = getattr(self._client, name)

        if not callable(attr):
            return attr

        if self._methods and name not in self._methods:
            return attr

        if name in self._excluded_methods:
            return attr

        @wraps(attr)
        async def wrapper(*args: object, **kwargs: object):
            start = time.monotonic()
            attempt = 0
            retries = 0

            while True:
                try:
                    result = await attr(*args, **kwargs)
                    self._stats.record_call(True, retries, time.monotonic() - start)
                    return result
                except Exception as e:
                    if not self._policy.should_retry(attempt, e):
                        self._stats.record_call(False, retries, time.monotonic() - start, e)
                        raise

                    delay = self._policy.get_delay(attempt)
                    if delay > 0:
                        await asyncio.sleep(delay)

                    attempt += 1
                    retries += 1

        return wrapper

    @property
    def stats(self) -> RetryStats:
        return self._stats


def create_retry_policy(
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL, **kwargs
) -> RetryPolicy:
    """Factory function for creating retry policies."""
    if strategy == RetryStrategy.FIXED:
        return FixedDelay(**kwargs)
    elif strategy == RetryStrategy.EXPONENTIAL:
        return ExponentialBackoff(**kwargs)
    elif strategy == RetryStrategy.FIBONACCI:
        return FibonacciBackoff(**kwargs)
    elif strategy == RetryStrategy.LINEAR:
        return LinearBackoff(**kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
