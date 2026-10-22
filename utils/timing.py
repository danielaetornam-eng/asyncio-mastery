"""
Timing Utilities

Time measurement, budgets, and rate limiting utilities.
"""

import asyncio
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass
class Timer:
    """High-resolution timer."""

    _start: float = field(default_factory=time.monotonic)
    _end: float | None = None
    _laps: list[float] = field(default_factory=list)
    _running: bool = True

    @property
    def elapsed(self) -> float:
        if self._end is not None:
            return self._end - self._start
        return time.monotonic() - self._start

    @property
    def running(self) -> bool:
        return self._running

    @property
    def laps(self) -> list[float]:
        return self._laps.copy()

    def stop(self) -> float:
        if self._running:
            self._end = time.monotonic()
            self._running = False
        return self.elapsed

    def lap(self) -> float:
        now = time.monotonic()
        last = self._laps[-1] if self._laps else self._start
        lap_time = now - last
        self._laps.append(now)
        return lap_time

    def reset(self) -> None:
        self._start = time.monotonic()
        self._end = None
        self._laps.clear()
        self._running = True

    def __enter__(self) -> "Timer":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    @classmethod
    def start_new(cls) -> "Timer":
        return cls()


@asynccontextmanager
async def measure_time(
    callback: Callable[[float], None] | None = None,
) -> AsyncGenerator[Timer, None]:
    """Context manager to measure execution time."""
    timer = Timer()
    try:
        yield timer
    finally:
        timer.stop()
        if callback:
            callback(timer.elapsed)


class TimeBudget:
    """Budget-based time management for operations."""

    def __init__(self, total_time: float):
        self._total_time = total_time
        self._start_time = time.monotonic()
        self._spent: dict[str, float] = {}
        _lock = asyncio.Lock()

    @property
    def total(self) -> float:
        return self._total_time

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start_time

    @property
    def remaining(self) -> float:
        return max(0, self._total_time - self.elapsed)

    @property
    def expired(self) -> bool:
        return self.remaining <= 0

    def allocate(self, name: str, time: float) -> bool:
        """Allocate time budget for an operation."""
        if self._spent.get(name, 0) + time > self.remaining:
            return False
        self._spent[name] = self._spent.get(name, 0) + time
        return True

    def spent(self, name: str) -> float:
        return self._spent.get(name, 0)

    def get_stats(self) -> dict[str, Any]:
        return {
            "total": self._total_time,
            "elapsed": self.elapsed,
            "remaining": self.remaining,
            "expired": self.expired,
            "spent": self._spent.copy(),
        }

    @asynccontextmanager
    async def spend(self, name: str, budget: float) -> AsyncGenerator[None, None]:
        """Context manager for spending time budget."""
        start = time.monotonic()

        if not self.allocate(name, budget):
            raise TimeoutError(f"Budget exceeded for {name}")

        try:
            yield
        finally:
            actual = time.monotonic() - start
            # Adjust allocation
            self._spent[name] = self._spent.get(name, 0) - budget + actual


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, rate: float, burst: int = 0, *, name: str | None = None):
        """
        Args:
            rate: Tokens per second
            burst: Maximum burst tokens (default: rate * 2)
        """
        self._rate = rate
        self._capacity = burst if burst > 0 else int(rate * 2)
        self._tokens = float(self._capacity)
        self._last_refill = time.monotonic()
        self._name = name or f"RateLimiter-{id(self)}"
        self._lock = asyncio.Lock()

    @property
    def rate(self) -> float:
        return self._rate

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def tokens(self) -> float:
        self._refill()
        return self._tokens

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_refill = now

    async def acquire(self, tokens: int = 1, timeout: float | None = None) -> bool:
        if tokens <= 0:
            return True

        start = time.monotonic()

        while True:
            async with self._lock:
                self._refill()

                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True

                needed = tokens - self._tokens
                wait_time = needed / self._rate

            if timeout is not None:
                if time.monotonic() - start + wait_time > timeout:
                    return False
                wait_time = min(wait_time, timeout - (time.monotonic() - start))

            if wait_time > 0:
                await asyncio.sleep(wait_time)

    def try_acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens without blocking."""
        self._refill()

        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False

    async def release(self, tokens: int = 1) -> None:
        """Release tokens back to bucket."""
        async with self._lock:
            self._refill()
            self._tokens = min(self._capacity, self._tokens + tokens)

    def get_stats(self) -> dict[str, Any]:
        return {
            "name": self._name,
            "rate": self._rate,
            "capacity": self._capacity,
            "tokens": self.tokens,
        }


class SlidingWindowRateLimiter:
    """Sliding window rate limiter."""

    def __init__(self, max_requests: int, window: float, *, name: str | None = None):
        self._max_requests = max_requests
        self._window = window
        self._name = name or f"SlidingWindowRateLimiter-{id(self)}"
        self._requests: list[float] = []
        self._lock = asyncio.Lock()

    def _clean_old(self) -> None:
        now = time.monotonic()
        cutoff = now - self._window
        while self._requests and self._requests[0] < cutoff:
            self._requests.pop(0)

    async def acquire(self, tokens: int = 1, timeout: float | None = None) -> bool:
        if tokens <= 0:
            return True

        start = time.monotonic()

        while True:
            async with self._lock:
                self._clean_old()

                if len(self._requests) + tokens <= self._max_requests:
                    now = time.monotonic()
                    for _ in range(tokens):
                        self._requests.append(now)
                    return True

                if not self._requests:
                    return False

                wait_time = self._requests[0] + self._window - time.monotonic()

            if timeout is not None:
                if time.monotonic() - start + wait_time > timeout:
                    return False
                wait_time = min(wait_time, timeout - (time.monotonic() - start))

            if wait_time > 0:
                await asyncio.sleep(wait_time)

    def try_acquire(self, tokens: int = 1) -> bool:
        self._clean_old()

        if len(self._requests) + tokens <= self._max_requests:
            now = time.monotonic()
            for _ in range(tokens):
                self._requests.append(now)
            return True
        return False

    def available(self) -> int:
        self._clean_old()
        return max(0, self._max_requests - len(self._requests))


class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts based on success/failure."""

    def __init__(
        self,
        initial_rate: float,
        min_rate: float = 0.1,
        max_rate: float = 1000.0,
        *,
        name: str | None = None,
        increase_factor: float = 1.1,
        decrease_factor: float = 0.5,
        window_size: float = 10.0,
    ):
        self._rate = initial_rate
        self._min_rate = min_rate
        self._max_rate = max_rate
        self._increase_factor = increase_factor
        self._decrease_factor = decrease_factor
        self._window_size = window_size
        self._name = name or f"AdaptiveRateLimiter-{id(self)}"

        self._token_bucket = RateLimiter(rate=initial_rate, burst=int(initial_rate * 2))

        self._success_count = 0
        self._failure_count = 0
        self._last_adjustment = time.monotonic()
        self._lock = asyncio.Lock()

    @property
    def rate(self) -> float:
        return self._rate

    async def acquire(self, tokens: int = 1, timeout: float | None = None) -> bool:
        return await self._token_bucket.acquire(tokens, timeout)

    def try_acquire(self, tokens: int = 1) -> bool:
        return self._token_bucket.try_acquire(tokens)

    def record_success(self) -> None:
        """Record a successful request."""
        self._success_count += 1
        self._maybe_adjust()

    def record_failure(self) -> None:
        """Record a failed request."""
        self._failure_count += 1
        self._maybe_adjust()

    def _maybe_adjust(self) -> None:
        now = time.monotonic()
        if now - self._last_adjustment < self._window_size:
            return

        total = self._success_count + self._failure_count
        if total == 0:
            return

        success_rate = self._success_count / total

        if success_rate > 0.95:
            # Increase rate
            self._rate = min(self._max_rate, self._rate * self._increase_factor)
        elif success_rate < 0.8:
            # Decrease rate
            self._rate = max(self._min_rate, self._rate * self._decrease_factor)

        # Update token bucket
        self._token_bucket = RateLimiter(rate=self._rate, burst=int(self._rate * 2))

        self._success_count = 0
        self._failure_count = 0
        self._last_adjustment = now

    def get_stats(self) -> dict[str, Any]:
        return {
            "name": self._name,
            "current_rate": self._rate,
            "min_rate": self._min_rate,
            "max_rate": self._max_rate,
            "success_count": self._success_count,
            "failure_count": self._failure_count,
        }


async def with_timeout(coro: Awaitable[T], timeout: float, *, default: T | None = None) -> T:
    """Run coroutine with timeout, with optional default."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except TimeoutError:
        if default is not None:
            return default
        raise


def timeout_decorator(seconds: float) -> Callable:
    """Decorator for adding timeout to async function."""

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)

        return wrapper

    return decorator


@asynccontextmanager
async def time_limit(seconds: float) -> AsyncGenerator[None, None]:
    """Context manager that enforces time limit."""
    start = time.monotonic()
    try:
        yield
    finally:
        elapsed = time.monotonic() - start
        if elapsed > seconds:
            raise TimeoutError(f"Operation took {elapsed:.3f}s, limit was {seconds}s")


class Stopwatch:
    """Simple stopwatch."""

    def __init__(self):
        self._start: float | None = None
        self._elapsed = 0.0
        self._running = False

    def start(self) -> None:
        if not self._running:
            self._start = time.monotonic()
            self._running = True

    def stop(self) -> float:
        if self._running:
            self._elapsed += time.monotonic() - self._start
            self._running = False
        return self._elapsed

    def reset(self) -> None:
        self._elapsed = 0.0
        self._running = False
        self._start = None

    @property
    def elapsed(self) -> float:
        if self._running:
            return self._elapsed + (time.monotonic() - self._start)
        return self._elapsed

    def __enter__(self) -> "Stopwatch":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
