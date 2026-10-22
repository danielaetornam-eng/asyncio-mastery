"""
Throttling and Rate Limiting Patterns

Advanced rate limiting algorithms for async systems.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, TypeVar

T = TypeVar("T")


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""

    pass


class RateLimiter(ABC):
    """Abstract base class for rate limiters."""

    @abstractmethod
    async def acquire(self, tokens: int = 1, timeout: float | None = None) -> bool:
        """Acquire tokens."""
        pass

    @abstractmethod
    def try_acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens without blocking."""
        pass

    @abstractmethod
    async def release(self, tokens: int = 1) -> None:
        """Release tokens back."""
        pass

    @abstractmethod
    def available_tokens(self) -> int:
        """Get available tokens."""
        pass

    @asynccontextmanager
    async def limit(self, tokens: int = 1, timeout: float | None = None):
        """Context manager for rate limiting."""
        acquired = await self.acquire(tokens, timeout)
        try:
            yield acquired
        finally:
            if acquired:
                await self.release(tokens)


class TokenBucket(RateLimiter):
    """Token bucket rate limiter."""

    def __init__(self, capacity: int, refill_rate: float, *, name: str | None = None) -> None:
        """
        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens added per second
        """
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if refill_rate <= 0:
            raise ValueError("refill_rate must be > 0")

        self._capacity = capacity
        self._refill_rate = refill_rate
        self._name = name or f"TokenBucket-{id(self)}"
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()
        self._waiters: deque[asyncio.Future] = deque()

    @property
    def name(self) -> str:
        return self._name

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def refill_rate(self) -> float:
        return self._refill_rate

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        new_tokens = elapsed * self._refill_rate
        self._tokens = min(self._capacity, self._tokens + new_tokens)
        self._last_refill = now

    async def acquire(self, tokens: int = 1, timeout: float | None = None) -> bool:
        if tokens <= 0:
            return True

        async with self._lock:
            self._refill()

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True

            # Need to wait
            needed = tokens - self._tokens
            wait_time = needed / self._refill_rate

            if timeout is not None and wait_time > timeout:
                return False

            waiter = asyncio.get_running_loop().create_future()
            self._waiters.append((waiter, tokens, time.monotonic() + (timeout or float("inf"))))

        try:
            if timeout is not None:
                await asyncio.wait_for(waiter, timeout=timeout)
            else:
                await waiter
            return True
        except TimeoutError:
            async with self._lock:
                # Remove our waiter
                self._waiters = deque((w, t, d) for w, t, d in self._waiters if w is not waiter)
            return False

    def try_acquire(self, tokens: int = 1) -> bool:
        if tokens <= 0:
            return True

        self._refill()

        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False

    async def release(self, tokens: int = 1) -> None:
        if tokens <= 0:
            return

        async with self._lock:
            self._refill()
            self._tokens = min(self._capacity, self._tokens + tokens)

            # Wake up waiters
            while self._waiters:
                waiter, needed, deadline = self._waiters[0]
                if time.monotonic() > deadline:
                    self._waiters.popleft()
                    if not waiter.done():
                        waiter.set_exception(TimeoutError())
                    continue

                if self._tokens >= needed:
                    self._tokens -= needed
                    self._waiters.popleft()
                    if not waiter.done():
                        waiter.set_result(True)
                else:
                    break

    def available_tokens(self) -> int:
        self._refill()
        return int(self._tokens)


class LeakyBucket(RateLimiter):
    """Leaky bucket rate limiter."""

    def __init__(self, capacity: int, leak_rate: float, *, name: str | None = None) -> None:
        """
        Args:
            capacity: Maximum bucket size
            leak_rate: Items processed per second
        """
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if leak_rate <= 0:
            raise ValueError("leak_rate must be > 0")

        self._capacity = capacity
        self._leak_rate = leak_rate
        self._name = name or f"LeakyBucket-{id(self)}"
        self._water = 0.0
        self._last_leak = time.monotonic()
        self._lock = asyncio.Lock()
        self._waiters: deque[asyncio.Future] = deque()

    @property
    def name(self) -> str:
        return self._name

    def _leak(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_leak
        leaked = elapsed * self._leak_rate
        self._water = max(0, self._water - leaked)
        self._last_leak = now

    async def acquire(self, tokens: int = 1, timeout: float | None = None) -> bool:
        if tokens <= 0:
            return True

        async with self._lock:
            self._leak()

            if self._water + tokens <= self._capacity:
                self._water += tokens
                return True

            # Need to wait for leak
            excess = self._water + tokens - self._capacity
            wait_time = excess / self._leak_rate

            if timeout is not None and wait_time > timeout:
                return False

            waiter = asyncio.get_running_loop().create_future()
            self._waiters.append((waiter, tokens, time.monotonic() + (timeout or float("inf"))))

        try:
            if timeout is not None:
                await asyncio.wait_for(waiter, timeout=timeout)
            else:
                await waiter
            return True
        except TimeoutError:
            async with self._lock:
                self._waiters = deque((w, t, d) for w, t, d in self._waiters if w is not waiter)
            return False

    def try_acquire(self, tokens: int = 1) -> bool:
        if tokens <= 0:
            return True

        self._leak()

        if self._water + tokens <= self._capacity:
            self._water += tokens
            return True
        return False

    async def release(self, tokens: int = 1) -> None:
        # Leaky bucket doesn't typically release, but we can reduce water
        if tokens <= 0:
            return

        async with self._lock:
            self._leak()
            self._water = max(0, self._water - tokens)

            while self._waiters:
                waiter, needed, deadline = self._waiters[0]
                if time.monotonic() > deadline:
                    self._waiters.popleft()
                    if not waiter.done():
                        waiter.set_exception(TimeoutError())
                    continue

                if self._water + needed <= self._capacity:
                    self._water += needed
                    self._waiters.popleft()
                    if not waiter.done():
                        waiter.set_result(True)
                else:
                    break

    def available_tokens(self) -> int:
        self._leak()
        return int(self._capacity - self._water)


class SlidingWindow(RateLimiter):
    """Sliding window rate limiter."""

    def __init__(self, max_requests: int, window_size: float, *, name: str | None = None) -> None:
        """
        Args:
            max_requests: Maximum requests in window
            window_size: Window size in seconds
        """
        if max_requests <= 0:
            raise ValueError("max_requests must be > 0")
        if window_size <= 0:
            raise ValueError("window_size must be > 0")

        self._max_requests = max_requests
        self._window_size = window_size
        self._name = name or f"SlidingWindow-{id(self)}"
        self._requests: deque[float] = deque()
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return self._name

    def _clean_old(self) -> None:
        now = time.monotonic()
        cutoff = now - self._window_size
        while self._requests and self._requests[0] < cutoff:
            self._requests.popleft()

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

                # Calculate wait time
                oldest = self._requests[0]
                wait_time = oldest + self._window_size - time.monotonic()

            if timeout is not None:
                if time.monotonic() - start + wait_time > timeout:
                    return False
                wait_time = min(wait_time, timeout - (time.monotonic() - start))

            if wait_time > 0:
                try:
                    await asyncio.sleep(wait_time)
                except asyncio.CancelledError:
                    return False

    def try_acquire(self, tokens: int = 1) -> bool:
        if tokens <= 0:
            return True

        self._clean_old()

        if len(self._requests) + tokens <= self._max_requests:
            now = time.monotonic()
            for _ in range(tokens):
                self._requests.append(now)
            return True
        return False

    async def release(self, tokens: int = 1) -> None:
        # Sliding window doesn't support release in traditional sense
        pass

    def available_tokens(self) -> int:
        self._clean_old()
        return max(0, self._max_requests - len(self._requests))


class SlidingWindowLog(RateLimiter):
    """Sliding window log rate limiter (more precise)."""

    def __init__(self, max_requests: int, window_size: float, *, name: str | None = None) -> None:
        if max_requests <= 0:
            raise ValueError("max_requests must be > 0")
        if window_size <= 0:
            raise ValueError("window_size must be > 0")

        self._max_requests = max_requests
        self._window_size = window_size
        self._name = name or f"SlidingWindowLog-{id(self)}"
        self._requests: deque[float] = deque()
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return self._name

    async def acquire(self, tokens: int = 1, timeout: float | None = None) -> bool:
        if tokens <= 0:
            return True

        start = time.monotonic()

        while True:
            async with self._lock:
                now = time.monotonic()
                cutoff = now - self._window_size

                # Remove old requests
                while self._requests and self._requests[0] < cutoff:
                    self._requests.popleft()

                if len(self._requests) + tokens <= self._max_requests:
                    for _ in range(tokens):
                        self._requests.append(now)
                    return True

                if not self._requests:
                    return False

                wait_time = self._requests[0] + self._window_size - now

            if timeout is not None and time.monotonic() - start + wait_time > timeout:
                return False

            if wait_time > 0:
                try:
                    await asyncio.sleep(wait_time)
                except asyncio.CancelledError:
                    return False

    def try_acquire(self, tokens: int = 1) -> bool:
        if tokens <= 0:
            return True

        now = time.monotonic()
        cutoff = now - self._window_size

        while self._requests and self._requests[0] < cutoff:
            self._requests.popleft()

        if len(self._requests) + tokens <= self._max_requests:
            for _ in range(tokens):
                self._requests.append(now)
            return True
        return False

    async def release(self, tokens: int = 1) -> None:
        pass

    def available_tokens(self) -> int:
        now = time.monotonic()
        cutoff = now - self._window_size

        while self._requests and self._requests[0] < cutoff:
            self._requests.popleft()

        return max(0, self._max_requests - len(self._requests))


class ConcurrencyLimiter:
    """Limit concurrent executions."""

    def __init__(self, max_concurrent: int, *, name: str | None = None) -> None:
        if max_concurrent <= 0:
            raise ValueError("max_concurrent must be > 0")
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._name = name or f"ConcurrencyLimiter-{id(self)}"
        self._active = 0
        self._lock = asyncio.Lock()
        self._max_concurrent = max_concurrent

    @property
    def name(self) -> str:
        return self._name

    @property
    def max_concurrent(self) -> int:
        return self._max_concurrent

    @property
    def available(self) -> int:
        return self._max_concurrent - self._active

    async def acquire(self) -> None:
        await self._semaphore.acquire()
        async with self._lock:
            self._active += 1

    async def release(self) -> None:
        self._semaphore.release()
        async with self._lock:
            self._active = max(0, self._active - 1)

    @asynccontextmanager
    async def limit(self):
        await self.acquire()
        try:
            yield
        finally:
            await self.release()

    async def run(self, coro: Awaitable[T]) -> T:
        """Run a coroutine with concurrency limit."""
        async with self.limit():
            return await coro

    async def map(
        self, func: Callable[..., Awaitable[T]], *iterables: Any, max_concurrent: int | None = None
    ) -> list[T]:
        """Map function over iterables with concurrency limit."""
        semaphore = asyncio.Semaphore(max_concurrent or self._max_concurrent)

        async def limited_func(*args):
            async with semaphore:
                return await func(*args)

        return await asyncio.gather(*[limited_func(*args) for args in zip(*iterables)])


class AdaptiveRateLimiter(RateLimiter):
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
    ) -> None:
        self._rate = initial_rate
        self._min_rate = min_rate
        self._max_rate = max_rate
        self._increase_factor = increase_factor
        self._decrease_factor = decrease_factor
        self._window_size = window_size
        self._name = name or f"AdaptiveRateLimiter-{id(self)}"

        self._token_bucket = TokenBucket(capacity=int(initial_rate * 2), refill_rate=initial_rate)

        self._success_count = 0
        self._failure_count = 0
        self._last_adjustment = time.monotonic()
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def current_rate(self) -> float:
        return self._rate

    async def acquire(self, tokens: int = 1, timeout: float | None = None) -> bool:
        return await self._token_bucket.acquire(tokens, timeout)

    def try_acquire(self, tokens: int = 1) -> bool:
        return self._token_bucket.try_acquire(tokens)

    async def release(self, tokens: int = 1) -> None:
        await self._token_bucket.release(tokens)

    def available_tokens(self) -> int:
        return self._token_bucket.available_tokens()

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
        self._token_bucket = TokenBucket(capacity=int(self._rate * 2), refill_rate=self._rate)

        self._success_count = 0
        self._failure_count = 0
        self._last_adjustment = now


class MultiRateLimiter:
    """Multiple rate limiters for different priority levels."""

    def __init__(self, *, name: str | None = None) -> None:
        self._name = name or f"MultiRateLimiter-{id(self)}"
        self._limiters: dict[str, RateLimiter] = {}
        self._default_limiter: RateLimiter | None = None

    def add_limiter(self, name: str, limiter: RateLimiter, default: bool = False) -> None:
        self._limiters[name] = limiter
        if default or self._default_limiter is None:
            self._default_limiter = limiter

    def get_limiter(self, name: str) -> RateLimiter | None:
        return self._limiters.get(name)

    async def acquire(self, name: str, tokens: int = 1, timeout: float | None = None) -> bool:
        limiter = self._limiters.get(name, self._default_limiter)
        if limiter is None:
            raise ValueError(f"No limiter for '{name}' and no default")
        return await limiter.acquire(tokens, timeout)

    def try_acquire(self, name: str, tokens: int = 1) -> bool:
        limiter = self._limiters.get(name, self._default_limiter)
        if limiter is None:
            raise ValueError(f"No limiter for '{name}' and no default")
        return limiter.try_acquire(tokens)


@asynccontextmanager
async def rate_limited(limiter: RateLimiter, tokens: int = 1, timeout: float | None = None):
    """Context manager for rate-limited operation."""
    acquired = await limiter.acquire(tokens, timeout)
    try:
        yield acquired
    finally:
        if acquired:
            await limiter.release(tokens)


def create_token_bucket(rate: float, burst: int = 0) -> TokenBucket:
    """Create token bucket with rate per second and burst capacity."""
    capacity = burst if burst > 0 else int(rate * 2)
    return TokenBucket(capacity=capacity, refill_rate=rate)


def create_leaky_bucket(rate: float, capacity: int = 0) -> LeakyBucket:
    """Create leaky bucket with rate per second."""
    cap = capacity if capacity > 0 else int(rate * 2)
    return LeakyBucket(capacity=cap, leak_rate=rate)


def create_sliding_window(requests: int, window_seconds: float) -> SlidingWindow:
    """Create sliding window rate limiter."""
    return SlidingWindow(max_requests=requests, window_size=window_seconds)


def create_concurrency_limiter(max_concurrent: int) -> ConcurrencyLimiter:
    """Create concurrency limiter."""
    return ConcurrencyLimiter(max_concurrent)
