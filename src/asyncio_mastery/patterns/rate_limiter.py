"""
Rate Limiter - Control request rates.

THE PROBLEM
-----------
APIs, databases, and downstream services get overwhelmed (or angry, or
bill you more) when you hit them too fast. A rate limiter paces your
calls so you stay inside an agreed budget like "at most 10 requests
per second".

REAL-WORLD ANALOGY
------------------
A **turnstile at a stadium**: only N people may enter per minute. When
you arrive and the turnstile is free, you walk through; when it is not,
you wait until the window resets. The rate limiter is that turnstile for
your requests.

THE TWO ALGORITHMS HERE
-----------------------
* **Token bucket**: a bucket holds "tokens" and refills at a constant
  rate. Each request spends a token. An empty bucket means "wait". A
  full-ish bucket lets a sudden **burst** through — good when spikes are
  OK but the *average* rate must be capped.
* **Sliding window**: we remember the timestamps of the last N requests.
  Each new request is allowed only if fewer than N timestamps fall
  within the trailing window. Smoother (no bursts) but uses memory.

WHY ASYNC PRIMITIVES ARE NEEDED
-------------------------------
When the budget is exhausted, ``await asyncio.sleep(wait_time)`` lets
the caller *suspend* until tokens arrive — the event loop keeps serving
other tasks instead of freezing. An :class:`asyncio.Lock` guarantees
two coroutines cannot both observe "1 token left" and both spend it
(a classic check-then-act race).

WHEN TO USE IT
--------------
* Calling third-party APIs with quotas (e.g., "600 requests/minute").
* Protecting your own service from overload (paired with a queue).
* Smoothing out a bursty producer before a rate-sensitive sink.

Minimal example
---------------
>>> import asyncio
>>> async def main() -> None:
...     limiter = RateLimiter(rate=5, burst=2)  # 5 req/s, burst of 2
...     async with limiter:                     # waits until a slot frees
...         print("request goes out now")
>>> asyncio.run(main())
"""

import asyncio
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass
class RateLimitConfig:
    """Plain data holder describing a rate limit.

    Attributes:
        rate: Allowed requests per second (steady-state budget).
        burst: How many requests may fire instantly before throttling
            applies (token bucket only). None means "pick a sensible
            default".
    """

    rate: float  # requests per second
    burst: int | None = None  # max burst (for token bucket)


class RateLimiterBase(ABC):
    """Abstract rate limiter.

    Defines the common vocabulary every limiter shares: ``acquire``
    (wait for permission), ``try_acquire`` (permission or nothing), and
    ``available`` (how much budget is left right now).

    NOTE: A concrete convenience class named :class:`RateLimiter`
    is defined later in this module — it composes one of the algorithmic
    limiters below and adds context-manager sugar.
    """

    @abstractmethod
    async def acquire(self, tokens: int = 1) -> None:
        """Acquire permission to proceed.

        Args:
            tokens: How many units of budget this operation costs
                (default 1). Suspends until the budget is available.
        """
        pass

    @abstractmethod
    def try_acquire(self, tokens: int = 1) -> bool:
        """Try to acquire without blocking.

        Args:
            tokens: Units of budget requested.

        Returns:
            bool: True if the budget was spent; False if the limiter is
            currently exhausted (the caller should skip or defer).
        """
        pass

    @property
    @abstractmethod
    def available(self) -> float:
        """How much budget is available right now.

        Returns:
            float: Tokens (bucket) or remaining slots (sliding window).
        """
        pass


class TokenBucket(RateLimiterBase):
    """Token bucket rate limiter.

    The bucket starts full and refills continuously at ``rate`` tokens
    per second, capped at ``burst``. Each ``acquire`` spends tokens;
    when the bucket is empty, we sleep until it refills — that is the
    pacing mechanism.

    Example:
        >>> bucket = TokenBucket(rate=10, burst=5)
        >>> await bucket.acquire()      # spends 1 token (fast)
        >>> bucket.try_acquire()        # False once bucket is empty
    """

    def __init__(self, rate: float, burst: int | None = None) -> None:
        """Create the bucket.

        Args:
            rate: Refill speed in tokens (requests) per second.
            burst: Bucket capacity — the maximum instantaneous burst.
                Defaults to ~2 seconds worth of rate.
        """
        self._rate = rate
        self._burst = burst or int(rate * 2)
        self._tokens = float(self._burst)
        self._last_update = time.monotonic()
        # The lock prevents the classic race where two coroutines both
        # read "1 token left" and both succeed (double-spending).
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> None:
        """Suspend until ``tokens`` tokens are available, then spend them.

        While waiting we *sleep inside the lock*. That serializes waiters:
        each woken waiter re-checks the bucket. it is simple and correct;
        the trade-off is fairness (first come, first served).

        Args:
            tokens: Number of tokens to spend (default 1).
        """
        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                # Wait for next token: how long until the bucket refills
                # enough? deficit / refill-rate is exactly that time.
                wait_time = (tokens - self._tokens) / self._rate
                await asyncio.sleep(wait_time)

    def try_acquire(self, tokens: int = 1) -> bool:
        """Spend ``tokens`` right now if possible; never wait.

        Args:
            tokens: Number of tokens to spend.

        Returns:
            bool: True on success; False if the bucket didn't have enough
            (nothing was spent in that case).
        """
        self._refill()
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False

    def _refill(self) -> None:
        """Top the bucket up based on how much time has passed.

        Tokens accrue lazily: instead of a background task adding tokens
        every second, we compute ``elapsed * rate`` whenever anyone
        asks. Cheaper and just as accurate.
        """
        now = time.monotonic()
        elapsed = now - self._last_update
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
        self._last_update = now

    @property
    def available(self) -> float:
        """Tokens currently in the bucket (after a lazy refill).

        Returns:
            float: Between 0 and ``burst``.
        """
        self._refill()
        return self._tokens


class SlidingWindow(RateLimiterBase):
    """Sliding window rate limiter.

    Keeps a deque of the timestamps of the last requests. On each
    request, expired timestamps (older than ``window`` seconds) are
    dropped, and the request is allowed only if the deque has room
    (``len < rate``). If not, we sleep until the oldest timestamp
    "falls out" of the window — exactly when a slot opens.

    Compared to :class:`TokenBucket`, this forbids bursts entirely:
    requests are spread evenly.

    Example:
        >>> window = SlidingWindow(rate=5, window=1.0)  # 5 per second
        >>> await window.acquire()
    """

    def __init__(self, rate: float, window: float = 1.0) -> None:
        """Create the limiter.

        Args:
            rate: Max requests allowed inside one window.
            window: Window length in seconds (default 1.0).
        """
        self._rate = rate
        self._window = window
        self._requests: deque[float] = deque()
        # Same check-then-act protection as TokenBucket: counting +
        # appending must be atomic relative to other coroutines.
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> None:
        """Suspend until there's room in the window, then record.

        Args:
            tokens: How many request-slots to claim (default 1).
        """
        async with self._lock:
            while True:
                self._cleanup()
                if len(self._requests) + tokens <= self._rate:
                    now = time.monotonic()
                    for _ in range(tokens):
                        self._requests.append(now)
                    return
                # Wait for oldest request to expire — that's the moment
                # a slot frees up.
                if self._requests:
                    wait_time = self._requests[0] + self._window - time.monotonic()
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                else:
                    # Empty queue but still over budget can only happen
                    # when tokens > rate: just poll briefly.
                    await asyncio.sleep(0.01)

    def try_acquire(self, tokens: int = 1) -> bool:
        """Record ``tokens`` requests if the window has room; never wait.

        Args:
            tokens: Request slots to claim.

        Returns:
            bool: True if recorded; False if the window is full.
        """
        self._cleanup()
        if len(self._requests) + tokens <= self._rate:
            now = time.monotonic()
            for _ in range(tokens):
                self._requests.append(now)
            return True
        return False

    def _cleanup(self) -> None:
        """Drop timestamps that have slid out of the window.

        The deque is ordered oldest-first, so we pop from the left until
        the front is inside the window again.
        """
        now = time.monotonic()
        cutoff = now - self._window
        while self._requests and self._requests[0] < cutoff:
            self._requests.popleft()

    @property
    def available(self) -> float:
        """Slots remaining in the current window.

        Returns:
            float: ``rate - len(requests)``, floored at 0.
        """
        self._cleanup()
        return max(0, self._rate - len(self._requests))


class RateLimiter:
    """High-level rate limiter with multiple algorithms.

    Wraps :class:`TokenBucket` or :class:`SlidingWindow` behind one
    interface and adds ``async with`` support, so a limited section of
    code reads like::

        async with limiter:
            await call_api()

    Attributes:
        _rate: Nominal rate in requests/second.
        _burst: Burst allowance (token bucket only).
        _limiter: The underlying algorithmic limiter instance.
    """

    def __init__(
        self, rate: float, *, burst: int | None = None, algorithm: str = "token_bucket"
    ) -> None:
        """Create the limiter.

        Args:
            rate: Allowed requests per second.
            burst: Burst size (only meaningful for ``token_bucket``).
            algorithm: ``"token_bucket"`` (bursts allowed) or
                ``"sliding_window"`` (strictly smooth).

        Raises:
            ValueError: If ``algorithm`` is not one of the supported names.
        """
        self._rate = rate
        self._burst = burst
        self._limiter: RateLimiterBase

        if algorithm == "token_bucket":
            self._limiter = TokenBucket(rate, burst)
        elif algorithm == "sliding_window":
            self._limiter = SlidingWindow(rate)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    async def acquire(self, tokens: int = 1) -> None:
        """Suspend until the rate budget permits, then spend it.

        Args:
            tokens: Units of rate budget to consume (default 1).
        """
        await self._limiter.acquire(tokens)

    def try_acquire(self, tokens: int = 1) -> bool:
        """Try to spend budget without waiting.

        Args:
            tokens: Units of rate budget to consume.

        Returns:
            bool: True if spent; False if the limiter is exhausted.
        """
        return self._limiter.try_acquire(tokens)

    @property
    def available(self) -> float:
        """Budget available immediately.

        Returns:
            float: Tokens or window slots, depending on the algorithm.
        """
        return self._limiter.available

    async def __aenter__(self) -> "RateLimiter":
        """Enter: wait for one unit of budget, then let the body run.

        Returns:
            RateLimiter: This limiter (so ``async with limiter as lm``
            works even though nothing else is needed from it).
        """
        await self.acquire()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit: nothing to release for a rate limiter."""
        pass


async def demonstrate_rate_limiter() -> None:
    """Run a self-contained demo of both limiter algorithms.

    Observe the timing differences: the token bucket emits a burst up
    front, while the sliding window spreads events evenly.
    """
    print("=== Rate Limiter Demo ===")

    # Token bucket
    print("\n--- Token Bucket ---")
    limiter = TokenBucket(rate=10, burst=5)  # 10/sec, burst of 5

    start = time.monotonic()
    for i in range(15):
        await limiter.acquire()
        elapsed = time.monotonic() - start
        print(f"  Request {i + 1} at {elapsed:.3f}s (available: {limiter.available:.1f})")

    # Sliding window
    print("\n--- Sliding Window ---")
    limiter2 = SlidingWindow(rate=10, window=1.0)

    start = time.monotonic()
    for i in range(15):
        await limiter2.acquire()
        elapsed = time.monotonic() - start
        print(f"  Request {i + 1} at {elapsed:.3f}s (available: {limiter2.available:.1f})")

    # High-level limiter
    print("\n--- High-level RateLimiter ---")
    limiter3 = RateLimiter(rate=5, burst=3, algorithm="token_bucket")

    async def make_request(id: int) -> None:
        async with limiter3:
            print(f"  Request {id} allowed")

    await asyncio.gather(*[make_request(i) for i in range(8)])

    # Try acquire
    print("\n--- Try Acquire ---")
    limiter4 = RateLimiter(rate=2, burst=2)
    for i in range(5):
        if limiter4.try_acquire():
            print(f"  Request {i + 1}: allowed")
        else:
            print(f"  Request {i + 1}: rejected (available: {limiter4.available})")


if __name__ == "__main__":
    asyncio.run(demonstrate_rate_limiter())
