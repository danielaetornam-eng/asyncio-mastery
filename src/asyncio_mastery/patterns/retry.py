"""
Retry Policy - Configurable retry logic with backoff.

THE PROBLEM
Network calls, file handles, and flaky services fail transiently:
the Wi-Fi hiccups, the server is restarting, the database is briefly
busy. A single failed call does not mean the operation is hopeless —
often a second attempt a moment later succeeds. Naively retrying in a
tight loop, however, can make a struggling service be overwhelmed even
more.

REAL-WORLD ANALOGY
Knocking on a friend's door. If they do not answer, you do not knock
again instantly 50 times — you wait a bit, knock again, wait longer,
then give up. That growing wait is backoff.

JARGON
- Exponential backoff: the delay between attempts grows
  geometrically (e.g., 1s, 2s, 4s, 8s...). It gives an overloaded
  service rapidly more room to recover between hits.
- Jitter: a small random wobble added to each delay. Without it,
  hundreds of clients that all failed at the same moment would all
  retry at the same moment — the thundering herd problem — knocking
  the service over again. Jitter spreads the retries out.

WHY ASYNC PRIMITIVES ARE NEEDED
Between attempts the caller awaits asyncio.sleep(delay) — the task
suspends and the event loop serves everyone else. A whole fleet of
retrying tasks costs almost no CPU while they wait.

WHEN TO USE IT
- Any idempotent remote call (safe to repeat) that may fail transiently.
- Combine with a circuit breaker: retries handle blips, the breaker
  handles outages.

Minimal example
>>> import asyncio
>>> async def flaky() -> str: return "ok"
>>> async def main() -> None:
...     policy = RetryPolicy(RetryConfig(max_attempts=3, base_delay=0.1))
...     print(await policy.execute(flaky))
>>> asyncio.run(main())
"""

import asyncio
import contextlib
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

T = TypeVar("T")


class RetryCondition(Enum):
    """When to retry.

    - ALWAYS — retry on any exception (until attempts run out).
    - ON_EXCEPTION — retry only when an exception was raised.
    - ON_RESULT — retry based on the value returned (e.g. retry
      while the health check returns "unavailable").
    - CUSTOM — user-supplied predicate decides.
    """

    ALWAYS = "always"
    ON_EXCEPTION = "on_exception"
    ON_RESULT = "on_result"
    CUSTOM = "custom"


@dataclass
class RetryConfig:
    """All the knobs controlling a retry policy.

    Attributes:
        max_attempts: Total tries (1 = no retries at all).
        base_delay: First delay, in seconds. Later delays derive from it.
        max_delay: Delay cap — backoff never waits longer than this.
        exponential_base: Growth factor for exponential backoff
            (2.0 means each delay doubles).
        jitter: Whether to add random wobble (thundering-herd antidote).
        jitter_factor: Jitter amplitude as a fraction of the delay
            (0.1 = plus/minus 10%).
        retry_on: Exception types that allow a retry.
        stop_on: Exception types that abort immediately, even if they would
            match retry_on (checked first).
        condition: Which RetryCondition rule to apply.
        custom_condition: User predicate for ON_RESULT / CUSTOM — takes
            the result (or last result) and returns True to retry.
    """

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_factor: float = 0.1
    retry_on: tuple[type[Exception], ...] = (Exception,)
    stop_on: tuple[type[Exception], ...] = ()
    condition: RetryCondition = RetryCondition.ON_EXCEPTION
    custom_condition: Callable[[Any], bool] | None = None


@dataclass
class RetryState:
    """Progress tracker threaded through the retry loop.

    Attributes:
        attempt: How many retries have been scheduled so far (0 during
            the first call).
        total_delay: Sum of all delays waited so far.
        last_exception: The exception from the most recent attempt, if
            any.
        last_result: The value returned by the most recent attempt.
    """

    attempt: int = 0
    total_delay: float = 0.0
    last_exception: Exception | None = None
    last_result: Any = None


class BackoffStrategy(ABC):
    """Abstract backoff strategy.

    A strategy answers one question: "given where we are in the retry
    sequence, how long should we wait before the next attempt?" Subclass
    and implement next_delay for your own schedule.
    """

    @abstractmethod
    def next_delay(self, state: RetryState) -> float:
        """Calculate next delay.

        Args:
            state: The current RetryState — most importantly,
                state.attempt tells you which retry this would be.

        Returns:
            float: Seconds to wait before the next attempt.
        """
        pass


class ExponentialBackoff(BackoffStrategy):
    """Exponential backoff with optional jitter.

    Delay for attempt N is base_delay * base ** attempt (1s, 2s,
    4s, 8s, ... with defaults), capped at max_delay, then wobbled
    by plus/minus jitter_factor if jitter is on.

    Example:
        >>> bo = ExponentialBackoff(base_delay=0.1, jitter=False)
        >>> bo.next_delay(RetryState(attempt=3))  # 0.1 * 2**3 = 0.8
        0.8
    """

    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        base: float = 2.0,
        jitter: bool = True,
        jitter_factor: float = 0.1,
    ) -> None:
        """Configure the strategy.

        Args:
            base_delay: Delay for the very first retry (attempt 0).
            max_delay: Hard ceiling for any single delay.
            base: Multiplicative growth factor (2.0 = double each time).
            jitter: Add randomness to desynchronize retrying clients.
            jitter_factor: Jitter amplitude as a fraction of the delay.
        """
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._base = base
        self._jitter = jitter
        self._jitter_factor = jitter_factor

    def next_delay(self, state: RetryState) -> float:
        """Delay = base_delay * base**attempt, capped, then jittered.

        Args:
            state: Current retry state (uses attempt).

        Returns:
            float: Seconds to sleep; never negative, never above
            max_delay (before jitter is applied).
        """
        delay = min(self._base_delay * (self._base**state.attempt), self._max_delay)
        if self._jitter:
            jitter_range = delay * self._jitter_factor
            delay += random.uniform(-jitter_range, jitter_range)
        return max(0, delay)


class LinearBackoff(BackoffStrategy):
    """Linear backoff.

    Delay grows by constant steps: base_delay, 2*base_delay,
    3*base_delay, ... Gentler than exponential and more predictable —
    useful when you expect short outages.

    Example:
        >>> bo = LinearBackoff(base_delay=0.5, jitter=False)
        >>> bo.next_delay(RetryState(attempt=2))  # 0.5 * 3
        1.5
    """

    def __init__(
        self, base_delay: float = 1.0, max_delay: float = 60.0, jitter: bool = True
    ) -> None:
        """Configure the strategy.

        Args:
            base_delay: Size of one step, in seconds.
            max_delay: Hard ceiling for any single delay.
            jitter: Randomly scale the delay between 50% and 150%.
        """
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._jitter = jitter

    def next_delay(self, state: RetryState) -> float:
        """Delay = base_delay * (attempt + 1), capped, then jittered.

        Args:
            state: Current retry state.

        Returns:
            float: Seconds to sleep (never negative).
        """
        delay = min(self._base_delay * (state.attempt + 1), self._max_delay)
        if self._jitter:
            delay *= random.uniform(0.5, 1.5)
        return max(0, delay)


class ConstantBackoff(BackoffStrategy):
    """Constant delay.

    Every retry waits exactly the same amount — simple, and appropriate
    when the poll rate itself is the pacing mechanism (e.g., waiting
    for a background job to finish).

    Example:
        >>> bo = ConstantBackoff(delay=2.0)
        >>> bo.next_delay(RetryState(attempt=5))
        2.0
    """

    def __init__(self, delay: float = 1.0, jitter: bool = False) -> None:
        """Configure the strategy.

        Args:
            delay: The fixed wait between attempts, in seconds.
            jitter: Randomly scale between 50% and 150%.
        """
        self._delay = delay
        self._jitter = jitter

    def next_delay(self, state: RetryState) -> float:
        """Return the fixed delay (optionally jittered).

        Args:
            state: Current retry state (ignored — the delay is constant).

        Returns:
            float: Seconds to sleep.
        """
        delay = self._delay
        if self._jitter:
            delay *= random.uniform(0.5, 1.5)
        return max(0, delay)


class RetryPolicy:
    """Retry policy with configurable behavior.

    Combines a RetryConfig (what to retry on, how many times)
    with a BackoffStrategy (how long to wait). One call to
    execute runs the whole try -> wait -> try-again loop for you.

    Example:
        >>> policy = RetryPolicy(RetryConfig(
        ...     max_attempts=4,
        ...     retry_on=(ConnectionError, TimeoutError),
        ...     stop_on=(KeyboardInterrupt,),
        ... ))
        >>> result = await policy.execute(call_flaky_api)
    """

    def __init__(
        self, config: RetryConfig | None = None, backoff: BackoffStrategy | None = None
    ) -> None:
        """Assemble the policy.

        Args:
            config: Retry rules; defaults to a sensible 3-attempt
                exponential-backoff setup.
            backoff: Delay schedule; defaults to ExponentialBackoff
                built from the config's delay settings.
        """
        self._config = config or RetryConfig()
        self._backoff = backoff or ExponentialBackoff(
            base_delay=self._config.base_delay,
            max_delay=self._config.max_delay,
            base=self._config.exponential_base,
            jitter=self._config.jitter,
            jitter_factor=self._config.jitter_factor,
        )

    async def execute(self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        """Execute with retry.

        Runs the function; on failure (or a retry-worthy result), waits
        per the backoff strategy and tries again, up to
        config.max_attempts total attempts.

        Order of exception checks matters: stop_on is consulted
        before retry_on so a subtype of a retryable exception can
        still be marked fatal.

        Args:
            func: The async (or value-returning) callable to attempt.
            *args: Positional arguments for func.
            **kwargs: Keyword arguments for func.

        Returns:
            T: The first successful result.

        Raises:
            Exception: The last exception encountered when attempts run
                out, or immediately for exceptions in stop_on.
        """
        state = RetryState()

        while True:
            try:
                state.last_result = await func(*args, **kwargs)
                # A "successful" call can still merit a retry if the
                # value signals incompleteness (ON_RESULT mode).
                if self._should_retry_result(state.last_result):
                    raise RetryableError("Result indicates retry needed")
                return state.last_result
            except self._config.stop_on:
                raise
            except self._config.retry_on as e:
                state.last_exception = e
                if not self._should_retry(state):
                    raise
                await self._wait_and_retry(state)

    def _should_retry(self, state: RetryState) -> bool:
        """Decide whether another attempt is permitted.

        Args:
            state: Current retry state including last exception/result.

        Returns:
            bool: True if a retry should happen per the configured
            condition and attempt budget.
        """
        if state.attempt >= self._config.max_attempts - 1:
            # Attempt budget exhausted: attempt counts already-taken
            # retries, so max_attempts - 1 is the hard stop.
            return False
        if self._config.condition == RetryCondition.ALWAYS:
            return True
        if self._config.condition == RetryCondition.ON_EXCEPTION:
            return state.last_exception is not None
        if self._config.condition == RetryCondition.CUSTOM and self._config.custom_condition:
            return self._config.custom_condition(state.last_result)
        return False

    def _should_retry_result(self, result: Any) -> bool:
        """Decide whether a returned value means "try again".

        Only active when the condition is ON_RESULT and a custom
        predicate exists. Example: retry while the status polling
        endpoint returns {"ready": False}.

        Args:
            result: The value just returned by the wrapped call.

        Returns:
            bool: True if this result means "retry".
        """
        if self._config.condition == RetryCondition.ON_RESULT and self._config.custom_condition:
            return self._config.custom_condition(result)
        return False

    async def _wait_and_retry(self, state: RetryState) -> None:
        """Sleep for the backoff delay, then advance the attempt counter.

        The await asyncio.sleep suspends only this task — other
        coroutines run normally while we wait.

        Args:
            state: Retry state to update (attempt and total_delay).
        """
        delay = self._backoff.next_delay(state)
        state.attempt += 1
        state.total_delay += delay
        await asyncio.sleep(delay)


class RetryableError(Exception):
    """Error indicating operation should be retried.

    Raised internally when a returned value (not an exception) says
    "try again" — it unifies result-based retry with the exception-based
    control flow in RetryPolicy.execute.
    """

    pass


async def demonstrate_retry() -> None:
    """Run a self-contained demo of the retry policy.

    Covers: basic retry-until-success, stopping on a specific exception,
    result-based retrying, and comparing the three backoff strategies.
    """
    print("=== Retry Demo ===")

    # Basic retry
    print("\n--- Basic Retry ---")
    attempt_count = 0

    async def flaky_operation() -> str:
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ConnectionError(f"Attempt {attempt_count} failed")
        return "success"

    policy = RetryPolicy(RetryConfig(max_attempts=5, base_delay=0.05))
    result = await policy.execute(flaky_operation)
    print(f"Result: {result} (attempts: {attempt_count})")

    # Retry on specific exceptions
    print("\n--- Specific Exceptions ---")
    attempt_count = 0

    async def specific_error() -> str:
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count == 1:
            raise ValueError("Stop on this")
        if attempt_count < 4:
            raise ConnectionError("Retry on this")
        return "success"

    config = RetryConfig(max_attempts=5, retry_on=(ConnectionError,), stop_on=(ValueError,))
    try:
        await RetryPolicy(config).execute(specific_error)
    except ValueError:
        print("Stopped on ValueError as expected")

    # Custom condition
    print("\n--- Custom Condition ---")
    attempt_count = 0

    async def returns_false_then_true() -> bool:
        nonlocal attempt_count
        attempt_count += 1
        return attempt_count >= 3

    config = RetryConfig(
        max_attempts=5,
        condition=RetryCondition.CUSTOM,
        custom_condition=lambda r: not r,  # Retry while False
    )
    result = await RetryPolicy(config).execute(returns_false_then_true)
    print(f"Result: {result} (attempts: {attempt_count})")

    # Different backoff strategies
    print("\n--- Backoff Strategies ---")
    for backoff in [ExponentialBackoff(0.1), LinearBackoff(0.1), ConstantBackoff(0.1)]:
        attempt_count = 0
        start = time.monotonic()

        async def slow() -> str:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError()
            return "ok"

        with contextlib.suppress(ConnectionError):
            await RetryPolicy(RetryConfig(max_attempts=3), backoff).execute(slow)

        elapsed = time.monotonic() - start
        print(f"  {backoff.__class__.__name__}: {elapsed:.3f}s total delay")


if __name__ == "__main__":
    asyncio.run(demonstrate_retry())
