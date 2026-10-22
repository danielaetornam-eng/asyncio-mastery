"""
Circuit Breaker - Prevent cascade failures.

THE PROBLEM
When a downstream service (a payment API, a database) starts failing,
blindly retrying every request makes things worse: each caller burns
time waiting on timeouts, the sick service gets hammered while it
struggles to recover, and the failure cascades up through your system.

REAL-WORLD ANALOGY
An electrical circuit breaker in your house. Too much current
(failed calls) and it trips, cutting power (rejecting calls instantly)
to protect the whole house. Later you flip it back part-way to test
whether the fault cleared before restoring full power. Exactly so here.

THE STATE MACHINE
::

    failures >= threshold
CLOSED --------------------> OPEN
  ^                            |
  |  successes >= threshold    | timeout elapsed
  |                            v
(close)                   HALF_OPEN
                             |
                   any failure v
                        back to OPEN

- CLOSED — normal operation; calls go straight through. We count
  failures; too many trips the breaker OPEN.
- OPEN — fail fast: calls are rejected immediately with
  CircuitBreakerOpenError instead of wasting time on a doomed
  call. After timeout seconds we cautiously try again.
- HALF_OPEN — probation: let a few calls through. Enough successes
  flips back to CLOSED; any failure re-trips to OPEN.

WHY ASYNC PRIMITIVES ARE NEEDED
Many coroutines share one breaker, so state reads/writes are guarded by
an asyncio.Lock — otherwise two tasks could both pass the
"can execute?" check when only one should (the half-open probe is meant
to be scarce). Because callers suspend instead of blocking threads,
rejected or waiting calls cost almost nothing.

WHEN TO USE IT
- Wrapping calls to any unreliable remote dependency.
- Pairing with a retry policy: retry inside, trip the breaker when
  the whole operation is hopeless (avoids retry storms).

Minimal example
>>> import asyncio
>>> async def flaky() -> str: return "ok"
>>> async def main() -> None:
...     breaker = CircuitBreaker("api")
...     print(await breaker.call(flaky))
>>> asyncio.run(main())

States: CLOSED (normal) -> OPEN (failing) -> HALF_OPEN (testing) -> CLOSED
"""

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states.

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery
    """

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Tuning knobs for a CircuitBreaker.

    Attributes:
        failure_threshold: Consecutive-ish failures (since the last
            success) that trip the breaker from CLOSED to OPEN.
        success_threshold: Successful calls in HALF_OPEN required
            before we trust the service again and go CLOSED.
        timeout: Seconds the breaker stays OPEN before allowing a
            probe call (transition to HALF_OPEN).
        excluded_exceptions: Exception types that should not count
            as failures (treated as neutral, e.g. a validation error
            is the caller's fault, not the service's).
    """

    failure_threshold: int = 5  # Failures to open
    success_threshold: int = 2  # Successes to close from half-open
    timeout: float = 30.0  # Seconds before half-open
    excluded_exceptions: tuple[type[Exception], ...] = ()  # Do not count these


@dataclass
class CircuitStats:
    """Mutable bookkeeping for a breaker.

    Attributes:
        state: Current CircuitState.
        failures: Failures counted since the last success (or close).
        successes: Successes counted in HALF_OPEN.
        total_calls: Calls actually attempted through the breaker.
        rejected_calls: Calls refused because the breaker was OPEN.
        last_failure_time: time.monotonic() of the most recent
            failure — the reference for the OPEN timeout.
        last_state_change: When the current state was entered.
    """

    state: CircuitState = CircuitState.CLOSED
    failures: int = 0
    successes: int = 0
    total_calls: int = 0
    rejected_calls: int = 0
    last_failure_time: float | None = None
    last_state_change: float = field(default_factory=time.monotonic)


class CircuitBreakerOpenError(Exception):
    """Raised when circuit is open.

    This is the "fail fast" path: instead of making a doomed call, the
    breaker refuses immediately and tells you when to try again.

    Attributes:
        retry_after: Seconds until the breaker may allow calls again.
    """

    def __init__(self, retry_after: float) -> None:
        """Create the error.

        Args:
            retry_after: Estimated seconds until the breaker probes again.
        """
        self.retry_after = retry_after
        super().__init__(f"Circuit open, retry after {retry_after:.1f}s")


class CircuitBreaker:
    """Circuit breaker implementation.

    Wrap calls to an unreliable dependency with call. The breaker
    counts outcomes and moves through CLOSED -> OPEN -> HALF_OPEN -> CLOSED,
    failing fast while the dependency is known to be down.

    Attributes:
        _name: Human-readable label (used in logs on state transitions).
        _config: The CircuitBreakerConfig tuning values.
        _stats: Mutable CircuitStats counters.
        _lock: Guards shared state across coroutines.
        _failure_times: Recent failure timestamps (deque available for
            windowed strategies).

    Example:
        >>> breaker = CircuitBreaker("payments", CircuitBreakerConfig(
        ...     failure_threshold=3, timeout=10.0))
        >>> try:
        ...     result = await breaker.call(charge_card, amount)
        ... except CircuitBreakerOpenError as e:
        ...     print(f"try again in {e.retry_after:.1f}s")
    """

    def __init__(self, name: str, config: CircuitBreakerConfig | None = None) -> None:
        """Create a breaker in CLOSED state.

        Args:
            name: Label for logging.
            config: Tuning values; defaults to CircuitBreakerConfig()
                (5 failures trips, 30s timeout, 2 successes to close).
        """
        self._name = name
        self._config = config or CircuitBreakerConfig()
        self._stats = CircuitStats()
        self._lock = asyncio.Lock()
        self._failure_times: deque[float] = deque()

    @property
    def name(self) -> str:
        """The breaker's label.

        Returns:
            str: Name passed to the constructor.
        """
        return self._name

    @property
    def state(self) -> CircuitState:
        """Current state of the state machine.

        Returns:
            CircuitState: CLOSED, OPEN, or HALF_OPEN.
        """
        return self._stats.state

    @property
    def stats(self) -> CircuitStats:
        """Live statistics object.

        Returns:
            CircuitStats: The shared stats (mutated as calls flow).
        """
        return self._stats

    async def call(self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        """Execute function with circuit breaker.

        Before calling, we check (under the lock) whether the state
        machine permits a call — this is where OPEN rejects fast and
        OPEN->HALF_OPEN promotion happens. After the call, we record
        success or failure, which may trip the breaker.

        Args:
            func: The async function to call.
            *args: Positional arguments for func.
            **kwargs: Keyword arguments for func.

        Returns:
            T: Whatever func returns.

        Raises:
            CircuitBreakerOpenError: If the breaker is OPEN — the call
                was rejected without invoking func.
            Exception: Whatever func itself raises (after recording
                the failure). Excluded exceptions from the config are
                re-raised but counted as neutral (success path).
        """
        async with self._lock:
            if not await self._can_execute():
                raise CircuitBreakerOpenError(self._retry_after())

            self._stats.total_calls += 1

        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except self._config.excluded_exceptions:
            await self._on_success()
            raise
        except Exception:
            await self._on_failure()
            raise

    async def _can_execute(self) -> bool:
        """Gate check: may a call proceed right now?

        CLOSED always allows; OPEN allows only after timeout has
        passed (promoting to HALF_OPEN on the way); HALF_OPEN allows
        probe calls.

        Returns:
            bool: True to let the call through; False to reject.
        """
        if self._stats.state == CircuitState.CLOSED:
            return True

        if self._stats.state == CircuitState.OPEN:
            # State transition: OPEN -> HALF_OPEN once the cooldown has
            # elapsed since the last failure.
            if time.monotonic() - self._stats.last_failure_time >= self._config.timeout:
                await self._transition_to(CircuitState.HALF_OPEN)
                return True
            return False

        # HALF_OPEN - allow one request
        return True

    def _retry_after(self) -> float:

        """How long until the OPEN breaker might allow calls again.

        Returns:
            float: Remaining seconds of cooldown (0 if not OPEN).
        """
        if self._stats.state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._stats.last_failure_time
            return max(0, self._config.timeout - elapsed)
        return 0

    async def _on_success(self) -> None:
        """Record a successful call.

        In HALF_OPEN, enough consecutive successes close the breaker.
        In CLOSED, a success resets the failure counter (failures only
        trip the breaker when they happen in a row).
        """
        async with self._lock:
            if self._stats.state == CircuitState.HALF_OPEN:
                self._stats.successes += 1
                if self._stats.successes >= self._config.success_threshold:
                    # State transition: HALF_OPEN -> CLOSED (service restored).
                    await self._transition_to(CircuitState.CLOSED)
            elif self._stats.state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_times.clear()
                self._stats.failures = 0

    async def _on_failure(self) -> None:
        """Record a failed call.

        In HALF_OPEN a single failure re-trips to OPEN (the service
        clearly is not better). In CLOSED, reaching failure_threshold
        trips to OPEN.
        """
        async with self._lock:
            now = time.monotonic()
            self._failure_times.append(now)
            self._stats.failures += 1
            self._stats.last_failure_time = now

            if self._stats.state == CircuitState.HALF_OPEN:
                # State transition: HALF_OPEN -> OPEN (still broken).
                await self._transition_to(CircuitState.OPEN)
            elif self._stats.state == CircuitState.CLOSED:
                if self._stats.failures >= self._config.failure_threshold:
                    # State transition: CLOSED -> OPEN (too many failures).
                    await self._transition_to(CircuitState.OPEN)

    async def _transition_to(self, new_state: CircuitState) -> None:
        """Move the state machine to a new state and reset counters.

        Entering states has side effects: CLOSED clears all failure
        history, HALF_OPEN clears success history. Transitions are
        logged because they are operationally important events.

        Args:
            new_state: The destination state.
        """
        old_state = self._stats.state
        self._stats.state = new_state
        self._stats.last_state_change = time.monotonic()

        if new_state == CircuitState.CLOSED:
            self._stats.failures = 0
            self._stats.successes = 0
            self._failure_times.clear()
        elif new_state == CircuitState.HALF_OPEN:
            self._stats.successes = 0

        print(f"Circuit '{self._name}': {old_state.value} -> {new_state.value}")

    async def reset(self) -> None:
        """Manually reset circuit.

        Forces the breaker back to CLOSED and clears all counters — e.g.
        after you have manually verified the dependency recovered.
        """
        async with self._lock:
            await self._transition_to(CircuitState.CLOSED)

    def force_open(self) -> None:
        """Force circuit open.

        Operator override: reject all calls immediately, e.g. before a
        planned dependency outage. Uses time.monotonic() so the normal
        OPEN cooldown clock starts now.
        """
        self._stats.state = CircuitState.OPEN
        self._stats.last_state_change = time.monotonic()
        self._stats.last_failure_time = time.monotonic()


async def demonstrate_circuit_breaker() -> None:
    """Run a self-contained demo of the breaker state machine.

    Watch the printed transitions: the service fails, the breaker trips
    OPEN (calls get rejected fast), then after the cooldown a half-open
    probe succeeds enough times to close the breaker again.
    """
    print("=== Circuit Breaker Demo ===")

    config = CircuitBreakerConfig(failure_threshold=3, success_threshold=2, timeout=0.5)
    breaker = CircuitBreaker("demo", config)

    call_count = 0

    async def unreliable_service() -> str:
        nonlocal call_count
        call_count += 1
        if call_count <= 4:
            raise ConnectionError(f"Failure {call_count}")
        return "success"

    # Trigger failures
    print("\n--- Triggering Failures ---")
    for i in range(5):
        try:
            await breaker.call(unreliable_service)
            print(f"Call {i + 1}: Success")
        except CircuitBreakerOpenError as e:
            print(f"Call {i + 1}: Circuit Open (retry after {e.retry_after:.1f}s)")
        except ConnectionError as e:
            print(f"Call {i + 1}: Failed - {e}")

    print(f"State: {breaker.state}")

    # Wait for half-open
    print("\n--- Waiting for Half-Open ---")
    await asyncio.sleep(0.6)
    print(f"State after timeout: {breaker.state}")

    # Test recovery
    print("\n--- Recovery ---")
    for i in range(3):
        try:
            result = await breaker.call(unreliable_service)
            print(f"Call: Success - {result}")
        except CircuitBreakerOpenError:
            print("Call: Circuit Open")
        except ConnectionError as e:
            print(f"Call: Failed - {e}")

    print(f"Final state: {breaker.state}")
    print(f"Stats: {breaker.stats}")


if __name__ == "__main__":
    asyncio.run(demonstrate_circuit_breaker())
