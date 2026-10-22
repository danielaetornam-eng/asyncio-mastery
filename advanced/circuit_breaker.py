"""
Circuit Breaker Pattern

Implementation of the circuit breaker pattern for fault tolerance.
"""

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open."""

    def __init__(self, message: str, retry_after: float) -> None:
        super().__init__(message)
        self.retry_after = retry_after


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes in half-open before closing
    timeout: float = 30.0  # Time in open state before half-open
    excluded_exceptions: tuple = ()  # Exceptions that don't count as failures
    expected_exception: type = Exception  # Exception type to catch
    name: str | None = None


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics."""

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    total_requests: int = 0
    rejected_requests: int = 0
    last_failure_time: float | None = None
    last_success_time: float | None = None
    state_changes: list[tuple[float, CircuitState, CircuitState]] = field(default_factory=list)


class CircuitBreaker(Generic[T]):
    """Circuit breaker for async operations."""

    def __init__(
        self,
        config: CircuitBreakerConfig | None = None,
        *,
        fallback: Callable[..., Awaitable[T]] | None = None,
    ) -> None:
        self._config = config or CircuitBreakerConfig()
        self._fallback = fallback
        self._state = CircuitState.CLOSED
        self._stats = CircuitBreakerStats(state=self._state)
        self._lock = asyncio.Lock()
        self._half_open_lock = asyncio.Lock()
        self._last_state_change = time.monotonic()

        # For metrics
        self._recent_failures: deque = deque(maxlen=100)
        self._recent_successes: deque = deque(maxlen=100)

    @property
    def name(self) -> str:
        return self._config.name or f"CircuitBreaker-{id(self)}"

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def stats(self) -> CircuitBreakerStats:
        return self._stats

    @property
    def is_closed(self) -> bool:
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        return self._state == CircuitState.HALF_OPEN

    async def call(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            await self._check_state_transition()

            if self._state == CircuitState.OPEN:
                self._stats.rejected_requests += 1
                retry_after = self._config.timeout - (time.monotonic() - self._last_state_change)
                raise CircuitBreakerOpen(
                    f"Circuit breaker '{self.name}' is OPEN", retry_after=max(0, retry_after)
                )

            self._stats.total_requests += 1

        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except self._config.excluded_exceptions:
            # Don't count excluded exceptions as failures
            raise
        except self._config.expected_exception as _e:
            await self._on_failure()
            raise

    async def _check_state_transition(self) -> None:
        """Check if state should transition."""
        now = time.monotonic()

        if self._state == CircuitState.OPEN:
            if now - self._last_state_change >= self._config.timeout:
                await self._transition_to_half_open()

    async def _transition_to_half_open(self) -> None:
        """Transition to half-open state."""
        old_state = self._state
        self._state = CircuitState.HALF_OPEN
        self._stats.state = self._state
        self._last_state_change = time.monotonic()
        self._stats.state_changes.append((self._last_state_change, old_state, self._state))
        self._stats.success_count = 0

    async def _transition_to_closed(self) -> None:
        """Transition to closed state."""
        old_state = self._state
        self._state = CircuitState.CLOSED
        self._stats.state = self._state
        self._last_state_change = time.monotonic()
        self._stats.state_changes.append((self._last_state_change, old_state, self._state))
        self._stats.failure_count = 0
        self._stats.success_count = 0

    async def _transition_to_open(self) -> None:
        """Transition to open state."""
        old_state = self._state
        self._state = CircuitState.OPEN
        self._stats.state = self._state
        self._last_state_change = time.monotonic()
        self._stats.state_changes.append((self._last_state_change, old_state, self._state))

    async def _on_success(self) -> None:
        """Handle successful call."""
        async with self._lock:
            self._stats.last_success_time = time.monotonic()
            self._recent_successes.append(self._stats.last_success_time)

            if self._state == CircuitState.HALF_OPEN:
                self._stats.success_count += 1
                if self._stats.success_count >= self._config.success_threshold:
                    await self._transition_to_closed()
            elif self._state == CircuitState.CLOSED:
                self._stats.failure_count = 0  # Reset failure count on success

    async def _on_failure(self) -> None:
        """Handle failed call."""
        async with self._lock:
            self._stats.last_failure_time = time.monotonic()
            self._recent_failures.append(self._stats.last_failure_time)

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open goes back to open
                await self._transition_to_open()
            elif self._state == CircuitState.CLOSED:
                self._stats.failure_count += 1
                if self._stats.failure_count >= self._config.failure_threshold:
                    await self._transition_to_open()

    async def execute(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """Execute with fallback support."""
        try:
            return await self.call(func, *args, **kwargs)
        except CircuitBreakerOpen:
            if self._fallback:
                return await self._fallback(*args, **kwargs)
            raise

    @asynccontextmanager
    async def protect(self):
        """Context manager for protecting a block of code."""
        await self._check_state_transition()

        if self._state == CircuitState.OPEN:
            retry_after = self._config.timeout - (time.monotonic() - self._last_state_change)
            raise CircuitBreakerOpen(
                f"Circuit breaker '{self.name}' is OPEN", retry_after=max(0, retry_after)
            )

        self._stats.total_requests += 1

        try:
            yield
            await self._on_success()
        except self._config.excluded_exceptions:
            raise
        except self._config.expected_exception as _e:
            await self._on_failure()
            raise
        except Exception as _e:
            await self._on_failure()
            raise

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self._state = CircuitState.CLOSED
        self._stats.state = self._state
        self._stats.failure_count = 0
        self._stats.success_count = 0
        self._last_state_change = time.monotonic()

    def force_open(self) -> None:
        """Force circuit breaker open."""
        if self._state != CircuitState.OPEN:
            self._transition_to_open()

    def force_closed(self) -> None:
        """Force circuit breaker closed."""
        if self._state != CircuitState.CLOSED:
            self._transition_to_closed()

    def get_metrics(self) -> dict[str, Any]:
        """Get detailed metrics."""
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._stats.failure_count,
            "success_count": self._stats.success_count,
            "total_requests": self._stats.total_requests,
            "rejected_requests": self._stats.rejected_requests,
            "failure_rate": (self._stats.failure_count / max(1, self._stats.total_requests)),
            "recent_failures": len(self._recent_failures),
            "recent_successes": len(self._recent_successes),
            "time_in_state": time.monotonic() - self._last_state_change,
        }


class AsyncCircuitBreaker(CircuitBreaker):
    """Async circuit breaker with additional async features."""

    def __init__(
        self,
        config: CircuitBreakerConfig | None = None,
        *,
        fallback: Callable[..., Awaitable[T]] | None = None,
        on_state_change: Callable[[CircuitState, CircuitState], Awaitable[None]] | None = None,
    ) -> None:
        super().__init__(config, fallback=fallback)
        self._on_state_change = on_state_change

    async def _transition_to_half_open(self) -> None:
        old_state = self._state
        await super()._transition_to_half_open()
        if self._on_state_change:
            await self._on_state_change(old_state, self._state)

    async def _transition_to_closed(self) -> None:
        old_state = self._state
        await super()._transition_to_closed()
        if self._on_state_change:
            await self._on_state_change(old_state, self._state)

    async def _transition_to_open(self) -> None:
        old_state = self._state
        await super()._transition_to_open()
        if self._on_state_change:
            await self._on_state_change(old_state, self._state)


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    def __init__(self) -> None:
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    def get_or_create(
        self, name: str, config: CircuitBreakerConfig | None = None
    ) -> CircuitBreaker:
        if name not in self._breakers:
            cfg = config or CircuitBreakerConfig(name=name)
            self._breakers[name] = CircuitBreaker(cfg)
        return self._breakers[name]

    def get(self, name: str) -> CircuitBreaker | None:
        return self._breakers.get(name)

    def remove(self, name: str) -> bool:
        if name in self._breakers:
            del self._breakers[name]
            return True
        return False

    def all_breakers(self) -> list[CircuitBreaker]:
        return list(self._breakers.values())

    def get_all_metrics(self) -> dict[str, dict[str, Any]]:
        return {name: cb.get_metrics() for name, cb in self._breakers.items()}

    async def reset_all(self) -> None:
        for cb in self._breakers.values():
            cb.reset()


_global_registry = CircuitBreakerRegistry()


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    return _global_registry


def circuit_breaker(
    failure_threshold: int = 5, success_threshold: int = 2, timeout: float = 30.0, **kwargs
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator for circuit breaker."""
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        success_threshold=success_threshold,
        timeout=timeout,
        **kwargs,
    )
    breaker = CircuitBreaker(config)

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await breaker.execute(func, *args, **kwargs)

        return wrapper

    return decorator


from functools import wraps
