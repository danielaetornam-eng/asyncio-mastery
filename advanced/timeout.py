"""
Timeout Patterns

Advanced timeout handling for async operations.
"""

import asyncio
import builtins
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class TimeoutError(Exception):
    """Raised when operation times out."""

    def __init__(
        self, message: str, timeout: float, elapsed: float, operation: str | None = None
    ) -> None:
        super().__init__(message)
        self.timeout = timeout
        self.elapsed = elapsed
        self.operation = operation


@dataclass
class TimeoutConfig:
    """Configuration for timeout behavior."""

    timeout: float
    default_timeout: float | None = None
    per_operation: bool = False
    cancel_on_timeout: bool = True
    return_on_timeout: Any | None = None
    raise_on_timeout: bool = True


@dataclass
class TimeoutStats:
    """Timeout statistics."""

    total_calls: int = 0
    timed_out: int = 0
    completed: int = 0
    cancelled: int = 0
    total_elapsed: float = 0.0
    max_elapsed: float = 0.0
    timeouts_by_operation: dict[str, int] = field(default_factory=dict)


class Timeout(Generic[T]):
    """Advanced timeout manager."""

    def __init__(
        self,
        timeout: float,
        *,
        default_timeout: float | None = None,
        cancel_on_timeout: bool = True,
        return_on_timeout: T | None = None,
        raise_on_timeout: bool = True,
    ) -> None:
        self._timeout = timeout
        self._default_timeout = default_timeout
        self._cancel_on_timeout = cancel_on_timeout
        self._return_on_timeout = return_on_timeout
        self._raise_on_timeout = raise_on_timeout
        self._stats = TimeoutStats()

    @property
    def timeout(self) -> float:
        return self._timeout

    @property
    def stats(self) -> TimeoutStats:
        return self._stats

    async def run(
        self, coro: Awaitable[T], timeout: float | None = None, operation: str | None = None
    ) -> T:
        """Run coroutine with timeout."""
        effective_timeout = timeout or self._timeout
        self._stats.total_calls += 1
        if operation:
            self._stats.timeouts_by_operation[operation] = (
                self._stats.timeouts_by_operation.get(operation, 0) + 1
            )

        start = time.monotonic()

        try:
            if effective_timeout is not None:
                result = await asyncio.wait_for(coro, timeout=effective_timeout)
            else:
                result = await coro

            elapsed = time.monotonic() - start
            self._stats.completed += 1
            self._stats.total_elapsed += elapsed
            self._stats.max_elapsed = max(self._stats.max_elapsed, elapsed)
            return result

        except builtins.TimeoutError:
            elapsed = time.monotonic() - start
            self._stats.timed_out += 1
            self._stats.total_elapsed += elapsed

            if self._raise_on_timeout:
                raise TimeoutError(
                    f"Operation {'(' + operation + ')' if operation else ''} "
                    f"timed out after {elapsed:.3f}s",
                    timeout=effective_timeout,
                    elapsed=elapsed,
                    operation=operation,
                )
                return self._return_on_timeout
            raise

        except asyncio.CancelledError:
            self._stats.cancelled += 1
            raise

    @asynccontextmanager
    async def limit(self, timeout: float | None = None, operation: str | None = None):
        """Context manager for timeout."""
        effective_timeout = timeout or self._timeout
        self._stats.total_calls += 1

        start = time.monotonic()
        task = asyncio.current_task()

        async def timeout_handler() -> None:
            try:
                await asyncio.sleep(effective_timeout)
                if task and not task.done() and self._cancel_on_timeout:
                    task.cancel()
            except asyncio.CancelledError:
                pass

        timeout_task = asyncio.create_task(timeout_handler())

        try:
            yield
            elapsed = time.monotonic() - start
            self._stats.completed += 1
            self._stats.total_elapsed += elapsed
            self._stats.max_elapsed = max(self._stats.max_elapsed, elapsed)
        except asyncio.CancelledError:
            elapsed = time.monotonic() - start
            self._stats.cancelled += 1
            if self._raise_on_timeout:
                raise TimeoutError(
                    f"Operation {'(' + operation + ')' if operation else ''} "
                    f"timed out after {elapsed:.3f}s",
                    timeout=effective_timeout,
                    elapsed=elapsed,
                    operation=operation,
                )
            raise
        except Exception:
            elapsed = time.monotonic() - start
            self._stats.total_elapsed += elapsed
            raise
        finally:
            timeout_task.cancel()
            with suppress(asyncio.CancelledError):
                await timeout_task

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_calls": self._stats.total_calls,
            "timed_out": self._stats.timed_out,
            "completed": self._stats.completed,
            "cancelled": self._stats.cancelled,
            "timeout_rate": self._stats.timed_out / max(1, self._stats.total_calls),
            "avg_elapsed": self._stats.total_elapsed / max(1, self._stats.total_calls),
            "max_elapsed": self._stats.max_elapsed,
        }


class AdaptiveTimeout(Generic[T]):
    """Timeout that adapts based on historical performance."""

    def __init__(
        self,
        initial_timeout: float = 5.0,
        min_timeout: float = 0.1,
        max_timeout: float = 60.0,
        percentile: float = 0.95,
        window_size: int = 100,
        *,
        name: str | None = None,
    ) -> None:
        self._timeout = initial_timeout
        self._min_timeout = min_timeout
        self._max_timeout = max_timeout
        self._percentile = percentile
        self._window_size = window_size
        self._name = name or f"AdaptiveTimeout-{id(self)}"
        self._durations: list[float] = []
        self._timeouts = 0
        self._successes = 0
        self._lock = asyncio.Lock()

    @property
    def timeout(self) -> float:
        return self._timeout

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "name": self._name,
            "current_timeout": self._timeout,
            "min_timeout": self._min_timeout,
            "max_timeout": self._max_timeout,
            "sample_count": len(self._durations),
            "timeouts": self._timeouts,
            "successes": self._successes,
            "success_rate": self._successes / max(1, self._successes + self._timeouts),
        }

    async def run(
        self, coro: Awaitable[T], timeout: float | None = None, operation: str | None = None
    ) -> T:
        effective_timeout = timeout or self._timeout
        start = time.monotonic()

        try:
            result = await asyncio.wait_for(coro, timeout=effective_timeout)
            elapsed = time.monotonic() - start
            await self._record_success(elapsed)
            return result
        except builtins.TimeoutError:
            elapsed = time.monotonic() - start
            await self._record_timeout(elapsed)
            raise TimeoutError(
                f"Adaptive timeout for {'(' + operation + ')' if operation else ''} "
                f"after {elapsed:.3f}s",
                timeout=effective_timeout,
                elapsed=elapsed,
                operation=operation,
            )

    async def _record_success(self, elapsed: float) -> None:
        async with self._lock:
            self._durations.append(elapsed)
            self._successes += 1
            if len(self._durations) > self._window_size:
                self._durations.pop(0)
            self._adjust_timeout()

    async def _record_timeout(self, elapsed: float) -> None:
        async with self._lock:
            self._timeouts += 1
            self._adjust_timeout()

    def _adjust_timeout(self) -> None:
        if not self._durations:
            return

        sorted_durations = sorted(self._durations)
        index = int(len(sorted_durations) * self._percentile)
        index = min(index, len(sorted_durations) - 1)

        target_timeout = sorted_durations[index] * 2  # 2x the percentile

        # Smooth adjustment
        self._timeout = max(self._min_timeout, min(self._max_timeout, target_timeout))


class TimeoutBudget(Generic[T]):
    """Budget-based timeout for operations with multiple steps."""

    def __init__(self, total_timeout: float, *, name: str | None = None) -> None:
        self._total_timeout = total_timeout
        self._name = name or f"TimeoutBudget-{id(self)}"
        self._start_time: float | None = None
        self._step_timeouts: dict[str, float] = {}
        self._step_start: dict[str, float] = {}

    async def __aenter__(self) -> "TimeoutBudget":
        self._start_time = time.monotonic()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    @property
    def remaining(self) -> float:
        if self._start_time is None:
            return self._total_timeout
        elapsed = time.monotonic() - self._start_time
        return max(0, self._total_timeout - elapsed)

    @property
    def elapsed(self) -> float:
        if self._start_time is None:
            return 0
        return time.monotonic() - self._start_time

    @property
    def expired(self) -> bool:
        return self.remaining <= 0

    def allocate_step(self, step_name: str, timeout: float) -> None:
        """Allocate timeout budget for a step."""
        self._step_timeouts[step_name] = timeout

    @asynccontextmanager
    async def step(self, step_name: str, timeout: float | None = None):
        """Context manager for a step with timeout."""
        effective_timeout = timeout or self._step_timeouts.get(step_name, self.remaining)
        effective_timeout = min(effective_timeout, self.remaining)

        self._step_start[step_name] = time.monotonic()

        try:
            async with asyncio.timeout(effective_timeout):
                yield
        finally:
            pass

    async def run_step(self, step_name: str, coro: Awaitable[T], timeout: float | None = None) -> T:
        """Run a step with allocated timeout."""
        effective_timeout = timeout or self._step_timeouts.get(step_name, self.remaining)
        effective_timeout = min(effective_timeout, self.remaining)

        if effective_timeout <= 0:
            raise TimeoutError(
                f"Timeout budget exhausted for step '{step_name}'",
                timeout=0,
                elapsed=self.elapsed,
                operation=step_name,
            )

        self._step_start[step_name] = time.monotonic()

        try:
            return await asyncio.wait_for(coro, timeout=effective_timeout)
        except builtins.TimeoutError:
            elapsed = time.monotonic() - self._step_start[step_name]
            raise TimeoutError(
                f"Step '{step_name}' timed out after {elapsed:.3f}s",
                timeout=effective_timeout,
                elapsed=elapsed,
                operation=step_name,
            )


async def with_timeout(
    coro: Awaitable[T],
    timeout: float,
    *,
    default: T | None = None,
    raise_on_timeout: bool = True,
    operation: str | None = None,
) -> T:
    """Run coroutine with timeout, with optional default return."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except builtins.TimeoutError:
        elapsed = timeout
        if raise_on_timeout:
            raise TimeoutError(
                f"Operation {'(' + operation + ')' if operation else ''} "
                f"timed out after {elapsed}s",
                timeout=timeout,
                elapsed=elapsed,
                operation=operation,
            )
        return default


def timeout(
    seconds: float, *, default: T | None = None, raise_on_timeout: bool = True
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator for adding timeout to async function."""

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await with_timeout(
                func(*args, **kwargs),
                seconds,
                default=default,
                raise_on_timeout=raise_on_timeout,
                operation=func.__name__,
            )

        return wrapper

    return decorator


class TimeoutManager:
    """Manage multiple timeouts with different policies."""

    def __init__(self) -> None:
        self._timeouts: dict[str, Timeout] = {}
        self._adaptive_timeouts: dict[str, AdaptiveTimeout] = {}
        self._budgets: dict[str, TimeoutBudget] = {}

    def add_timeout(self, name: str, timeout: float, **kwargs) -> Timeout:
        t = Timeout(timeout, **kwargs)
        self._timeouts[name] = t
        return t

    def add_adaptive_timeout(
        self, name: str, initial_timeout: float = 5.0, **kwargs
    ) -> AdaptiveTimeout:
        at = AdaptiveTimeout(initial_timeout, **kwargs)
        self._adaptive_timeouts[name] = at
        return at

    def create_budget(self, name: str, total_timeout: float) -> TimeoutBudget:
        budget = TimeoutBudget(total_timeout, name=name)
        self._budgets[name] = budget
        return budget

    def get_timeout(self, name: str) -> Timeout | None:
        return self._timeouts.get(name)

    def get_adaptive_timeout(self, name: str) -> AdaptiveTimeout | None:
        return self._adaptive_timeouts.get(name)

    def get_budget(self, name: str) -> TimeoutBudget | None:
        return self._budgets.get(name)

    def get_all_stats(self) -> dict[str, Any]:
        return {
            "timeouts": {n: t.get_stats() for n, t in self._timeouts.items()},
            "adaptive": {n: t.stats for n, t in self._adaptive_timeouts.items()},
            "budgets": {
                n: {"remaining": b.remaining, "elapsed": b.elapsed}
                for n, b in self._budgets.items()
            },
        }


_global_timeout_manager = TimeoutManager()


def get_timeout_manager() -> TimeoutManager:
    return _global_timeout_manager
