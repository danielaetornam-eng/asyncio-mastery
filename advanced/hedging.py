"""
Request Hedging

Send duplicate requests to mitigate tail latency.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from functools import wraps
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass
class HedgingConfig:
    """Configuration for request hedging."""

    hedge_after: float = 0.1  # Time before sending hedge request
    max_hedges: int = 2  # Maximum number of hedge requests
    cancel_on_first_success: bool = True
    exceptions: tuple[type, ...] = (Exception,)


@dataclass
class HedgingStats:
    """Hedging statistics."""

    total_requests: int = 0
    primary_success: int = 0
    hedge_success: int = 0
    all_failed: int = 0
    hedges_sent: int = 0
    cancelled_hedges: int = 0


class HedgingPolicy(Generic[T]):
    """Policy for hedging requests."""

    def __init__(
        self, func: Callable[..., Awaitable[T]], config: HedgingConfig | None = None
    ) -> None:
        self._func = func
        self._config = config or HedgingConfig()
        self._stats = HedgingStats()

    @property
    def stats(self) -> HedgingStats:
        return self._stats

    async def execute(self, *args, **kwargs) -> T:
        """Execute with hedging."""
        self._stats.total_requests += 1

        # Start primary request
        primary_task = asyncio.create_task(self._func(*args, **kwargs))

        hedge_tasks: list[asyncio.Task] = []
        hedge_delay = self._config.hedge_after

        try:
            # Wait for primary with hedge delay
            result = await asyncio.wait_for(primary_task, timeout=hedge_delay)
            self._stats.primary_success += 1
            return result
        except TimeoutError:
            # Primary didn't complete in time, send hedges
            pass
        except self._config.exceptions:
            # Primary failed immediately
            primary_task.cancel()
            # Still send hedges

        # Send hedge requests
        for i in range(self._config.max_hedges):
            if primary_task.done() and self._config.cancel_on_first_success:
                break

            hedge_task = asyncio.create_task(self._func(*args, **kwargs))
            hedge_tasks.append(hedge_task)
            self._stats.hedges_sent += 1

        # Wait for any to complete
        all_tasks = [primary_task, *hedge_tasks]

        try:
            if self._config.cancel_on_first_success:
                done, pending = await asyncio.wait(all_tasks, return_when=asyncio.FIRST_COMPLETED)

                # Cancel pending
                for task in pending:
                    task.cancel()
                    self._stats.cancelled_hedges += 1

                # Get result from completed
                for task in done:
                    try:
                        result = task.result()
                        if task is primary_task:
                            self._stats.primary_success += 1
                        else:
                            self._stats.hedge_success += 1
                        return result
                    except self._config.exceptions:
                        continue

                # All completed but failed
                self._stats.all_failed += 1
                raise Exception("All hedged requests failed")
            else:
                # Wait for all
                results = await asyncio.gather(*all_tasks, return_exceptions=True)

                for i, result in enumerate(results):
                    if not isinstance(result, Exception):
                        if i == 0:
                            self._stats.primary_success += 1
                        else:
                            self._stats.hedge_success += 1
                        return result

                self._stats.all_failed += 1
                raise Exception("All hedged requests failed")

        except asyncio.CancelledError:
            for task in all_tasks:
                task.cancel()
            raise


class AdaptiveHedging(Generic[T]):
    """Adaptive hedging based on latency percentiles."""

    def __init__(
        self,
        func: Callable[..., Awaitable[T]],
        percentile: float = 0.95,
        min_hedge_delay: float = 0.01,
        max_hedge_delay: float = 1.0,
        max_hedges: int = 2,
        window_size: int = 100,
    ) -> None:
        self._func = func
        self._percentile = percentile
        self._min_hedge_delay = min_hedge_delay
        self._max_hedge_delay = max_hedge_delay
        self._max_hedges = max_hedges
        self._window_size = window_size
        self._latencies: list[float] = []
        self._stats = HedgingStats()
        self._lock = asyncio.Lock()

    @property
    def stats(self) -> HedgingStats:
        return self._stats

    def _calculate_hedge_delay(self) -> float:
        if not self._latencies:
            return self._min_hedge_delay

        sorted_latencies = sorted(self._latencies)
        index = int(len(sorted_latencies) * self._percentile)
        index = min(index, len(sorted_latencies) - 1)

        delay = sorted_latencies[index]
        return max(self._min_hedge_delay, min(self._max_hedge_delay, delay))

    async def execute(self, *args, **kwargs) -> T:
        self._stats.total_requests += 1

        hedge_delay = self._calculate_hedge_delay()

        primary_task = asyncio.create_task(self._func(*args, **kwargs))

        try:
            start = time.monotonic()
            result = await asyncio.wait_for(primary_task, timeout=hedge_delay)
            elapsed = time.monotonic() - start

            async with self._lock:
                self._latencies.append(elapsed)
                if len(self._latencies) > self._window_size:
                    self._latencies.pop(0)

            self._stats.primary_success += 1
            return result
        except TimeoutError:
            pass
        except Exception:
            primary_task.cancel()

        # Send hedges
        hedge_tasks = []
        for i in range(self._max_hedges):
            if primary_task.done():
                break
            hedge_task = asyncio.create_task(self._func(*args, **kwargs))
            hedge_tasks.append(hedge_task)
            self._stats.hedges_sent += 1

        all_tasks = [primary_task, *hedge_tasks]
        done, pending = await asyncio.wait(all_tasks, return_when=asyncio.FIRST_COMPLETED)

        for task in pending:
            task.cancel()
            self._stats.cancelled_hedges += 1

        for task in done:
            try:
                result = task.result()
                if task is primary_task:
                    self._stats.primary_success += 1
                else:
                    self._stats.hedge_success += 1
                return result
            except Exception:
                continue

        self._stats.all_failed += 1
        raise Exception("All hedged requests failed")


class ParallelHedging(Generic[T]):
    """Run multiple implementations in parallel, return first success."""

    def __init__(
        self,
        functions: list[Callable[..., Awaitable[T]]],
        exceptions: tuple[type, ...] = (Exception,),
        cancel_on_success: bool = True,
    ) -> None:
        if not functions:
            raise ValueError("At least one function required")
        self._functions = functions
        self._exceptions = exceptions
        self._cancel_on_success = cancel_on_success
        self._stats = {"total": 0, "by_index": [0] * len(functions), "cancelled": 0}

    @property
    def stats(self) -> dict[str, Any]:
        return self._stats.copy()

    async def execute(self, *args, **kwargs) -> T:
        self._stats["total"] += 1

        tasks = [asyncio.create_task(func(*args, **kwargs)) for func in self._functions]

        try:
            if self._cancel_on_success:
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

                for task in pending:
                    task.cancel()
                    self._stats["cancelled"] += 1

                for task in done:
                    try:
                        result = task.result()
                        idx = tasks.index(task)
                        self._stats["by_index"][idx] += 1
                        return result
                    except self._exceptions:
                        continue

                raise Exception("All parallel requests failed")
            else:
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for i, result in enumerate(results):
                    if not isinstance(result, Exception):
                        self._stats["by_index"][i] += 1
                        return result

                raise Exception("All parallel requests failed")

        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            raise


def hedge(
    hedge_after: float = 0.1,
    max_hedges: int = 2,
    cancel_on_first_success: bool = True,
    exceptions: tuple[type, ...] = (Exception,),
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator for request hedging."""

    config = HedgingConfig(
        hedge_after=hedge_after,
        max_hedges=max_hedges,
        cancel_on_first_success=cancel_on_first_success,
        exceptions=exceptions,
    )

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        policy = HedgingPolicy(func, config)

        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await policy.execute(*args, **kwargs)

        return wrapper

    return decorator


class HedgingClient(Generic[T]):
    """Client wrapper with automatic hedging."""

    def __init__(
        self,
        client: T,
        hedge_after: float = 0.1,
        max_hedges: int = 2,
        methods: list[str] | None = None,
        excluded_methods: list[str] | None = None,
    ) -> None:
        self._client = client
        self._hedge_after = hedge_after
        self._max_hedges = max_hedges
        self._methods = set(methods) if methods else None
        self._excluded_methods = set(excluded_methods) if excluded_methods else set()

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
            return await HedgingPolicy(
                lambda: attr(*args, **kwargs),
                HedgingConfig(hedge_after=self._hedge_after, max_hedges=self._max_hedges),
            ).execute()

        return wrapper


@asynccontextmanager
async def hedged_request(
    func: Callable[..., Awaitable[T]],
    hedge_after: float = 0.1,
    max_hedges: int = 2,
    *args,
    **kwargs,
):
    """Context manager for hedged request."""
    policy = HedgingPolicy(func, HedgingConfig(hedge_after=hedge_after, max_hedges=max_hedges))

    task = asyncio.create_task(policy.execute(*args, **kwargs))

    try:
        yield task
    finally:
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


def parallel_hedge(
    *functions: Callable[..., Awaitable[T]],
    exceptions: tuple[type, ...] = (Exception,),
    cancel_on_success: bool = True,
) -> Callable[..., Awaitable[T]]:
    """Create a function that runs multiple implementations in parallel."""
    parallel = ParallelHedging(list(functions), exceptions, cancel_on_success)

    @wraps(functions[0] if functions else None)
    async def wrapper(*args, **kwargs) -> T:
        return await parallel.execute(*args, **kwargs)

    return wrapper
