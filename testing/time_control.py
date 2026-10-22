"""
Time Control for Testing

Utilities for controlling time in async tests.
"""

import asyncio
import functools
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from threading import Lock
from typing import TypeVar
from unittest.mock import patch

T = TypeVar("T")


class TimeController:
    """Control time in async tests."""

    def __init__(self) -> None:
        self._time = time.time()
        self._lock = Lock()
        self._paused = False
        self._callbacks: list[Callable[[float], None]] = []

    @property
    def time(self) -> float:
        return self._time

    @property
    def paused(self) -> bool:
        return self._paused

    def set_time(self, t: float) -> None:
        with self._lock:
            self._time = t
            for cb in self._callbacks:
                with suppress(Exception):
                    cb(t)

    def advance(self, delta: float) -> float:
        with self._lock:
            if delta < 0:
                raise ValueError("Cannot advance by negative amount")
            self._time += delta
            new_time = self._time
            for cb in self._callbacks:
                with suppress(Exception):
                    cb(new_time)
            return new_time

    def pause(self) -> None:
        with self._lock:
            self._paused = True

    def resume(self) -> None:
        with self._lock:
            self._paused = False

    def add_callback(self, callback: Callable[[float], None]) -> None:
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[float], None]) -> bool:
        try:
            self._callbacks.remove(callback)
            return True
        except ValueError:
            return False


# Global time controller
_global_time_controller: TimeController | None = None
_time_patched = False


def get_time_controller() -> TimeController:
    global _global_time_controller
    if _global_time_controller is None:
        _global_time_controller = TimeController()
    return _global_time_controller


@asynccontextmanager
async def freeze_time(at: float | None = None) -> AsyncGenerator[TimeController, None]:
    """Freeze time at a specific point."""
    global _time_patched

    controller = get_time_controller()

    if at is not None:
        controller.set_time(at)

    # Patch time functions

    def mock_time():
        return controller.time

    def mock_monotonic():
        return controller.time

    async def mock_sleep(delay) -> None:
        controller.advance(delay)
        await asyncio.sleep(0)  # Yield control

    with (
        patch("time.time", mock_time),
        patch("time.monotonic", mock_monotonic),
        patch("asyncio.sleep", mock_sleep),
    ):
        _time_patched = True
        try:
            yield controller
        finally:
            _time_patched = False


@asynccontextmanager
async def advance_time(delta: float) -> AsyncGenerator[TimeController, None]:
    """Advance time by delta."""
    controller = get_time_controller()
    controller.advance(delta)
    yield controller


@asynccontextmanager
async def async_time_travel(
    start: float | None = None, end: float | None = None, step: float = 1.0
) -> AsyncGenerator[TimeController, None]:
    """Time travel through a range."""
    controller = get_time_controller()

    if start is not None:
        controller.set_time(start)

    def mock_time():
        return controller.time

    def mock_monotonic():
        return controller.time

    async def mock_sleep(delay) -> None:
        controller.advance(delay)
        await asyncio.sleep(0)

    with (
        patch("time.time", mock_time),
        patch("time.monotonic", mock_monotonic),
        patch("asyncio.sleep", mock_sleep),
    ):
        _time_patched = True
        try:
            yield controller

            # Advance through time range
            if end is not None and end > controller.time:
                while controller.time < end:
                    controller.advance(step)
                    await asyncio.sleep(0)
        finally:
            _time_patched = False


class MockEventLoop:
    """Mock event loop with time control."""

    def __init__(self, controller: TimeController | None = None) -> None:
        self._controller = controller or TimeController()
        self._tasks: list[asyncio.Task] = []
        self._running = False
        self._callbacks: list[tuple[float, Callable]] = []  # (time, callback)
        self._lock = asyncio.Lock()

    @property
    def time(self) -> float:
        return self._controller.time

    def create_task(self, coro: Awaitable) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._tasks.append(task)
        task.add_done_callback(self._tasks.remove)
        return task

    def call_later(self, delay: float, callback: Callable, *args) -> asyncio.Handle:
        when = self._controller.time + delay
        handle = asyncio.get_event_loop().call_at(
            when, lambda: self._execute_callback(callback, args)
        )
        return handle

    def call_at(self, when: float, callback: Callable, *args) -> asyncio.Handle:
        handle = asyncio.get_event_loop().call_at(
            when, lambda: self._execute_callback(callback, args)
        )
        return handle

    def _execute_callback(self, callback: Callable, args: tuple) -> None:
        with suppress(Exception):
            callback(*args)

    async def advance_time(self, delta: float) -> None:
        """Advance time and execute ready callbacks."""
        await self._controller.advance(delta)

        # Execute ready callbacks
        now = self._controller.time
        ready = [(t, cb, args) for t, cb, args in self._callbacks if t <= now]

        for t, cb, args in ready:
            with suppress(Exception):
                cb(*args)

        # Remove executed
        self._callbacks = [(t, cb, args) for t, cb, args in self._callbacks if t > now]

    def run_until_complete(self, coro: Awaitable[T]) -> T:
        return asyncio.get_event_loop().run_until_complete(coro)

    def close(self) -> None:
        for task in self._tasks:
            task.cancel()


@asynccontextmanager
async def mock_event_loop(start_time: float | None = None) -> AsyncGenerator[MockEventLoop, None]:
    """Context manager for mock event loop."""
    controller = TimeController()
    if start_time is not None:
        controller.set_time(start_time)

    loop = MockEventLoop(controller)

    original_loop = None
    with suppress(RuntimeError):
        original_loop = asyncio.get_running_loop()

    asyncio.set_event_loop(loop)
    try:
        yield loop
    finally:
        if original_loop:
            asyncio.set_event_loop(original_loop)
        loop.close()


def time_travel_test(
    start: float | None = None, end: float | None = None, step: float = 1.0
) -> Callable:
    """Decorator for time travel tests."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            async def run():
                async with async_time_travel(start, end, step) as controller:
                    return await func(*args, controller=controller, **kwargs)

            return asyncio.run(run())

        return wrapper

    return decorator


def freeze_time_test(at: float | None = None) -> Callable:
    """Decorator to freeze time during test."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            async def run():
                async with freeze_time(at) as controller:
                    return await func(*args, controller=controller, **kwargs)

            return asyncio.run(run())

        return wrapper

    return decorator


# Sleep mocking
class SleepMocker:
    """Mock asyncio.sleep for time control."""

    def __init__(self, controller: TimeController | None = None) -> None:
        self._controller = controller or get_time_controller()
        self._original_sleep = asyncio.sleep
        self._sleeps: list[float] = []

    def __enter__(self):
        async def mock_sleep(delay) -> None:
            self._sleeps.append(delay)
            self._controller.advance(delay)
            await asyncio.sleep(0)  # Yield

        self._patch = patch("asyncio.sleep", mock_sleep)
        self._patch.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._patch.__exit__(exc_type, exc_val, exc_tb)

    @property
    def total_slept(self) -> float:
        return sum(self._sleeps)

    @property
    def sleep_count(self) -> int:
        return len(self._sleeps)

    def assert_slept(self, expected: float, tolerance: float = 0.01) -> None:
        actual = self.total_slept
        assert abs(actual - expected) <= tolerance, (
            f"Expected to sleep {expected}s, actually slept {actual}s"
        )


def mock_sleep(controller: TimeController | None = None) -> SleepMocker:
    """Create sleep mocker."""
    return SleepMocker(controller)


# Pytest fixtures
try:
    import pytest

    @pytest.fixture
    def time_controller():
        """Time controller fixture."""
        return TimeController()

    @pytest.fixture
    async def frozen_time():
        """Frozen time fixture."""
        async with freeze_time() as controller:
            yield controller

    @pytest.fixture
    async def time_travel():
        """Time travel fixture."""
        async with async_time_travel() as controller:
            yield controller

    @pytest.fixture
    def sleep_mocker():
        """Sleep mocker fixture."""
        with SleepMocker() as mocker:
            yield mocker

except ImportError:
    pass


# Async sleep that respects time controller
async def controlled_sleep(delay: float, controller: TimeController | None = None) -> None:
    """Sleep that advances controlled time."""
    controller = controller or get_time_controller()
    controller.advance(delay)
    await asyncio.sleep(0)


def with_time_control(start: float | None = None, auto_advance: bool = False) -> Callable:
    """Decorator to run function with time control."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            async def run():
                async with freeze_time(start) as controller:
                    if auto_advance:
                        # Patch sleep to auto-advance
                        original_sleep = asyncio.sleep

                        async def auto_sleep(delay) -> None:
                            controller.advance(delay)
                            await original_sleep(0)

                        with patch("asyncio.sleep", auto_sleep):
                            return await func(*args, controller=controller, **kwargs)
                    else:
                        return await func(*args, controller=controller, **kwargs)

            return asyncio.run(run())

        return wrapper

    return decorator
