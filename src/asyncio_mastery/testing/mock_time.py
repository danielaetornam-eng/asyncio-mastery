"""
Mock Time - Control the flow of time inside async tests.

Time-dependent async code is hard to test. Real sleeps make tests slow.
Real time is non-deterministic and causes flaky tests. The solution is
to replace the clock with a controlled mock clock. You can fast-forward
time instantly with clock.advance(60) and verify code reactions.

Key terms:
- patch / monkey-patch: Temporarily swap a real function for a fake one.
- context manager: An object used with with or async with that sets up
  on entry and cleans up on exit.
- time.monotonic: A clock that only moves forward; asyncio uses it for
  scheduling.

Components:
- MockClock: A hand-controlled clock with a fake sleep method.
- TimeTravel: Context manager that patches asyncio.sleep and time.monotonic.
- patch_time: Wrapper around TimeTravel starting from a chosen initial time.
- AsyncTimeMachine: Explicit start/stop version that also patches time.time.

Example:
    with TimeTravel() as travel:
        task = asyncio.create_task(long_running())
        travel.advance(60.0)
        await task
"""

import asyncio
import contextlib
import time
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch


class MockClock:
    """A clock you set and advance by hand, with a mock sleep.

    Instead of reading real time, code sees the time you choose. Tasks
    sleeping via sleep register a wake-up time. When you advance the
    clock past that time, they wake up.

    Args:
        initial_time: The clock's starting value in seconds (default 0.0).

    Attributes:
        time: The current fake time. Readable and writable.

    Example:
        clock = MockClock(100.0)
        clock.advance(5.0)
        clock.time  # 105.0
    """

    def __init__(self, initial_time: float = 0.0) -> None:
        self._time = initial_time
        self._sleep_tasks: dict[asyncio.Task, float] = {}

    @property
    def time(self) -> float:
        """Current fake time on this clock, in seconds."""
        return self._time

    @time.setter
    def time(self, value: float) -> None:
        self._time = value

    def advance(self, delta: float) -> None:
        """Move the clock forward by delta seconds and wake due sleepers.

        Args:
            delta: Seconds to fast-forward. Should be positive.
        """
        self._time += delta
        self._wake_sleepers()

    def set_time(self, value: float) -> None:
        """Jump the clock to an absolute time and wake due sleepers.

        Args:
            value: New absolute time in seconds.
        """
        self._time = value
        self._wake_sleepers()

    def _wake_sleepers(self) -> None:
        """Cancel every sleeping task whose wake-up time has arrived.

        Cancellation wakes the real asyncio.sleep inside sleep. The
        sleeping coroutine resumes immediately.
        """
        current = self._time
        for task, wake_time in list(self._sleep_tasks.items()):
            if wake_time <= current and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    task.result()
                del self._sleep_tasks[task]

    async def sleep(self, delay: float) -> None:
        """Drop-in fake for asyncio.sleep tied to this clock.

        Registers the current task's wake-up time, then does a real sleep
        which the clock's advance interrupts via cancellation.

        Args:
            delay: How many fake seconds to sleep.
        """
        if delay <= 0:
            return
        task = asyncio.current_task()
        if task:
            wake_time = self._time + delay
            self._sleep_tasks[task] = wake_time
            try:
                await asyncio.sleep(delay)
            finally:
                self._sleep_tasks.pop(task, None)


class TimeTravel:
    """Context manager that replaces real time with a MockClock.

    Inside the with block, asyncio.sleep and time.monotonic are swapped
    for mock versions. On exit the originals are restored.

    Attributes:
        clock: The MockClock driving the fake time.

    Example:
        with TimeTravel() as travel:
            travel.advance(30.0)
    """

    def __init__(self) -> None:
        self._clock = MockClock()
        self._original_sleep = asyncio.sleep
        self._original_time = time.monotonic

    def __enter__(self) -> "TimeTravel":
        """Install the fake sleep and monotonic; return self."""

        async def mock_sleep(delay: float) -> None:
            await self._clock.sleep(delay)

        asyncio.sleep = mock_sleep
        time.monotonic = lambda: self._clock.time
        return self

    def __exit__(self, *args: object) -> None:
        """Restore the real asyncio.sleep and time.monotonic."""
        asyncio.sleep = self._original_sleep
        time.monotonic = self._original_time

    @property
    def clock(self) -> MockClock:
        """The underlying mock clock."""
        return self._clock

    def advance(self, delta: float) -> None:
        """Fast-forward fake time by delta seconds."""
        self._clock.advance(delta)

    def set_time(self, value: float) -> None:
        """Jump fake time to an absolute value."""
        self._clock.set_time(value)


@contextmanager
def patch_time(initial: float = 0.0) -> Any:
    """Context manager that patches time and yields the mock clock.

    Wrapper around TimeTravel that starts the clock at initial seconds
    and yields the clock itself.

    Args:
        initial: Starting fake time in seconds.

    Yields:
        The MockClock for calling clock.advance(...).

    Example:
        with patch_time(1000.0) as clock:
            assert time.monotonic() == 1000.0
            clock.advance(10.0)
    """
    travel = TimeTravel()
    travel.clock.set_time(initial)
    travel.__enter__()
    try:
        yield travel.clock
    finally:
        travel.__exit__(None, None, None)


class AsyncTimeMachine:
    """Explicit start/stop time control that also patches time.time.

    More thorough than TimeTravel: uses unittest.mock.patch and
    additionally replaces time.time, asyncio.sleep, and time.monotonic.

    Usable as a context manager (with AsyncTimeMachine():) or via
    explicit start/stop calls (useful in test setUp/tearDown).

    Example:
        machine = AsyncTimeMachine()
        machine.start()
        machine.advance(3600)
        machine.stop()
    """

    def __init__(self) -> None:
        self._clock = MockClock()
        self._patches: list[object] = []

    def start(self) -> None:
        """Install all time-related patches."""
        self._patches.append(patch("asyncio.sleep", self._mock_sleep))
        self._patches.append(patch("time.monotonic", lambda: self._clock.time))
        self._patches.append(patch("time.time", lambda: self._clock.time))
        for p in self._patches:
            p.start()

    def stop(self) -> None:
        """Remove all patches, restoring the real functions."""
        for p in self._patches:
            p.stop()
        self._patches.clear()

    async def _mock_sleep(self, delay: float) -> None:
        await self._clock.sleep(delay)

    def advance(self, delta: float) -> None:
        """Fast-forward fake time by delta seconds, waking due sleepers."""
        self._clock.advance(delta)

    def travel_to(self, timestamp: float) -> None:
        """Jump fake time to an absolute timestamp (seconds)."""
        self._clock.set_time(timestamp)

    def __enter__(self) -> "AsyncTimeMachine":
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()


async def demonstrate_mock_time() -> None:
    """Narrated tour of time control: TimeTravel, patch_time, AsyncTimeMachine.

    Run python mock_time.py. Notice how advance lets sleeps complete
    nearly instantly.
    """
    print("=== Mock Time Demo ===")

    print("\n--- Time Travel ---")
    with TimeTravel() as travel:
        print(f"Initial time: {time.monotonic()}")

        async def test_sleep() -> None:
            await asyncio.sleep(0.5)
            print(f"After 0.5s sleep: {time.monotonic()}")

        task = asyncio.create_task(test_sleep())
        travel.advance(0.5)
        await task

    print("\n--- Patch Time ---")
    with patch_time(1000.0) as clock:
        print(f"Time: {time.monotonic()}")
        clock.advance(10.0)
        print(f"After advance: {time.monotonic()}")

        async def test() -> None:
            await asyncio.sleep(5.0)
            print(f"After sleep: {time.monotonic()}")

        task = asyncio.create_task(test())
        clock.advance(5.0)
        await task

    print("\n--- Async Time Machine ---")
    machine = AsyncTimeMachine()
    machine.start()
    try:
        print(f"Time: {time.monotonic()}")

        async def delayed() -> str:
            await asyncio.sleep(1.0)
            return "done"

        task = asyncio.create_task(delayed())
        machine.advance(1.0)
        result = await task
        print(f"Result: {result}, Time: {time.monotonic()}")
    finally:
        machine.stop()


if __name__ == "__main__":
    asyncio.run(demonstrate_mock_time())
