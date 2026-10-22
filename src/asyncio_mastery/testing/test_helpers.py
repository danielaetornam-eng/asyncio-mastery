"""
Test Helpers - Common utilities for testing asynchronous code.

Testing async code differs from testing synchronous code. An async function
returns a coroutine object when called. The coroutine must run on an event
loop. This module provides tools to run coroutines in tests.

The AsyncTestCase class creates a fresh event loop for each test method.
The run_async_test function runs a single coroutine in a new event loop.
The assert_eventually function polls a condition until it becomes true or
a timeout occurs.

Example:
    class MyTests(AsyncTestCase):
        def test_addition(self):
            async def main():
                await asyncio.sleep(0.01)
                return 1 + 1
            self.assertEqual(self.run_async(main()), 2)
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar
from unittest import TestCase

T = TypeVar("T")


class AsyncTestCase(TestCase):
    """Base class for unittest-style tests that run coroutines.

    Creates a fresh event loop before each test method. Closes the loop
    after each test method. Provides run_async method to execute coroutines.

    Attributes:
        loop: The per-test event loop created in setUp. Closed in tearDown.
    """

    def setUp(self) -> None:
        """Create a new event loop for this test.

        Called automatically by unittest before every test_* method.
        Each test gets its own loop to prevent interference.
        """
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self) -> None:
        """Close the event loop after the test.

        Called automatically by unittest after every test_* method.
        Closes the loop and unregisters it.
        """
        self.loop.close()
        asyncio.set_event_loop(None)

    def run_async(self, coro: Awaitable[T]) -> T:
        """Run a coroutine on this test's event loop and return its result.

        Args:
            coro: A coroutine or awaitable to execute.

        Returns:
            The coroutine's return value.

        Example:
            self.run_async(my_async_function())
        """
        return self.loop.run_until_complete(coro)

    async def assert_eventually(
        self,
        condition: Callable[[], bool],
        timeout: float = 5.0,
        interval: float = 0.1,
        message: str = "Condition not met",
    ) -> None:
        """Assert that condition becomes true within timeout seconds.

        Useful for concurrency tests where completion time varies. The
        condition is checked every interval seconds.

        Args:
            condition: A zero-argument callable returning True or False.
            timeout: Maximum seconds to wait before failing.
            interval: Seconds to sleep between checks.
            message: Failure message prefix if condition never holds.

        Raises:
            AssertionError: If condition is false after timeout.

        Example:
            await self.assert_eventually(lambda: task.done(), timeout=1.0)
        """
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if condition():
                return
            await asyncio.sleep(interval)
        self.fail(f"{message} within {timeout}s")

    async def assert_raises_async(
        self, exc_type: type[BaseException], coro: Awaitable[Any], *args: Any, **kwargs: Any
    ) -> BaseException:
        """Assert that awaiting coro raises an exception of type exc_type.

        This is the async equivalent of unittest.TestCase.assertRaises.

        Args:
            exc_type: The expected exception class (e.g., ValueError).
            coro: The coroutine to await (already called, e.g., f(1)).
            *args: Unused extra positional arguments.
            **kwargs: Unused extra keyword arguments.

        Returns:
            The exception instance that was raised.

        Raises:
            AssertionError: If no exception or wrong exception type is raised.

        Example:
            err = await self.assert_raises_async(ValueError, bad_coro())
            self.assertIn("boom", str(err))
        """
        try:
            await coro
        except exc_type as e:
            return e
        except Exception as e:
            self.fail(f"Expected {exc_type.__name__}, got {type(e).__name__}: {e}")
        else:
            self.fail(f"Expected {exc_type.__name__}, but no exception raised")


def run_async_test(coro: Awaitable[T]) -> T:
    """Run a coroutine in a new event loop and return its result.

    Standalone alternative to AsyncTestCase for plain function tests.
    The loop is always closed afterwards.

    Args:
        coro: The coroutine to run.

    Returns:
        The coroutine's return value.

    Example:
        result = run_async_test(fetch_data())
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def assert_eventually(
    condition: Callable[[], bool],
    timeout: float = 5.0,
    interval: float = 0.1,
    message: str = "Condition not met",
) -> None:
    """Assert that condition becomes true within timeout seconds.

    Standalone version of AsyncTestCase.assert_eventually for pytest-style
    or plain async tests.

    Args:
        condition: A zero-argument callable returning True or False.
        timeout: Maximum seconds to wait before failing.
        interval: Seconds to sleep between checks.
        message: Failure message prefix.

    Raises:
        AssertionError: If condition never becomes true in time.

    Example:
        await assert_eventually(lambda: queue.qsize() > 0, timeout=2.0)
    """
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if condition():
            return
        await asyncio.sleep(interval)
    raise AssertionError(f"{message} within {timeout}s")


async def wait_for_condition(
    condition: Callable[[], Awaitable[bool] | bool], timeout: float = 5.0, interval: float = 0.1
) -> bool:
    """Wait until condition is true, without failing if it never happens.

    Unlike assert_eventually, this returns a bool instead of raising.
    The condition may be a normal function or an async function.

    Args:
        condition: Callable returning a bool or a coroutine that returns a bool.
        timeout: Maximum seconds to keep polling.
        interval: Seconds to sleep between polls.

    Returns:
        True if condition became true within timeout; False otherwise.

    Example:
        ok = await wait_for_condition(lambda: server.ready, timeout=3.0)
        if not ok: print("server never became ready")
    """
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        result = condition()
        if asyncio.iscoroutine(result):
            result = await result
        if result:
            return True
        await asyncio.sleep(interval)
    return False


class AsyncIterator:
    """Turn a synchronous list into an async iterator.

    An async iterator works with async for loops. It implements __aiter__
    and __anext__. Useful in tests to fake an async database cursor.

    Args:
        items: The list of items to yield one at a time.

    Example:
        async for row in AsyncIterator([1, 2, 3]):
            print(row)
    """

    def __init__(self, items: list[Any]) -> None:
        self._items = items
        self._index = 0

    def __aiter__(self) -> "AsyncIterator":
        return self

    async def __anext__(self) -> Any:
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item


async def collect_async(aiter: Any) -> list[Any]:
    """Collect every item from an async iterable into a list.

    This is the async version of list() for synchronous iterables.

    Args:
        aiter: Any async iterable (usable with async for).

    Returns:
        A list containing all items in order.

    Example:
        items = await collect_async(AsyncIterator(["a", "b"]))
    """
    return [item async for item in aiter]


def async_test(func: Callable[..., Awaitable[Any]]) -> Callable[..., Any]:
    """Decorator that lets you write an async test as async def.

    Applied with @async_test above a coroutine function, it makes the
    test callable by a synchronous test runner. Each call runs the
    coroutine in a fresh event loop via run_async_test.

    Args:
        func: The async test function to wrap.

    Returns:
        A synchronous wrapper function that runs the coroutine.

    Example:
        @async_test
        async def test_fetches():
            assert await fetch() == 200
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return run_async_test(func(*args, **kwargs))

    return wrapper


class AsyncContextManager:
    """A minimal async context manager for test use.

    An async context manager works with async with. It implements
    __aenter__ (run on entry) and __aexit__ (run on exit). Records
    whether it was entered/exited so tests can verify resource handling.

    Args:
        enter_result: The value yielded by async with ... as value:.

    Attributes:
        _entered: True after the block was entered.
        _exited: True after the block was exited.
        _exit_args: The (exc_type, exc_value, traceback) tuple passed to
            __aexit__. All None if the block finished without error.

    Example:
        cm = AsyncContextManager(enter_result="resource")
        async with cm as value: ...
        assert cm._entered and cm._exited
    """

    def __init__(self, enter_result: Any = None) -> None:
        self._enter_result = enter_result
        self._entered = False
        self._exited = False
        self._exit_args: tuple | None = None

    async def __aenter__(self) -> Any:
        self._entered = True
        return self._enter_result

    async def __aexit__(self, *args: Any) -> bool:
        self._exited = True
        self._exit_args = args
        return False


class ExampleTests(AsyncTestCase):
    """Runnable examples showing patterns from this module.

    Run this file directly (python test_helpers.py) to execute these
    tests via unittest.main().
    """

    def test_basic_async(self) -> None:
        """Show the simplest pattern: define a coroutine, run it, assert."""

        async def test() -> None:
            result = await asyncio.sleep(0.01, result="done")
            assert result == "done"

        self.run_async(test())

    def test_eventually(self) -> None:
        """Show waiting for a background task to flip a flag."""

        async def test() -> None:
            flag = {"value": False}

            async def set_flag() -> None:
                await asyncio.sleep(0.1)
                flag["value"] = True

            asyncio.create_task(set_flag())
            await assert_eventually(lambda: flag["value"], timeout=1.0)

        self.run_async(test())


if __name__ == "__main__":
    import unittest

    unittest.main()
