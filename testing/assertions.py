"""
Async Assertions

Advanced assertions for async testing.
"""

import asyncio
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine
from contextlib import asynccontextmanager
from typing import Any, TypeVar

T = TypeVar("T")


class AsyncAssertions:
    """Collection of async assertion methods."""

    @staticmethod
    async def assert_equal(
        actual: T, expected: T, msg: str | None = None, *, approx: float | None = None
    ) -> None:
        """Assert two values are equal."""
        if approx is not None:
            if abs(actual - expected) > approx:
                raise AssertionError(msg or f"Expected {expected} ± {approx}, got {actual}")
        elif actual != expected:
            raise AssertionError(msg or f"Expected {expected}, got {actual}")

    @staticmethod
    async def assert_not_equal(actual: T, expected: T, msg: str | None = None) -> None:
        """Assert two values are not equal."""
        if actual == expected:
            raise AssertionError(msg or f"Expected not {expected}, got {actual}")

    @staticmethod
    async def assert_true(condition: bool, msg: str | None = None) -> None:
        """Assert condition is true."""
        if not condition:
            raise AssertionError(msg or "Expected True, got False")

    @staticmethod
    async def assert_false(condition: bool, msg: str | None = None) -> None:
        """Assert condition is false."""
        if condition:
            raise AssertionError(msg or "Expected False, got True")

    @staticmethod
    async def assert_is_none(value: Any, msg: str | None = None) -> None:
        """Assert value is None."""
        if value is not None:
            raise AssertionError(msg or f"Expected None, got {value}")

    @staticmethod
    async def assert_is_not_none(value: Any, msg: str | None = None) -> None:
        """Assert value is not None."""
        if value is None:
            raise AssertionError(msg or "Expected not None, got None")

    @staticmethod
    async def assert_in(
        item: Any, container: list | set | dict | str, msg: str | None = None
    ) -> None:
        """Assert item is in container."""
        if item not in container:
            raise AssertionError(msg or f"Expected {item} in {container}")

    @staticmethod
    async def assert_not_in(
        item: Any, container: list | set | dict | str, msg: str | None = None
    ) -> None:
        """Assert item is not in container."""
        if item in container:
            raise AssertionError(msg or f"Expected {item} not in {container}")

    @staticmethod
    async def assert_is_instance(obj: Any, cls: type, msg: str | None = None) -> None:
        """Assert object is instance of class."""
        if not isinstance(obj, cls):
            raise AssertionError(msg or f"Expected {cls.__name__}, got {type(obj).__name__}")

    @staticmethod
    async def assert_not_is_instance(obj: Any, cls: type, msg: str | None = None) -> None:
        """Assert object is not instance of class."""
        if isinstance(obj, cls):
            raise AssertionError(msg or f"Expected not {cls.__name__}, got {cls.__name__}")

    @staticmethod
    async def assert_greater(a: int | float, b: int | float, msg: str | None = None) -> None:
        """Assert a > b."""
        if not a > b:
            raise AssertionError(msg or f"Expected {a} > {b}")

    @staticmethod
    async def assert_less(a: int | float, b: int | float, msg: str | None = None) -> None:
        """Assert a < b."""
        if not a < b:
            raise AssertionError(msg or f"Expected {a} < {b}")

    @staticmethod
    async def assert_greater_equal(a: int | float, b: int | float, msg: str | None = None) -> None:
        """Assert a >= b."""
        if not a >= b:
            raise AssertionError(msg or f"Expected {a} >= {b}")

    @staticmethod
    async def assert_less_equal(a: int | float, b: int | float, msg: str | None = None) -> None:
        """Assert a <= b."""
        if not a <= b:
            raise AssertionError(msg or f"Expected {a} <= {b}")

    @staticmethod
    async def assert_raises(
        exception_type: type, coro: Coroutine[Any, Any, T], msg: str | None = None
    ) -> None:
        """Assert coroutine raises exception."""
        try:
            await coro
        except exception_type:
            return
        except Exception as e:
            raise AssertionError(
                msg or f"Expected {exception_type.__name__}, got {type(e).__name__}: {e}"
            )

        raise AssertionError(msg or f"Expected {exception_type.__name__} but no exception raised")

    @staticmethod
    async def assert_raises_regex(
        exception_type: type, pattern: str, coro: Coroutine[Any, Any, T], msg: str | None = None
    ) -> None:
        """Assert coroutine raises exception matching regex."""
        import re

        try:
            await coro
        except exception_type as e:
            if re.search(pattern, str(e)):
                return
            raise AssertionError(
                msg or f"Exception message '{e}' does not match pattern '{pattern}'"
            )
        except Exception as e:
            raise AssertionError(
                msg or f"Expected {exception_type.__name__}, got {type(e).__name__}: {e}"
            )

        raise AssertionError(msg or f"Expected {exception_type.__name__} but no exception raised")

    @staticmethod
    async def assert_completes_within(
        coro: Coroutine[Any, Any, T], timeout: float, msg: str | None = None
    ) -> T:
        """Assert coroutine completes within timeout."""
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except TimeoutError:
            raise AssertionError(msg or f"Coroutine did not complete within {timeout}s")

    @staticmethod
    async def assert_completes_after(
        coro: Coroutine[Any, Any, T], min_time: float, msg: str | None = None
    ) -> T:
        """Assert coroutine takes at least min_time to complete."""
        start = time.monotonic()
        result = await coro
        elapsed = time.monotonic() - start

        if elapsed < min_time:
            raise AssertionError(
                msg or f"Coroutine completed in {elapsed:.3f}s, expected at least {min_time}s"
            )
        return result

    @staticmethod
    async def assert_eventually(
        condition: Callable[[], bool],
        timeout: float = 5.0,
        interval: float = 0.1,
        msg: str | None = None,
    ) -> None:
        """Assert condition eventually becomes true."""
        start = time.monotonic()

        while time.monotonic() - start < timeout:
            if condition():
                return
            await asyncio.sleep(interval)

        raise AssertionError(msg or f"Condition not met within {timeout}s")

    @staticmethod
    async def assert_eventually_async(
        condition: Callable[[], Awaitable[bool]],
        timeout: float = 5.0,
        interval: float = 0.1,
        msg: str | None = None,
    ) -> None:
        """Assert async condition eventually becomes true."""
        start = time.monotonic()

        while time.monotonic() - start < timeout:
            if await condition():
                return
            await asyncio.sleep(interval)

        raise AssertionError(msg or f"Condition not met within {timeout}s")

    @staticmethod
    async def assert_never(
        condition: Callable[[], bool],
        duration: float,
        interval: float = 0.1,
        msg: str | None = None,
    ) -> None:
        """Assert condition never becomes true within duration."""
        start = time.monotonic()

        while time.monotonic() - start < duration:
            if condition():
                raise AssertionError(msg or "Condition became true unexpectedly")
            await asyncio.sleep(interval)

    @staticmethod
    async def assert_never_async(
        condition: Callable[[], Awaitable[bool]],
        duration: float,
        interval: float = 0.1,
        msg: str | None = None,
    ) -> None:
        """Assert async condition never becomes true within duration."""
        start = time.monotonic()

        while time.monotonic() - start < duration:
            if await condition():
                raise AssertionError(msg or "Condition became true unexpectedly")
            await asyncio.sleep(interval)

    @staticmethod
    async def assert_all_complete(
        *coros: Coroutine[Any, Any, T],
        timeout: float | None = None,
        return_exceptions: bool = False,
    ) -> list[T | BaseException]:
        """Assert all coroutines complete."""
        return await asyncio.gather(*coros, return_exceptions=return_exceptions)

    @staticmethod
    async def assert_any_complete(
        *coros: Coroutine[Any, Any, T], timeout: float | None = None
    ) -> T:
        """Assert at least one coroutine completes."""
        done, pending = await asyncio.wait(
            [asyncio.create_task(c) for c in coros],
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )

        if not done:
            raise AssertionError("No coroutines completed")

        # Cancel pending
        for task in pending:
            task.cancel()

        return done.pop().result()

    @staticmethod
    async def assert_exact_count(
        items: list[T], expected_count: int, msg: str | None = None
    ) -> None:
        """Assert exact count of items."""
        actual = len(items)
        if actual != expected_count:
            raise AssertionError(msg or f"Expected {expected_count} items, got {actual}")

    @staticmethod
    async def assert_count_between(
        items: list[T], min_count: int, max_count: int, msg: str | None = None
    ) -> None:
        """Assert count is between min and max."""
        actual = len(items)
        if not (min_count <= actual <= max_count):
            raise AssertionError(
                msg or f"Expected count between {min_count} and {max_count}, got {actual}"
            )

    @staticmethod
    async def assert_contains_all(
        container: list[T], expected_items: list[T], msg: str | None = None
    ) -> None:
        """Assert container contains all expected items."""
        missing = [item for item in expected_items if item not in container]
        if missing:
            raise AssertionError(msg or f"Missing items: {missing}")

    @staticmethod
    async def assert_contains_any(
        container: list[T], expected_items: list[T], msg: str | None = None
    ) -> None:
        """Assert container contains at least one expected item."""
        if not any(item in container for item in expected_items):
            raise AssertionError(msg or f"Expected at least one of {expected_items} in {container}")

    @staticmethod
    async def assert_sorted(
        items: list[T], key: Callable | None = None, reverse: bool = False, msg: str | None = None
    ) -> None:
        """Assert list is sorted."""
        sorted_items = sorted(items, key=key, reverse=reverse)
        if items != sorted_items:
            raise AssertionError(msg or f"List not sorted: {items}")

    @staticmethod
    async def assert_unique(items: list[T], msg: str | None = None) -> None:
        """Assert all items are unique."""
        if len(items) != len(set(items)):
            raise AssertionError(msg or f"List contains duplicates: {items}")

    @staticmethod
    async def assert_called(mock: Any, times: int | None = None, msg: str | None = None) -> None:
        """Assert mock was called."""
        if not hasattr(mock, "call_count"):
            raise AssertionError(msg or "Object is not a mock")

        if times is not None:
            if mock.call_count != times:
                raise AssertionError(msg or f"Expected {times} calls, got {mock.call_count}")
        elif mock.call_count == 0:
            raise AssertionError(msg or "Mock was not called")

    @staticmethod
    async def assert_not_called(mock: Any, msg: str | None = None) -> None:
        """Assert mock was not called."""
        if not hasattr(mock, "call_count"):
            raise AssertionError(msg or "Object is not a mock")

        if mock.call_count > 0:
            raise AssertionError(msg or f"Mock was called {mock.call_count} times")

    @staticmethod
    async def assert_called_with(mock: Any, *args, **kwargs) -> None:
        """Assert mock was called with specific arguments."""
        if not hasattr(mock, "assert_called_with"):
            raise AssertionError("Object is not a mock")

        mock.assert_called_with(*args, **kwargs)

    @staticmethod
    async def assert_awaited(mock: Any, times: int | None = None, msg: str | None = None) -> None:
        """Assert async mock was awaited."""
        if not hasattr(mock, "awaited"):
            raise AssertionError(msg or "Object is not an async mock")

        if not mock.awaited:
            raise AssertionError(msg or "Mock was not awaited")

        if times is not None and hasattr(mock, "await_count") and mock.await_count != times:
            raise AssertionError(msg or f"Expected {times} awaits, got {mock.await_count}")


# Standalone assertion functions
async def assert_async_equal(
    actual: T, expected: T, msg: str | None = None, *, approx: float | None = None
) -> None:
    """Assert two values are equal."""
    await AsyncAssertions.assert_equal(actual, expected, msg, approx=approx)


async def assert_async_not_equal(actual: T, expected: T, msg: str | None = None) -> None:
    """Assert two values are not equal."""
    await AsyncAssertions.assert_not_equal(actual, expected, msg)


async def assert_async_true(condition: bool, msg: str | None = None) -> None:
    """Assert condition is true."""
    await AsyncAssertions.assert_true(condition, msg)


async def assert_async_false(condition: bool, msg: str | None = None) -> None:
    """Assert condition is false."""
    await AsyncAssertions.assert_false(condition, msg)


async def assert_async_is_none(value: Any, msg: str | None = None) -> None:
    """Assert value is None."""
    await AsyncAssertions.assert_is_none(value, msg)


async def assert_async_is_not_none(value: Any, msg: str | None = None) -> None:
    """Assert value is not None."""
    await AsyncAssertions.assert_is_not_none(value, msg)


async def assert_async_in(
    item: Any, container: list | set | dict | str, msg: str | None = None
) -> None:
    """Assert item is in container."""
    await AsyncAssertions.assert_in(item, container, msg)


async def assert_async_not_in(
    item: Any, container: list | set | dict | str, msg: str | None = None
) -> None:
    """Assert item is not in container."""
    await AsyncAssertions.assert_not_in(item, container, msg)


async def assert_async_is_instance(obj: Any, cls: type, msg: str | None = None) -> None:
    """Assert object is instance of class."""
    await AsyncAssertions.assert_is_instance(obj, cls, msg)


async def assert_async_raises(
    exception_type: type, coro: Coroutine[Any, Any, T], msg: str | None = None
) -> None:
    """Assert coroutine raises exception."""
    await AsyncAssertions.assert_raises(exception_type, coro, msg)


async def assert_async_raises_regex(
    exception_type: type, pattern: str, coro: Coroutine[Any, Any, T], msg: str | None = None
) -> None:
    """Assert coroutine raises exception matching regex."""
    await AsyncAssertions.assert_raises_regex(exception_type, pattern, coro, msg)


async def assert_async_completes_within(
    coro: Coroutine[Any, Any, T], timeout: float, msg: str | None = None
) -> T:
    """Assert coroutine completes within timeout."""
    return await AsyncAssertions.assert_completes_within(coro, timeout, msg)


async def assert_async_eventually(
    condition: Callable[[], bool],
    timeout: float = 5.0,
    interval: float = 0.1,
    msg: str | None = None,
) -> None:
    """Assert condition eventually becomes true."""
    await AsyncAssertions.assert_eventually(condition, timeout, interval, msg)


async def assert_async_eventually_async(
    condition: Callable[[], Awaitable[bool]],
    timeout: float = 5.0,
    interval: float = 0.1,
    msg: str | None = None,
) -> None:
    """Assert async condition eventually becomes true."""
    await AsyncAssertions.assert_eventually_async(condition, timeout, interval, msg)


async def assert_async_never(
    condition: Callable[[], bool], duration: float, interval: float = 0.1, msg: str | None = None
) -> None:
    """Assert condition never becomes true."""
    await AsyncAssertions.assert_never(condition, duration, interval, msg)


async def assert_async_awaited(mock: Any, times: int | None = None, msg: str | None = None) -> None:
    """Assert async mock was awaited."""
    await AsyncAssertions.assert_awaited(mock, times, msg)


# Context managers for assertions
@asynccontextmanager
async def assert_timeout(timeout: float, msg: str | None = None) -> AsyncGenerator[None, None]:
    """Context manager that asserts timeout."""
    try:
        yield
    except TimeoutError:
        pass
    except Exception:
        raise AssertionError(msg or "Expected timeout but got different exception")
    else:
        raise AssertionError(msg or "Expected timeout but operation completed")


@asynccontextmanager
async def assert_completes_in(
    min_time: float, max_time: float, msg: str | None = None
) -> AsyncGenerator[None, None]:
    """Context manager that asserts completion time bounds."""
    start = time.monotonic()
    try:
        yield
    finally:
        elapsed = time.monotonic() - start
        if not (min_time <= elapsed <= max_time):
            raise AssertionError(
                msg
                or f"Operation took {elapsed:.3f}s, expected between {min_time}s and {max_time}s"
            )


@asynccontextmanager
async def assert_no_exception(msg: str | None = None) -> AsyncGenerator[None, None]:
    """Context manager that asserts no exception is raised."""
    try:
        yield
    except Exception as e:
        raise AssertionError(msg or f"Unexpected exception: {e}")


@asynccontextmanager
async def assert_exception_type(
    exception_type: type, msg: str | None = None
) -> AsyncGenerator[None, None]:
    """Context manager that asserts specific exception type."""
    try:
        yield
    except exception_type:
        pass
    except Exception as e:
        raise AssertionError(
            msg or f"Expected {exception_type.__name__}, got {type(e).__name__}: {e}"
        )
    else:
        raise AssertionError(msg or f"Expected {exception_type.__name__} but no exception raised")


@asynccontextmanager
async def assert_changes(
    value_getter: Callable[[], T], expected_change: T | None = None, msg: str | None = None
) -> AsyncGenerator[None, None]:
    """Context manager that asserts value changes."""
    before = value_getter()
    try:
        yield
    finally:
        after = value_getter()
        if before == after:
            raise AssertionError(msg or f"Value did not change: {before}")

        if expected_change is not None:
            actual_change = after - before if hasattr(after, "__sub__") else after
            if actual_change != expected_change:
                raise AssertionError(
                    msg or f"Expected change {expected_change}, got {actual_change}"
                )


# Matchers for complex assertions
class AsyncMatcher:
    """Base class for async matchers."""

    def __init__(self, description: str = "") -> None:
        self._description = description

    async def match(self, actual: Any) -> bool:
        raise NotImplementedError

    def describe(self) -> str:
        return self._description or self.__class__.__name__

    def describe_mismatch(self, actual: Any) -> str:
        return f"Expected {self.describe()}, got {actual}"


class Equals(AsyncMatcher):
    """Match equal values."""

    def __init__(self, expected: Any) -> None:
        super().__init__(f"equals {expected}")
        self._expected = expected

    async def match(self, actual: Any) -> bool:
        return actual == self._expected


class Contains(AsyncMatcher):
    """Match container containing item."""

    def __init__(self, item: Any) -> None:
        super().__init__(f"contains {item}")
        self._item = item

    async def match(self, actual: Any) -> bool:
        return self._item in actual


class MatchesRegex(AsyncMatcher):
    """Match string against regex."""

    def __init__(self, pattern: str) -> None:
        import re

        super().__init__(f"matches regex '{pattern}'")
        self._pattern = re.compile(pattern)

    async def match(self, actual: str) -> bool:
        return bool(self._pattern.search(actual))


class IsInstance(AsyncMatcher):
    """Match instance of type."""

    def __init__(self, cls: type) -> None:
        super().__init__(f"is instance of {cls.__name__}")
        self._cls = cls

    async def match(self, actual: Any) -> bool:
        return isinstance(actual, self._cls)


class HasLength(AsyncMatcher):
    """Match container length."""

    def __init__(self, length: int) -> None:
        super().__init__(f"has length {length}")
        self._length = length

    async def match(self, actual: Any) -> bool:
        return len(actual) == self._length


class AllMatch(AsyncMatcher):
    """Match all items against matcher."""

    def __init__(self, matcher: AsyncMatcher) -> None:
        super().__init__(f"all match {matcher.describe()}")
        self._matcher = matcher

    async def match(self, actual: list[Any]) -> bool:
        for item in actual:
            if not await self._matcher.match(item):
                return False
        return True


class AnyMatch(AsyncMatcher):
    """Match any item against matcher."""

    def __init__(self, matcher: AsyncMatcher) -> None:
        super().__init__(f"any match {matcher.describe()}")
        self._matcher = matcher

    async def match(self, actual: list[Any]) -> bool:
        for item in actual:
            if await self._matcher.match(item):
                return True
        return False


def assert_that(actual: Any, matcher: AsyncMatcher, msg: str | None = None) -> None:
    """Assert that actual matches matcher."""
    import asyncio

    async def check() -> None:
        if not await matcher.match(actual):
            raise AssertionError(msg or f"{matcher.describe_mismatch(actual)}")

    asyncio.run(check())


# Pytest integration
try:
    import pytest

    @pytest.fixture
    def async_assert() -> AsyncAssertions:
        """Async assertions fixture."""
        return AsyncAssertions()

except ImportError:
    pass


# Assertion helpers for common patterns
class AssertHelpers:
    """Helper methods for common assertion patterns."""

    @staticmethod
    async def assert_task_completed(task: asyncio.Task, timeout: float | None = None) -> Any:
        """Assert task completes and return result."""
        if timeout:
            return await asyncio.wait_for(task, timeout=timeout)
        return await task

    @staticmethod
    async def assert_task_cancelled(task: asyncio.Task, timeout: float | None = None) -> None:
        """Assert task is cancelled."""
        if not task.done():
            task.cancel()

        try:
            if timeout:
                await asyncio.wait_for(task, timeout=timeout)
            else:
                await task
        except asyncio.CancelledError:
            return

        raise AssertionError("Task was not cancelled")

    @staticmethod
    async def assert_all_tasks_done(
        tasks: list[asyncio.Task], timeout: float | None = None
    ) -> list[Any]:
        """Assert all tasks complete and return results."""
        if timeout:
            done, pending = await asyncio.wait(
                tasks, timeout=timeout, return_when=asyncio.ALL_COMPLETED
            )
            if pending:
                raise AssertionError(f"{len(pending)} tasks did not complete")
        else:
            done = await asyncio.gather(*tasks, return_exceptions=True)
            return list(done)

    @staticmethod
    async def assert_queue_empty(queue: asyncio.Queue) -> None:
        """Assert queue is empty."""
        if not queue.empty():
            raise AssertionError(f"Queue not empty, size: {queue.qsize()}")

    @staticmethod
    async def assert_queue_size(queue: asyncio.Queue, expected: int) -> None:
        """Assert queue has exact size."""
        actual = queue.qsize()
        if actual != expected:
            raise AssertionError(f"Expected queue size {expected}, got {actual}")
