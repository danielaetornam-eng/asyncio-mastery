"""
Async Test Framework

Testing utilities for async code with pytest and unittest compatibility.
"""

import asyncio
import functools
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine
from contextlib import asynccontextmanager
from typing import Any, TypeVar
from unittest import TestCase

T = TypeVar("T")


def async_test(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., T]:
    """Decorator to run async test function."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> T:
        # Get or create event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(func(*args, **kwargs))

    return wrapper


class AsyncTestCase(TestCase):
    """Base test case for async tests."""

    def setUp(self) -> None:
        super().setUp()
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

    def tearDown(self) -> None:
        if self._loop and not self._loop.is_closed():
            self._loop.close()
        super().tearDown()

    def run_async(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run coroutine in test event loop."""
        return self._loop.run_until_complete(coro)

    def create_task(self, coro: Coroutine[Any, Any, T]) -> asyncio.Task[T]:
        """Create task in test event loop."""
        return self._loop.create_task(coro)

    async def assert_eventually(
        self,
        condition: Callable[[], bool],
        timeout: float = 5.0,
        interval: float = 0.1,
        msg: str | None = None,
    ) -> None:
        """Assert that condition eventually becomes true."""
        start = time.monotonic()

        while time.monotonic() - start < timeout:
            if condition():
                return
            await asyncio.sleep(interval)

        self.fail(msg or f"Condition not met within {timeout}s")

    async def assert_completes_within(
        self, coro: Coroutine[Any, Any, T], timeout: float, msg: str | None = None
    ) -> T:
        """Assert coroutine completes within timeout."""
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except TimeoutError:
            self.fail(msg or f"Coroutine did not complete within {timeout}s")

    async def assert_raises_async(
        self, exception_type: type, coro: Coroutine[Any, Any, T], msg: str | None = None
    ) -> None:
        """Assert coroutine raises exception."""
        try:
            await coro
        except exception_type:
            return
        except Exception as e:
            self.fail(msg or f"Wrong exception type: {type(e).__name__}")

        self.fail(msg or f"Expected {exception_type.__name__} but no exception raised")

    @asynccontextmanager
    async def assert_timeout(
        self, timeout: float, msg: str | None = None
    ) -> AsyncGenerator[None, None]:
        """Context manager that asserts timeout."""
        try:
            yield
        except TimeoutError:
            pass
        except Exception:
            self.fail(msg or "Expected timeout but got different exception")
        else:
            self.fail(msg or "Expected timeout but operation completed")


def assert_eventually(
    condition: Callable[[], bool],
    timeout: float = 5.0,
    interval: float = 0.1,
    msg: str | None = None,
) -> None:
    """Assert that condition eventually becomes true."""
    start = time.monotonic()

    while time.monotonic() - start < timeout:
        if condition():
            return
        time.sleep(interval)

    raise AssertionError(msg or f"Condition not met within {timeout}s")


async def assert_eventually_async(
    condition: Callable[[], Awaitable[bool]],
    timeout: float = 5.0,
    interval: float = 0.1,
    msg: str | None = None,
) -> None:
    """Assert that async condition eventually becomes true."""
    start = time.monotonic()

    while time.monotonic() - start < timeout:
        if await condition():
            return
        await asyncio.sleep(interval)

    raise AssertionError(msg or f"Condition not met within {timeout}s")


def assert_completes_within(
    coro: Coroutine[Any, Any, T], timeout: float, msg: str | None = None
) -> T:
    """Assert coroutine completes within timeout."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        return loop.run_until_complete(asyncio.wait_for(coro, timeout=timeout))
    except TimeoutError:
        raise AssertionError(msg or f"Coroutine did not complete within {timeout}s")


async def assert_completes_within_async(
    coro: Coroutine[Any, Any, T], timeout: float, msg: str | None = None
) -> T:
    """Assert coroutine completes within timeout (async version)."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except TimeoutError:
        raise AssertionError(msg or f"Coroutine did not complete within {timeout}s")


def assert_raises_async(
    exception_type: type, coro: Coroutine[Any, Any, T], msg: str | None = None
) -> None:
    """Assert coroutine raises exception."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(coro)
    except exception_type:
        return
    except Exception as e:
        raise AssertionError(msg or f"Wrong exception type: {type(e).__name__}")

    raise AssertionError(msg or f"Expected {exception_type.__name__} but no exception raised")


async def assert_raises_async_async(
    exception_type: type, coro: Coroutine[Any, Any, T], msg: str | None = None
) -> None:
    """Assert coroutine raises exception (async version)."""
    try:
        await coro
    except exception_type:
        return
    except Exception as e:
        raise AssertionError(msg or f"Wrong exception type: {type(e).__name__}")

    raise AssertionError(msg or f"Expected {exception_type.__name__} but no exception raised")


class AsyncAssertionContext:
    """Context manager for async assertions."""

    def __init__(self, timeout: float | None = None) -> None:
        self._timeout = timeout
        self._start = time.monotonic()

    def __enter__(self) -> "AsyncAssertionContext":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is asyncio.TimeoutError:
            return True  # Suppress timeout
        return False


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


class AsyncMockHelper:
    """Helper for creating async mocks."""

    @staticmethod
    def create_mock(return_value: Any = None, side_effect: Any = None) -> "AsyncMock":
        return AsyncMock(return_value=return_value, side_effect=side_effect)

    @staticmethod
    def create_magic_mock(**kwargs) -> "AsyncMagicMock":
        return AsyncMagicMock(**kwargs)


class AsyncMock:
    """Async mock object."""

    def __init__(
        self, return_value: Any = None, side_effect: Any = None, name: str | None = None
    ) -> None:
        self._return_value = return_value
        self._side_effect = side_effect
        self._name = name
        self._call_count = 0
        self._calls: list[tuple] = []
        self._async_called = False

    def __call__(self, *args, **kwargs) -> "AsyncMock":
        self._call_count += 1
        self._calls.append((args, kwargs))

        if self._side_effect:
            if callable(self._side_effect):
                result = self._side_effect(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    return result
                return result
            elif isinstance(self._side_effect, Exception):
                raise self._side_effect
            elif isinstance(self._side_effect, list):
                if self._call_count <= len(self._side_effect):
                    result = self._side_effect[self._call_count - 1]
                    if asyncio.iscoroutine(result):
                        return result
                    return result

        return self._return_value

    async def __call_async__(self, *args, **kwargs) -> Any:
        return self(*args, **kwargs)

    def __await__(self):
        async def _awaiter():
            return self._return_value

        return _awaiter().__await__()

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def calls(self) -> list[tuple]:
        return self._calls

    def assert_called(self) -> None:
        assert self._call_count > 0, "Mock not called"

    def assert_called_once(self) -> None:
        assert self._call_count == 1, f"Mock called {self._call_count} times, expected 1"

    def assert_called_with(self, *args, **kwargs) -> None:
        assert self._calls, "Mock not called"
        last_call = self._calls[-1]
        assert last_call[0] == args, f"Args mismatch: {last_call[0]} != {args}"
        assert last_call[1] == kwargs, f"Kwargs mismatch: {last_call[1]} != {kwargs}"

    def assert_awaited(self) -> None:
        assert self._async_called, "Mock not awaited"

    def reset_mock(self) -> None:
        self._call_count = 0
        self._calls.clear()
        self._async_called = False


class AsyncMagicMock(AsyncMock):
    """Async magic mock with attribute access."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._mocks: dict[str, AsyncMock] = {}

    def __getattr__(self, name: str) -> "AsyncMagicMock":
        if name not in self._mocks:
            self._mocks[name] = AsyncMagicMock(name=f"{self._name}.{name}" if self._name else name)
        return self._mocks[name]

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_") or name in (
            "call_count",
            "calls",
            "_mocks",
            "_return_value",
            "_side_effect",
            "_name",
            "_async_called",
        ):
            super().__setattr__(name, value)
        else:
            self._mocks[name] = (
                value if isinstance(value, AsyncMock) else AsyncMagicMock(return_value=value)
            )


class AsyncContextManagerMock:
    """Mock for async context managers."""

    def __init__(self, enter_result: Any = None, exit_result: bool = False) -> None:
        self._enter_result = enter_result
        self._exit_result = exit_result
        self._enter_called = False
        self._exit_called = False

    async def __aenter__(self) -> Any:
        self._enter_called = True
        if asyncio.iscoroutine(self._enter_result):
            return await self._enter_result
        return self._enter_result

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        self._exit_called = True
        return self._exit_result

    def assert_entered(self) -> None:
        assert self._enter_called, "Context manager not entered"

    def assert_exited(self) -> None:
        assert self._exit_called, "Context manager not exited"


def patch_async(target: str, **kwargs) -> Callable:
    """Patch async function or method."""
    from unittest.mock import patch

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with patch(target, **kwargs) as mock:
                # Ensure mock supports async
                if asyncio.iscoroutinefunction(kwargs.get("new", func)):
                    mock.return_value = AsyncMock(return_value=kwargs.get("return_value"))
                return func(*args, **kwargs, **{target.rsplit(".", maxsplit=1)[-1]: mock})

        return wrapper

    return decorator


# Pytest fixtures
try:
    import pytest

    @pytest.fixture
    def event_loop():
        """Create event loop for tests."""
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()

    @pytest.fixture
    async def async_event_loop():
        """Async event loop fixture."""
        loop = asyncio.get_running_loop()
        return loop

    @pytest.fixture
    def async_mock():
        """Async mock fixture."""
        return AsyncMock()

    @pytest.fixture
    def async_magic_mock():
        """Async magic mock fixture."""
        return AsyncMagicMock()

except ImportError:
    pass


def parametrize_async(*args, **kwargs):
    """Parametrize async tests."""
    try:
        import pytest

        return pytest.mark.parametrize(*args, **kwargs)
    except ImportError:

        def decorator(func):
            return func

        return decorator


# Async test runner
class AsyncTestRunner:
    """Run async tests with custom configuration."""

    def __init__(self, *, timeout: float = 30.0, verbose: bool = True) -> None:
        self._timeout = timeout
        self._verbose = verbose
        self._results: list[dict[str, Any]] = []

    def run(self, test_func: Callable, *args, **kwargs) -> dict[str, Any]:
        """Run a single async test."""
        start = time.monotonic()

        try:
            result = async_test(test_func)(*args, **kwargs)
            elapsed = time.monotonic() - start

            return {
                "name": getattr(test_func, "__name__", str(test_func)),
                "status": "passed",
                "elapsed": elapsed,
                "result": result,
            }
        except AssertionError as e:
            elapsed = time.monotonic() - start
            return {
                "name": getattr(test_func, "__name__", str(test_func)),
                "status": "failed",
                "elapsed": elapsed,
                "error": str(e),
            }
        except Exception as e:
            elapsed = time.monotonic() - start
            return {
                "name": getattr(test_func, "__name__", str(test_func)),
                "status": "error",
                "elapsed": elapsed,
                "error": str(e),
            }

    def run_suite(self, tests: list[Callable]) -> dict[str, Any]:
        """Run a suite of tests."""
        results = []
        passed = 0
        failed = 0
        errors = 0

        for test in tests:
            result = self.run(test)
            results.append(result)

            if result["status"] == "passed":
                passed += 1
            elif result["status"] == "failed":
                failed += 1
            else:
                errors += 1

        return {
            "total": len(tests),
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "results": results,
        }
