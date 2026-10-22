"""
Async Mocking Utilities

Advanced mocking for async code including context managers, iterators, and streams.
"""

import asyncio
import functools
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import Any, TypeVar
from unittest.mock import patch as unittest_patch

T = TypeVar("T")


class AsyncMock:
    """Advanced async mock with full protocol support."""

    def __init__(
        self,
        *,
        return_value: Any = None,
        side_effect: Any = None,
        name: str | None = None,
        spec: type | None = None,
    ) -> None:
        self._return_value = return_value
        self._side_effect = side_effect
        self._name = name
        self._spec = spec
        self._call_count = 0
        self._calls: list[tuple[tuple, dict]] = []
        self._async_called = False
        self._awaited = False
        self._attributes: dict[str, AsyncMock] = {}
        self._call_args: tuple | None = None
        self._call_args_list: list[tuple] = []

    def __call__(self, *args, **kwargs) -> "AsyncMock":
        self._call_count += 1
        self._calls.append((args, kwargs))
        self._call_args = (args, kwargs)
        self._call_args_list.append((args, kwargs))

        # Handle side effect
        if self._side_effect is not None:
            if callable(self._side_effect):
                result = self._side_effect(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    return result
                if isinstance(result, Exception):
                    raise result
                return result
            elif isinstance(self._side_effect, Exception):
                raise self._side_effect
            elif isinstance(self._side_effect, list):
                if self._call_count <= len(self._side_effect):
                    item = self._side_effect[self._call_count - 1]
                    if isinstance(item, Exception):
                        raise item
                    if asyncio.iscoroutine(item):
                        return item
                    return item

        # Return mock for chaining
        return self

    def __await__(self):
        self._awaited = True

        async def _awaiter():
            return self._return_value

        return _awaiter().__await__()

    async def __call_async__(self, *args, **kwargs) -> Any:
        return self(*args, **kwargs)

    def __getattr__(self, name: str) -> "AsyncMock":
        if name.startswith("_"):
            raise AttributeError(name)

        if name not in self._attributes:
            self._attributes[name] = AsyncMock(name=f"{self._name}.{name}" if self._name else name)

        return self._attributes[name]

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_") or name in (
            "_return_value",
            "_side_effect",
            "_name",
            "_spec",
            "_call_count",
            "_calls",
            "_async_called",
            "_awaited",
            "_attributes",
            "_call_args",
            "_call_args_list",
        ):
            super().__setattr__(name, value)
        else:
            self._attributes[name] = (
                value if isinstance(value, AsyncMock) else AsyncMock(return_value=value)
            )

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def calls(self) -> list[tuple[tuple, dict]]:
        return self._calls

    @property
    def call_args(self) -> tuple | None:
        return self._call_args

    @property
    def call_args_list(self) -> list[tuple]:
        return self._call_args_list

    @property
    def awaited(self) -> bool:
        return self._awaited

    def assert_called(self) -> None:
        assert self._call_count > 0, f"Expected {self._name or 'mock'} to be called"

    def assert_not_called(self) -> None:
        assert self._call_count == 0, f"Expected {self._name or 'mock'} not to be called"

    def assert_called_once(self) -> None:
        assert self._call_count == 1, (
            f"Expected {self._name or 'mock'} to be called once. Called {self._call_count} times."
        )

    def assert_called_with(self, *args, **kwargs) -> None:
        assert self._call_count > 0, f"Expected {self._name or 'mock'} to be called"

        last_call = self._calls[-1]
        assert last_call[0] == args, f"Args mismatch: {last_call[0]} != {args}"
        assert last_call[1] == kwargs, f"Kwargs mismatch: {last_call[1]} != {kwargs}"

    def assert_called_once_with(self, *args, **kwargs) -> None:
        self.assert_called_once()
        self.assert_called_with(*args, **kwargs)

    def assert_any_call(self, *args, **kwargs) -> None:
        expected = (args, kwargs)
        assert expected in self._calls, f"Call {expected} not found in {self._calls}"

    def assert_has_calls(self, calls: list[tuple], any_order: bool = False) -> None:
        if any_order:
            for call in calls:
                assert call in self._calls, f"Call {call} not found"
        else:
            # Check sequential
            for i, call in enumerate(calls):
                if i >= len(self._calls):
                    raise AssertionError(
                        f"Expected call {call} at index {i}, but only {len(self._calls)} calls made"
                    )
                assert self._calls[i] == call, (
                    f"Call at index {i} mismatch: {self._calls[i]} != {call}"
                )

    def assert_awaited(self) -> None:
        assert self._awaited, f"Expected {self._name or 'mock'} to be awaited"

    def assert_awaited_once(self) -> None:
        assert self._awaited, f"Expected {self._name or 'mock'} to be awaited once"

    def assert_awaited_with(self, *args, **kwargs) -> None:
        self.assert_awaited()
        self.assert_called_with(*args, **kwargs)

    def assert_awaited_once_with(self, *args, **kwargs) -> None:
        self.assert_awaited_once()
        self.assert_called_once_with(*args, **kwargs)

    def reset_mock(self) -> None:
        self._call_count = 0
        self._calls.clear()
        self._async_called = False
        self._awaited = False
        self._call_args = None
        self._call_args_list.clear()
        for attr in self._attributes.values():
            if isinstance(attr, AsyncMock):
                attr.reset_mock()

    def configure_mock(self, **kwargs) -> "AsyncMock":
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self._attributes[key] = (
                    value if isinstance(value, AsyncMock) else AsyncMock(return_value=value)
                )
        return self

    def __repr__(self) -> str:
        name = self._name or "AsyncMock"
        return f"<AsyncMock name='{name}' call_count={self._call_count}>"


class AsyncMagicMock(AsyncMock):
    """Async magic mock with automatic attribute creation."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._mock_children: dict[str, AsyncMagicMock] = {}

    def __getattr__(self, name: str) -> "AsyncMagicMock":
        if name.startswith("_"):
            return super().__getattr__(name)

        if name not in self._mock_children:
            self._mock_children[name] = AsyncMagicMock(
                name=f"{self._name}.{name}" if self._name else name
            )

        return self._mock_children[name]

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_") or name in (
            "_return_value",
            "_side_effect",
            "_name",
            "_spec",
            "_call_count",
            "_calls",
            "_async_called",
            "_awaited",
            "_attributes",
            "_call_args",
            "_call_args_list",
            "_mock_children",
        ):
            super().__setattr__(name, value)
        elif isinstance(value, AsyncMock):
            self._mock_children[name] = value
        else:
            self._mock_children[name] = AsyncMagicMock(return_value=value)


class AsyncContextManagerMock(AsyncMock):
    """Mock for async context managers."""

    def __init__(self, enter_result: Any = None, exit_result: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self._enter_result = enter_result
        self._exit_result = exit_result
        self._enter_called = False
        self._exit_called = False
        self._exc_info: tuple | None = None

    async def __aenter__(self) -> Any:
        self._enter_called = True
        if asyncio.iscoroutine(self._enter_result):
            return await self._enter_result
        return self._enter_result

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        self._exit_called = True
        self._exc_info = (exc_type, exc_val, exc_tb)

        if asyncio.iscoroutine(self._exit_result):
            return await self._exit_result
        return self._exit_result

    def assert_entered(self) -> None:
        assert self._enter_called, "Async context manager not entered"

    def assert_exited(self) -> None:
        assert self._exit_called, "Async context manager not exited"

    def assert_exited_with(self, exc_type=None, exc_val=None, exc_tb=None) -> None:
        self.assert_exited()
        if exc_type is not None:
            assert self._exc_info[0] == exc_type
        if exc_val is not None:
            assert self._exc_info[1] == exc_val
        if exc_tb is not None:
            assert self._exc_info[2] == exc_tb


class AsyncIteratorMock(AsyncMock):
    """Mock for async iterators."""

    def __init__(
        self, items: list[Any] | None = None, side_effect: Callable | None = None, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self._items = items or []
        self._side_effect = side_effect
        self._index = 0
        self._exhausted = False

    def __aiter__(self) -> "AsyncIteratorMock":
        self._index = 0
        self._exhausted = False
        return self

    async def __anext__(self) -> Any:
        if self._exhausted:
            raise StopAsyncIteration

        if self._side_effect:
            result = await self._side_effect()
            if result is None:
                self._exhausted = True
                raise StopAsyncIteration
            return result

        if self._index >= len(self._items):
            self._exhausted = True
            raise StopAsyncIteration

        item = self._items[self._index]
        self._index += 1

        if asyncio.iscoroutine(item):
            return await item
        return item

    def add_item(self, item: Any) -> None:
        self._items.append(item)

    def set_items(self, items: list[Any]) -> None:
        self._items = items
        self._index = 0
        self._exhausted = False


class AsyncGeneratorMock(AsyncMock):
    """Mock for async generators."""

    def __init__(
        self, generator_func: Callable | None = None, items: list[Any] | None = None, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self._generator_func = generator_func
        self._items = items or []
        self._generator = None
        self._started = False

    async def __call__(self, *args, **kwargs) -> AsyncGenerator[Any, None]:
        self._started = True

        if self._generator_func:
            gen = self._generator_func(*args, **kwargs)
            async for item in gen:
                yield item
        else:
            for item in self._items:
                if asyncio.iscoroutine(item):
                    yield await item
                else:
                    yield item

    def set_items(self, items: list[Any]) -> None:
        self._items = items

    def add_item(self, item: Any) -> None:
        self._items.append(item)


class AsyncStreamMock(AsyncMock):
    """Mock for async streams (readers/writers)."""

    def __init__(self, data: bytes | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._buffer = data or b""
        self._position = 0
        self._closed = False
        self._write_buffer: list[bytes] = []

    async def read(self, n: int = -1) -> bytes:
        if self._closed:
            return b""

        if n == -1:
            result = self._buffer[self._position :]
            self._position = len(self._buffer)
        else:
            result = self._buffer[self._position : self._position + n]
            self._position += len(result)

        return result

    async def readline(self) -> bytes:
        if self._closed:
            return b""

        newline_pos = self._buffer.find(b"\n", self._position)
        if newline_pos == -1:
            result = self._buffer[self._position :]
            self._position = len(self._buffer)
        else:
            result = self._buffer[self._position : newline_pos + 1]
            self._position = newline_pos + 1

        return result

    async def readexactly(self, n: int) -> bytes:
        if self._closed:
            raise asyncio.IncompleteReadError(b"", n)

        if self._position + n > len(self._buffer):
            raise asyncio.IncompleteReadError(self._buffer[self._position :], n)

        result = self._buffer[self._position : self._position + n]
        self._position += n
        return result

    def write(self, data: bytes) -> None:
        self._write_buffer.append(data)

    async def drain(self) -> None:
        pass

    def close(self) -> None:
        self._closed = True

    async def wait_closed(self) -> None:
        pass

    @property
    def closed(self) -> bool:
        return self._closed

    def get_written(self) -> bytes:
        return b"".join(self._write_buffer)

    def set_data(self, data: bytes) -> None:
        self._buffer = data
        self._position = 0
        self._closed = False


def create_async_mock(return_value: Any = None, side_effect: Any = None, **kwargs) -> AsyncMock:
    """Create async mock."""
    return AsyncMock(return_value=return_value, side_effect=side_effect, **kwargs)


def create_async_magic_mock(**kwargs) -> AsyncMagicMock:
    """Create async magic mock."""
    return AsyncMagicMock(**kwargs)


def create_async_context_manager_mock(
    enter_result: Any = None, exit_result: bool = False, **kwargs
) -> AsyncContextManagerMock:
    """Create async context manager mock."""
    return AsyncContextManagerMock(enter_result=enter_result, exit_result=exit_result, **kwargs)


def create_async_iterator_mock(items: list[Any] | None = None, **kwargs) -> AsyncIteratorMock:
    """Create async iterator mock."""
    return AsyncIteratorMock(items=items, **kwargs)


def create_async_stream_mock(data: bytes | None = None, **kwargs) -> AsyncStreamMock:
    """Create async stream mock."""
    return AsyncStreamMock(data=data, **kwargs)


# Patch decorators
def patch_async(target: str, **kwargs) -> Callable:
    """Patch async function with async mock."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kw):
            with unittest_patch(target, **kwargs) as mock:
                # Ensure mock is async-compatible
                if asyncio.iscoroutinefunction(
                    kwargs.get("new", getattr(unittest_patch, "target", None))
                ):
                    mock.return_value = AsyncMock(return_value=kwargs.get("return_value"))
                return func(*args, mock=mock, **kw)

        return wrapper

    return decorator


def patch_async_context_manager(target: str, **kwargs) -> Callable:
    """Patch async context manager."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kw):
            with unittest_patch(target, **kwargs) as mock:
                mock.__aenter__ = AsyncMock(return_value=kwargs.get("enter_result"))
                mock.__aexit__ = AsyncMock(return_value=kwargs.get("exit_result", False))
                return func(*args, mock=mock, **kw)

        return wrapper

    return decorator


@asynccontextmanager
async def async_patch(target: str, **kwargs):
    """Async context manager for patching."""
    with unittest_patch(target, **kwargs) as mock:
        yield mock


# Utility functions
def mock_async_function(
    func: Callable, return_value: Any = None, side_effect: Any = None
) -> AsyncMock:
    """Replace async function with mock."""
    mock = AsyncMock(return_value=return_value, side_effect=side_effect)
    original = func

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        return await mock(*args, **kwargs)

    wrapper.mock = mock
    wrapper.original = original
    return wrapper


def mock_async_method(obj: object, method_name: str, **kwargs) -> AsyncMock:
    """Mock an async method on an object."""
    original = getattr(obj, method_name)
    mock = AsyncMock(**kwargs)

    @functools.wraps(original)
    async def wrapper(*args, **kw):
        return await mock(*args, **kw)

    setattr(obj, method_name, wrapper)
    wrapper.mock = mock
    wrapper.original = original
    return mock


def restore_mocked_method(obj: object, method_name: str) -> bool:
    """Restore previously mocked method."""
    method = getattr(obj, method_name, None)
    if method and hasattr(method, "original"):
        setattr(obj, method_name, method.original)
        return True
    return False


class MockRegistry:
    """Registry for managing multiple mocks."""

    def __init__(self) -> None:
        self._mocks: dict[str, AsyncMock] = {}
        self._patches: list[Any] = []

    def add(self, name: str, mock: AsyncMock) -> None:
        self._mocks[name] = mock

    def get(self, name: str) -> AsyncMock | None:
        return self._mocks.get(name)

    def remove(self, name: str) -> bool:
        if name in self._mocks:
            del self._mocks[name]
            return True
        return False

    def reset_all(self) -> None:
        for mock in self._mocks.values():
            mock.reset_mock()

    def assert_all_called(self) -> None:
        for name, mock in self._mocks.items():
            mock.assert_called()

    @asynccontextmanager
    async def mock_context(self, name: str, **kwargs) -> AsyncGenerator[AsyncMock, None]:
        mock = AsyncMock(**kwargs)
        self.add(name, mock)
        try:
            yield mock
        finally:
            self.remove(name)


_global_mock_registry = MockRegistry()


def get_mock_registry() -> MockRegistry:
    return _global_mock_registry
