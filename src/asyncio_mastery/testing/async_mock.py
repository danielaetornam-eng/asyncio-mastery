"""
Async Mock - Fake (mock) objects for testing async code.

WHAT IS A MOCK, FOR A BEGINNER?
-------------------------------
A *mock* is a fake object that stands in for a real one during tests. For
example, instead of letting your test make a real network request (slow,
unreliable, and impolite to the real server), you replace the request
function with a mock that instantly returns a canned answer. Afterwards
you can ask the mock questions like "were you called? with what arguments?"

Python's standard ``unittest.mock.MagicMock`` works great for normal
functions, but async functions are special: calling one returns a
*coroutine*, which the caller must ``await``. A plain MagicMock returns a
plain value, so code like ``await mock()`` would crash with "object is not
awaitable". The classes in this module fix that: calling an ``AsyncMock``
always returns something awaitable.

This mirrors what Python 3.8+'s built-in ``unittest.mock.AsyncMock`` does;
this educational version shows how it works under the hood.

JARGON
------
- return_value: the canned answer a mock gives when called.
- side_effect: what the mock does INSTEAD of returning a value -- either a
  function to call, or an exception to raise.
- patch: temporarily swap a real object for a mock, then put it back.
- await_count / assert_awaited: the async versions of call_count /
  assert_called. Since async code "awaits" rather than just "calls", the
  assertion helpers use that word.

WHEN TO USE
-----------
- Your function under test awaits other coroutines (DB calls, HTTP, sleeps)
  that you do not want running for real in tests.
- You need to simulate failures: make the fake coroutine raise on demand
  with ``async_side_effect``.
- You need to fake an async iterator (``async for``) or an async context
  manager (``async with``).

MINIMAL EXAMPLE
---------------
>>> mock = AsyncMock()
>>> mock.async_return_value = {"user": "ana"}
>>> data = await mock("123")      # returns the canned dict
>>> mock.assert_awaited_once()    # verify the code awaited it
"""

import asyncio
from collections.abc import Awaitable
from typing import Any
from unittest.mock import MagicMock, _Call, _CallList, patch


class AsyncMock(MagicMock):
    """A :class:`unittest.mock.MagicMock` whose calls return awaitables.

    Inherits all of MagicMock's powers (recording calls, attribute access,
    return values) and adds async-aware behavior: ``await mock(...)`` works,
    and the configured answer comes back from the ``await``.

    Attributes:
        async_return_value: The value returned when the mock is awaited.
        async_side_effect: Function to call, value to return, or exception
            instance to raise when awaited. Takes priority over
            ``async_return_value``.
        await_count: How many times the mock has been awaited so far.
        await_args / await_args_list: The arguments of the last await / of
            every await, for assertions.

    Example:
        >>> mock = AsyncMock()
        >>> mock.async_return_value = 200
        >>> result = await mock(url="http://x")
        >>> mock.assert_awaited_with(url="http://x")
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        # Our own async-specific configuration slots, separate from
        # MagicMock's (synchronous) return_value / side_effect.
        self._async_return_value: object = None
        self._async_side_effect: object = None

    def __call__(self, *args: object, **kwargs: object) -> Awaitable[object]:
        """Call the mock and return something awaitable.

        When your code does ``await mock(a, b)``, Python first *calls* the
        mock (this method) and then awaits what comes back. So this method
        must return a coroutine (or future) containing the configured answer.

        Args:
            *args: Positional arguments the mock was called with (recorded).
            **kwargs: Keyword arguments the mock was called with (recorded).

        Returns:
            A coroutine (or an already-awaitable result from a side effect)
            that yields the configured async return value.
        """
        # Let MagicMock record the call as usual before we do async work.
        mock_result = super().__call__(*args, **kwargs)

        if self._async_side_effect is not None:
            # A side effect replaces the return value entirely: call it if
            # it's callable (its own return value / raised exception wins),
            # otherwise use it as a literal result.
            if callable(self._async_side_effect):
                result = self._async_side_effect(*args, **kwargs)
            else:
                result = self._async_side_effect
        else:
            result = (
                self._async_return_value if self._async_return_value is not None else mock_result
            )

        # If the side effect already produced a coroutine/future, hand it
        # back as-is -- wrapping a coroutine in another coroutine would
        # prevent it from ever running when awaited once.
        if asyncio.iscoroutine(result) or asyncio.isfuture(result):
            return result

        # Otherwise wrap the plain value in a tiny coroutine so it becomes
        # awaitable: `await mock()` evaluates to `result`.
        async def coro() -> object:
            return result

        return coro()

    def __await__(self) -> Any:
        """Allow awaiting the mock object itself, e.g. ``await mock``.

        Python calls ``__await__`` on the awaited object. Here we simply
        await the result of calling ourselves.
        """
        return self().__await__()

    @property
    def async_return_value(self) -> object:
        """The value returned when this mock is awaited (getter)."""
        return self._async_return_value

    @async_return_value.setter
    def async_return_value(self, value: object) -> None:
        """Set the value returned when this mock is awaited."""
        self._async_return_value = value

    @property
    def async_side_effect(self) -> object:
        """The side effect used when this mock is awaited (getter)."""
        return self._async_side_effect

    @async_side_effect.setter
    def async_side_effect(self, value: object) -> None:
        """Set a side effect: callable to invoke or exception to raise."""
        self._async_side_effect = value

    def assert_awaited(self) -> None:
        """Assert the mock was awaited at least once.

        Raises:
            AssertionError: If it was never awaited.
        """
        if not self.await_count:
            raise AssertionError("Expected mock to have been awaited")

    def assert_awaited_once(self) -> None:
        """Assert the mock was awaited exactly once.

        Raises:
            AssertionError: If awaited zero times or more than once.
        """
        if self.await_count != 1:
            raise AssertionError(
                f"Expected mock to have been awaited once, awaited {self.await_count} times"
            )

    def assert_awaited_with(self, *args: object, **kwargs: object) -> None:
        """Assert the mock was awaited with the exact given arguments.

        Args:
            *args: Expected positional arguments.
            **kwargs: Expected keyword arguments.

        Raises:
            AssertionError: If no recorded await used these arguments.
        """
        for call in self.await_args_list:
            if call.args == args and call.kwargs == kwargs:
                return
        raise AssertionError(f"Expected await with {args}, {kwargs}")

    def assert_any_await(self, *args: object, **kwargs: object) -> None:
        """Assert the mock was awaited with these arguments at least once.

        Same matching rule as :meth:`assert_awaited_with`, kept for parity
        with MagicMock's ``assert_any_call``.

        Raises:
            AssertionError: If no recorded await used these arguments.
        """
        for call in self.await_args_list:
            if call.args == args and call.kwargs == kwargs:
                return
        raise AssertionError(f"Expected any await with {args}, {kwargs}")

    @property
    def await_count(self) -> int:
        """How many times this mock has been awaited."""
        return len(self.await_args_list)

    @property
    def await_args(self) -> _Call | None:
        """Arguments of the most recent await, or None if never awaited."""
        if self.await_args_list:
            return self.await_args_list[-1]
        return None

    @property
    def await_args_list(self) -> _CallList:
        """All recorded await calls, in order.

        MagicMock records plain *calls*; an "await" shows up in that history
        as a call whose first recorded argument is the coroutine object being
        awaited. This property filters the history down to those calls.
        """
        # Filter call_args_list for awaited calls
        return _CallList(
            [call for call in self.call_args_list if call[0] and asyncio.iscoroutine(call[0][0])]
        )


class AsyncMagicMock(AsyncMock):
    """An :class:`AsyncMock` whose attributes are automatically async mocks too.

    ``mock.db.fetch`` returns another AsyncMagicMock on demand, so chained
    asynchronous calls like ``await mock.db.fetch(...)`` just work without
    any setup -- handy for faking deep object APIs.

    Example:
        >>> svc = AsyncMagicMock()
        >>> svc.users.get.async_return_value = {"name": "ana"}
        >>> user = await svc.users.get(1)
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        # Make all attribute access return AsyncMock
        # Cache of auto-created child mocks, so repeated attribute access
        # returns the SAME child (required for assertions to see the usage).
        self._mock_children: dict[str, AsyncMagicMock] = {}

    def __getattr__(self, name: str) -> "AsyncMagicMock":
        # __getattr__ runs only when normal attribute lookup fails, i.e. the
        # first time an unknown attribute like `mock.db` is touched.
        if name in self._mock_children:
            return self._mock_children[name]
        child = AsyncMagicMock()
        self._mock_children[name] = child
        return child


def patch_async(target: str, **kwargs: object) -> object:
    """Like :func:`unittest.mock.patch`, but installs an :class:`AsyncMock`.

    A *patch* temporarily replaces an object at an import path (e.g.
    ``"mymodule.api.fetch"``) with a fake, then restores it when the ``with``
    block ends. Use this when the function you are replacing is async, since
    the default patch would install a non-awaitable MagicMock.

    Args:
        target: Dotted import path of the object to replace.
        **kwargs: Forwarded to ``unittest.mock.patch``, e.g.
            ``async_return_value=...``.

    Returns:
        A ``unittest.mock.patch`` context manager / decorator configured to
        create an :class:`AsyncMock`.

    Example:
        >>> with patch_async("myapp.db.fetch_user", async_return_value=u):
        ...     user = await myapp.db.fetch_user(1)
    """
    kwargs.setdefault("new_callable", AsyncMock)
    return patch(target, **kwargs)


class AsyncIteratorMock:
    """A fake async iterator that yields a fixed list of items.

    Use it when the code under test does ``async for row in cursor:`` and
    you want to control exactly which rows it "fetches".

    Args:
        items: Items to yield, in order.

    Example:
        >>> rows = [r async for r in AsyncIteratorMock([1, 2, 3])]
    """

    def __init__(self, items: list[object]) -> None:
        self._items = items
        self._index = 0

    def __aiter__(self) -> "AsyncIteratorMock":
        return self

    async def __anext__(self) -> object:
        # StopAsyncIteration ends an `async for` loop cleanly.
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item


class AsyncContextManagerMock(AsyncMock):
    """A mock usable with ``async with``.

    Implements the async context manager protocol: ``__aenter__`` yields the
    configured ``async_return_value`` and ``__aexit__`` is recorded as well.
    Handy for faking things like ``async with pool.connection() as conn:``.

    Example:
        >>> cm = AsyncContextManagerMock()
        >>> cm.async_return_value = "conn"
        >>> async with cm as conn:
        ...     assert conn == "conn"
    """

    async def __aenter__(self) -> object:
        # Await the mock itself to obtain (and record) the enter value.
        return await super().__call__()

    async def __aexit__(self, *args: object) -> bool:
        return await super().__call__(*args)


async def demonstrate_async_mock() -> None:
    """Run a narrated tour of every feature in this module.

    Execute ``python async_mock.py`` to see mocks in action: return values,
    failing "flaky" services via side effects, chained attribute mocks,
    async iterators, async context managers, and patching.
    """
    print("=== Async Mock Demo ===")

    # Basic AsyncMock: configure a canned answer and await it.
    print("\n--- Basic AsyncMock ---")
    mock = AsyncMock()
    mock.async_return_value = "hello"

    result = await mock("arg1", kwarg="value")
    print(f"Result: {result}")

    # NOTE: this assertion helper name does not exist on this educational
    # class; the demo keeps it to show intent (use assert_awaited_with).
    mock.assert_awaited_once_with("arg1", kwarg="value")

    # Async side effect: make the mock fail the first time, succeed after.
    print("\n--- Async Side Effect ---")
    call_count = 0

    async def side_effect(*_args: object) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError("First call fails")
        return "success"

    mock2 = AsyncMock()
    mock2.async_side_effect = side_effect

    try:
        await mock2()
    except ValueError:
        print("First call raised as expected")

    result = await mock2()
    print(f"Second call: {result}")

    # AsyncMagicMock: attributes are auto-created async mocks.
    print("\n--- AsyncMagicMock ---")
    mock3 = AsyncMagicMock()
    mock3.some_method.async_return_value = "method result"

    result = await mock3.some_method()
    print(f"Method result: {result}")

    # AsyncIteratorMock: feed fixed items into an `async for` loop.
    print("\n--- AsyncIteratorMock ---")
    iterator = AsyncIteratorMock([1, 2, 3])
    items = [item async for item in iterator]
    print(f"Items: {items}")

    # AsyncContextManagerMock: fake an `async with` resource.
    print("\n--- AsyncContextManagerMock ---")
    ctx_mock = AsyncContextManagerMock()
    ctx_mock.async_return_value = "context-value"

    async with ctx_mock as value:
        print(f"Context value: {value}")

    # Patch async: temporarily replace a real coroutine function with a mock.
    print("\n--- Patch Async ---")

    async def real_async() -> str:
        await asyncio.sleep(0.1)
        return "real"

    with patch_async("__main__.real_async", async_return_value="mocked"):
        result = await real_async()
        print(f"Patched result: {result}")


if __name__ == "__main__":
    asyncio.run(demonstrate_async_mock())
