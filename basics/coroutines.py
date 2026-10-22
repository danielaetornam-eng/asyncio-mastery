"""
Coroutines and Async Primitives

Advanced coroutine patterns, decorators, and utilities.
"""

import asyncio
import functools
import inspect
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Coroutine, Generator
from collections.abc import Awaitable as ABCAwaitable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, Generic, TypeVar, cast, overload

T = TypeVar("T")
R = TypeVar("R")


async def simple_coroutine() -> str:
    """Simplest coroutine. Returns a value.

    This coroutine does not await. It shows the minimal coroutine.
    It is a sync function wrapped in async def.

    Returns:
        A greeting string.
    """
    return "Hello from coroutine"


async def coroutine_with_args(name: str, delay: float = 0.1) -> str:
    """Coroutine that accepts arguments and uses await.

    The coroutine sleeps to simulate work.
    While sleeping, the event loop runs other coroutines.

    Args:
        name: The name to greet.
        delay: Seconds to sleep before greeting. Default is 0.1.

    Returns:
        A greeting string in the form "Hello, {name}!".
    """
    # await gives control to the event loop
    # The event loop runs other coroutines while this one sleeps
    # This is cooperative multitasking
    await asyncio.sleep(delay)
    return f"Hello, {name}!"


async def coroutine_returning_value() -> dict[str, Any]:
    """Coroutine that returns a complex value.

    Coroutines return values of any type. This is the same as a sync function.

    Returns:
        A dictionary with status, data, and a nested dictionary.
    """
    await asyncio.sleep(0.01)
    return {"status": "success", "data": [1, 2, 3], "nested": {"key": "value"}}


async def coroutine_raising_exception() -> None:
    """Coroutine that raises an exception.

    If a coroutine raises, the exception goes to the awaiting caller.
    The coroutine does not finish normally.
    """
    await asyncio.sleep(0.01)
    # Exceptions propagate like in sync code. await does not swallow them.
    raise ValueError("Intentional error for demonstration")


class CoroutineExample:
    """Class that shows coroutine patterns.

    A class-based wrapper for related coroutines.
    The instance can be awaited (see __await__).
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.call_count = 0

    async def method_coroutine(self, value: int) -> int:
        """Instance method as coroutine.

        Methods can be coroutines like sync functions, but with async.
        """
        self.call_count += 1
        await asyncio.sleep(0.01)
        return value * 2

    @staticmethod
    async def static_coroutine(x: int, y: int) -> int:
        """Static method as coroutine.

        A staticmethod is a coroutine without self.
        """
        await asyncio.sleep(0.01)
        return x + y

    @classmethod
    async def class_coroutine(cls, multiplier: int) -> type:
        """Class method as coroutine.

        A classmethod is a coroutine with cls.
        """
        await asyncio.sleep(0.01)
        return cls

    def __await__(self) -> Generator[Any, None, str]:
        """Make instance awaitable.

        This method tells Python how to await the instance directly.
        """

        async def inner() -> str:
            await asyncio.sleep(0.01)
            return f"Awaited {self.name}"

        # Delegate to the coroutine await machinery
        return inner().__await__()


async def coroutine_with_yield() -> AsyncGenerator[int, None]:
    """Async generator that yields multiple values over time.

    An async generator pauses and yields values in an async for loop.
    It pauses between each yield.

    Yields:
        The integers 0 through 4.
    """
    for i in range(5):
        await asyncio.sleep(0.01)
        yield i


async def coroutine_with_yield_from() -> AsyncIterator[str]:
    """Async iterator that yields values over time.

    Yields:
        Strings that represent items in a sequence.
    """
    items = ["a", "b", "c"]
    for item in items:
        await asyncio.sleep(0.01)
        yield item


async def nested_coroutines() -> list[str]:
    """Awaits multiple coroutines sequentially.

    Each await waits for the previous coroutine to finish.
    The steps add up because execution is sequential.

    Returns:
        A list of results, one string per coroutine.
    """
    results = []
    # Await coroutines sequentially. Order is preserved.
    results.append(await coroutine_with_args("first"))
    results.append(await coroutine_with_args("second"))
    results.append(await coroutine_with_args("third"))
    return results


async def concurrent_coroutines() -> tuple[str, str, str]:
    """Runs coroutines concurrently with gather.

    asyncio.gather creates Tasks. The coroutines start when passed to gather.
    Their sleep delays overlap.
    """
    # asyncio.gather runs coroutines concurrently. Sleep delays overlap.
    return await asyncio.gather(
        coroutine_with_args("first"), coroutine_with_args("second"), coroutine_with_args("third")
    )


def coroutine(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
    """Decorator to mark a function as a coroutine (Python 3.4+ style)."""
    return func


def is_coroutine_function(func: Callable) -> bool:
    """Checks if a function is a coroutine function."""
    return inspect.iscoroutinefunction(func)


def is_coroutine(obj: object) -> bool:
    """Checks if an object is a coroutine."""
    return asyncio.iscoroutine(obj)


def awaitable(obj: object) -> bool:
    """Checks if an object is awaitable."""
    return isinstance(obj, ABCAwaitable)


class CoroutineWrapper(Generic[T]):
    """Wrapper for coroutine objects with utilities."""

    def __init__(self, coro: Coroutine[object, object, T]) -> None:
        self._coro = coro
        self._task: asyncio.Task | None = None

    def __await__(self):
        return self._coro.__await__()

    def __aiter__(self):
        return cast("AsyncIterator[T]", self._coro).__aiter__()

    async def __anext__(self):
        return await cast("AsyncIterator[T]", self._coro).__anext__()

    @property
    def coro(self) -> Coroutine[Any, Any, T]:
        return self._coro

    def to_task(self, *, loop: asyncio.AbstractEventLoop | None = None) -> asyncio.Task[T]:
        """Converts to a Task."""
        if self._task is None:
            if loop is None:
                loop = asyncio.get_running_loop()
            self._task = loop.create_task(self._coro)
        return self._task

    def cancel(self, msg: str | None = None) -> bool:
        """Cancels the coroutine if it runs as a task."""
        if self._task:
            return self._task.cancel(msg)
        return False


@overload
def async_generator(
    func: Callable[..., AsyncGenerator[T, None]],
) -> Callable[..., AsyncGenerator[T, None]]: ...
@overload
def async_generator(func: Callable[..., AsyncIterator[T]]) -> Callable[..., AsyncIterator[T]]: ...


def async_generator(func: Callable) -> Callable:
    """Decorator to mark an async generator function."""
    return func


class AsyncContextManager(Generic[T]):
    """Base class for async context managers."""

    async def __aenter__(self) -> T:
        raise NotImplementedError

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool | None:
        raise NotImplementedError


def async_context_manager(
    func: Callable[..., AsyncIterator[T]],
) -> Callable[..., AbstractAsyncContextManager[T]]:
    """Decorator to create async context manager from generator."""
    return asynccontextmanager(func)


class AwaitableWrapper(Generic[T]):
    """Makes any object awaitable."""

    def __init__(self, value: T) -> None:
        self._value = value

    def __await__(self):
        async def _awaiter():
            return self._value

        return _awaiter().__await__()


class CoroutineChain(Generic[T]):
    """Chains multiple coroutines together."""

    def __init__(self) -> None:
        self._coros: list[Callable[..., Coroutine[object, object, object]]] = []

    def add(self, coro: Callable[..., Coroutine[object, object, object]]) -> "CoroutineChain":
        self._coros.append(coro)
        return self

    async def run(self, *args: object, **kwargs: object) -> T:
        result = None
        for coro in self._coros:
            if inspect.iscoroutinefunction(coro):
                result = await coro(*args, **kwargs)
            else:
                result = coro(*args, **kwargs)
        return result


async def run_coroutine_with_result(
    coro: Coroutine[Any, Any, T], *, return_exceptions: bool = False
) -> T | BaseException:
    """Runs coroutine and returns result or exception."""
    try:
        return await coro
    except asyncio.CancelledError:
        raise
    except BaseException as e:
        if return_exceptions:
            return e
        raise


async def run_coroutines_concurrently(
    *coros: Coroutine[object, object, T],
    max_concurrent: int | None = None,
    return_exceptions: bool = False,
) -> list[T | BaseException]:
    """Runs multiple coroutines concurrently with optional concurrency limit."""
    if max_concurrent is None:
        return await asyncio.gather(*coros, return_exceptions=return_exceptions)

    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_coro(coro: Coroutine[object, object, T]) -> T | BaseException:
        async with semaphore:
            try:
                return await coro
            except asyncio.CancelledError:
                raise
            except BaseException as e:
                if return_exceptions:
                    return e
                raise

    return await asyncio.gather(
        *[limited_coro(c) for c in coros], return_exceptions=return_exceptions
    )


async def run_coroutines_sequentially(
    *coros: Coroutine[object, object, T], stop_on_error: bool = True
) -> list[T | BaseException]:
    """Runs coroutines sequentially."""
    results = []
    for coro in coros:
        try:
            result = await coro
            results.append(result)
        except asyncio.CancelledError:
            raise
        except BaseException as e:
            results.append(e)
            if stop_on_error:
                break
    return results


class AsyncRetry:
    """Async retry decorator with exponential backoff."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        exceptions: tuple = (Exception,),
    ) -> None:
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.exceptions = exceptions

    def __call__(
        self, func: Callable[..., Coroutine[object, object, T]]
    ) -> Callable[..., Coroutine[object, object, T]]:
        @functools.wraps(func)
        async def wrapper(*args: object, **kwargs: object) -> T:
            last_exception = None
            for attempt in range(self.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except self.exceptions as e:
                    last_exception = e
                    if attempt < self.max_attempts - 1:
                        delay = min(
                            self.base_delay * (self.exponential_base**attempt), self.max_delay
                        )
                        if self.jitter:
                            import random

                            delay *= 0.5 + random.random()
                        await asyncio.sleep(delay)
            raise last_exception

        return wrapper


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: tuple = (Exception,),
) -> Callable[
    [Callable[..., Coroutine[object, object, T]]], Callable[..., Coroutine[object, object, T]]
]:
    """Decorator for async retry with exponential backoff."""
    return AsyncRetry(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        exceptions=exceptions,
    )


class AsyncCache(Generic[T]):
    """Async LRU cache for coroutine results."""

    def __init__(self, maxsize: int = 128) -> None:
        self._cache: dict[object, T] = {}
        self._maxsize = maxsize
        self._lock = asyncio.Lock()

    async def get(self, key: object, default: T | None = None) -> T | None:
        async with self._lock:
            return self._cache.get(key, default)

    async def set(self, key: object, value: T) -> None:
        async with self._lock:
            if len(self._cache) >= self._maxsize:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = value

    async def delete(self, key: object) -> bool:
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self) -> None:
        async with self._lock:
            self._cache.clear()

    def __call__(
        self, func: Callable[..., Coroutine[object, object, T]]
    ) -> Callable[..., Coroutine[object, object, T]]:
        @functools.wraps(func)
        async def wrapper(*args: object, **kwargs: object) -> T:
            key = (args, frozenset(kwargs.items()))
            cached = await self.get(key)
            if cached is not None:
                return cached
            result = await func(*args, **kwargs)
            await self.set(key, result)
            return result

        return wrapper


async def first_completed(*coros: Coroutine[object, object, T]) -> T:
    """Return result of first completed coroutine, cancel others."""
    tasks = [asyncio.create_task(c) for c in coros]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        return done.pop().result()
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()


async def all_completed(
    *coros: Coroutine[object, object, T], return_exceptions: bool = False
) -> list[T]:
    """Wait for all coroutines to complete."""
    return await asyncio.gather(*coros, return_exceptions=return_exceptions)


async def demonstrate_coroutines() -> None:
    """Coroutine fundamentals tutorial for asyncio beginners.

    This function teaches the most important concept in asyncio: the coroutine.

    WHAT IS A COROUTINE?
    A coroutine is a function defined with async def that can pause and resume.
    Calling a coroutine function does not execute its body. It returns a coroutine
    object. You must await that object to start execution.

    MENTAL MODEL: The Juggler
    The event loop is like a juggler. It starts one coroutine. When that
    coroutine hits await asyncio.sleep(...), it yields control. The loop
    pauses that coroutine and starts another. When the sleep finishes, the
    first resumes. This is cooperative multitasking. Coroutines voluntarily
    yield control.

    KEY CONCEPTS COVERED:
    1. Coroutine objects vs functions: Calling async def foo() returns
       a coroutine object. It does not run yet.
    2. await: The mechanism that yields control to the event loop.
    3. Arguments: Coroutines can accept arguments like sync functions.
    4. Return values: Coroutines can return any type.
    5. Exception handling: Exceptions propagate to the awaiting caller.
    6. Class-based coroutines: staticmethod, classmethod, and
       instance methods can all be coroutines.
    7. __await__: Making custom objects awaitable.
    8. Async generators: Using async for with coroutines that yield.
    9. Async iterators: Behind-the-scenes __anext__ calls.
    10. Sequential vs concurrent: Demonstrated with timing to show the
        performance difference.

    WHAT YOU WILL EXPERIENCE:
    Live output showing each concept:
    - Creating coroutine objects (printing the object itself)
    - Awaiting and getting results
    - Coroutines with arguments
    - Complex return values (dicts, nested structures)
    - Exception catching in async code
    - Class method coroutines (static, class, instance)
    - Awaiting a coroutine instance (via __await__)
    - Async generator iteration
    - Async iterator consumption
    - Sequential execution timing
    - Concurrent execution timing (asyncio.gather())
    - Coroutine introspection (cr_code.co_name, etc.)

    PERFORMANCE DEMONSTRATION:
    The function measures and prints the time for both sequential and
    concurrent execution. You will see that concurrent execution overlaps
    the asyncio.sleep() delays, making it significantly faster.

    SEE ALSO:
    - demonstrate_tasks() - the next layer of abstraction (Tasks)
    - asyncio.gather() - runs coroutines concurrently
    - asyncio.create_task() - schedules coroutines as Tasks
    - Real Python: Async IO in Python (https://realpython.com/async-io-python/)
    """
    print("=== Coroutine Basics ===")

    # 1. Creating coroutine objects (does not execute yet)
    # Awaiting a coroutine object runs the coroutine code.
    coro_obj = simple_coroutine()
    print(f"Coroutine object: {coro_obj}")
    print(f"Is coroutine function: {inspect.iscoroutinefunction(simple_coroutine)}")
    print(f"Is coroutine object: {inspect.iscoroutine(coro_obj)}")

    # 2. Executing coroutine with await
    # Hand the coroutine object to await, and the event loop runs it.
    result = await coro_obj
    print(f"Result: {result}")

    # 3. Coroutine with arguments
    result = await coroutine_with_args("World", 0.01)
    print(f"With args: {result}")

    # 4. Coroutine returning complex value
    result = await coroutine_returning_value()
    print(f"Complex return: {result}")

    # 5. Exception handling in coroutines
    # Catching a coroutine exception works like sync code.
    try:
        await coroutine_raising_exception()
    except ValueError as e:
        print(f"Caught exception: {e}")

    # 6. Class-based coroutines
    example = CoroutineExample("test")
    # Awaiting a coroutine method is the same as awaiting a function.
    result = await example.method_coroutine(21)
    print(f"Method coroutine: {result}")

    result = await CoroutineExample.static_coroutine(10, 20)
    print(f"Static coroutine: {result}")

    result = await example  # Uses __await__
    # await example makes an instance awaitable.
    print(f"Awaitable instance: {result}")

    # 7. Async generators
    # async for consumes an async generator. Each iteration yields the next value.
    print("\n--- Async Generator ---")
    async for value in coroutine_with_yield():
        print(f"Yielded: {value}")

    # 8. Async iterators
    # Iterating over an async iterator. Behind the scenes each __anext__ is awaited.
    print("\n--- Async Iterator ---")
    async for value in coroutine_with_yield_from():
        print(f"Iterated: {value}")

    # 9. Sequential vs concurrent
    import time

    # Sequential execution: each await blocks this coroutine until the nested
    # coroutines finish. The steps add up because execution is sequential.
    print("\n--- Sequential Execution ---")
    start = time.perf_counter()
    await nested_coroutines()
    print(f"Sequential time: {time.perf_counter() - start:.3f}s")

    # Concurrent execution: asyncio.gather() overlaps the sleeps. The delays
    # overlap, proving concurrency works.
    print("\n--- Concurrent Execution ---")
    start = time.perf_counter()
    await concurrent_coroutines()
    print(f"Concurrent time: {time.perf_counter() - start:.3f}s")

    # 10. Coroutine inspection
    print("\n--- Coroutine Inspection ---")
    coro = coroutine_with_args("inspect")
    print(f"Coroutine name: {coro.cr_code.co_name}")
    print(f"Coroutine file: {coro.cr_code.co_filename}")
    print(f"Coroutine line: {coro.cr_code.co_firstlineno}")
    # Clean up unstarted coroutine
    coro.close()
