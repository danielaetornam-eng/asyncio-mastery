"""
Coroutines. The foundation of asyncio.

A coroutine is a function that can pause and resume later. In Python, you define a coroutine with async def. When you call a coroutine, the code does not run immediately. You get a coroutine object. You must await the object to execute it.

A coroutine is like a recipe. Calling the function produces the recipe. You hand the recipe to the event loop. The event loop runs the recipe when convenient.

await some_coroutine() means wait for the coroutine to finish. While waiting, let other coroutines run. The event loop is like a juggler. The loop starts one coroutine. When the coroutine says await, the loop pauses it and starts another.

Key terms:
- Coroutine function. The function defined with async def.
- Coroutine object. The object you get by calling a coroutine function.
- Await. Yields control back to the event loop.
- Coroutine. The conceptual building block that lets you pause and resume.

Runnable example:
import asyncio

async def say_hello():
    await asyncio.sleep(1)
    return "hello"

asyncio.run(say_hello())

Most asyncio code uses coroutines as the basic unit of async execution. The next layer is a Task. A Task adds scheduling and management on top.

See also: asyncio.gather(), asyncio.create_task()
"""

import asyncio
import inspect
from collections.abc import AsyncGenerator, AsyncIterator, Generator
from typing import Any


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
    # await yields control to the event loop
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
        """Makes instance awaitable.

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


async def demonstrate_coroutines() -> None:
    """Coroutine fundamentals tutorial for asyncio beginners.

    This is the primary teaching function for the core concept of asyncio: coroutines.
    It takes you from the absolute basics through to concurrent execution.
    It builds intuition step by step.

    What you will learn:
    1. Coroutine objects vs coroutine functions. Creation does not execute.
    2. How await runs a coroutine via the event loop.
    3. Passing arguments to coroutines.
    4. Return values from coroutines.
    5. Exception handling in async code.
    6. Class-based coroutines (methods, static/class methods).
    7. Making instances awaitable via __await__.
    8. Async generators with async for.
    9. Async iterators behind the scenes.
    10. The performance difference: sequential vs concurrent execution.

    Conceptual mental model:
    Think of a coroutine as a pausable function. When you define one with async def,
    calling it does not run the body immediately. It returns a coroutine object.
    You must await that object to start execution. While one coroutine is awaiting,
    the event loop can run other coroutines. This is cooperative multitasking.
    Coroutines voluntarily yield control via await.

    The event loop is like a juggler. It starts one coroutine. When that coroutine
    says await asyncio.sleep(...), the loop pauses it and starts another.
    When the sleep finishes, the first resumes.

    Why this matters:
    This pattern lets you write concurrent code that is simpler than threading.
    It does not suffer from most threading problems. No locks needed for simple data.
    No race conditions from implicit preemption. It is Python way of doing
    high-performance I/O-bound concurrency.

    Runnable example:
    The function includes live demonstrations with print() output showing each
    concept in action. Times are shown to prove that concurrent execution is
    faster than sequential.

    See also:
    - asyncio.gather() for running coroutines in parallel
    - asyncio.create_task() for scheduling coroutines as Tasks
    - demonstrate_tasks() for the next layer of asyncio abstraction
    """
    print("=== Coroutine Basics ===")

    # 1. Creating coroutine objects (does not execute yet)
    # Awaiting a coroutine object actually runs the coroutine code.
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
    # Catching a coroutine exception works just like sync code.
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
    print("\n--- Sequential Execution ---")
    import time

    # Sequential execution: each await blocks this coroutine until the nested
    # coroutines finish. The steps add up because execution is sequential.
    start = time.perf_counter()
    await nested_coroutines()
    print(f"Sequential time: {time.perf_counter() - start:.3f}s")

    print("\n--- Concurrent Execution ---")
    # Concurrent execution overlaps the sleeps. gather starts all of the coroutines.
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


if __name__ == "__main__":
    asyncio.run(demonstrate_coroutines())
