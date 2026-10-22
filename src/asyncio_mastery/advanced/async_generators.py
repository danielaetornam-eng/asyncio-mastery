"""
Async Generators & Iterators - Produce/consume sequences asynchronously.

================================================================================
WHAT IS AN ASYNC GENERATOR?
================================================================================

An async generator is a function written with ``async def`` and ``yield``.
Like a regular generator, it produces a *sequence lazily* (one item at a
time, on demand). The async part means it may ``await`` between yields --
perfect for sequences whose items arrive over time (network pages, database
rows, messages).

MENTAL MODEL
------------
An async generator is a conveyor belt delivering items over time. Each
``async for`` loop iteration grabs the next item off the belt; if the belt
is momentarily empty, the loop patiently ``await``s for the next item without
freezing your whole program.

An async ITERATOR is the lower-level machinery underneath: any object with
``__aiter__`` (returns the iterator) and ``async __anext__`` (returns the next
item, or raises ``StopAsyncIteration`` to signal "belt is done"). Writing
``async def`` + ``yield`` gives you all that for free.

WHEN YOU NEED THIS
------------------
- Streaming large/unbounded data without loading it all into memory.
- Consuming web sockets, message queues, paginated APIs.
- Building pipelines: map/filter/zip/chain over async sources.

MINIMAL EXAMPLE
---------------
>>> import asyncio
>>> async def countdown(n):
...     while n > 0:
...         await asyncio.sleep(1)   # wait, then deliver next item
...         yield n
...         n -= 1
>>> async def main():
...     async for x in countdown(3):
...         print(x)

Run the full demo with: ``python -m asyncio_mastery.advanced.async_generators``
"""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator, Awaitable, Callable
from typing import Any, TypeVar

T = TypeVar("T")
U = TypeVar("U")


async def async_range(
    start: int = 0, stop: int | None = None, step: int = 1, delay: float = 0
) -> AsyncGenerator[int, None]:
    """Async version of the built-in ``range()``.

    Args:
        start: First value (or the stop value if ``stop`` is None, matching
            range()'s one-argument form).
        stop: End value (exclusive). If None, behaves like ``range(start)``.
        step: Increment between values (may be negative).
        delay: Seconds to ``await asyncio.sleep`` before each yield --
            simulates items arriving over time.

    Yields:
        Integers from start towards stop.

    Example:
        >>> async for i in async_range(3, delay=0.5):
        ...     print(i)   # 0, 1, 2, half a second apart
    """
    if stop is None:
        stop = start
        start = 0

    current = start
    while (step > 0 and current < stop) or (step < 0 and current > stop):
        if delay:
            await asyncio.sleep(delay)
        yield current
        current += step


async def async_map(
    func: Callable[[T], Awaitable[U]], iterable: AsyncIterable[T]
) -> AsyncGenerator[U, None]:
    """Like ``map()``, but the function is async.

    Args:
        func: An async function applied to each item.
        iterable: Any async iterable (e.g. an async generator).

    Yields:
        ``await func(item)`` for each item, one at a time (lazy: nothing is
        computed until the consumer asks for the next value).
    """
    async for item in iterable:
        yield await func(item)


async def async_filter(
    predicate: Callable[[T], Awaitable[bool]], iterable: AsyncIterable[T]
) -> AsyncGenerator[T, None]:
    """Like ``filter()``, but the test function is async.

    Args:
        predicate: Async function returning True to keep an item.
        iterable: Source async iterable.

    Yields:
        Only the items for which ``await predicate(item)`` is True.
    """
    async for item in iterable:
        if await predicate(item):
            yield item


async def async_chain(*iterables: AsyncIterable[T]) -> AsyncGenerator[T, None]:
    """Concatenate async iterables end-to-end (like ``itertools.chain``).

    Yields every item from the first iterable, then the second, and so on.
    """
    for it in iterables:
        async for item in it:
            yield item


async def async_zip(*iterables: AsyncIterable[Any]) -> AsyncGenerator[tuple, None]:
    """Like ``zip()``: yields tuples of the i-th item from each iterable.

    Stops as soon as the SHORTEST iterable is exhausted. Note how the next
    item is fetched from every iterable concurrently via ``asyncio.gather``.

    Yields:
        Tuples ``(item_from_first, item_from_second, ...)``.
    """
    # Get the underlying iterator objects so we can call __anext__ manually.
    iterators = [it.__aiter__() for it in iterables]
    while True:
        try:
            # Advance all iterators at once; if any raises StopAsyncIteration,
            # gather propagates it and we treat it as "belt done".
            items = await asyncio.gather(*[it.__anext__() for it in iterators])
            yield tuple(items)
        except StopAsyncIteration:
            break


class AsyncGenerator:
    """Utility belt of static helpers for consuming async generators.

    All methods take an async generator and either drain it into concrete
    Python values (``collect``, ``first``, ``take``) or wrap it in another
    lazy generator (``skip``, ``batch``).
    """

    @staticmethod
    async def collect(gen: AsyncGenerator[T, None]) -> list[T]:
        """Drain the whole generator into a list.

        Careful: defeats laziness -- do not use on infinite generators!

        Returns:
            A list of every yielded item.
        """
        # Async comprehension: the async-for version of [x for x in gen].
        return [item async for item in gen]

    @staticmethod
    async def first(gen: AsyncGenerator[T, None], default: T | None = None) -> T | None:
        """Return the first item, or ``default`` if the generator is empty.

        Stops after one item, so the generator is not fully consumed.
        """
        async for item in gen:
            return item
        return default

    @staticmethod
    async def take(n: int, gen: AsyncGenerator[T, None]) -> list[T]:
        """Return a list of at most the first ``n`` items.

        Safe on infinite generators because it stops early.

        Args:
            n: Maximum number of items to consume.
            gen: The async generator to draw from.
        """
        result = []
        async for item in gen:
            result.append(item)
            if len(result) >= n:
                break  # leaving the loop closes our consumption of the belt
        return result

    @staticmethod
    async def skip(n: int, gen: AsyncGenerator[T, None]) -> AsyncGenerator[T, None]:
        """Lazily discard the first ``n`` items, yielding the rest."""
        skipped = 0
        async for item in gen:
            if skipped < n:
                skipped += 1
                continue
            yield item

    @staticmethod
    async def batch(size: int, gen: AsyncGenerator[T, None]) -> AsyncGenerator[list[T], None]:
        """Group items into lists of ``size`` (the last batch may be smaller).

        Example:
            ``batch(2, async_range(5))`` yields ``[0,1]``, ``[2,3]``, ``[4]``.
        """
        batch = []
        async for item in gen:
            batch.append(item)
            if len(batch) >= size:
                yield batch
                batch = []  # start a fresh batch (NOT batch.clear()!)
        if batch:  # flush any leftovers that didn't fill a whole batch
            yield batch


class AsyncIterator:
    """Base class showing the async iterator protocol by hand.

    ``async for`` needs two things from an object:
      1. ``__aiter__()`` -> returns the iterator (usually ``self``).
      2. ``await __anext__()`` -> next item, or raise ``StopAsyncIteration``
         to end the loop cleanly (this is the async analogue of ``StopIteration``).

    Subclass this and override ``__anext__`` to build custom async sequences.
    """

    def __aiter__(self) -> AsyncIterator[T]:
        return self  # iterators are their own iterable, just like sync ones

    async def __anext__(self) -> T:
        raise StopAsyncIteration


class AsyncCounter(AsyncIterator):
    """An iterator that counts up (or down), pausing cooperatively each step.

    Example:
        >>> async for i in AsyncCounter(0, 1, 3):
        ...     print(i)   # 0, 1, 2 then StopAsyncIteration ends the loop

    Args:
        start: First value produced.
        step: Amount added after each value.
        limit: Stop bound; None means count forever.
    """

    def __init__(self, start: int = 0, step: int = 1, limit: int | None = None) -> None:
        self._current = start
        self._step = step
        self._limit = limit

    async def __anext__(self) -> int:
        if self._limit is not None:
            if (self._step > 0 and self._current >= self._limit) or (
                self._step < 0 and self._current <= self._limit
            ):
                raise StopAsyncIteration

        value = self._current
        self._current += self._step
        # sleep(0) yields control back to the event loop for one round, so
        # other tasks can run even while this counter is being consumed fast.
        await asyncio.sleep(0)  # Allow context switch
        return value


class AsyncFileReader(AsyncIterator):
    """Read a text file line by line using ``async for``.

    Note: file I/O here uses the normal blocking ``open()`` for teaching
    simplicity -- real applications should use ``aiofiles`` or
    ``asyncio.to_thread`` so reads don't block the event loop.

    Example:
        >>> async with AsyncFileReader("data.txt") as reader:
        ...     async for line in reader:
        ...         print(line)
    """

    def __init__(self, filepath: str, encoding: str = "utf-8") -> None:
        self._filepath = filepath
        self._encoding = encoding
        self._file: Any = None

    async def __aenter__(self) -> "AsyncFileReader":
        # Async context manager protocol: __aenter__/__aexit__ let us use
        # `async with` so the file is guaranteed to be closed on exit.
        self._file = open(self._filepath, encoding=self._encoding)
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._file:
            self._file.close()

    async def __anext__(self) -> str:
        if not self._file:
            raise StopAsyncIteration
        line = self._file.readline()
        if not line:
            # readline() returns "" at end of file -> signal the loop to stop.
            raise StopAsyncIteration
        return line.rstrip("\n")


async def demonstrate_async_generators() -> None:
    """Narrated tour of async generators and iterators.

    Run with: ``python -m asyncio_mastery.advanced.async_generators``.
    """
    print("=== Async Generators Demo ===")

    # Basic async generator
    print("\n--- async_range ---")
    async for i in async_range(5, delay=0.01):
        print(f"  {i}")

    # Async map
    print("\n--- async_map ---")

    async def double(x: int) -> int:
        await asyncio.sleep(0.01)
        return x * 2

    async for result in async_map(double, async_range(5)):
        print(f"  {result}")

    # Async filter
    print("\n--- async_filter ---")

    async def is_even(x: int) -> bool:
        await asyncio.sleep(0.01)
        return x % 2 == 0

    async for result in async_filter(is_even, async_range(10)):
        print(f"  {result}")

    # Async zip
    print("\n--- async_zip ---")
    async for a, b in async_zip(async_range(3), async_range(10, 13)):
        print(f"  ({a}, {b})")

    # Async chain
    print("\n--- async_chain ---")
    async for item in async_chain(async_range(3), async_range(10, 13)):
        print(f"  {item}")

    # Utilities
    print("\n--- Generator Utilities ---")
    gen = async_range(100)
    first_five = await AsyncGenerator.take(5, gen)
    print(f"First 5: {first_five}")

    gen = async_range(10)
    batched = [batch async for batch in AsyncGenerator.batch(3, gen)]
    print(f"Batched: {batched}")

    # Async iterators
    print("\n--- AsyncCounter ---")
    async for i in AsyncCounter(0, 1, 5):
        print(f"  {i}")

    print("\n--- AsyncFileReader ---")
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("line 1\nline 2\nline 3\n")
        temp_path = f.name

    async with AsyncFileReader(temp_path) as reader:
        async for line in reader:
            print(f"  Read: {line}")

    import os

    os.unlink(temp_path)


if __name__ == "__main__":
    asyncio.run(demonstrate_async_generators())
