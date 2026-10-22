"""
Async Collections

Async versions of common collection types.
"""

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterable
from typing import Any, Generic, TypeVar

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


class AsyncList(Generic[T]):
    """Async list with thread-safe operations."""

    def __init__(self, iterable: Iterable[T] | None = None) -> None:
        self._list: list[T] = list(iterable) if iterable else []
        self._lock = asyncio.Lock()

    async def append(self, item: T) -> None:
        async with self._lock:
            self._list.append(item)

    async def extend(self, items: Iterable[T]) -> None:
        async with self._lock:
            self._list.extend(items)

    async def insert(self, index: int, item: T) -> None:
        async with self._lock:
            self._list.insert(index, item)

    async def remove(self, item: T) -> None:
        async with self._lock:
            self._list.remove(item)

    async def pop(self, index: int = -1) -> T:
        async with self._lock:
            return self._list.pop(index)

    async def clear(self) -> None:
        async with self._lock:
            self._list.clear()

    async def index(self, item: T, start: int = 0, end: int | None = None) -> int:
        async with self._lock:
            return self._list.index(item, start, end)

    async def count(self, item: T) -> int:
        async with self._lock:
            return self._list.count(item)

    async def sort(self, *, key: Callable | None = None, reverse: bool = False) -> None:
        async with self._lock:
            self._list.sort(key=key, reverse=reverse)

    async def reverse(self) -> None:
        async with self._lock:
            self._list.reverse()

    async def copy(self) -> list[T]:
        async with self._lock:
            return self._list.copy()

    async def __len__(self) -> int:
        async with self._lock:
            return len(self._list)

    async def __getitem__(self, index: int | slice) -> T | list[T]:
        async with self._lock:
            return self._list[index]

    async def __setitem__(self, index: int | slice, value: T | Iterable[T]) -> None:
        async with self._lock:
            self._list[index] = value

    async def __delitem__(self, index: int | slice) -> None:
        async with self._lock:
            del self._list[index]

    async def __contains__(self, item: T) -> bool:
        async with self._lock:
            return item in self._list

    async def __iter__(self) -> AsyncIterator[T]:
        async with self._lock:
            items = self._list.copy()
        for item in items:
            yield item

    def __len__(self) -> int:
        return len(self._list)

    def __getitem__(self, index: int | slice) -> T | list[T]:
        return self._list[index]


class AsyncDict(Generic[K, V]):
    """Async dictionary with thread-safe operations."""

    def __init__(self, mapping: dict[K, V] | None = None) -> None:
        self._dict: dict[K, V] = dict(mapping) if mapping else {}
        self._lock = asyncio.Lock()

    async def __len__(self) -> int:
        async with self._lock:
            return len(self._dict)

    async def __contains__(self, key: K) -> bool:
        async with self._lock:
            return key in self._dict

    async def __getitem__(self, key: K) -> V:
        async with self._lock:
            return self._dict[key]

    async def __setitem__(self, key: K, value: V) -> None:
        async with self._lock:
            self._dict[key] = value

    async def __delitem__(self, key: K) -> None:
        async with self._lock:
            del self._dict[key]

    async def get(self, key: K, default: V | None = None) -> V | None:
        async with self._lock:
            return self._dict.get(key, default)

    async def setdefault(self, key: K, default: V | None = None) -> V:
        async with self._lock:
            return self._dict.setdefault(key, default)

    async def update(self, mapping: dict[K, V]) -> None:
        async with self._lock:
            self._dict.update(mapping)

    async def pop(self, key: K, default: V | None = None) -> V:
        async with self._lock:
            return self._dict.pop(key, default)

    async def popitem(self) -> tuple:
        async with self._lock:
            return self._dict.popitem()

    async def clear(self) -> None:
        async with self._lock:
            self._dict.clear()

    async def copy(self) -> dict[K, V]:
        async with self._lock:
            return self._dict.copy()

    async def keys(self) -> list[K]:
        async with self._lock:
            return list(self._dict.keys())

    async def values(self) -> list[V]:
        async with self._lock:
            return list(self._dict.values())

    async def items(self) -> list[tuple]:
        async with self._lock:
            return list(self._dict.items())

    async def __iter__(self) -> AsyncIterator[K]:
        async with self._lock:
            keys = list(self._dict.keys())
        for key in keys:
            yield key

    def __len__(self) -> int:
        return len(self._dict)

    def __contains__(self, key: K) -> bool:
        return key in self._dict


class AsyncSet(Generic[T]):
    """Async set with thread-safe operations."""

    def __init__(self, iterable: Iterable[T] | None = None) -> None:
        self._set: set[T] = set(iterable) if iterable else set()
        self._lock = asyncio.Lock()

    async def __len__(self) -> int:
        async with self._lock:
            return len(self._set)

    async def __contains__(self, item: T) -> bool:
        async with self._lock:
            return item in self._set

    async def add(self, item: T) -> None:
        async with self._lock:
            self._set.add(item)

    async def remove(self, item: T) -> None:
        async with self._lock:
            self._set.remove(item)

    async def discard(self, item: T) -> None:
        async with self._lock:
            self._set.discard(item)

    async def pop(self) -> T:
        async with self._lock:
            return self._set.pop()

    async def clear(self) -> None:
        async with self._lock:
            self._set.clear()

    async def update(self, iterable: Iterable[T]) -> None:
        async with self._lock:
            self._set.update(iterable)

    async def intersection_update(self, iterable: Iterable[T]) -> None:
        async with self._lock:
            self._set.intersection_update(iterable)

    async def difference_update(self, iterable: Iterable[T]) -> None:
        async with self._lock:
            self._set.difference_update(iterable)

    async def symmetric_difference_update(self, iterable: Iterable[T]) -> None:
        async with self._lock:
            self._set.symmetric_difference_update(iterable)

    async def union(self, iterable: Iterable[T]) -> set[T]:
        async with self._lock:
            return self._set.union(iterable)

    async def intersection(self, iterable: Iterable[T]) -> set[T]:
        async with self._lock:
            return self._set.intersection(iterable)

    async def difference(self, iterable: Iterable[T]) -> set[T]:
        async with self._lock:
            return self._set.difference(iterable)

    async def symmetric_difference(self, iterable: Iterable[T]) -> set[T]:
        async with self._lock:
            return self._set.symmetric_difference(iterable)

    async def issubset(self, iterable: Iterable[T]) -> bool:
        async with self._lock:
            return self._set.issubset(iterable)

    async def issuperset(self, iterable: Iterable[T]) -> bool:
        async with self._lock:
            return self._set.issuperset(iterable)

    async def isdisjoint(self, iterable: Iterable[T]) -> bool:
        async with self._lock:
            return self._set.isdisjoint(iterable)

    async def copy(self) -> set[T]:
        async with self._lock:
            return self._set.copy()

    async def __iter__(self) -> AsyncIterator[T]:
        async with self._lock:
            items = list(self._set)
        for item in items:
            yield item

    async def __len__(self) -> int:
        async with self._lock:
            return len(self._set)

    def __len__(self) -> int:
        return len(self._set)


class AsyncQueue(Generic[T]):
    """Enhanced async queue with additional features."""

    def __init__(self, maxsize: int = 0, *, name: str | None = None) -> None:
        self._queue = asyncio.Queue(maxsize=maxsize)
        self._name = name or f"AsyncQueue-{id(self)}"
        self._stats = {"puts": 0, "gets": 0, "put_wait_time": 0.0, "get_wait_time": 0.0}
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def maxsize(self) -> int:
        return self._queue.maxsize

    @property
    def qsize(self) -> int:
        return self._queue.qsize()

    @property
    def empty(self) -> bool:
        return self._queue.empty()

    @property
    def full(self) -> bool:
        return self._queue.full()

    def stats(self) -> dict[str, Any]:
        return self._stats.copy()

    async def put(self, item: T, timeout: float | None = None) -> bool:
        import time

        start = time.monotonic()
        try:
            if timeout is not None:
                await asyncio.wait_for(self._queue.put(item), timeout=timeout)
            else:
                await self._queue.put(item)

            async with self._lock:
                self._stats["puts"] += 1
                self._stats["put_wait_time"] += time.monotonic() - start
            return True
        except TimeoutError:
            return False

    async def put_nowait(self, item: T) -> None:
        self._queue.put_nowait(item)
        async with self._lock:
            self._stats["puts"] += 1

    async def get(self, timeout: float | None = None) -> T:
        import time

        start = time.monotonic()
        try:
            if timeout is not None:
                item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
            else:
                item = await self._queue.get()

            async with self._lock:
                self._stats["gets"] += 1
                self._stats["get_wait_time"] += time.monotonic() - start
            return item
        except TimeoutError:
            raise asyncio.QueueEmpty("Queue get timed out")

    async def get_nowait(self) -> T:
        try:
            item = self._queue.get_nowait()
            async with self._lock:
                self._stats["gets"] += 1
            return item
        except asyncio.QueueEmpty:
            raise asyncio.QueueEmpty("Queue is empty")

    def task_done(self) -> None:
        self._queue.task_done()

    async def join(self) -> None:
        await self._queue.join()

    async def clear(self) -> list[T]:
        items = []
        while not self._queue.empty():
            try:
                items.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return items

    async def peek(self) -> T | None:
        """Peek at next item without removing."""
        # This requires access to internal queue
        return None  # Not directly supported by asyncio.Queue

    async def __aiter__(self) -> AsyncIterator[T]:
        while True:
            item = await self.get()
            yield item

    def __len__(self) -> int:
        return self.qsize()


class AsyncPriorityQueue(Generic[T]):
    """Async priority queue."""

    def __init__(self, maxsize: int = 0) -> None:
        self._queue = asyncio.PriorityQueue(maxsize=maxsize)
        self._counter = 0

    @property
    def maxsize(self) -> int:
        return self._queue.maxsize

    @property
    def qsize(self) -> int:
        return self._queue.qsize()

    @property
    def empty(self) -> bool:
        return self._queue.empty()

    @property
    def full(self) -> bool:
        return self._queue.full()

    async def put(
        self, item: T, priority: float | None = None, timeout: float | None = None
    ) -> bool:
        if priority is None:
            priority = self._counter
            self._counter += 1

        entry = (priority, self._counter, item)

        try:
            if timeout is not None:
                await asyncio.wait_for(self._queue.put(entry), timeout=timeout)
            else:
                await self._queue.put(entry)
            return True
        except TimeoutError:
            return False

    async def put_nowait(self, item: T, priority: float | None = None) -> None:
        if priority is None:
            priority = self._counter
        entry = (priority, self._counter, item)
        self._counter += 1
        self._queue.put_nowait(entry)

    async def get(self, timeout: float | None = None) -> T:
        try:
            if timeout is not None:
                _, _, item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
            else:
                _, _, item = await self._queue.get()
            return item
        except TimeoutError:
            raise asyncio.QueueEmpty("PriorityQueue get timed out")

    async def get_nowait(self) -> T:
        try:
            _, _, item = self._queue.get_nowait()
            return item
        except asyncio.QueueEmpty:
            raise asyncio.QueueEmpty("PriorityQueue is empty")

    def task_done(self) -> None:
        self._queue.task_done()

    async def join(self) -> None:
        await self._queue.join()


class AsyncDeque(Generic[T]):
    """Async double-ended queue."""

    def __init__(self, iterable: Iterable[T] | None = None, maxlen: int | None = None) -> None:
        self._deque: deque[T] = deque(iterable, maxlen=maxlen) if iterable else deque(maxlen=maxlen)
        self._lock = asyncio.Lock()

    @property
    def maxlen(self) -> int | None:
        return self._deque.maxlen

    async def append(self, item: T) -> None:
        async with self._lock:
            self._deque.append(item)

    async def appendleft(self, item: T) -> None:
        async with self._lock:
            self._deque.appendleft(item)

    async def pop(self) -> T:
        async with self._lock:
            return self._deque.pop()

    async def popleft(self) -> T:
        async with self._lock:
            return self._deque.popleft()

    async def extend(self, items: Iterable[T]) -> None:
        async with self._lock:
            self._deque.extend(items)

    async def extendleft(self, items: Iterable[T]) -> None:
        async with self._lock:
            self._deque.extendleft(items)

    async def remove(self, item: T) -> None:
        async with self._lock:
            self._deque.remove(item)

    async def rotate(self, n: int = 1) -> None:
        async with self._lock:
            self._deque.rotate(n)

    async def clear(self) -> None:
        async with self._lock:
            self._deque.clear()

    async def copy(self) -> deque[T]:
        async with self._lock:
            return self._deque.copy()

    async def count(self, item: T) -> int:
        async with self._lock:
            return self._deque.count(item)

    async def index(self, item: T, start: int = 0, end: int | None = None) -> int:
        async with self._lock:
            return self._deque.index(item, start, end)

    async def __len__(self) -> int:
        async with self._lock:
            return len(self._deque)

    async def __getitem__(self, index: int | slice) -> T | deque[T]:
        async with self._lock:
            return self._deque[index]

    async def __setitem__(self, index: int | slice, value: T | Iterable[T]) -> None:
        async with self._lock:
            self._deque[index] = value

    async def __delitem__(self, index: int | slice) -> None:
        async with self._lock:
            del self._deque[index]

    async def __contains__(self, item: T) -> bool:
        async with self._lock:
            return item in self._deque

    async def __iter__(self) -> AsyncIterator[T]:
        async with self._lock:
            items = list(self._deque)
        for item in items:
            yield item
