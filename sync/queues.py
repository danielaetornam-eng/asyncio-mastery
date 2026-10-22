"""
Advanced Queue Implementations

High-performance async queues with various semantics.
"""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class QueueFull(Exception):
    """Raised when queue is full and cannot accept more items."""

    pass


class QueueEmpty(Exception):
    """Raised when queue is empty and cannot provide items."""

    pass


class Queue(Generic[T]):
    """Enhanced asyncio Queue with monitoring and utilities."""

    def __init__(
        self, maxsize: int = 0, *, name: str | None = None, track_stats: bool = True
    ) -> None:
        self._queue = asyncio.Queue(maxsize=maxsize)
        self._name = name or f"Queue-{id(self)}"
        self._track_stats = track_stats
        self._stats = (
            {
                "puts": 0,
                "gets": 0,
                "put_wait_time": 0.0,
                "get_wait_time": 0.0,
                "max_size_reached": 0,
            }
            if track_stats
            else None
        )

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

    def stats(self) -> dict[str, Any] | None:
        if self._stats:
            return self._stats.copy()
        return None

    async def put(self, item: T, timeout: float | None = None) -> bool:
        import time

        start = time.monotonic()
        try:
            if timeout is not None:
                await asyncio.wait_for(self._queue.put(item), timeout=timeout)
            else:
                await self._queue.put(item)

            if self._stats:
                self._stats["puts"] += 1
                self._stats["put_wait_time"] += time.monotonic() - start
                self._stats["max_size_reached"] = max(
                    self._stats["max_size_reached"], self._queue.qsize()
                )
            return True
        except TimeoutError:
            return False

    def put_nowait(self, item: T) -> None:
        self._queue.put_nowait(item)
        if self._stats:
            self._stats["puts"] += 1
            self._stats["max_size_reached"] = max(
                self._stats["max_size_reached"], self._queue.qsize()
            )

    async def get(self, timeout: float | None = None) -> T:
        import time

        start = time.monotonic()
        try:
            if timeout is not None:
                item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
            else:
                item = await self._queue.get()

            if self._stats:
                self._stats["gets"] += 1
                self._stats["get_wait_time"] += time.monotonic() - start
            return item
        except TimeoutError:
            raise QueueEmpty("Queue get timed out")

    def get_nowait(self) -> T:
        try:
            item = self._queue.get_nowait()
            if self._stats:
                self._stats["gets"] += 1
            return item
        except asyncio.QueueEmpty:
            raise QueueEmpty("Queue is empty")

    def task_done(self) -> None:
        self._queue.task_done()

    async def join(self) -> None:
        await self._queue.join()

    async def clear(self) -> list[T]:
        """Clear queue and return all items."""
        items = []
        while not self._queue.empty():
            try:
                items.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return items

    def __len__(self) -> int:
        return self.qsize()

    def __aiter__(self) -> AsyncIterator[T]:
        return self._async_iterator()

    async def _async_iterator(self) -> AsyncIterator[T]:
        while True:
            item = await self.get()
            yield item

    @asynccontextmanager
    async def batch(self, max_size: int, timeout: float):
        """Context manager for batching gets."""
        items = []
        try:
            while len(items) < max_size:
                item = await asyncio.wait_for(self.get(), timeout=timeout)
                items.append(item)
            yield items
        finally:
            pass


class PriorityQueue(Generic[T]):
    """Priority queue with async support."""

    def __init__(
        self,
        maxsize: int = 0,
        *,
        name: str | None = None,
        priority_key: Callable[[T], Any] | None = None,
    ) -> None:
        self._queue = asyncio.PriorityQueue(maxsize=maxsize)
        self._name = name or f"PriorityQueue-{id(self)}"
        self._priority_key = priority_key or (lambda x: x)
        self._counter = 0

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

    async def put(
        self, item: T, priority: float | None = None, timeout: float | None = None
    ) -> bool:
        if priority is None:
            priority = self._priority_key(item)

        entry = (priority, self._counter, item)
        self._counter += 1

        try:
            if timeout is not None:
                await asyncio.wait_for(self._queue.put(entry), timeout=timeout)
            else:
                await self._queue.put(entry)
            return True
        except TimeoutError:
            return False

    def put_nowait(self, item: T, priority: float | None = None) -> None:
        if priority is None:
            priority = self._priority_key(item)
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
            raise QueueEmpty("PriorityQueue get timed out")

    def get_nowait(self) -> T:
        try:
            _, _, item = self._queue.get_nowait()
            return item
        except asyncio.QueueEmpty:
            raise QueueEmpty("PriorityQueue is empty")

    def task_done(self) -> None:
        self._queue.task_done()

    async def join(self) -> None:
        await self._queue.join()


class LifoQueue(Generic[T]):
    """Last-In-First-Out queue (stack)."""

    def __init__(self, maxsize: int = 0, *, name: str | None = None) -> None:
        self._queue = asyncio.LifoQueue(maxsize=maxsize)
        self._name = name or f"LifoQueue-{id(self)}"

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

    async def put(self, item: T, timeout: float | None = None) -> bool:
        try:
            if timeout is not None:
                await asyncio.wait_for(self._queue.put(item), timeout=timeout)
            else:
                await self._queue.put(item)
            return True
        except TimeoutError:
            return False

    def put_nowait(self, item: T) -> None:
        self._queue.put_nowait(item)

    async def get(self, timeout: float | None = None) -> T:
        try:
            if timeout is not None:
                return await asyncio.wait_for(self._queue.get(), timeout=timeout)
            return await self._queue.get()
        except TimeoutError:
            raise QueueEmpty("LifoQueue get timed out")

    def get_nowait(self) -> T:
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            raise QueueEmpty("LifoQueue is empty")

    def task_done(self) -> None:
        self._queue.task_done()

    async def join(self) -> None:
        await self._queue.join()


class Channel(Generic[T]):
    """Go-style channel for communication between tasks."""

    def __init__(self, *, name: str | None = None) -> None:
        self._queue = asyncio.Queue()
        self._name = name or f"Channel-{id(self)}"
        self._closed = False
        self._receivers: set[asyncio.Task] = set()
        self._senders: set[asyncio.Task] = set()

    @property
    def name(self) -> str:
        return self._name

    @property
    def closed(self) -> bool:
        return self._closed

    async def send(self, value: T, timeout: float | None = None) -> bool:
        if self._closed:
            raise RuntimeError("Cannot send on closed channel")

        self._senders.add(asyncio.current_task())
        try:
            if timeout is not None:
                await asyncio.wait_for(self._queue.put(value), timeout=timeout)
            else:
                await self._queue.put(value)
            return True
        except TimeoutError:
            return False
        finally:
            self._senders.discard(asyncio.current_task())

    async def recv(self, timeout: float | None = None) -> T:
        if self._closed and self._queue.empty():
            raise StopAsyncIteration("Channel closed and empty")

        self._receivers.add(asyncio.current_task())
        try:
            if timeout is not None:
                return await asyncio.wait_for(self._queue.get(), timeout=timeout)
            return await self._queue.get()
        except TimeoutError:
            raise QueueEmpty("Channel recv timed out")
        finally:
            self._receivers.discard(asyncio.current_task())

    def close(self) -> None:
        self._closed = True

    def __aiter__(self) -> AsyncIterator[T]:
        return self._async_iterator()

    async def _async_iterator(self) -> AsyncIterator[T]:
        while not self._closed or not self._queue.empty():
            try:
                yield await self.recv()
            except StopAsyncIteration:
                break

    async def __aenter__(self) -> "Channel[T]":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class BufferedChannel(Generic[T]):
    """Buffered channel with configurable capacity."""

    def __init__(self, capacity: int, *, name: str | None = None) -> None:
        if capacity <= 0:
            raise ValueError("Capacity must be > 0")
        self._queue = asyncio.Queue(maxsize=capacity)
        self._name = name or f"BufferedChannel-{id(self)}"
        self._closed = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def capacity(self) -> int:
        return self._queue.maxsize

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def qsize(self) -> int:
        return self._queue.qsize()

    async def send(self, value: T, timeout: float | None = None) -> bool:
        if self._closed:
            raise RuntimeError("Cannot send on closed channel")

        try:
            if timeout is not None:
                await asyncio.wait_for(self._queue.put(value), timeout=timeout)
            else:
                await self._queue.put(value)
            return True
        except TimeoutError:
            return False

    def send_nowait(self, value: T) -> bool:
        if self._closed:
            raise RuntimeError("Cannot send on closed channel")
        try:
            self._queue.put_nowait(value)
            return True
        except asyncio.QueueFull:
            return False

    async def recv(self, timeout: float | None = None) -> T:
        if self._closed and self._queue.empty():
            raise StopAsyncIteration("Channel closed and empty")

        try:
            if timeout is not None:
                return await asyncio.wait_for(self._queue.get(), timeout=timeout)
            return await self._queue.get()
        except TimeoutError:
            raise QueueEmpty("Channel recv timed out")

    def recv_nowait(self) -> T:
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            raise QueueEmpty("Channel is empty")

    def close(self) -> None:
        self._closed = True

    def __aiter__(self) -> AsyncIterator[T]:
        return self._async_iterator()

    async def _async_iterator(self) -> AsyncIterator[T]:
        while not self._closed or not self._queue.empty():
            try:
                yield await self.recv()
            except StopAsyncIteration:
                break


class BroadcastChannel(Generic[T]):
    """Broadcast channel - sends to all current receivers."""

    def __init__(self, *, name: str | None = None) -> None:
        self._name = name or f"BroadcastChannel-{id(self)}"
        self._receivers: list[asyncio.Queue] = []
        self._closed = False
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def receiver_count(self) -> int:
        return len(self._receivers)

    def subscribe(self, maxsize: int = 0) -> asyncio.Queue:
        """Create a new subscription."""
        queue = asyncio.Queue(maxsize=maxsize)
        self._receivers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> bool:
        """Remove a subscription."""
        try:
            self._receivers.remove(queue)
            return True
        except ValueError:
            return False

    async def broadcast(self, value: T) -> int:
        """Broadcast value to all subscribers."""
        if self._closed:
            raise RuntimeError("Cannot broadcast on closed channel")

        sent = 0
        dead_receivers = []

        for receiver in self._receivers:
            try:
                receiver.put_nowait(value)
                sent += 1
            except asyncio.QueueFull:
                dead_receivers.append(receiver)

        for dead in dead_receivers:
            self._receivers.remove(dead)

        return sent

    def close(self) -> None:
        self._closed = True
        for receiver in self._receivers:
            with suppress(BaseException):
                receiver.put_nowait(StopAsyncIteration)
        self._receivers.clear()

    @asynccontextmanager
    async def subscription(self, maxsize: int = 0) -> AsyncGenerator[asyncio.Queue, None]:
        """Context manager for temporary subscription."""
        queue = self.subscribe(maxsize)
        try:
            yield queue
        finally:
            self.unsubscribe(queue)


class MultiQueue(Generic[T]):
    """Multiple queues with fan-out/fan-in patterns."""

    def __init__(self, *, name: str | None = None) -> None:
        self._name = name or f"MultiQueue-{id(self)}"
        self._queues: dict[str, asyncio.Queue] = {}
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return self._name

    async def create_queue(self, name: str, maxsize: int = 0) -> asyncio.Queue:
        async with self._lock:
            if name in self._queues:
                raise ValueError(f"Queue '{name}' already exists")
            queue = asyncio.Queue(maxsize=maxsize)
            self._queues[name] = queue
            return queue

    async def get_queue(self, name: str) -> asyncio.Queue | None:
        async with self._lock:
            return self._queues.get(name)

    async def put(self, name: str, item: T, timeout: float | None = None) -> bool:
        queue = await self.get_queue(name)
        if queue is None:
            raise KeyError(f"Queue '{name}' not found")

        try:
            if timeout is not None:
                await asyncio.wait_for(queue.put(item), timeout=timeout)
            else:
                await queue.put(item)
            return True
        except TimeoutError:
            return False

    async def get(self, name: str, timeout: float | None = None) -> T:
        queue = await self.get_queue(name)
        if queue is None:
            raise KeyError(f"Queue '{name}' not found")

        try:
            if timeout is not None:
                return await asyncio.wait_for(queue.get(), timeout=timeout)
            return await queue.get()
        except TimeoutError:
            raise QueueEmpty(f"Queue '{name}' get timed out")

    async def fan_out(self, item: T, exclude: set[str] | None = None) -> int:
        """Send item to all queues except excluded."""
        exclude = exclude or set()
        count = 0
        async with self._lock:
            for name, queue in self._queues.items():
                if name not in exclude:
                    try:
                        queue.put_nowait(item)
                        count += 1
                    except asyncio.QueueFull:
                        pass
        return count

    async def close_all(self) -> None:
        async with self._lock:
            for queue in self._queues.values():
                while not queue.empty():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
            self._queues.clear()


class RingBuffer(Generic[T]):
    """Fixed-size ring buffer with async access."""

    def __init__(self, capacity: int, *, name: str | None = None) -> None:
        if capacity <= 0:
            raise ValueError("Capacity must be > 0")
        self._capacity = capacity
        self._buffer: list[T | None] = [None] * capacity
        self._head = 0
        self._tail = 0
        self._size = 0
        self._name = name or f"RingBuffer-{id(self)}"
        self._not_full = asyncio.Condition()
        self._not_empty = asyncio.Condition()

    @property
    def name(self) -> str:
        return self._name

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        return self._size

    @property
    def full(self) -> bool:
        return self._size == self._capacity

    @property
    def empty(self) -> bool:
        return self._size == 0

    async def put(self, item: T, timeout: float | None = None, overwrite: bool = False) -> bool:
        async with self._not_full:
            while self.full and not overwrite:
                try:
                    if timeout is not None:
                        await asyncio.wait_for(self._not_full.wait(), timeout=timeout)
                    else:
                        await self._not_full.wait()
                except TimeoutError:
                    return False

            if self.full and overwrite:
                self._tail = (self._tail + 1) % self._capacity
                self._size -= 1

            self._buffer[self._head] = item
            self._head = (self._head + 1) % self._capacity
            self._size += 1
            self._not_empty.notify()
            return True

    def put_nowait(self, item: T, overwrite: bool = False) -> bool:
        if self.full and not overwrite:
            raise QueueFull("RingBuffer is full")

        if self.full and overwrite:
            self._tail = (self._tail + 1) % self._capacity
            self._size -= 1

        self._buffer[self._head] = item
        self._head = (self._head + 1) % self._capacity
        self._size += 1
        return True

    async def get(self, timeout: float | None = None) -> T:
        async with self._not_empty:
            while self.empty:
                try:
                    if timeout is not None:
                        await asyncio.wait_for(self._not_empty.wait(), timeout=timeout)
                    else:
                        await self._not_empty.wait()
                except TimeoutError:
                    raise QueueEmpty("RingBuffer get timed out")

            item = self._buffer[self._tail]
            self._buffer[self._tail] = None
            self._tail = (self._tail + 1) % self._capacity
            self._size -= 1
            self._not_full.notify()
            return item

    def get_nowait(self) -> T:
        if self.empty:
            raise QueueEmpty("RingBuffer is empty")

        item = self._buffer[self._tail]
        self._buffer[self._tail] = None
        self._tail = (self._tail + 1) % self._capacity
        self._size -= 1
        return item

    def peek(self) -> T | None:
        if self.empty:
            return None
        return self._buffer[self._tail]

    def clear(self) -> list[T]:
        items = []
        while not self.empty:
            items.append(self.get_nowait())
        return items


@asynccontextmanager
async def queue_context(
    maxsize: int = 0, *, queue_type: str = "fifo"
) -> AsyncGenerator[Queue[Any], None]:
    """Context manager for temporary queue."""
    if queue_type == "fifo":
        q = Queue(maxsize=maxsize)
    elif queue_type == "lifo":
        q = LifoQueue(maxsize=maxsize)
    elif queue_type == "priority":
        q = PriorityQueue(maxsize=maxsize)
    else:
        raise ValueError(f"Unknown queue type: {queue_type}")

    try:
        yield q
    finally:
        await q.clear()


def create_queue(
    maxsize: int = 0, *, queue_type: str = "fifo", name: str | None = None
) -> Queue | PriorityQueue | LifoQueue:
    """Factory function for creating queues."""
    if queue_type == "fifo":
        return Queue(maxsize=maxsize, name=name)
    elif queue_type == "lifo":
        return LifoQueue(maxsize=maxsize, name=name)
    elif queue_type == "priority":
        return PriorityQueue(maxsize=maxsize, name=name)
    else:
        raise ValueError(f"Unknown queue type: {queue_type}")
