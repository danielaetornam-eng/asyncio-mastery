"""
Producer-Consumer - Classic pattern with async queues.

THE PROBLEM
-----------
Producers (code that *creates* work) and consumers (code that *does* work)
often run at different speeds. If a producer floods a consumer directly,
the consumer gets overwhelmed. If they are wired together with plain
function calls, you lose the ability to run them concurrently.

REAL-WORLD ANALOGY
------------------
Think of a restaurant: **chefs** (consumers) cook dishes and **waiters**
(producers) pin order tickets onto a rail (the queue). Chefs take orders
off the rail at their own pace. During a rush, the rail fills up — and
waiters simply have to wait for space (that is **backpressure**: when the
downstream side cannot keep up, the upstream side is slowed down instead
of everything crashing).

JARGON
------
* **Backpressure**: The resistance a system applies when it is producing
  data faster than it can be consumed. Here, a bounded ``asyncio.Queue``
  applies backpressure automatically: ``await queue.put(item)`` *pauses*
  the producer (without blocking the event loop!) until a consumer frees
  up a slot.

WHY ASYNC PRIMITIVES ARE NEEDED
-------------------------------
``asyncio.Queue`` is the perfect middleman: producers ``await put()``
(suspending if the queue is full), consumers ``await get()`` (suspending
if the queue is empty). No busy-waiting, no CPU burn — the event loop
only wakes a task when work is actually available.

WHEN TO USE IT
--------------
* Decoupling fast event ingress (webhooks, sensors) from slower handlers.
* You want graceful shutdown: stop accepting new work, drain the queue,
  let in-flight work finish.

Minimal example
---------------
>>> import asyncio
>>> async def consume(item: int) -> None:
...     print("got", item)
>>> async def main() -> None:
...     async with producer_consumer(consume, num_consumers=2) as pc:
...         await pc.produce_many(range(10))
...         while pc.queue_size > 0:
...             await asyncio.sleep(0.01)
>>> asyncio.run(main())
"""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

T = TypeVar("T")


class ShutdownSignal(Enum):
    """How to shut the producer-consumer system down.

    GRACEFUL = "graceful"  # Finish current work
    IMMEDIATE = "immediate"  # Stop now
    """

    GRACEFUL = "graceful"  # Finish current work
    IMMEDIATE = "immediate"  # Stop now


@dataclass
class ProducerConsumerStats:
    """Observable statistics about the running system.

    Attributes:
        produced: Total items ever placed on the queue.
        consumed: Items successfully processed by the consumer function.
        errors: How many items raised an exception while being consumed.
        queue_size: Items currently sitting in the queue.
        active_producers: Producer tasks running right now.
        active_consumers: Consumer tasks running right now.
    """

    produced: int = 0
    consumed: int = 0
    errors: int = 0
    queue_size: int = 0
    active_producers: int = 0
    active_consumers: int = 0


class ProducerConsumer(Generic[T]):
    """Producer-consumer with managed lifecycle.

    Owns a bounded :class:`asyncio.Queue`, a fleet of consumer tasks, and
    the start/stop lifecycle. Producers call :meth:`produce`; consumers
    run the user-provided ``consumer_func`` in a loop.

    Type Parameters:
        T: The type of item flowing through the queue.

    Example:
        >>> async def handle(item: str) -> None: ...
        >>> pc = ProducerConsumer(handle, max_queue_size=10, num_consumers=2)
        >>> await pc.start()
        >>> await pc.produce("hello")
        >>> await pc.stop()  # drains remaining items (graceful default)
    """

    def __init__(
        self,
        consumer_func: Callable[[T], Awaitable[None]],
        *,
        max_queue_size: int = 100,
        num_consumers: int = 4,
        name: str = "producer-consumer",
    ) -> None:
        """Set up the system (does not start consumers yet).

        Args:
            consumer_func: Async function called once per dequeued item.
            max_queue_size: Upper bound of the queue. When full,
                :meth:`produce` suspends producers — backpressure in action.
            num_consumers: How many concurrent consumer tasks to run. More
                consumers = more parallelism, but they compete for the
                same queue.
            name: Prefix for task names, useful for debugging.
        """
        self._consumer_func = consumer_func
        self._max_queue_size = max_queue_size
        self._num_consumers = num_consumers
        self._name = name
        self._queue: asyncio.Queue[T] = asyncio.Queue(maxsize=max_queue_size)
        self._consumer_tasks: list[asyncio.Task] = []
        self._producer_tasks: list[asyncio.Task] = []
        self._running = False
        # An Event acts like a broadcast flag: one set() wakes every task
        # waiting on it. Here it signals "shut down" to all consumers.
        self._shutdown_event = asyncio.Event()
        self._stats = ProducerConsumerStats()

    async def start(self) -> None:
        """Start consumers.

        Idempotent: calling it twice is safe (second call is a no-op).

        Creates ``num_consumers`` background tasks, each pulling items off
        the shared queue and invoking ``consumer_func``.
        """
        if self._running:
            return
        self._running = True
        self._shutdown_event.clear()

        for i in range(self._num_consumers):
            task = asyncio.create_task(self._consumer_loop(i), name=f"{self._name}-consumer-{i}")
            self._consumer_tasks.append(task)

    async def stop(self, signal: ShutdownSignal = ShutdownSignal.GRACEFUL) -> None:
        """Stop producers and consumers.

        With ``GRACEFUL``, consumers finish the items already in the queue
        (the loop condition ``while self._running or not self._queue.empty()``
        keeps them going until drained). With ``IMMEDIATE``, tasks are
        cancelled mid-item.

        Args:
            signal: GRACEFUL (default) to drain, IMMEDIATE to cancel now.
        """
        self._running = False
        self._shutdown_event.set()

        if signal == ShutdownSignal.IMMEDIATE:
            # Cancel all tasks
            for task in self._consumer_tasks + self._producer_tasks:
                task.cancel()

        # Wait for consumers
        if self._consumer_tasks:
            # return_exceptions=True so one misbehaving consumer shutdown
            # doesn't prevent us from collecting the others.
            await asyncio.gather(*self._consumer_tasks, return_exceptions=True)
        self._consumer_tasks.clear()
        self._producer_tasks.clear()

    async def produce(self, item: T) -> None:
        """Produce item (blocks if queue full).

        "Blocks" here means *asynchronously waits* — your producer task
        suspends and the event loop keeps running other tasks. That
        suspension IS backpressure: production naturally slows to the
        pace of consumption.

        Args:
            item: The work item to enqueue.

        Raises:
            RuntimeError: If the system has not been started.
        """
        if not self._running:
            raise RuntimeError("Not running")
        await self._queue.put(item)
        self._stats.produced += 1
        self._stats.queue_size = self._queue.qsize()

    async def produce_many(self, items: list[T]) -> None:
        """Produce multiple items.

        Args:
            items: Items enqueued one by one (each obeying backpressure).
        """
        for item in items:
            await self.produce(item)

    def try_produce(self, item: T) -> bool:
        """Try to produce without blocking.

        Useful when you would rather *drop* or *handle* overflow yourself
        than have the producer suspend.

        Args:
            item: The work item to enqueue.

        Returns:
            bool: True if the item was accepted; False if the system is
            not running or the queue is full (no backpressure applied).
        """
        if not self._running or self._queue.full():
            return False
        self._queue.put_nowait(item)
        self._stats.produced += 1
        self._stats.queue_size = self._queue.qsize()
        return True

    async def _consumer_loop(self, worker_id: int) -> None:
        """Consumer worker loop.

        Repeatedly pulls one item off the queue and processes it. The
        0.1s ``wait_for`` timeout is a polling trick: it lets the loop
        re-check the ``self._running`` flag / shutdown condition
        regularly instead of sleeping forever on an empty queue.

        ``queue.task_done()`` is the queue's bookkeeping for each item
        retrieved — a good habit if you ever switch to ``queue.join()``.

        Args:
            worker_id: Numeric id of this consumer (used in error logs).
        """
        self._stats.active_consumers += 1
        try:
            while self._running or not self._queue.empty():
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                except TimeoutError:
                    continue

                try:
                    await self._consumer_func(item)
                    self._stats.consumed += 1
                except Exception as e:
                    self._stats.errors += 1
                    print(f"Consumer {worker_id} error: {e}")
                finally:
                    self._queue.task_done()
                    self._stats.queue_size = self._queue.qsize()
        finally:
            self._stats.active_consumers -= 1

    @property
    def stats(self) -> ProducerConsumerStats:
        """Live statistics snapshot.

        Returns:
            ProducerConsumerStats: The shared stats object (updated live).
        """
        return self._stats

    @property
    def queue_size(self) -> int:
        """Number of items currently waiting in the queue.

        Returns:
            int: Items enqueued but not yet consumed.
        """
        return self._queue.qsize()


@asynccontextmanager
async def producer_consumer(
    consumer_func: Callable[[T], Awaitable[None]],
    *,
    max_queue_size: int = 100,
    num_consumers: int = 4,
) -> AsyncIterator[ProducerConsumer[T]]:
    """Context manager for producer-consumer.

    Starts the system on entry, gracefully drains it on exit — so you can
    never forget to shut down cleanly.

    Args:
        consumer_func: Async function called per item.
        max_queue_size: Queue capacity (backpressure threshold).
        num_consumers: Number of concurrent consumer tasks.

    Yields:
        ProducerConsumer: The running system, ready for :meth:`produce`.

    Example:
        >>> async with producer_consumer(handle, num_consumers=2) as pc:
        ...     await pc.produce_many(items)
        ...     while pc.queue_size > 0:
        ...         await asyncio.sleep(0.01)
    """
    pc = ProducerConsumer(consumer_func, max_queue_size=max_queue_size, num_consumers=num_consumers)
    await pc.start()
    try:
        yield pc
    finally:
        await pc.stop()


async def demonstrate_producer_consumer() -> None:
    """Run a self-contained demo of the producer-consumer pattern.

    Covers: basic usage, backpressure with a tiny queue, and multiple
    concurrent producers. Run ``python -m asyncio_mastery.patterns.producer_consumer``.
    """
    print("=== Producer-Consumer Demo ===")

    processed: list[int] = []

    async def consumer(item: int) -> None:
        await asyncio.sleep(0.01)
        processed.append(item)

    # Basic usage
    print("\n--- Basic ---")
    async with producer_consumer(consumer, max_queue_size=10, num_consumers=2) as pc:
        await pc.produce_many(range(20))
        # Wait for queue to drain
        while pc.queue_size > 0:
            await asyncio.sleep(0.01)

    print(f"Processed: {processed}")
    print(f"Stats: produced={pc.stats.produced}, consumed={pc.stats.consumed}")

    # Backpressure
    print("\n--- Backpressure ---")
    processed.clear()

    async with producer_consumer(consumer, max_queue_size=5, num_consumers=1) as pc:
        # Produce faster than consume
        for i in range(10):
            if not pc.try_produce(i):
                print("  Queue full, waiting...")
                await asyncio.sleep(0.05)
                await pc.produce(i)

        while pc.queue_size > 0:
            await asyncio.sleep(0.01)

    print(f"Processed: {processed}")

    # Multiple producers
    print("\n--- Multiple Producers ---")
    processed.clear()

    async def producer(pc: ProducerConsumer[int], start: int, count: int) -> None:
        for i in range(start, start + count):
            await pc.produce(i)

    async with producer_consumer(consumer, max_queue_size=20, num_consumers=3) as pc:
        await asyncio.gather(producer(pc, 0, 10), producer(pc, 100, 10), producer(pc, 200, 10))
        while pc.queue_size > 0:
            await asyncio.sleep(0.01)

    print(f"Processed {len(processed)} items from 3 producers")


if __name__ == "__main__":
    asyncio.run(demonstrate_producer_consumer())
