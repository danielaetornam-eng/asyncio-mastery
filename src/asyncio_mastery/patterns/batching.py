"""
Batching - Accumulate items and process in batches.

THE PROBLEM
-----------
Many operations are much cheaper in bulk: one ``INSERT ... VALUES
(...), (...), (...)`` beats 100 single inserts; one bulk API call beats
100 HTTP round-trips. But your code receives items one at a time. How do
you accumulate items into efficient batches without delaying each caller
too long?

REAL-WORLD ANALOGY
------------------
A **hotel shuttle bus**. Sending a bus per passenger is wasteful, so the
bus departs either when it is full (``max_size``) or after a scheduled
wait (``max_wait``) — whichever comes first. Passengers know their final
destination individually, but they travel together. Here, each passenger
even gets their own "ticket stub" (a Future) that gets stamped with
their individual result when the batch returns.

HOW IT WORKS
------------
Each call to :meth:`Batcher.add` returns a Future. The Batcher bundles
items until either the batch is full (flush now) or a timer fires
(flush the partial batch). The user-supplied processor then handles the
whole list at once and its results are distributed back to each waiting
Future — so callers still write simple one-item-at-a-time code.

WHY ASYNC PRIMITIVES ARE NEEDED
-------------------------------
* **Futures** decouple "caller wants a result" from "batch may contain
  hundreds of callers' items" — each caller awaits only its own Future.
* **A lock** guards the "current batch" pointer: without it, two
  coroutines adding simultaneously could both see batch-is-full and
  both try to flush, corrupting the batch. The lock makes the
  read-modify-write sequence atomic relative to other tasks.
* **A background flush task** implements the ``max_wait`` timeout.

WHEN TO USE IT
--------------
* High-throughput writes (database upserts, metrics, logs).
* Calls to APIs with bulk endpoints or per-request overhead.
* Anywhere you can trade a little latency for a lot of throughput.

Minimal example
---------------
>>> import asyncio
>>> async def bulk_insert(items: list[int]) -> list[str]:
...     return [f"saved-{x}" for x in items]
>>> async def main() -> None:
...     batcher = Batcher(bulk_insert, BatchConfig(max_size=2, max_wait=0.1))
...     print(await asyncio.gather(batcher.add(1), batcher.add(2)))
...     # both calls resolved together when the batch flushed
>>> asyncio.run(main())
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class BatchConfig:
    """Knobs that decide when a batch gets flushed.

    A flush happens on *whichever trigger fires first*:

    Attributes:
        max_size: Flush immediately once the batch holds this many items.
            Guarantees throughput does not wait forever for a timer.
        max_wait: Seconds of patience. If the batch is not full by then,
            flush whatever has accumulated (bounded latency guarantee).
        max_batch_size: Optional downstream cap (passed to processors
            that chunk very large batches; None = no cap).
    """

    max_size: int = 100
    max_wait: float = 0.1  # seconds
    max_batch_size: int | None = None


@dataclass
class Batch(Generic[T, R]):
    """One queued batch of items plus each caller's result Future.

    Each item was added by a different coroutine that is *awaiting its
    own Future*. When the processor finishes the batch, results are
    fanned back out one-for-one.

    Type Parameters:
        T: The item type.

    Attributes:
        items: The queued items (sent to the processor as one list).
        created_at: ``time.monotonic()`` when the batch was opened —
            used for the age metric.
        futures: Per-item Futures, parallel to ``items``; index i of
            ``futures`` belongs to index i of ``items``.
    """

    items: list[T] = field(default_factory=list)
    created_at: float = field(default_factory=time.monotonic)
    futures: list[asyncio.Future[R]] = field(default_factory=list)

    def add(self, item: T) -> asyncio.Future[R]:
        """Add item and return future for result.

        The caller keeps the Future; when the batch eventually flushes,
        this Future resolves with the caller's individual result.

        Args:
            item: The work item to include in this batch.

        Returns:
            asyncio.Future: A Future that will resolve with this item's
            individual result once the batch is processed.
        """
        future: asyncio.Future[R] = asyncio.Future()
        self.items.append(item)
        self.futures.append(future)
        return future

    @property
    def size(self) -> int:
        """How many items are currently queued in this batch.

        Returns:
            int: ``len(items)`` (also == ``len(futures)``).
        """
        return len(self.items)

    @property
    def age(self) -> float:
        """Seconds this batch has been accumulating.

        Returns:
            float: ``time.monotonic() - created_at``.
        """
        return time.monotonic() - self.created_at

    def set_results(self, results: list[R]) -> None:
        """Set results for all futures.

        Called by the Batcher after the bulk processor returns. Result i
        is delivered to Future i — that is how each original caller gets
        *their* answer from the shared batch.

        Args:
            results: Results in the same order as ``items``.
        """
        for future, result in zip(self.futures, results):
            if not future.done():
                future.set_result(result)

    def set_exception(self, exc: Exception) -> None:
        """Set exception for all futures.

        If the batch processor itself blew up (network down, bad SQL),
        *every caller* of this batch needs to see the failure.

        Args:
            exc: The exception the processor raised.
        """
        for future in self.futures:
            if not future.done():
                future.set_exception(exc)


class Batcher(Generic[T, R]):
    """Batch items and process together.

    Accepts items via :meth:`add`, and flushes (calls the batch
    processor) when the batch reaches ``max_size`` or ages past
    ``max_wait``. All state mutations happen under a lock because many
    coroutines call ``add`` concurrently.

    Type Parameters:
        T: Item type. R: Per-item result type.

    Example:
        >>> batcher = Batcher(bulk_insert, BatchConfig(max_size=50))
        >>> result = await batcher.add(row)  # resolves after a flush
    """

    def __init__(
        self, processor: Callable[[list[T]], Awaitable[list[R]]], config: BatchConfig | None = None
    ) -> None:
        """Create a batcher.

        Args:
            processor: Async function taking the whole list of items and
                returning one result per item, in the same order.
            config: Flush triggers; defaults to :class:`BatchConfig()`.
        """
        self._processor = processor
        self._config = config or BatchConfig()
        self._current_batch: Batch[T, R] | None = None
        # The lock makes "check batch, add item, maybe flush" atomic:
        # without it two concurrent add() calls could both decide the
        # batch isn't full, leading to oversized batches or lost flushes.
        self._lock = asyncio.Lock()
        self._flush_task: asyncio.Task[Any] | None = None

    async def add(self, item: T) -> R:
        """Add item and wait for batch result.

        Appends to the current batch. If that fills it, flushes
        immediately *in the background* (a new task) so this coroutine
        is not stuck doing the whole batch's work. Otherwise starts the
        auto-flush timer if this item opened a fresh batch.

        Args:
            item: The item to process.

        Returns:
            R: This item's individual result, delivered via Future once
            the containing batch is processed.

        Raises:
            Exception: Whatever the batch processor raised — the failure
            is broadcast to every Future in that batch.
        """
        async with self._lock:
            if self._current_batch is None:
                # First item of a new batch: open it and start the
                # max_wait countdown timer.
                self._current_batch = Batch[T, R]()
                self._flush_task = asyncio.create_task(self._auto_flush())

            future = self._current_batch.add(item)

            # Check if batch is full
            if self._current_batch.size >= self._config.max_size:
                batch = self._current_batch
                self._current_batch = None
                if self._flush_task:
                    # Full batch flushes now; no need for the timer.
                    self._flush_task.cancel()
                # Fire-and-forget: process in background so add() can
                # return the Future to its caller immediately.
                asyncio.create_task(self._process_batch(batch))

        return await future

    async def _auto_flush(self) -> None:
        """Flush batch after max_wait.

        This is the latency-guarantee mechanism: even if the batch never
        fills, callers are only ever asked to wait ``max_wait`` seconds
        for their result. Timer is cancelled if the batch fills first
        (see :meth:`add`).
        """
        await asyncio.sleep(self._config.max_wait)
        async with self._lock:
            if self._current_batch and self._current_batch.size > 0:
                batch = self._current_batch
                self._current_batch = None
                asyncio.create_task(self._process_batch(batch))

    async def _process_batch(self, batch: Batch[T, R]) -> None:
        """Process a batch.

        Invokes the user-supplied bulk processor and fans the results
        out to each caller's Future. A result-count mismatch means the
        processor is buggy and we fail loudly; any other processor
        exception is broadcast to all waiters as well.

        Args:
            batch: The batch to process.
        """
        try:
            results = await self._processor(batch.items)
            if len(results) != batch.size:
                raise ValueError(
                    f"Processor returned {len(results)} results for {batch.size} items"
                )
            batch.set_results(results)
        except Exception as e:
            batch.set_exception(e)

    async def flush(self) -> None:
        """Force flush current batch.

        Escape hatch for callers that cannot accept ``max_wait`` latency
        right now (e.g., at shutdown): process whatever has accumulated.
        Unlike the auto-flush, this *awaits* the processor so you know
        the work completed.
        """
        async with self._lock:
            if self._current_batch and self._current_batch.size > 0:
                batch = self._current_batch
                self._current_batch = None
                if self._flush_task:
                    self._flush_task.cancel()
                await self._process_batch(batch)

    async def __aenter__(self) -> "Batcher[T, R]":
        """Enter context (single-use friendly).

        Returns:
            Batcher: This instance.
        """
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit context: flush anything still pending."""
        await self.flush()


class BatchProcessor(Generic[T, R]):
    """High-level batch processor with multiple strategies.

    A friendlier facade over :class:`Batcher` that takes milliseconds
    instead of seconds and exposes ``process``/``process_many`` verbs.

    Example:
        >>> proc = BatchProcessor(bulk_handler, max_batch_size=20, max_wait_ms=50)
        >>> await proc.process_many(range(100))  # auto-chunked and flushed
    """

    def __init__(
        self,
        processor: Callable[[list[T]], Awaitable[list[R]]],
        *,
        max_batch_size: int = 100,
        max_wait_ms: float = 100,
    ) -> None:
        """Create the facade (wraps a ``Batcher``).

        Args:
            processor: Bulk async function (list in, list out, 1:1).
            max_batch_size: Flush trigger — items per batch.
            max_wait_ms: Flush trigger — max age of a batch, in ms.
        """
        self._batcher = Batcher(processor, BatchConfig(max_batch_size, max_wait_ms / 1000))

    async def process(self, item: T) -> R:
        """Queue one item and await its individual result.

        Args:
            item: The item to process.

        Returns:
            R: This item's result after its batch flushes.
        """
        return await self._batcher.add(item)

    async def process_many(self, items: list[T]) -> list[R]:
        """Queue many items concurrently and collect all results.

        Args:
            items: The items to process.

        Returns:
            list[R]: Results aligned with the input order.
        """
        return await asyncio.gather(*[self.process(item) for item in items])

    async def flush(self) -> None:
        """Force-flush whatever is currently pending."""
        await self._batcher.flush()


async def demonstrate_batching() -> None:
    """Run a self-contained demo of batching.

    Notice in "Rapid Adds": 10 individual calls but only ~4 processor
    invocations — that is the whole point of the pattern.
    """
    print("=== Batching Demo ===")

    processed_batches: list[list[int]] = []

    async def batch_processor(items: list[int]) -> list[str]:
        await asyncio.sleep(0.01)  # Simulate batch work
        processed_batches.append(items.copy())
        return [f"result-{x}" for x in items]

    config = BatchConfig(max_size=3, max_wait=0.05)
    batcher = Batcher(batch_processor, config)

    # Add items rapidly
    print("\n--- Rapid Adds ---")
    start = time.monotonic()
    tasks = [batcher.add(i) for i in range(10)]
    results = await asyncio.gather(*tasks)
    elapsed = time.monotonic() - start

    print(f"Processed {len(results)} items in {elapsed:.3f}s")
    print(f"Batches: {processed_batches}")
    print(f"Results: {results[:5]}...")

    # Manual flush
    print("\n--- Manual Flush ---")
    processed_batches.clear()
    batcher2 = Batcher(batch_processor, BatchConfig(max_size=5, max_wait=1.0))

    await batcher2.add(1)
    await batcher2.add(2)
    await batcher2.flush()  # Force flush partial batch

    await batcher2.add(3)
    await batcher2.add(4)
    await batcher2.add(5)
    # Auto-flush after max_wait

    await asyncio.sleep(0.1)
    print(f"Batches: {processed_batches}")

    # BatchProcessor
    print("\n--- BatchProcessor ---")
    processor = BatchProcessor(batch_processor, max_batch_size=3, max_wait_ms=50)
    results = await processor.process_many(list(range(7)))
    print(f"Results: {results}")


if __name__ == "__main__":
    asyncio.run(demonstrate_batching())
