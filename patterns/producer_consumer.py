"""
Producer-Consumer Patterns

Flexible producer-consumer implementations with backpressure support.
"""

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class BackpressureStrategy(Enum):
    """Strategy for handling backpressure."""

    BLOCK = "block"  # Block until space available
    DROP_OLDEST = "drop_oldest"  # Drop oldest item
    DROP_NEWEST = "drop_newest"  # Drop newest item
    RAISE = "raise"  # Raise exception


@dataclass
class ProducerConfig:
    """Configuration for producer."""

    max_queue_size: int = 100
    backpressure: BackpressureStrategy = BackpressureStrategy.BLOCK
    batch_size: int = 1
    batch_timeout: float = 0.0


@dataclass
class ConsumerConfig:
    """Configuration for consumer."""

    max_concurrent: int = 1
    timeout: float | None = None
    retry_attempts: int = 0
    retry_delay: float = 1.0


class Producer(Generic[T]):
    """Async producer with configurable backpressure."""

    def __init__(
        self,
        source: Callable[[], AsyncGenerator[T, None]],
        config: ProducerConfig | None = None,
        *,
        name: str | None = None,
    ) -> None:
        self._source = source
        self._config = config or ProducerConfig()
        self._name = name or f"Producer-{id(self)}"
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=self._config.max_queue_size)
        self._task: asyncio.Task | None = None
        self._running = False
        self._stats = {"produced": 0, "dropped": 0, "errors": 0}

    @property
    def name(self) -> str:
        return self._name

    @property
    def running(self) -> bool:
        return self._running

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def stats(self) -> dict[str, int]:
        return self._stats.copy()

    async def start(self) -> None:
        """Start the producer."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Stop the producer."""
        self._running = False
        if self._task:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task

    async def _run(self) -> None:
        try:
            async for item in self._source():
                if not self._running:
                    break

                await self._put_item(item)
        except asyncio.CancelledError:
            raise
        except Exception:
            self._stats["errors"] += 1
            raise

    async def _put_item(self, item: T) -> bool:
        try:
            if self._config.backpressure == BackpressureStrategy.BLOCK:
                await self._queue.put(item)
            elif self._config.backpressure == BackpressureStrategy.DROP_OLDEST:
                try:
                    self._queue.put_nowait(item)
                except asyncio.QueueFull:
                    self._queue.get_nowait()
                    self._queue.put_nowait(item)
                    self._stats["dropped"] += 1
            elif self._config.backpressure == BackpressureStrategy.DROP_NEWEST:
                try:
                    self._queue.put_nowait(item)
                except asyncio.QueueFull:
                    self._stats["dropped"] += 1
                    return False
            elif self._config.backpressure == BackpressureStrategy.RAISE:
                self._queue.put_nowait(item)

            self._stats["produced"] += 1
            return True
        except Exception:
            self._stats["errors"] += 1
            raise

    def get_queue(self) -> asyncio.Queue:
        """Get the underlying queue."""
        return self._queue

    async def __aenter__(self) -> "Producer[T]":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()


class Consumer(Generic[T, R]):
    """Async consumer with configurable concurrency."""

    def __init__(
        self,
        handler: Callable[[T], Awaitable[R]],
        config: ConsumerConfig | None = None,
        *,
        name: str | None = None,
    ) -> None:
        self._handler = handler
        self._config = config or ConsumerConfig()
        self._name = name or f"Consumer-{id(self)}"
        self._semaphore = asyncio.Semaphore(self._config.max_concurrent)
        self._task: asyncio.Task | None = None
        self._running = False
        self._stats = {"consumed": 0, "failed": 0, "retried": 0}

    @property
    def name(self) -> str:
        return self._name

    @property
    def running(self) -> bool:
        return self._running

    @property
    def stats(self) -> dict[str, int]:
        return self._stats.copy()

    async def consume(self, item: T) -> R:
        """Consume a single item."""
        async with self._semaphore:
            return await self._consume_with_retry(item)

    async def _consume_with_retry(self, item: T) -> R:
        last_exception = None

        for attempt in range(self._config.retry_attempts + 1):
            try:
                if self._config.timeout:
                    result = await asyncio.wait_for(
                        self._handler(item), timeout=self._config.timeout
                    )
                else:
                    result = await self._handler(item)

                self._stats["consumed"] += 1
                if attempt > 0:
                    self._stats["retried"] += attempt
                return result
            except asyncio.CancelledError:
                raise
            except Exception as e:
                last_exception = e
                if attempt < self._config.retry_attempts:
                    await asyncio.sleep(self._config.retry_delay * (attempt + 1))
                else:
                    self._stats["failed"] += 1
                    raise

        raise last_exception

    async def consume_batch(self, items: list[T]) -> list[R]:
        """Consume a batch of items concurrently."""
        tasks = [self.consume(item) for item in items]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def __aenter__(self) -> "Consumer[T, R]":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        pass


class ProducerConsumer(Generic[T, R]):
    """Combined producer-consumer pipeline."""

    def __init__(
        self, producer: Producer[T], consumer: Consumer[T, R], *, name: str | None = None
    ) -> None:
        self._producer = producer
        self._consumer = consumer
        self._name = name or f"ProducerConsumer-{id(self)}"
        self._task: asyncio.Task | None = None
        self._running = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def running(self) -> bool:
        return self._running

    async def start(self) -> None:
        """Start the pipeline."""
        if self._running:
            return
        self._running = True
        await self._producer.start()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Stop the pipeline."""
        self._running = False
        await self._producer.stop()
        if self._task:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task

    async def _run(self) -> None:
        queue = self._producer.get_queue()

        while self._running:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=1.0)
                try:
                    await self._consumer.consume(item)
                except Exception:
                    pass
                finally:
                    queue.task_done()
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def __aenter__(self) -> "ProducerConsumer[T, R]":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()


def create_producer_consumer(
    source: Callable[[], AsyncGenerator[T, None]],
    handler: Callable[[T], Awaitable[R]],
    *,
    producer_config: ProducerConfig | None = None,
    consumer_config: ConsumerConfig | None = None,
    name: str | None = None,
) -> ProducerConsumer[T, R]:
    """Factory function to create a producer-consumer pair."""
    producer = Producer(source, producer_config, name=f"{name}-producer" if name else None)
    consumer = Consumer(handler, consumer_config, name=f"{name}-consumer" if name else None)
    return ProducerConsumer(producer, consumer, name=name)


class MultiProducerConsumer(Generic[T, R]):
    """Multiple producers feeding multiple consumers."""

    def __init__(
        self,
        producers: list[Producer[T]],
        consumers: list[Consumer[T, R]],
        *,
        name: str | None = None,
    ) -> None:
        self._producers = producers
        self._consumers = consumers
        self._name = name or f"MultiProducerConsumer-{id(self)}"
        self._shared_queue = asyncio.Queue()
        self._consumer_tasks: list[asyncio.Task] = []
        self._running = False

    @property
    def name(self) -> str:
        return self._name

    async def start(self) -> None:
        if self._running:
            return
        self._running = True

        for producer in self._producers:
            await producer.start()
            # Redirect producer output to shared queue
            asyncio.create_task(self._forward_producer(producer))

        for consumer in self._consumers:
            task = asyncio.create_task(self._run_consumer(consumer))
            self._consumer_tasks.append(task)

    async def _forward_producer(self, producer: Producer[T]) -> None:
        queue = producer.get_queue()
        while self._running:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=1.0)
                await self._shared_queue.put(item)
                queue.task_done()
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def _run_consumer(self, consumer: Consumer[T, R]) -> None:
        while self._running:
            try:
                item = await asyncio.wait_for(self._shared_queue.get(), timeout=1.0)
                try:
                    await consumer.consume(item)
                except Exception:
                    pass
                finally:
                    self._shared_queue.task_done()
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def stop(self) -> None:
        self._running = False
        for producer in self._producers:
            await producer.stop()

        for task in self._consumer_tasks:
            task.cancel()
        await asyncio.gather(*self._consumer_tasks, return_exceptions=True)
        self._consumer_tasks.clear()


@asynccontextmanager
async def producer_context(
    source: Callable[[], AsyncGenerator[T, None]], **kwargs
) -> AsyncGenerator[Producer[T], None]:
    """Context manager for producer."""
    producer = Producer(source, **kwargs)
    await producer.start()
    try:
        yield producer
    finally:
        await producer.stop()


@asynccontextmanager
async def consumer_context(
    handler: Callable[[T], Awaitable[R]], **kwargs
) -> AsyncGenerator[Consumer[T, R], None]:
    """Context manager for consumer."""
    consumer = Consumer(handler, **kwargs)
    try:
        yield consumer
    finally:
        pass


class BufferedProducer(Generic[T]):
    """Producer with internal buffering for batch output."""

    def __init__(
        self,
        source: Callable[[], AsyncGenerator[T, None]],
        buffer_size: int = 100,
        flush_interval: float = 1.0,
        *,
        name: str | None = None,
    ) -> None:
        self._source = source
        self._buffer_size = buffer_size
        self._flush_interval = flush_interval
        self._name = name or f"BufferedProducer-{id(self)}"
        self._buffer: list[T] = []
        self._task: asyncio.Task | None = None
        self._running = False
        self._output_queue = asyncio.Queue()
        self._lock = asyncio.Lock()
        self._stats = {"produced": 0, "flushed": 0}

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._run())
        asyncio.create_task(self._flush_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
        await self._flush()

    async def _run(self) -> None:
        try:
            async for item in self._source():
                if not self._running:
                    break

                async with self._lock:
                    self._buffer.append(item)
                    self._stats["produced"] += 1

                    if len(self._buffer) >= self._buffer_size:
                        await self._flush()
        except asyncio.CancelledError:
            raise

    async def _flush_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self._flush_interval)
            await self._flush()

    async def _flush(self) -> None:
        async with self._lock:
            if self._buffer:
                batch = self._buffer.copy()
                self._buffer.clear()
                await self._output_queue.put(batch)
                self._stats["flushed"] += 1

    def get_output_queue(self) -> asyncio.Queue:
        return self._output_queue

    @property
    def stats(self) -> dict[str, int]:
        return self._stats.copy()


class PriorityProducerConsumer(Generic[T, R]):
    """Producer-consumer with priority support."""

    def __init__(
        self,
        source: Callable[[], AsyncGenerator[tuple[float, T], None]],
        handler: Callable[[T], Awaitable[R]],
        max_concurrent: int = 1,
        *,
        name: str | None = None,
    ) -> None:
        self._source = source
        self._handler = handler
        self._max_concurrent = max_concurrent
        self._name = name or f"PriorityProducerConsumer-{id(self)}"
        self._queue = asyncio.PriorityQueue()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._producer_task: asyncio.Task | None = None
        self._consumer_tasks: list[asyncio.Task] = []
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._producer_task = asyncio.create_task(self._run_producer())
        for i in range(self._max_concurrent):
            task = asyncio.create_task(self._run_consumer(i))
            self._consumer_tasks.append(task)

    async def stop(self) -> None:
        self._running = False
        if self._producer_task:
            self._producer_task.cancel()
        for task in self._consumer_tasks:
            task.cancel()
        await asyncio.gather(
            *(
                [self._producer_task, *self._consumer_tasks]
                if self._producer_task
                else self._consumer_tasks
            ),
            return_exceptions=True,
        )

    async def _run_producer(self) -> None:
        try:
            async for priority, item in self._source():
                if not self._running:
                    break
                await self._queue.put((priority, item))
        except asyncio.CancelledError:
            raise

    async def _run_consumer(self, worker_id: int) -> None:
        while self._running:
            try:
                _priority, item = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                async with self._semaphore:
                    with suppress(Exception):
                        await self._handler(item)
                self._queue.task_done()
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break
