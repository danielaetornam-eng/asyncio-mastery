"""
Pipeline Patterns

Composable async pipeline stages for data processing.
"""

import asyncio
import contextlib
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any, Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")
U = TypeVar("U")


class PipelineStage(Generic[T, R], ABC):
    """Base class for pipeline stages."""

    def __init__(self, name: str | None = None) -> None:
        self._name = name or self.__class__.__name__
        self._input_queue: asyncio.Queue = asyncio.Queue()
        self._output_queue: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._running = False
        self._stats = {"processed": 0, "errors": 0, "latency_sum": 0.0}

    @property
    def name(self) -> str:
        return self._name

    @property
    def input_queue(self) -> asyncio.Queue:
        return self._input_queue

    @property
    def output_queue(self) -> asyncio.Queue:
        return self._output_queue

    @property
    def running(self) -> bool:
        return self._running

    @property
    def stats(self) -> dict[str, Any]:
        return self._stats.copy()

    @abstractmethod
    async def process(self, item: T) -> R:
        """Process a single item."""
        pass

    async def _run(self) -> None:
        while self._running:
            try:
                item = await asyncio.wait_for(self._input_queue.get(), timeout=0.1)
                import time

                start = time.monotonic()
                try:
                    result = await self.process(item)
                    await self._output_queue.put(result)
                    self._stats["processed"] += 1
                except Exception:
                    self._stats["errors"] += 1
                    raise
                finally:
                    self._stats["latency_sum"] += time.monotonic() - start
                    self._input_queue.task_done()
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def start(self) -> None:
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    def __or__(self, other: "PipelineStage[R, U]") -> "Pipeline[T, U]":
        """Pipe operator for composing stages."""
        return Pipeline(self, other)

    async def __aenter__(self) -> "PipelineStage[T, R]":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()


class MapStage(PipelineStage[T, R]):
    """Map stage - transforms each item."""

    def __init__(self, func: Callable[[T], R | Awaitable[R]], *, name: str | None = None) -> None:
        super().__init__(name)
        self._func = func

    async def process(self, item: T) -> R:
        if asyncio.iscoroutinefunction(self._func):
            return await self._func(item)
        return self._func(item)


class FilterStage(PipelineStage[T, T]):
    """Filter stage - passes items that match predicate."""

    def __init__(
        self, predicate: Callable[[T], bool | Awaitable[bool]], *, name: str | None = None
    ) -> None:
        super().__init__(name)
        self._predicate = predicate

    async def process(self, item: T) -> T:
        if asyncio.iscoroutinefunction(self._predicate):
            if await self._predicate(item):
                return item
        elif self._predicate(item):
            return item
        # Item filtered out - return sentinel to skip
        raise StopAsyncIteration


class BatchStage(PipelineStage[T, list[T]]):
    """Batch stage - groups items into batches."""

    def __init__(self, batch_size: int, timeout: float = 0.0, *, name: str | None = None) -> None:
        super().__init__(name)
        self._batch_size = batch_size
        self._timeout = timeout
        self._buffer: list[T] = []

    async def process(self, item: T) -> list[T]:
        self._buffer.append(item)

        if len(self._buffer) >= self._batch_size:
            batch = self._buffer.copy()
            self._buffer.clear()
            return batch

        if self._timeout > 0:
            # Wait for more items or timeout
            try:
                await asyncio.wait_for(self._input_queue.get(), timeout=self._timeout)
                # Put it back
                await self._input_queue.put(item)
            except TimeoutError:
                pass

        if self._buffer:
            batch = self._buffer.copy()
            self._buffer.clear()
            return batch

        raise StopAsyncIteration


class FanOutStage(PipelineStage[T, tuple[str, T]]):
    """Fan-out stage - routes items to multiple outputs."""

    def __init__(
        self,
        routes: dict[str, Callable[[T], bool]],
        default: str | None = None,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name)
        self._routes = routes
        self._default = default

    async def process(self, item: T) -> tuple[str, T]:
        for route_name, predicate in self._routes.items():
            if asyncio.iscoroutinefunction(predicate):
                if await predicate(item):
                    return (route_name, item)
            elif predicate(item):
                return (route_name, item)

        if self._default:
            return (self._default, item)

        raise StopAsyncIteration


class FanInStage(PipelineStage[tuple[str, T], T]):
    """Fan-in stage - merges multiple input streams."""

    def __init__(self, *, name: str | None = None) -> None:
        super().__init__(name)
        self._streams: dict[str, asyncio.Queue] = {}

    def add_stream(self, name: str) -> asyncio.Queue:
        queue = asyncio.Queue()
        self._streams[name] = queue
        return queue

    async def process(self, item: tuple[str, T]) -> T:
        _, value = item
        return value


class ReduceStage(PipelineStage[T, R]):
    """Reduce stage - accumulates items into a single result."""

    def __init__(
        self, reducer: Callable[[R, T], R | Awaitable[R]], initial: R, *, name: str | None = None
    ) -> None:
        super().__init__(name)
        self._reducer = reducer
        self._accumulator = initial
        self._first = True

    async def process(self, item: T) -> R:
        if self._first:
            self._accumulator = item  # type: ignore
            self._first = False
            return self._accumulator

        if asyncio.iscoroutinefunction(self._reducer):
            self._accumulator = await self._reducer(self._accumulator, item)
        else:
            self._accumulator = self._reducer(self._accumulator, item)
        return self._accumulator


class WindowStage(PipelineStage[T, list[T]]):
    """Sliding window stage."""

    def __init__(self, window_size: int, step: int = 1, *, name: str | None = None) -> None:
        super().__init__(name)
        self._window_size = window_size
        self._step = step
        self._buffer: deque = deque(maxlen=window_size)
        self._count = 0

    async def process(self, item: T) -> list[T]:
        self._buffer.append(item)
        self._count += 1

        if len(self._buffer) == self._window_size and self._count % self._step == 0:
            return list(self._buffer)

        raise StopAsyncIteration


class Pipeline(Generic[T, R]):
    """Composable pipeline of stages."""

    def __init__(self, *stages: PipelineStage) -> None:
        if not stages:
            raise ValueError("Pipeline must have at least one stage")

        self._stages = list(stages)
        self._name = f"Pipeline({len(stages)} stages)"
        self._running = False

        # Connect stages
        for i in range(len(stages) - 1):
            stages[i].output_queue = stages[i + 1].input_queue

    @property
    def name(self) -> str:
        return self._name

    @property
    def stages(self) -> list[PipelineStage]:
        return self._stages

    @property
    def first_stage(self) -> PipelineStage:
        return self._stages[0]

    @property
    def last_stage(self) -> PipelineStage:
        return self._stages[-1]

    @property
    def running(self) -> bool:
        return self._running

    @property
    def input_queue(self) -> asyncio.Queue:
        return self._stages[0].input_queue

    @property
    def output_queue(self) -> asyncio.Queue:
        return self._stages[-1].output_queue

    async def start(self) -> None:
        if not self._running:
            self._running = True
            for stage in self._stages:
                await stage.start()

    async def stop(self) -> None:
        self._running = False
        for stage in reversed(self._stages):
            await stage.stop()

    async def put(self, item: T) -> None:
        await self.input_queue.put(item)

    async def get(self, timeout: float | None = None) -> R:
        try:
            if timeout:
                return await asyncio.wait_for(self.output_queue.get(), timeout=timeout)
            return await self.output_queue.get()
        except TimeoutError:
            raise TimeoutError("Pipeline get timed out")

    async def process(self, item: T) -> R:
        """Process single item through pipeline."""
        await self.put(item)
        return await self.get()

    async def process_batch(self, items: list[T]) -> list[R]:
        """Process batch through pipeline."""
        for item in items:
            await self.put(item)

        results = []
        for _ in items:
            results.append(await self.get())
        return results

    def __or__(self, other: PipelineStage[R, U]) -> "Pipeline[T, U]":
        return Pipeline(*self._stages, other)

    async def __aenter__(self) -> "Pipeline[T, R]":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()

    def stats(self) -> dict[str, Any]:
        return {
            "stages": [s.stats for s in self._stages],
            "total_processed": sum(s.stats["processed"] for s in self._stages),
            "total_errors": sum(s.stats["errors"] for s in self._stages),
        }


class AsyncPipeline(Generic[T, R]):
    """High-level async pipeline with source and sink."""

    def __init__(
        self,
        source: AsyncGenerator[T, None],
        pipeline: Pipeline[T, R],
        *,
        name: str | None = None,
        max_concurrent: int = 10,
    ) -> None:
        self._source = source
        self._pipeline = pipeline
        self._name = name or "AsyncPipeline"
        self._max_concurrent = max_concurrent
        self._task: asyncio.Task | None = None
        self._running = False
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def start(self) -> None:
        if not self._running:
            self._running = True
            await self._pipeline.start()
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        await self._pipeline.stop()

    async def _run(self) -> None:
        try:
            async for item in self._source:
                if not self._running:
                    break
                await self._pipeline.put(item)
        except asyncio.CancelledError:
            raise

    async def results(self) -> AsyncGenerator[R, None]:
        """Get results as async generator."""
        while self._running or not self._pipeline.output_queue.empty():
            try:
                result = await asyncio.wait_for(self._pipeline.get(), timeout=0.1)
                yield result
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def collect(self, max_results: int | None = None) -> list[R]:
        """Collect all results into a list."""
        results = []
        async for result in self.results():
            results.append(result)
            if max_results and len(results) >= max_results:
                break
        return results

    async def __aenter__(self) -> "AsyncPipeline[T, R]":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()


def pipeline(*stages: PipelineStage) -> Pipeline:
    """Create a pipeline from stages."""
    return Pipeline(*stages)


def map_stage(func: Callable[[T], R | Awaitable[R]], **kwargs) -> MapStage[T, R]:
    """Create a map stage."""
    return MapStage(func, **kwargs)


def filter_stage(predicate: Callable[[T], bool | Awaitable[bool]], **kwargs) -> FilterStage[T]:
    """Create a filter stage."""
    return FilterStage(predicate, **kwargs)


def batch_stage(batch_size: int, timeout: float = 0.0, **kwargs) -> BatchStage[T]:
    """Create a batch stage."""
    return BatchStage(batch_size, timeout, **kwargs)


def reduce_stage(
    reducer: Callable[[R, T], R | Awaitable[R]], initial: R, **kwargs
) -> ReduceStage[T, R]:
    """Create a reduce stage."""
    return ReduceStage(reducer, initial, **kwargs)


def window_stage(window_size: int, step: int = 1, **kwargs) -> WindowStage[T]:
    """Create a window stage."""
    return WindowStage(window_size, step, **kwargs)


class PipelineBuilder(Generic[T]):
    """Fluent builder for pipelines."""

    def __init__(self) -> None:
        self._stages: list[PipelineStage] = []
        self._source: AsyncGenerator[T, None] | None = None

    def source(self, gen: AsyncGenerator[T, None]) -> "PipelineBuilder[T]":
        self._source = gen
        return self

    def map(self, func: Callable[[T], R | Awaitable[R]]) -> "PipelineBuilder[R]":
        self._stages.append(MapStage(func))
        return self  # type: ignore

    def filter(self, predicate: Callable[[T], bool | Awaitable[bool]]) -> "PipelineBuilder[T]":
        self._stages.append(FilterStage(predicate))
        return self

    def batch(self, batch_size: int, timeout: float = 0.0) -> "PipelineBuilder[list[T]]":
        self._stages.append(BatchStage(batch_size, timeout))
        return self  # type: ignore

    def window(self, window_size: int, step: int = 1) -> "PipelineBuilder[list[T]]":
        self._stages.append(WindowStage(window_size, step))
        return self  # type: ignore

    def reduce(
        self, reducer: Callable[[R, T], R | Awaitable[R]], initial: R
    ) -> "PipelineBuilder[R]":
        self._stages.append(ReduceStage(reducer, initial))
        return self  # type: ignore

    def build(self) -> Pipeline:
        if not self._stages:
            raise ValueError("Pipeline must have at least one stage")
        return Pipeline(*self._stages)

    def build_async(self, max_concurrent: int = 10) -> AsyncPipeline:
        if not self._source:
            raise ValueError("Source required for async pipeline")
        if not self._stages:
            raise ValueError("Pipeline must have at least one stage")
        return AsyncPipeline(self._source, self.build(), max_concurrent=max_concurrent)


def create_pipeline() -> PipelineBuilder:
    """Create a new pipeline builder."""
    return PipelineBuilder()
