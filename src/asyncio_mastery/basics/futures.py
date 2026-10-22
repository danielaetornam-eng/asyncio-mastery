"""
Futures and Executor Integration

Advanced Future patterns and executor utilities.
"""

import asyncio
import concurrent.futures
import functools
import inspect
import os
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine, Iterable
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager, suppress
from typing import Any, Generic, TypeVar, cast

T = TypeVar("T")
R = TypeVar("R")
S = TypeVar("S")


class Future(Generic[T]):
    """Enhanced Future with additional utilities."""

    def __init__(self, *, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._loop = loop or asyncio.get_running_loop()
        self._future: asyncio.Future[T] = self._loop.create_future()
        self._callbacks: list[Callable[[asyncio.Future[Any]], None]] = []

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        return self._loop

    def __await__(self) -> Any:
        return self._future.__await__()

    def add_done_callback(self, callback: Callable[[asyncio.Future[Any]], None]) -> None:
        self._callbacks.append(callback)
        self._future.add_done_callback(callback)

    def remove_done_callback(self, callback: Callable[[asyncio.Future[Any]], None]) -> int:
        try:
            self._callbacks.remove(callback)
            return self._future.remove_done_callback(callback)
        except ValueError:
            return 0

    def cancel(self, msg: str | None = None) -> bool:
        return self._future.cancel(msg)

    def cancelled(self) -> bool:
        return self._future.cancelled()

    def done(self) -> bool:
        return self._future.done()

    def result(self) -> T:
        return self._future.result()

    def exception(self) -> BaseException | None:
        return self._future.exception()

    def set_result(self, result: T) -> None:
        self._future.set_result(result)

    def set_exception(self, exception: BaseException) -> None:
        self._future.set_exception(exception)

    def get_loop(self) -> asyncio.AbstractEventLoop:
        return self._future.get_loop()


def ensure_future(
    awaitable: Awaitable[T], *, loop: asyncio.AbstractEventLoop | None = None
) -> asyncio.Future[T]:
    """Ensure an awaitable is a Future."""
    return asyncio.ensure_future(awaitable, loop=loop)


def wrap_future(
    future: concurrent.futures.Future[T], *, loop: asyncio.AbstractEventLoop | None = None
) -> asyncio.Future[T]:
    """Wrap a concurrent.futures.Future as an asyncio Future."""
    return asyncio.wrap_future(future, loop=loop)


class ExecutorManager:
    """Manage executors with lifecycle control."""

    def __init__(self) -> None:
        self._thread_executor: ThreadPoolExecutor | None = None
        self._process_executor: ProcessPoolExecutor | None = None
        self._custom_executors: dict[str, Executor] = {}
        self._lock = asyncio.Lock()

    @property
    def thread_executor(self) -> ThreadPoolExecutor:
        if self._thread_executor is None:
            self._thread_executor = ThreadPoolExecutor(
                max_workers=min(32, (os.cpu_count() or 1) + 4), thread_name_prefix="asyncio_mastery"
            )
        return self._thread_executor

    @property
    def process_executor(self) -> ProcessPoolExecutor:
        if self._process_executor is None:
            self._process_executor = ProcessPoolExecutor(max_workers=os.cpu_count() or 1)
        return self._process_executor

    def get_executor(self, name: str) -> Executor | None:
        return self._custom_executors.get(name)

    def register_executor(self, name: str, executor: Executor) -> None:
        self._custom_executors[name] = executor

    async def run_in_threadpool(self, func: Callable[..., T], *args: object, **kwargs: object) -> T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.thread_executor, functools.partial(func, *args, **kwargs)
        )

    async def run_in_processpool(
        self, func: Callable[..., T], *args: object, **kwargs: object
    ) -> T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.process_executor, functools.partial(func, *args, **kwargs)
        )

    async def run_in_executor(
        self, executor_name: str, func: Callable[..., T], *args: object, **kwargs: object
    ) -> T:
        executor = self._custom_executors.get(executor_name)
        if executor is None:
            raise ValueError(f"Executor '{executor_name}' not found")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, functools.partial(func, *args, **kwargs))

    async def map_in_threadpool(
        self,
        func: Callable[..., T],
        *iterables: Iterable[object],
        timeout: float | None = None,
        chunksize: int = 1,
    ) -> list[T]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.thread_executor,
            lambda: list(
                self.thread_executor.map(func, *iterables, timeout=timeout, chunksize=chunksize)
            ),
        )

    async def map_in_processpool(
        self,
        func: Callable[..., T],
        *iterables: Iterable[Any],
        timeout: float | None = None,
        chunksize: int = 1,
    ) -> list[T]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.process_executor,
            lambda: list(
                self.process_executor.map(func, *iterables, timeout=timeout, chunksize=chunksize)
            ),
        )

    async def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        """Shutdown all executors."""
        async with self._lock:
            if self._thread_executor:
                self._thread_executor.shutdown(wait=wait, cancel_futures=cancel_futures)
                self._thread_executor = None
            if self._process_executor:
                self._process_executor.shutdown(wait=wait, cancel_futures=cancel_futures)
                self._process_executor = None
            for executor in self._custom_executors.values():
                executor.shutdown(wait=wait, cancel_futures=cancel_futures)
            self._custom_executors.clear()


_global_executor_manager = ExecutorManager()


def get_executor_manager() -> ExecutorManager:
    """Get the global executor manager."""
    return _global_executor_manager


async def run_in_threadpool(func: Callable[..., T], *args: object, **kwargs: object) -> T:
    """Run a function in the global thread pool."""
    return await _global_executor_manager.run_in_threadpool(func, *args, **kwargs)


async def run_in_processpool(func: Callable[..., T], *args: object, **kwargs: object) -> T:
    """Run a function in the global process pool."""
    return await _global_executor_manager.run_in_processpool(func, *args, **kwargs)


async def run_in_executor(
    executor: Executor | None, func: Callable[..., T], *args: object, **kwargs: object
) -> T:
    """Run a function in a specific executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, functools.partial(func, *args, **kwargs))


class AsyncExecutor(Generic[T]):
    """Async wrapper for synchronous iterators/generators."""

    def __init__(
        self, iterator: Iterable[T], *, executor: Executor | None = None, prefetch: int = 1
    ) -> None:
        self._iterator = iter(iterator)
        self._executor = executor
        self._prefetch = prefetch
        self._buffer: list[T] = []
        self._exhausted = False
        self._lock = asyncio.Lock()

    def __aiter__(self) -> "AsyncExecutor[T]":
        return self

    async def __anext__(self) -> T:
        async with self._lock:
            if self._buffer:
                return self._buffer.pop(0)

            if self._exhausted:
                raise StopAsyncIteration

            try:
                if self._executor:
                    loop = asyncio.get_running_loop()
                    item = await loop.run_in_executor(self._executor, next, self._iterator)
                else:
                    item = next(self._iterator)
                return item
            except StopIteration:
                self._exhausted = True
                raise StopAsyncIteration


def async_iter(iterator: Iterable[T], *, executor: Executor | None = None) -> AsyncExecutor[T]:
    """Convert a synchronous iterator to async."""
    return AsyncExecutor(iterator, executor=executor)


class FutureResult(Generic[T]):
    """Container for future results with metadata."""

    def __init__(self, future: asyncio.Future[T], metadata: dict[str, Any] | None = None) -> None:
        self._future = future
        self.metadata = metadata or {}
        self._start_time = asyncio.get_running_loop().time()

    @property
    def future(self) -> asyncio.Future[T]:
        return self._future

    async def result(self) -> T:
        return await self._future

    def done(self) -> bool:
        return self._future.done()

    def cancelled(self) -> bool:
        return self._future.cancelled()

    def exception(self) -> BaseException | None:
        return self._future.exception()

    @property
    def elapsed_time(self) -> float:
        return asyncio.get_running_loop().time() - self._start_time


async def as_completed_with_results(
    futures: list[asyncio.Future[T]], *, timeout: float | None = None
) -> AsyncGenerator[FutureResult[T], None]:
    """Yield futures as they complete with metadata."""
    done = set()
    pending = {f for f in futures}

    while pending:
        done_batch, pending = await asyncio.wait(
            pending, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
        )
        for future in done_batch:
            yield FutureResult(future)
        done.update(done_batch)
        if timeout is not None and not done_batch:
            break


async def first_result(futures: list[asyncio.Future[T]], *, cancel_others: bool = True) -> T:
    """Get the first successful result, waiting past failures if needed."""
    pending = set(futures)
    failures: list[BaseException] = []

    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for future in done:
            if future.cancelled():
                continue
            exception = future.exception()
            if exception is None:
                if cancel_others:
                    for p in pending:
                        p.cancel()
                return future.result()
            failures.append(exception)

    if failures:
        raise failures[0]
    raise RuntimeError("All futures cancelled or failed")


async def all_results(
    futures: list[asyncio.Future[T]], *, return_exceptions: bool = False
) -> list[T | BaseException]:
    """Get results from all futures."""
    return await asyncio.gather(*futures, return_exceptions=return_exceptions)


class FutureChain(Generic[T]):
    """Chain futures together with transformations."""

    def __init__(self, future: asyncio.Future[T]) -> None:
        self._future = future

    def then(self, func: Callable[[T], Any]) -> "FutureChain[S]":
        """Add a transformation to the chain."""

        async def transform() -> S:
            result = await self._future
            if inspect.iscoroutinefunction(func):
                return cast(S, await func(result))
            return cast(S, func(result))

        return FutureChain(asyncio.ensure_future(transform()))

    def catch(self, handler: Callable[[BaseException], Any]) -> "FutureChain[S]":
        """Add error handler to the chain."""

        async def handle() -> S:
            try:
                return cast(S, await self._future)
            except asyncio.CancelledError:
                raise
            except BaseException as e:
                if inspect.iscoroutinefunction(handler):
                    return cast(S, await handler(e))
                return cast(S, handler(e))

        return FutureChain(asyncio.ensure_future(handle()))

    def finally_(self, callback: Callable[[], Awaitable[None] | None]) -> "FutureChain[T]":
        """Add finally callback."""

        async def with_finally() -> T:
            try:
                return await self._future
            finally:
                if inspect.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()

        return FutureChain(asyncio.ensure_future(with_finally()))

    async def result(self) -> T:
        return await self._future


def chain_future(future: asyncio.Future[T]) -> FutureChain[T]:
    """Create a future chain."""
    return FutureChain(future)


@contextmanager
def future_context() -> Any:
    """Context manager for future operations."""
    _loop = asyncio.get_running_loop()
    futures: list[asyncio.Future[Any]] = []
    try:
        yield futures
    finally:
        for f in futures:
            if not f.done():
                f.cancel()


class BatchedFuture(Generic[T, R]):
    """Batch multiple futures for efficient execution."""

    def __init__(self, max_batch_size: int = 100, max_wait: float = 0.01) -> None:
        self._max_batch_size = max_batch_size
        self._max_wait = max_wait
        self._queue: asyncio.Queue[tuple[T, asyncio.Future[R]]] = asyncio.Queue()
        self._processor_task: asyncio.Task[Any] | None = None
        self._running = False

    async def start(self, processor: Callable[[list[T]], Coroutine[Any, Any, list[R]]]) -> None:
        """Start the batch processor."""
        self._running = True
        self._processor_task = asyncio.create_task(self._process_batches(processor))

    async def stop(self) -> None:
        """Stop the batch processor, cancelling any queued submissions."""
        self._running = False
        if self._processor_task:
            self._processor_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._processor_task
            self._processor_task = None
        while not self._queue.empty():
            _item, future = self._queue.get_nowait()
            if not future.done():
                future.cancel()

    async def _process_batches(
        self, processor: Callable[[list[T]], Coroutine[Any, Any, list[R]]]
    ) -> None:
        while self._running:
            batch: list[tuple[T, asyncio.Future[R]]] = []
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=self._max_wait)
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            batch.append(item)

            while len(batch) < self._max_batch_size:
                try:
                    batch.append(self._queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            items, futures = zip(*batch)
            try:
                results = await processor(list(items))
                for future, result in zip(futures, results, strict=True):
                    if not future.done():
                        future.set_result(result)
            except asyncio.CancelledError:
                for future in futures:
                    if not future.done():
                        future.cancel()
                raise
            except Exception as e:
                for future in futures:
                    if not future.done():
                        future.set_exception(e)

    async def submit(self, item: T) -> R:
        """Submit an item for batch processing."""
        future: asyncio.Future[R] = asyncio.get_running_loop().create_future()
        await self._queue.put((item, future))
        return await future


class FutureWrapper:
    """Wrap an asyncio Future with convenience helpers and a lifecycle."""

    def __init__(self, *, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._loop = loop or asyncio.get_running_loop()
        self._future: asyncio.Future[str] = self._loop.create_future()
        self._handle: asyncio.TimerHandle | None = None

    @property
    def future(self) -> asyncio.Future[str]:
        return self._future

    def start_operation(
        self, delay: float = 0.1, result: str = "Operation complete"
    ) -> asyncio.Future[str]:
        """Start a simulated operation that completes after ``delay`` seconds."""
        if self._handle is not None or self._future.done():
            raise RuntimeError("Operation already started")
        self._handle = self._loop.call_later(delay, self._complete, result)
        return self._future

    def _complete(self, result: str) -> None:
        if not self._future.done():
            self._future.set_result(result)

    def is_done(self) -> bool:
        """Whether the wrapped future has finished."""
        return self._future.done()

    def cancel(self) -> bool:
        """Cancel the pending operation and the wrapped future."""
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None
        return self._future.cancel()

    def result(self) -> str:
        """Return the future's result once it is done."""
        return self._future.result()

    def exception(self) -> BaseException | None:
        """Return the future's exception, if any."""
        return self._future.exception()

    def __await__(self) -> Any:
        return self._future.__await__()


async def future_basics() -> None:
    """Demonstrate creating a Future and resolving it from a callback."""
    loop = asyncio.get_running_loop()
    future: asyncio.Future[str] = loop.create_future()
    print(f"Before: done={future.done()}, cancelled={future.cancelled()}")

    def resolve() -> None:
        if not future.done():
            future.set_result("resolved by callback")

    loop.call_later(0.05, resolve)
    result = await future
    print(f"After: done={future.done()}, result={result!r}")
    print(f"future.result() -> {future.result()!r}")


async def future_with_callback() -> None:
    """Demonstrate done-callbacks firing when a Future completes."""
    loop = asyncio.get_running_loop()
    future: asyncio.Future[int] = loop.create_future()
    log: list[str] = []

    def on_done(done_future: asyncio.Future[int]) -> None:
        log.append(f"callback saw result={done_future.result()}")

    future.add_done_callback(on_done)
    loop.call_soon(future.set_result, 42)
    await future
    await asyncio.sleep(0)
    print(f"Callback log: {log}")


async def future_from_coroutine() -> None:
    """Demonstrate ``asyncio.ensure_future`` wrapping a coroutine."""

    async def compute() -> int:
        await asyncio.sleep(0.01)
        return 21 * 2

    future = asyncio.ensure_future(compute())
    print(f"Wrapped type: {type(future).__name__}, done={future.done()}")
    result = await future
    print(f"Result via Future: {result}")


async def future_chaining() -> None:
    """Demonstrate chaining transformations across Futures."""
    loop = asyncio.get_running_loop()
    base: asyncio.Future[int] = loop.create_future()
    loop.call_later(0.01, base.set_result, 10)

    async def double(value: int) -> int:
        await asyncio.sleep(0.01)
        return value * 2

    chained: FutureChain[int] = FutureChain(base).then(double).then(lambda value: value + 1)
    result = await chained.result()
    print(f"Chained result: {result}")


async def demonstrate_futures() -> None:
    """🎓 Future fundamentals tutorial for asyncio beginners.

    A ``Future`` is a placeholder for a value that will arrive later. Unlike a
    coroutine, a Future does not need a coroutine to produce its value: any
    code can call ``set_result`` or ``set_exception`` on it.

    WHAT YOU'LL LEARN:
    ------------------
    1. Creating a Future and resolving it from a callback
    2. Registering done-callbacks with ``add_done_callback``
    3. Wrapping coroutines as Futures with ``asyncio.ensure_future``
    4. Chaining transformations with ``FutureChain``
    5. The ``FutureWrapper`` helper for lifecycle-managed operations

    SEE ALSO:
    ---------
    - ``Future`` for the enhanced Future wrapper in this module
    - ``as_completed_with_results`` / ``first_result`` / ``all_results``
    - ``BatchedFuture`` for high-throughput batch processing
    """
    print("=== Future Basics ===")
    await future_basics()

    print("\n=== Future Callbacks ===")
    await future_with_callback()

    print("\n=== Future from Coroutine ===")
    await future_from_coroutine()

    print("\n=== Chaining Futures ===")
    await future_chaining()
