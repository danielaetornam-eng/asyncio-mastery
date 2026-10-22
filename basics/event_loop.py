"""
Event Loop Management

Comprehensive event loop utilities for asyncio mastery.
"""

import asyncio
import concurrent.futures
import inspect
import threading
from collections.abc import Callable, Coroutine, Generator
from contextlib import contextmanager, suppress
from functools import wraps
from typing import Any, TypeVar

T = TypeVar("T")


class EventLoopPolicy:
    """Custom event loop policy for advanced control."""

    def __init__(self, loop_factory: Callable[[], asyncio.AbstractEventLoop] | None = None) -> None:
        self._loop_factory = loop_factory or asyncio.new_event_loop
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread_local = threading.local()

    def get_event_loop(self) -> asyncio.AbstractEventLoop:
        """Gets or creates event loop for current thread."""
        loop: asyncio.AbstractEventLoop | None = getattr(self._thread_local, "loop", None)
        if loop is None:
            loop = self._loop_factory()
            self._thread_local.loop = loop
        return loop

    def set_event_loop(self, loop: asyncio.AbstractEventLoop | None) -> None:
        """Sets event loop for current thread."""
        self._thread_local.loop = loop

    def new_event_loop(self) -> asyncio.AbstractEventLoop:
        """Creates a new event loop."""
        return self._loop_factory()

    def close_loop(self) -> None:
        """Closes the current thread's event loop."""
        if hasattr(self._thread_local, "loop") and self._thread_local.loop:
            self._thread_local.loop.close()
            self._thread_local.loop = None

    @contextmanager
    def loop_context(self) -> Generator[asyncio.AbstractEventLoop, None, None]:
        """Context manager for temporary event loop."""
        old_loop = getattr(self._thread_local, "loop", None)
        loop = self.new_event_loop()
        self.set_event_loop(loop)
        try:
            yield loop
        finally:
            loop.close()
            self.set_event_loop(old_loop)


class EventLoopMixin:
    """Mixin for classes that need event loop access."""

    def __init__(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._loop = loop
        self._owns_loop = loop is None

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """Gets the event loop, creating one if needed."""
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
            self._owns_loop = True
        return self._loop

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Sets a specific event loop."""
        if self._owns_loop and self._loop:
            self._loop.close()
        self._loop = loop
        self._owns_loop = False

    def close_loop(self) -> None:
        """Closes the owned event loop."""
        if self._owns_loop and self._loop:
            self._loop.close()
            self._loop = None
            self._owns_loop = False


def get_event_loop() -> asyncio.AbstractEventLoop:
    """Gets the current event loop, creating one if necessary."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


def create_event_loop(*, policy: EventLoopPolicy | None = None) -> asyncio.AbstractEventLoop:
    """Creates a new event loop with optional custom policy."""
    if policy:
        return policy.new_event_loop()
    return asyncio.new_event_loop()


def run_coroutine(
    coro: Coroutine[Any, Any, T], *, loop: asyncio.AbstractEventLoop | None = None
) -> T:
    """Runs a coroutine and returns its result."""
    if loop is None:
        loop = get_event_loop()
    return loop.run_until_complete(coro)


def run_coroutine_threadsafe(
    coro: Coroutine[Any, Any, T], loop: asyncio.AbstractEventLoop
) -> concurrent.futures.Future[T]:
    """Submits a coroutine to a specific event loop from another thread."""
    return asyncio.run_coroutine_threadsafe(coro, loop)


def run_in_executor(
    executor: concurrent.futures.Executor | None,
    func: Callable[..., T],
    *args: object,
    loop: asyncio.AbstractEventLoop | None = None,
    **kwargs: object,
) -> asyncio.Future[Any]:
    """Runs a function in an executor."""
    if loop is None:
        loop = get_event_loop()
    return loop.run_in_executor(executor, lambda: func(*args, **kwargs))


def run_in_threadpool(
    func: Callable[..., T],
    *args: object,
    loop: asyncio.AbstractEventLoop | None = None,
    **kwargs: object,
) -> asyncio.Future[Any]:
    """Runs a function in the default thread pool executor."""
    return run_in_executor(None, func, *args, loop=loop, **kwargs)


def run_in_processpool(
    func: Callable[..., T],
    *args: object,
    loop: asyncio.AbstractEventLoop | None = None,
    **kwargs: object,
) -> asyncio.Future[Any]:
    """Runs a function in a process pool executor."""
    executor = concurrent.futures.ProcessPoolExecutor()
    return run_in_executor(executor, func, *args, loop=loop, **kwargs)


@contextmanager
def temporary_event_loop(
    loop: asyncio.AbstractEventLoop | None = None,
) -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Context manager for temporary event loop.

    A loop created by this helper is closed on exit; a caller-provided loop
    is left open and the previous loop is restored.
    """
    owns_loop = loop is None
    old_loop: asyncio.AbstractEventLoop | None = None
    with suppress(RuntimeError):
        old_loop = asyncio.get_running_loop()

    if loop is None:
        loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield loop
    finally:
        asyncio.set_event_loop(old_loop)
        if owns_loop:
            loop.close()


async def sleep(delay: float, result: object = None) -> object:
    """Async sleep with optional result."""
    await asyncio.sleep(delay)
    return result


async def run_with_timeout(coro: Coroutine[object, object, T], timeout_seconds: float) -> T:
    """Runs coroutine with timeout."""
    return await asyncio.wait_for(coro, timeout=timeout_seconds)


def async_wrapper(func: Callable[..., Coroutine[object, object, T]]) -> Callable[..., T]:
    """Decorator to run async function synchronously."""

    @wraps(func)
    def wrapper(*args: object, **kwargs: object) -> T:
        return run_coroutine(func(*args, **kwargs))

    return wrapper


def ensure_async(func: Callable[..., T]) -> Callable[..., Coroutine[object, object, T]]:
    """Ensure a function is async."""
    if inspect.iscoroutinefunction(func):
        return func

    @wraps(func)
    async def async_func(*args: object, **kwargs: object) -> T:
        return func(*args, **kwargs)

    return async_func
