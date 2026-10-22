"""
The module explains how to use the event loop.

An event loop is a scheduler. It runs the coroutines of one program.
The event loop also runs the callback functions and the timers of the program.
The event loop runs one program at a time in one thread.

The event loop is like a juggler. It starts one coroutine.
When the coroutine says `await`, the event loop pauses the coroutine.
Then the event loop starts the next coroutine.

The module shows these items.

- The active event loop. Use `asyncio.get_running_loop()`.
- The event loop clock. Use `loop.time()`.
- The event loop policies. Use `EventLoopPolicy`.
- The event loop life cycle. Use `EventLoopManager`.
- The delayed callbacks. Use `loop.call_later()` and `loop.call_at()`.
- The executor calls. Use `run_in_executor()` and `run_in_threadpool()`.

Read the docstring of `demonstrate_event_loop()` first.
The function runs all the examples of this module.

See also: `asyncio.run()`, `asyncio.get_running_loop()`, `asyncio.wait_for()`
"""

import asyncio
import concurrent.futures
import inspect
import threading
import warnings
from collections.abc import Callable, Coroutine, Generator
from contextlib import contextmanager, suppress
from functools import wraps
from typing import Any, TypeVar

T = TypeVar("T")

RUNNING_LOOP_ERROR = "Cannot run a coroutine on a running loop; await it instead"


class EventLoopPolicy:
    """The class gives control over the creation of the event loop.

    The class stores the event loop of each thread.
    The class uses `asyncio.new_event_loop()` if you do not supply a factory.
    """

    def __init__(self, loop_factory: Callable[[], asyncio.AbstractEventLoop] | None = None) -> None:
        """The function stores the event loop factory.

        The function uses `asyncio.new_event_loop()` if you do not supply a factory.
        The function creates the thread data for the event loops.
        """

        self._loop_factory = loop_factory or asyncio.new_event_loop
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread_local = threading.local()

    def get_event_loop(self) -> asyncio.AbstractEventLoop:
        """The function returns the event loop of the current thread.

        The function creates the event loop if the thread has no event loop.
        """

        loop: asyncio.AbstractEventLoop | None = getattr(self._thread_local, "loop", None)
        if loop is None:
            loop = self._loop_factory()
            self._thread_local.loop = loop
        return loop

    def set_event_loop(self, loop: asyncio.AbstractEventLoop | None) -> None:
        """The function stores the event loop for the current thread.

        The function stores `None` if you pass it. The thread then has no event loop.
        """

        self._thread_local.loop = loop

    def new_event_loop(self) -> asyncio.AbstractEventLoop:
        """The function creates a new event loop.

        The function does not install the new event loop as the current one.
        """

        return self._loop_factory()

    def close_loop(self) -> None:
        """The function closes the event loop of the current thread.

        The function does nothing if the thread has no event loop.
        """

        if hasattr(self._thread_local, "loop") and self._thread_local.loop:
            self._thread_local.loop.close()
            self._thread_local.loop = None

    @contextmanager
    def loop_context(self) -> Generator[asyncio.AbstractEventLoop, None, None]:
        """The function supplies a temporary event loop for a block of code.

        The function closes the temporary event loop at the end of the block.
        The function restores the previous event loop of the thread.
        """

        old_loop = getattr(self._thread_local, "loop", None)
        loop = self.new_event_loop()
        self.set_event_loop(loop)
        try:
            yield loop
        finally:
            loop.close()
            self.set_event_loop(old_loop)


class EventLoopMixin:
    """The class supplies the event loop to other classes.

    The class creates the event loop if you do not supply one.
    The class closes the event loop only if the class creates it.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """The function stores the event loop.

        The class closes this event loop only if the class creates it.
        """

        self._loop = loop
        self._owns_loop = loop is None

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """The property returns the event loop of the class.

        The property creates the event loop if the class has no event loop.
        """

        if self._loop is None:
            self._loop = asyncio.new_event_loop()
            self._owns_loop = True
        return self._loop

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """The function stores a new event loop for the class.

        The function closes the old event loop if the class owns it.
        """

        if self._owns_loop and self._loop:
            self._loop.close()
        self._loop = loop
        self._owns_loop = False

    def close_loop(self) -> None:
        """The function closes the event loop if the class owns it."""

        if self._owns_loop and self._loop:
            self._loop.close()
            self._loop = None
            self._owns_loop = False


def get_event_loop() -> asyncio.AbstractEventLoop:
    """The function returns the active event loop.

    The function creates a new event loop if no event loop runs.
    The function installs the new event loop as the current one.
    """

    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


def create_event_loop(*, policy: EventLoopPolicy | None = None) -> asyncio.AbstractEventLoop:
    """The function creates a new event loop.

    The function uses the policy to create the event loop if you supply one.
    """

    if policy:
        return policy.new_event_loop()
    return asyncio.new_event_loop()


def run_coroutine(
    coro: Coroutine[Any, Any, T], *, loop: asyncio.AbstractEventLoop | None = None
) -> T:
    """The function runs the coroutine until the coroutine ends.

    The function returns the result of the coroutine.
    The function uses the current event loop if you do not supply one.
    """

    if loop is None:
        loop = get_event_loop()
    return loop.run_until_complete(coro)


def run_coroutine_threadsafe(
    coro: Coroutine[Any, Any, T], loop: asyncio.AbstractEventLoop
) -> concurrent.futures.Future[T]:
    """The function submits the coroutine to the event loop.

    You must call this function from a thread that is not the event loop thread.
    The function returns a `Future` object for the result.
    """

    return asyncio.run_coroutine_threadsafe(coro, loop)


def run_in_executor(
    executor: concurrent.futures.Executor | None,
    func: Callable[..., T],
    *args: object,
    loop: asyncio.AbstractEventLoop | None = None,
    **kwargs: object,
) -> asyncio.Future[Any]:
    """The function runs the function in the executor.

    The executor runs the function in a thread or in a process.
    The function returns a `Future` object for the result.
    The function uses the current event loop if you do not supply one.
    """

    if loop is None:
        loop = get_event_loop()
    return loop.run_in_executor(executor, lambda: func(*args, **kwargs))


def run_in_threadpool(
    func: Callable[..., T],
    *args: object,
    loop: asyncio.AbstractEventLoop | None = None,
    **kwargs: object,
) -> asyncio.Future[Any]:
    """The function runs the function in the default thread pool.

    The thread pool uses one thread for each call.
    The function returns a `Future` object for the result.
    """

    return run_in_executor(None, func, *args, loop=loop, **kwargs)


def run_in_processpool(
    func: Callable[..., T],
    *args: object,
    loop: asyncio.AbstractEventLoop | None = None,
    **kwargs: object,
) -> asyncio.Future[Any]:
    """The function runs the function in a new process pool.

    The process pool uses one process for each call.
    The function returns a `Future` object for the result.
    """

    executor = concurrent.futures.ProcessPoolExecutor()
    return run_in_executor(executor, func, *args, loop=loop, **kwargs)


@contextmanager
def temporary_event_loop(
    loop: asyncio.AbstractEventLoop | None = None,
) -> Generator[asyncio.AbstractEventLoop, None, None]:
    """The function supplies a temporary event loop for a block of code.

    The function closes the temporary event loop at the end of the block.
    The function keeps an event loop that you supply open.
    The function restores the previous event loop of the thread.
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
    """The function waits for the delay in seconds.

    The event loop runs the other coroutines during the wait.
    The function returns the result after the wait ends.
    """

    await asyncio.sleep(delay)
    return result


async def run_with_timeout(coro: Coroutine[object, object, T], timeout_seconds: float) -> T:
    """The function runs the coroutine until the timeout ends.

    The function returns the result of the coroutine.
    The function raises `asyncio.TimeoutError` if the timeout ends first.
    """

    return await asyncio.wait_for(coro, timeout=timeout_seconds)


def async_wrapper(func: Callable[..., Coroutine[object, object, T]]) -> Callable[..., T]:
    """The function makes a normal function from an async function.

    The function runs the coroutine and returns the result.
    The function blocks the current thread until the coroutine ends.
    """

    @wraps(func)
    def wrapper(*args: object, **kwargs: object) -> T:
        return run_coroutine(func(*args, **kwargs))

    return wrapper


def ensure_async(func: Callable[..., T]) -> Callable[..., Coroutine[object, object, T]]:
    """The function makes an async function from a normal function.

    The function returns the same function if the function is already async.
    """

    if inspect.iscoroutinefunction(func):
        return func

    @wraps(func)
    async def async_func(*args: object, **kwargs: object) -> T:
        # The function stays `async` on purpose. The caller awaits the result.
        # The function never awaits itself because the wrapped call is blocking.
        return func(*args, **kwargs)

    return async_func


class EventLoopManager:
    """The class manages the life cycle of an event loop.

    The class uses the active event loop if the code runs in an event loop.
    The class creates a new event loop in the other cases.
    The class closes the new event loop at the end of the block.
    """

    def __init__(self, *, debug: bool = False) -> None:
        """The function stores the debug flag and the event loop data."""

        self._debug = debug
        self._loop: asyncio.AbstractEventLoop | None = None
        self._owns_loop = False
        self._previous_debug: bool | None = None

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """The property returns the managed event loop.

        The property creates the event loop if the class has no event loop.
        """

        if self._loop is None:
            self._loop = asyncio.new_event_loop()
            self._owns_loop = True
        return self._loop

    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        """The function runs the coroutine on the managed event loop.

        The function returns the result of the coroutine.
        """

        return self.loop.run_until_complete(coro)

    def time(self) -> float:
        """The function returns the monotonic clock of the managed event loop."""

        return self.loop.time()

    def close(self) -> None:
        """The function closes the managed event loop if the class owns it."""

        if self._owns_loop and self._loop is not None and not self._loop.is_closed():
            self._loop.close()
        self._loop = None
        self._owns_loop = False

    def __enter__(self) -> asyncio.AbstractEventLoop:
        """The function starts the managed event loop.

        The function turns on the debug mode if you set the debug flag.
        """

        try:
            loop = asyncio.get_running_loop()
            self._owns_loop = False
        except RuntimeError:
            loop = self.loop
            asyncio.set_event_loop(loop)
        self._loop = loop
        if self._debug:
            self._previous_debug = loop.get_debug()
            loop.set_debug(True)
        return loop

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """The function stops the managed event loop.

        The function restores the debug mode of the event loop.
        The function closes the event loop if the class owns it.
        """

        loop = self._loop
        if loop is None:
            return
        if self._debug and self._previous_debug is not None:
            loop.set_debug(self._previous_debug)
            self._previous_debug = None
        if self._owns_loop:
            loop.close()
            asyncio.set_event_loop(None)
        self._loop = None
        self._owns_loop = False


def create_new_loop() -> asyncio.AbstractEventLoop:
    """The function creates a new event loop.

    The function does not install the new event loop as the current one.
    """

    return asyncio.new_event_loop()


def get_event_loop_policy() -> asyncio.AbstractEventLoopPolicy:
    """The function returns the event loop policy of the process.

    The function hides the deprecation warning of Python 3.14 and later.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return asyncio.get_event_loop_policy()


def run_in_loop(
    coro: Coroutine[Any, Any, T], *, loop: asyncio.AbstractEventLoop | None = None
) -> T:
    """The function runs the coroutine until the coroutine ends.

    The function uses the current event loop if you do not supply one.
    The function raises `RuntimeError` if the event loop already runs.
    """

    if loop is None:
        loop = get_event_loop()
    if loop.is_running():
        raise RuntimeError(RUNNING_LOOP_ERROR)
    return loop.run_until_complete(coro)


async def call_later_example() -> None:
    """The function shows a callback function that runs later.

    The function calls `loop.call_later()` with a delay of 0.05 seconds.
    The event loop calls the callback function after the delay ends.
    The function waits for the `Future` object and prints the result.
    """

    loop = asyncio.get_running_loop()
    future: asyncio.Future[str] = loop.create_future()

    def on_timeout() -> None:
        """The function sets the result of the `Future` object.

        The function does nothing if the `Future` object already has a result.
        """

        if not future.done():
            future.set_result("callback fired after 0.05s")

    # The event loop stores the callback function in its timer list.
    # The event loop calls the callback function after the delay ends.
    handle = loop.call_later(0.05, on_timeout)
    print(f"Scheduled handle: {handle!r}")

    # The `await` statement pauses this coroutine until the result arrives.
    result = await future
    print(f"call_later result: {result}")


async def call_at_example() -> None:
    """The function shows a callback function that runs at a set time.

    The function reads the event loop clock and adds 0.05 seconds.
    The function calls `loop.call_at()` with the new time.
    The event loop calls the callback function at the set time.
    """

    loop = asyncio.get_running_loop()
    future: asyncio.Future[str] = loop.create_future()
    when = loop.time() + 0.05

    def on_deadline() -> None:
        """The function sets the result of the `Future` object.

        The function does nothing if the `Future` object already has a result.
        """

        if not future.done():
            future.set_result("deadline reached")

    # The event loop calls the callback function at the set time.
    # The set time comes from the monotonic clock of the event loop.
    loop.call_at(when, on_deadline)
    result = await future
    print(f"call_at result: {result} (scheduled for loop time {when:.3f})")


async def loop_time_example() -> None:
    """The function shows the monotonic clock of the event loop.

    The monotonic clock never goes back. It is the correct clock for timeouts.
    The function reads the clock before the wait and after the wait.
    """

    loop = asyncio.get_running_loop()
    start = loop.time()
    await asyncio.sleep(0.05)
    elapsed = loop.time() - start
    print(f"loop.time() advanced by {elapsed:.3f}s (monotonic)")


def run_sample_in_new_loop() -> object:
    """The function runs a coroutine on a new event loop.

    The function must run in a thread that has no active event loop.
    The function closes the new event loop at the end of the call.
    """

    other_loop = create_new_loop()
    try:
        return run_in_loop(sleep(0.01, "the coroutine result"), loop=other_loop)
    finally:
        other_loop.close()


async def demonstrate_event_loop() -> None:
    """The function runs a lesson about the event loop.

    An event loop is a scheduler. It runs the coroutines of one program.
    The event loop runs one coroutine at a time in one thread.
    The event loop pauses a coroutine at each `await` and starts the next one.

    The function shows these items.

    1. The active event loop. Use `asyncio.get_running_loop()`.
    2. The event loop clock. Use `loop.time()`.
    3. The debug mode. Use `EventLoopManager` and the `debug` flag.
    4. The delayed callback. Use `loop.call_later()`.
    5. The set time callback. Use `loop.call_at()`.
    6. The call from normal code. Use `run_in_loop()` in a new thread.

    HOW to run the lesson:

    1. Call `asyncio.run(demonstrate_event_loop())` in a program.
    2. Read the output. The output shows the clock value and the results.

    CAUTION: Do not call `run_until_complete()` on an active event loop.
    The call raises a `RuntimeError`.

    SEE ALSO:

    - `EventLoopManager`. The class manages the life cycle of an event loop.
    - `create_new_loop()`. The function creates an event loop.
    - `get_event_loop_policy()`. The function returns the policy of the process.
    - `temporary_event_loop()`. The function supplies a temporary event loop.
    """

    print("=== Event Loop Basics ===")

    # The `get_running_loop()` function returns the event loop of this thread.
    # The function raises a `RuntimeError` if no event loop runs.
    loop = asyncio.get_running_loop()
    print(f"Running loop: {loop!r}")
    print(f"Monotonic loop time: {loop.time():.3f}")
    print(f"Debug mode: {loop.get_debug()}")
    print(f"Policy: {type(get_event_loop_policy()).__name__}")

    # The manager sets the debug mode on the event loop.
    # The manager restores the old debug mode at the end of the block.
    print("\n--- Debug Mode: EventLoopManager ---")
    with EventLoopManager(debug=True) as managed_loop:
        print(f"Managed loop: {managed_loop!r}")
        print(f"Debug mode in the block: {managed_loop.get_debug()}")
    print(f"Debug mode after the block: {loop.get_debug()}")

    # The event loop calls the callback function after the delay ends.
    print("\n--- Deferred Callback: call_later ---")
    await call_later_example()

    # The event loop calls the callback function at the set time.
    print("\n--- Absolute Scheduling: call_at ---")
    await call_at_example()

    # The monotonic clock never goes back.
    print("\n--- Monotonic Clock: loop.time ---")
    await loop_time_example()

    # The `run_in_threadpool()` function runs the function in a new thread.
    # The new thread has no active event loop.
    # The `run_in_loop()` function runs the coroutine in that thread.
    print("\n--- Normal Code: run_in_loop ---")
    value = await run_in_threadpool(run_sample_in_new_loop)
    print(f"run_in_loop result: {value}")
