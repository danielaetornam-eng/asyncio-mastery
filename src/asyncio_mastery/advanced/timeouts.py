"""
Timeouts - Control execution time of async operations.

================================================================================
WHAT ARE TIMEOUTS AND WHY DO WE NEED THEM?
================================================================================

A timeout is a safety mechanism that automatically cancels an operation if it
takes too long to complete. Think of it like setting a kitchen timer - if your
code does not finish before the timer rings, it gets stopped automatically.

WHY ARE TIMEOUTS CRITICAL IN ASYNC PROGRAMMING?
------------------------------------------------
1. **Prevent Hanging**: Network requests, database queries, or external APIs
   can hang indefinitely without timeouts
2. **Resource Protection**: Stuck operations consume memory, file descriptors,
   and connection pool slots
3. **User Experience**: Users should not wait forever for a response
4. **Cascading Failures**: One slow service can bring down your entire system
5. **SLAs**: Service Level Agreements often require response time guarantees

PYTHON 3.11+ TIMEOUT FEATURES
------------------------------
Python 3.11 introduced two powerful context managers:
- `asyncio.timeout(seconds)` - Set a relative timeout (e.g., "5 seconds from now")
- `asyncio.timeout_at(deadline)` - Set an absolute deadline (e.g., "by 3:00 PM")

Both raise `TimeoutError` when the time expires, which you can catch and handle.

================================================================================
QUICK START GUIDE
================================================================================

BASIC USAGE (Python 3.11+):
----------------------------
```python
import asyncio

async def fetch_data():
    # This might hang forever without a timeout!
    await asyncio.sleep(10)
    return "data"

async def main():
    try:
        # Give it at most 2 seconds
        async with asyncio.timeout(2):
            result = await fetch_data()
            print(f"Got: {result}")
    except TimeoutError:
        print("Operation took too long!")

asyncio.run(main())
```

WITH A DEFAULT VALUE:
---------------------
```python
async def main():
    try:
        async with asyncio.timeout(2):
            result = await fetch_data()
    except TimeoutError:
        result = "fallback value"  # Use a default instead
    print(result)
```

================================================================================
COMMON PATTERNS
================================================================================

1. **Relative Timeout**: `asyncio.timeout(5)` - "5 seconds from now"
2. **Absolute Deadline**: `asyncio.timeout_at(loop.time() + 5)` - "by this timestamp"
3. **No Timeout**: `asyncio.timeout(None)` - Wait forever (default behavior)
4. **Shielding**: `asyncio.shield(coro)` - Protect critical code from timeout

================================================================================
"""

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")


# =============================================================================
# FIRST DEFINITION BLOCK (documented for learning; OVERRIDDEN below)
# =============================================================================
# Heads-up for beginners: `Timeout`, `timeout_after`, `run_with_timeout`, and
# `TimeoutManager` are defined twice in this file. These FIRST definitions
# (which use the ASYNC context-manager protocol) are replaced by the second
# definitions near the bottom of the module — Python's last-assignment-wins
# rule applies at module scope. Read these docstrings to learn the concepts;
# the active behavior comes from the later block.
# =============================================================================

@dataclass
class TimeoutResult:
    """
    Result of a timed operation - a structured way to handle timeouts.

    Instead of raising exceptions, this returns a result object that tells you
    exactly what happened. This is often cleaner than try/except blocks.

    ATTRIBUTES:
    -----------
    success: bool
        True if the operation completed within the timeout
    value: Any
        The return value if successful, or the default value if timed out
    error: Exception | None
        The exception that occurred (TimeoutError or other), or None if success
    duration: float
        How long the operation actually ran (in seconds)
    timed_out: bool
        True specifically if a TimeoutError occurred

    EXAMPLE:
    --------
    ```python
    result = await run_with_timeout(slow_operation(), timeout=2.0, default="fallback")

    if result.success:
        print(f"Got result: {result.value}")
    elif result.timed_out:
        print(f"Timed out after {result.duration:.2f}s, using default: {result.value}")
    else:
        print(f"Failed with error: {result.error}")
    ```
    """

    success: bool
    value: Any = None
    error: Exception | None = None
    duration: float = 0.0
    timed_out: bool = False


class Timeout:
    """
    Reusable async timeout context manager.

    Unlike `asyncio.timeout()` which creates a new context manager each time,
    this class can be reused and provides additional properties like `expired`
    and `remaining` to check timeout status without waiting for an exception.

    USAGE:
    ------
    ```python
    timeout = Timeout(5.0)  # 5 second timeout

    # Use as ASYNC context manager
    async with timeout:
        await some_operation()

    # Or check status programmatically
    if timeout.expired:
        print("Time's up!")
    print(f"Seconds left: {timeout.remaining}")
    ```

    This implements the ASYNC context manager protocol (__aenter__/__aexit__)
    to properly work with asyncio.timeout().
    """

    def __init__(self, seconds: float | None) -> None:
        """
        Initialize a timeout.

        Args:
            seconds: Timeout in seconds, or None for no timeout (wait forever)
        """
        self._seconds = seconds
        self._deadline: float | None = None
        self._cm: Any = None

    async def __aenter__(self) -> "Timeout":
        """Enter the timeout context (starts the timer)."""
        if self._seconds is not None:
            self._deadline = time.monotonic() + self._seconds
            self._cm = asyncio.timeout(self._seconds)
            await self._cm.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        """Exit the timeout context (stops the timer)."""
        if self._cm:
            return await self._cm.__aexit__(*args)
        return None

    @property
    def expired(self) -> bool:
        """
        Check if the timeout has expired.

        Returns:
            True if the deadline has passed, False otherwise.
            Returns False if no timeout was set (seconds=None).
        """
        if self._deadline is None:
            return False
        return time.monotonic() >= self._deadline

    @property
    def remaining(self) -> float | None:
        """
        Get remaining time in seconds.

        Returns:
            Seconds remaining until timeout, or None if no timeout set.
            Returns 0 if already expired.
        """
        if self._deadline is None:
            return None
        return max(0, self._deadline - time.monotonic())


@asynccontextmanager
async def timeout_after(seconds: float | None) -> AsyncIterator[None]:
    """
    Async context manager for timeout (alias for asyncio.timeout).

    This is a thin wrapper that handles the `None` case (no timeout) gracefully.
    Using `asyncio.timeout(None)` directly would raise an error, but this
    function allows `None` to mean "wait forever".

    Args:
        seconds: Timeout in seconds, or None for no timeout

    Yields:
        Nothing - just controls the timeout scope

    EXAMPLE:
    --------
    ```python
    # With timeout
    async with timeout_after(5.0):
        await long_operation()

    # Without timeout (wait forever)
    async with timeout_after(None):
        await long_operation()
    ```
    """
    if seconds is None:
        yield
    else:
        async with asyncio.timeout(seconds):
            yield


async def run_with_timeout(
    coro: asyncio.coroutines.Coroutine[Any, Any, T],
    timeout: float | None,
    *,
    default: T | None = None,
) -> TimeoutResult:
    """
    Run a coroutine with a timeout, returning a structured result.

    This is the RECOMMENDED way to handle timeouts in most cases because it:
    1. Never raises TimeoutError - returns a result object instead
    2. Lets you specify a default value for timeout cases
    3. Captures all exceptions, not just timeouts
    4. Tracks how long the operation actually took

    Args:
        coro: The coroutine to run (e.g., `fetch_data()`)
        timeout: Maximum seconds to wait, or None for no timeout
        default: Value to return if timeout occurs (default: None)

    Returns:
        TimeoutResult with success, value, error, duration, and timed_out fields

    EXAMPLES:
    ---------
    ```python
    # Basic usage
    result = await run_with_timeout(fetch_data(), 2.0)
    if result.success:
        print(result.value)
    else:
        print(f"Failed: {result.error}")

    # With default value for timeout
    result = await run_with_timeout(fetch_data(), 2.0, default="cached")
    print(result.value)  # Either real data or "cached"

    # No timeout
    result = await run_with_timeout(fetch_data(), None)
    ```
    """
    start = time.monotonic()
    try:
        async with timeout_after(timeout):
            value = await coro
        return TimeoutResult(success=True, value=value, duration=time.monotonic() - start)
    except TimeoutError as e:
        return TimeoutResult(
            success=False, value=default, error=e, duration=time.monotonic() - start, timed_out=True
        )
    except Exception as e:
        return TimeoutResult(success=False, error=e, duration=time.monotonic() - start)


class TimeoutManager:
    """
    Manage multiple named timeouts.

    Useful when you have multiple concurrent operations each with their own
    timeout, and you want to track/cancel them by name.

    EXAMPLE:
    --------
    ```python
    manager = TimeoutManager()

    async def worker(name: str, timeout: float):
        async with manager.create(name, timeout):
            await do_work()

    # Run multiple workers with different timeouts
    await asyncio.gather(
        worker("fast", 1.0),
        worker("slow", 5.0),
    )

    # Check status
    if manager.get("fast").expired:
        print("Fast worker timed out!")
    ```
    """

    def __init__(self) -> None:
        self._timeouts: dict[str, Timeout] = {}

    def create(self, name: str, seconds: float | None) -> Timeout:
        """
        Create a named timeout.

        Args:
            name: Unique identifier for this timeout
            seconds: Timeout in seconds, or None for no timeout

        Returns:
            Timeout object that can be used as a context manager
        """
        timeout = Timeout(seconds)
        self._timeouts[name] = timeout
        return timeout

    def get(self, name: str) -> Timeout | None:
        """Get a timeout by name, or None if not found."""
        return self._timeouts.get(name)

    async def cancel(self, name: str) -> bool:
        """
        Cancel a timeout by entering and exiting immediately.

        This effectively stops the timeout from firing. The timeout is also
        removed from the manager.

        Args:
            name: Name of the timeout to cancel

        Returns:
            True if timeout was found and cancelled, False otherwise
        """
        if name in self._timeouts:
            timeout = self._timeouts[name]
            await timeout.__aenter__()
            await timeout.__aexit__(None, None, None)
            del self._timeouts[name]
            return True
        return False


async def demonstrate_timeouts() -> None:
    """
    Comprehensive demonstration of timeout patterns.

    Run this function to see all timeout features in action:
    `python -m asyncio_mastery.advanced.timeouts`
    """
    print("=== Timeouts Demo ===")

    # -------------------------------------------------------------------------
    # BASIC TIMEOUT (Python 3.11+)
    # -------------------------------------------------------------------------
    print("\n--- asyncio.timeout ---")

    async def slow_operation() -> str:
        """An operation that takes 1 second."""
        await asyncio.sleep(1)
        return "done"

    try:
        # Give it only 0.1 seconds - it will timeout!
        async with asyncio.timeout(0.1):
            await slow_operation()
    except TimeoutError:
        print("Operation timed out! (expected)")

    # -------------------------------------------------------------------------
    # ABSOLUTE DEADLINE WITH timeout_at
    # -------------------------------------------------------------------------
    print("\n--- asyncio.timeout_at ---")
    # deadline = current time + 0.1 seconds
    deadline = time.monotonic() + 0.1
    try:
        async with asyncio.timeout_at(deadline):
            await slow_operation()
    except TimeoutError:
        print("Deadline exceeded! (expected)")

    # -------------------------------------------------------------------------
    # run_with_timeout HELPER (RECOMMENDED)
    # -------------------------------------------------------------------------
    print("\n--- run_with_timeout ---")

    # This will timeout
    result = await run_with_timeout(slow_operation(), 0.1)
    print(
        f"Success: {result.success}, Timed out: {result.timed_out}, "
        f"Duration: {result.duration:.3f}s"
    )

    # This will succeed
    result = await run_with_timeout(asyncio.sleep(0.01, result="fast"), 1.0)
    print(f"Success: {result.success}, Value: {result.value}")

    # -------------------------------------------------------------------------
    # TIMEOUT WITH DEFAULT VALUE
    # -------------------------------------------------------------------------
    print("\n--- Timeout with Default ---")
    result = await run_with_timeout(slow_operation(), 0.1, default="fallback")
    print(f"Value: {result.value}")  # Will be "fallback"

    # -------------------------------------------------------------------------
    # TIMEOUT MANAGER
    # -------------------------------------------------------------------------
    print("\n--- Timeout Manager ---")
    manager = TimeoutManager()

    async def worker(id: int, timeout: float) -> str:
        # Each worker gets its own named timeout
        # Note: Timeout now uses async context manager protocol
        async with manager.create(f"worker-{id}", timeout):
            await asyncio.sleep(timeout * 2)  # Sleep longer than timeout
            return f"worker-{id} done"

    # Run workers concurrently - one will timeout, one won't
    results = await asyncio.gather(worker(1, 0.5), worker(2, 0.1), return_exceptions=True)
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            print(f"Worker {i + 1}: {type(r).__name__}")
        else:
            print(f"Worker {i + 1}: {r}")

    # -------------------------------------------------------------------------
    # SHIELD FROM TIMEOUT (CRITICAL OPERATIONS)
    # -------------------------------------------------------------------------
    print("\n--- Shield from Timeout ---")

    async def critical_operation() -> str:
        """This operation MUST complete even if timeout expires."""
        await asyncio.sleep(0.2)
        return "critical done"

    try:
        # The timeout applies to the shield() call, not the inner operation
        async with asyncio.timeout(0.1):
            # shield() protects the operation from being cancelled
            result = await asyncio.shield(critical_operation())
            print(f"Shielded result: {result}")
    except TimeoutError:
        # The timeout fired, but critical_operation() continues in background!
        print("Timeout fired, but shielded operation continues in background...")

    # Wait a bit for shielded operation to complete
    await asyncio.sleep(0.3)
    print("Shielded operation should be done now.")


# =============================================================================
# BEGINNER TIPS AND BEST PRACTICES
# =============================================================================
#
# 1. ALWAYS SET TIMEOUTS ON EXTERNAL CALLS
#    - HTTP requests, database queries, RPC calls
#    - Default: 5-30 seconds depending on operation
#
# 2. USE run_with_timeout() FOR CLEANER CODE
#    - Avoids try/except blocks everywhere
#    - Makes timeout handling explicit in the return type
#
# 3. CONSIDER SHIELD FOR CRITICAL OPERATIONS
#    - Database commits, payment processing, cleanup
#    - But be careful - shielded code can still run forever!
#
# 4. USE timeout_at FOR COORDINATED DEADLINES
#    - When multiple operations must complete by the same absolute time
#    - Pass the deadline down through your call chain
#
# 5. do not FORGET TO HANDLE TimeoutError
#    - it is a BaseException subclass (like KeyboardInterrupt)
#    - Catch it explicitly, not with bare `except:`
#
# 6. TIMEOUTS ARE COOPERATIVE
#    - They only work at `await` points
#    - CPU-bound loops without await will not be interrupted
#    - Use `asyncio.to_thread()` for CPU-bound work with timeouts
#
# 7. TEST YOUR TIMEOUTS
#    - Verify they actually fire when expected
#    - Test both success and timeout paths
#
# =============================================================================


if __name__ == "__main__":
    asyncio.run(demonstrate_timeouts())


# =============================================================================
# SECOND DEFINITION BLOCK (ACTIVE / OVERRIDING)
# =============================================================================
#
# IMPORTANT NOTE FOR READERS:
# ---------------------------
# The `Timeout`, `timeout_after`, `run_with_timeout`, and `TimeoutManager`
# names are REDEFINED below. In Python, later definitions at module scope
# silently replace earlier ones, so the versions BELOW this comment are the
# ones that actually get used if you import from this module.
# Importing `Timeout` from this module therefore yields the block below.
#
# Key difference from the first block: this `Timeout` class implements the
# regular SYNCHRONOUS context manager protocol (__enter__/__exit__) instead
# of the async one (__aenter__/__aexit__), and `TimeoutManager.cancel()` is
# a plain synchronous method. The duplicates exist to demonstrate that Python
# allows redefinition; the first block's broad docstrings above still
# explain the concepts and apply equally here.
#
# =============================================================================

class Timeout:
    """
    Reusable timeout context manager (ACTIVE definition).

    NOTE: This is the second definition of `Timeout` in this module. It
    overrides the async-context-manager version defined earlier (around
    line 132). This version uses the plain SYNCHRONOUS protocol
    (``with`` / ``__enter__`` / ``__exit__``). The underlying timer is still
    ``asyncio.timeout`` from Python 3.11+.

    THE KITCHEN-TIMER ANALOGY:
    --------------------------
    Think of this class as a kitchen timer: you twist it to N seconds
    (``Timeout(5.0)``), start it when you ``with timeout:`` (enter the
    context), and if your eggs... er, code... aren't done when the timer
    rings, the timer goes off and the code inside the block is cancelled,
    raising ``TimeoutError`` (see module docstring for why TimeoutError and
    asyncio.TimeoutError are the SAME exception since Python 3.11).
    Unlike a kitchen timer, you can peek at it while cooking: ``remaining``
    tells you how much time is left, and ``expired`` tells you whether it
    already rang — without any exception being raised.

    ``asyncio.wait_for`` VS ``asyncio.timeout``:
    --------------------------------------------
    - ``asyncio.wait_for(coro, timeout)`` wraps a SINGLE coroutine and is
      the classic pre-3.11 approach; use it when you must support Python
      <= 3.10 or want a one-liner around one awaitable.
    - ``asyncio.timeout(seconds)`` (3.11+) wraps a WHOLE BLOCK of code —
      many awaits, loops, whatever — and is the modern, preferred style.
    This class wraps the modern context-manager style, adding reusability
    and the ``expired`` / ``remaining`` inspection properties.

    Args (constructor):
        seconds (float | None): Timeout in seconds. ``None`` means
            "no timeout — wait forever", which is handy when the timeout
            duration comes from configuration that may disable it.

    Attributes / Properties:
        expired (bool): True if the deadline has already passed.
        remaining (float | None): Seconds left before the timer rings,
            or None when no timeout was configured.

    Examples:
        Basic usage (synchronous protocol)::

            timeout = Timeout(5.0)      # wind the timer to 5 seconds
            with timeout:               # timer starts on entry
                await some_operation()  # must finish within 5s
            # TimeoutError raised here if it took longer

        Inspecting status without exceptions::

            with timeout:
                ...
            if timeout.expired:
                print("Time's up!")
            print(f"Seconds remaining: {timeout.remaining}")

        No timeout at all::

            with Timeout(None):         # timer disabled
                await some_operation()  # may take as long as it wants
    """

    def __init__(self, seconds: float | None) -> None:
        """
        Initialize (but do not start) the timer.

        Args:
            seconds (float | None): How many seconds the block may take,
                or None to disable the timeout entirely.
        """
        # Store the requested duration; the deadline is only computed when
        # the context is entered, because "5 seconds" means "5 seconds FROM
        # NOW", and "now" only makes sense once we start.
        self._seconds = seconds
        self._deadline: float | None = None
        # Will hold the real asyncio.timeout context manager instance.
        self._cm: Any = None

    def __enter__(self) -> "Timeout":
        """
        Enter the context — this is where the timer actually starts.

        Returns:
            Timeout: ``self``, so ``with Timeout(5) as t:`` gives you the
            same object back (useful for checking ``t.remaining`` inside).
        """
        if self._seconds is not None:
            # monotonic() is the right clock for durations: it never jumps
            # backwards due to NTP adjustments or daylight saving.
            self._deadline = time.monotonic() + self._seconds
            # Delegate the real timeout machinery to asyncio.timeout (3.11+).
            self._cm = asyncio.timeout(self._seconds)
            self._cm.__enter__()
        return self

    def __exit__(self, *args: Any) -> bool | None:
        """
        Exit the context — stops the timer / forwards the exit to asyncio.

        If the timeout fired inside the block, ``asyncio.timeout``'s exit
        converts the internal cancellation into a ``TimeoutError`` here.
        Returns whatever the inner context manager returns (its exit value
        decides whether an exception gets suppressed — normally it doesn't).
        """
        if self._cm:
            # Pass through exc_type/exc_value/traceback unchanged.
            return self._cm.__exit__(*args)
        return None

    @property
    def expired(self) -> bool:
        """
        Has the timer already rung?

        Returns:
            bool: True if the deadline has passed; False if the timeout was
            never set (``seconds=None``) or time is still left.
        """
        if self._deadline is None:
            # No deadline can never expire.
            return False
        return time.monotonic() >= self._deadline

    @property
    def remaining(self) -> float | None:
        """
        How much time is left on the timer, in seconds.

        Returns:
            float | None: Seconds remaining (clamped at 0 once expired),
            or None if no timeout was configured.
        """
        if self._deadline is None:
            return None
        # max(0, ...) avoids returning a tiny negative number right after expiry.
        return max(0, self._deadline - time.monotonic())


@asynccontextmanager
async def timeout_after(seconds: float | None) -> AsyncIterator[None]:
    """
    Async context manager for a timeout — friendly alias for ``asyncio.timeout``.

    The ONLY reason this wrapper exists is the ``None`` case: passing
    ``None`` to ``asyncio.timeout`` in some setups is awkward; here it
    cleanly means "no timeout — run the block for as long as it needs".
    For any real number this is a thin, zero-behavior-change wrapper around
    ``asyncio.timeout(seconds)`` (Python 3.11+), which raises the unified
    builtin ``TimeoutError`` when the block overruns.

    Args:
        seconds (float | None): Maximum time the wrapped block may run,
            in seconds, or None to disable the timeout.

    Yields:
        None: The context manager exists purely for its side effect
        (enforcing the time limit around the ``async with`` block).

    Examples:
        With a real timeout::

            async with timeout_after(5.0):
                await long_operation()   # TimeoutError if > 5s

        Timeout disabled — same code path, waits forever::

            async with timeout_after(None):
                await long_operation()
    """
    if seconds is None:
        # No timeout requested: just yield so the block runs unhindered.
        yield
    else:
        # Real timeout: nest inside asyncio.timeout's async context manager.
        async with asyncio.timeout(seconds):
            yield


async def run_with_timeout(
    coro: asyncio.coroutines.Coroutine[Any, Any, T],
    timeout: float | None,
    *,
    default: T | None = None,
) -> TimeoutResult:
    """
    Run a coroutine with a timeout and return a structured TimeoutResult.

    RECOMMENDED for beginners: instead of writing ``try/except TimeoutError``
    everywhere, this helper catches everything and returns an inspectable
    result object. It also gives you a ``default`` fallback value for the
    timeout case — like ordering pizza but having leftovers in the fridge:
    if the delivery (your coroutine) doesn't arrive before the timer rings,
    you eat the leftovers (the default).

    Uses ``timeout_after`` internally, which in turn uses the modern
    ``asyncio.timeout`` context manager (Python 3.11+). Because 3.11 unified
    ``asyncio.TimeoutError`` with the builtin ``TimeoutError``, a single
    ``except TimeoutError`` here catches it — no import juggling required.
    (If you must target Python <= 3.10, prefer ``asyncio.wait_for`` instead.)

    Args:
        coro: The coroutine to run, e.g. ``fetch_data()``. Note you pass
            the CALL (``fetch_data()``), not the function itself.
        timeout (float | None): Maximum seconds to wait, or None for
            no timeout (wait forever).
        default (T | None, keyword-only): Fallback value stored in
            ``result.value`` when a timeout occurs. Defaults to None.

    Returns:
        TimeoutResult: A dataclass with fields:
            - ``success``: True only if the coroutine finished in time.
            - ``value``: The coroutine's result, or ``default`` on timeout.
            - ``error``: The exception raised (TimeoutError or other), or None.
            - ``duration``: Actual wall time in seconds.
            - ``timed_out``: True specifically for timeouts (vs. other errors).

    Examples::

        result = await run_with_timeout(fetch_data(), 2.0, default="cached")
        if result.success:
            print(result.value)          # real data
        elif result.timed_out:
            print(result.value)          # "cached" fallback
        else:
            print(f"Unexpected error: {result.error}")
    """
    start = time.monotonic()  # monotonic clock: immune to wall-clock jumps
    try:
        async with timeout_after(timeout):
            value = await coro
        # Happy path: finished within the deadline.
        return TimeoutResult(success=True, value=value, duration=time.monotonic() - start)
    except TimeoutError as e:
        # Timer rang. In 3.11+ this is the builtin TimeoutError (same object
        # as asyncio.TimeoutError), so one except clause suffices.
        return TimeoutResult(
            success=False, value=default, error=e, duration=time.monotonic() - start, timed_out=True
        )
    except Exception as e:
        # Something else went wrong (bug, network error, ...) — report it
        # too, but not flagged as a timeout.
        return TimeoutResult(success=False, error=e, duration=time.monotonic() - start)


class TimeoutManager:
    """
    Manage multiple named timeouts (ACTIVE definition).

    NOTE: Second definition in this module — overrides the earlier one
    (around line 299). This version's ``cancel()`` is a PLAIN synchronous
    method, matching the synchronous ``__enter__``/``__exit__`` protocol of
    the active ``Timeout`` class below... err, above.

    Useful when several concurrent operations each need their own timer and
    you want to track or cancel them by name — like a chef with several
    kitchen timers, one per pot on the stove, each labelled so they can
    check "has the pasta timer gone off yet?".

    Attributes:
        _timeouts (dict[str, Timeout]): name -> Timeout registry.

    Example::

        manager = TimeoutManager()
        with manager.create("db-query", 5.0):
            await run_query()
        with manager.create("api-call", 10.0):
            await call_api()
        if manager.get("db-query").expired:
            print("Database was too slow!")
        manager.cancel("api-call")  # stop that timer by name
    """

    def __init__(self) -> None:
        """Initialize an empty registry of named timeouts."""
        self._timeouts: dict[str, Timeout] = {}

    def create(self, name: str, seconds: float | None) -> Timeout:
        """
        Create a named timeout and register it for later lookup/cancellation.

        Args:
            name (str): Unique label, e.g. ``"worker-1"``. Reusing a name
                silently replaces the old timeout (dict semantics).
            seconds (float | None): Timeout duration, or None for "no limit".

        Returns:
            Timeout: The new timeout object, ready to use with ``with``.
        """
        timeout = Timeout(seconds)
        self._timeouts[name] = timeout
        return timeout

    def get(self, name: str) -> Timeout | None:
        """
        Look up a timeout by name.

        Args:
            name (str): The label passed to :meth:`create`.

        Returns:
            Timeout | None: The timeout if the name exists, else None.
        """
        # dict.get returns None instead of raising KeyError for unknown names.
        return self._timeouts.get(name)

    def cancel(self, name: str) -> bool:
        """
        Cancel a named timeout by entering and immediately exiting it.

        This takes the pot off the stove: the timer is stopped before it can
        fire, and it is removed from the registry so ``get(name)`` returns
        None afterwards. (Note: unlike the first definition block's version,
        this method is synchronous.)

        Args:
            name (str): Name of the timeout to cancel.

        Returns:
            bool: True if the timeout existed and was cancelled,
            False if no timeout with that name was found.
        """
        if name in self._timeouts:
            timeout = self._timeouts[name]
            # Enter then immediately exit: exits the asyncio.timeout context
            # cleanly, which disarms it.
            timeout.__enter__()
            timeout.__exit__(None, None, None)
            del self._timeouts[name]
            return True
        return False

    def __init__(self, seconds: float | None) -> None:
        self._seconds = seconds
        self._deadline: float | None = None
        self._cm: Any = None

    def __enter__(self) -> "Timeout":
        if self._seconds is not None:
            self._deadline = time.monotonic() + self._seconds
            self._cm = asyncio.timeout(self._seconds)
            self._cm.__enter__()
        return self

    def __exit__(self, *args: Any) -> bool | None:
        if self._cm:
            return self._cm.__exit__(*args)
        return None

    @property
    def expired(self) -> bool:
        if self._deadline is None:
            return False
        return time.monotonic() >= self._deadline

    @property
    def remaining(self) -> float | None:
        if self._deadline is None:
            return None
        return max(0, self._deadline - time.monotonic())


@asynccontextmanager
async def timeout_after(seconds: float | None) -> AsyncIterator[None]:
    """Context manager for timeout (alias for asyncio.timeout)."""
    if seconds is None:
        yield
    else:
        async with asyncio.timeout(seconds):
            yield


async def run_with_timeout(
    coro: asyncio.coroutines.Coroutine[Any, Any, T],
    timeout: float | None,
    *,
    default: T | None = None,
) -> TimeoutResult:
    """Run coroutine with timeout, returning result object."""
    start = time.monotonic()
    try:
        async with timeout_after(timeout):
            value = await coro
        return TimeoutResult(success=True, value=value, duration=time.monotonic() - start)
    except TimeoutError as e:
        return TimeoutResult(
            success=False, value=default, error=e, duration=time.monotonic() - start, timed_out=True
        )
    except Exception as e:
        return TimeoutResult(success=False, error=e, duration=time.monotonic() - start)


class TimeoutManager:
    """Manage multiple timeouts."""

    def __init__(self) -> None:
        self._timeouts: dict[str, Timeout] = {}

    def create(self, name: str, seconds: float | None) -> Timeout:
        """Create named timeout."""
        timeout = Timeout(seconds)
        self._timeouts[name] = timeout
        return timeout

    def get(self, name: str) -> Timeout | None:
        return self._timeouts.get(name)

    def cancel(self, name: str) -> bool:
        """Cancel timeout by entering and exiting immediately."""
        if name in self._timeouts:
            timeout = self._timeouts[name]
            timeout.__enter__()
            timeout.__exit__(None, None, None)
            del self._timeouts[name]
            return True
        return False


async def demonstrate_timeouts() -> None:
    """
    Demonstrate all timeout patterns (ACTIVE definition — outputs to console).

    NOTE: This second definition overrides the earlier, more verbose demo
    (around line 371). Run the module to see every pattern live:

        python -m asyncio_mastery.advanced.timeouts

    Patterns shown, in order:
    1. ``asyncio.timeout(0.1)`` — relative timeout (3.11+); the 1-second
       operation overruns the 0.1s limit and raises ``TimeoutError``.
    2. ``asyncio.timeout_at(deadline)`` — absolute deadline computed as
       ``time.monotonic() + 0.1``; same TimeoutError on overrun.
    3. ``run_with_timeout`` — the structured-result helper: no exceptions,
       inspect ``result.success`` / ``result.timed_out`` / ``result.value``.
    4. Timeout with a ``default`` fallback value ("fallback" string).
    5. ``TimeoutManager`` — two concurrent workers with different named
       timeouts; results collected via ``gather(..., return_exceptions=True)``.
    6. ``asyncio.shield`` — protects a critical operation from the timeout:
       the TimeoutError fires, but the shielded coroutine keeps running in
       the background to completion.

    Returns:
        None: Everything is printed to stdout for learning purposes.
    """
    print("=== Timeouts Demo ===")

    # Basic timeout (Python 3.11+)
    print("\n--- asyncio.timeout ---")

    async def slow_operation() -> str:
        await asyncio.sleep(1)
        return "done"

    try:
        async with asyncio.timeout(0.1):
            await slow_operation()
    except TimeoutError:
        print("Operation timed out!")

    # timeout_at with deadline
    print("\n--- asyncio.timeout_at ---")
    deadline = time.monotonic() + 0.1
    try:
        async with asyncio.timeout_at(deadline):
            await slow_operation()
    except TimeoutError:
        print("Deadline exceeded!")

    # run_with_timeout helper
    print("\n--- run_with_timeout ---")
    result = await run_with_timeout(slow_operation(), 0.1)
    print(
        f"Success: {result.success}, Timed out: {result.timed_out}, "
        f"Duration: {result.duration:.3f}s"
    )

    result = await run_with_timeout(asyncio.sleep(0.01, result="fast"), 1.0)
    print(f"Success: {result.success}, Value: {result.value}")

    # Timeout with default
    print("\n--- Timeout with Default ---")
    result = await run_with_timeout(slow_operation(), 0.1, default="fallback")
    print(f"Value: {result.value}")

    # Timeout Manager
    print("\n--- Timeout Manager ---")
    manager = TimeoutManager()

    async def worker(id: int, timeout: float) -> str:
        async with manager.create(f"worker-{id}", timeout):
            await asyncio.sleep(timeout * 2)
            return f"worker-{id} done"

    results = await asyncio.gather(worker(1, 0.5), worker(2, 0.1), return_exceptions=True)
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            print(f"Worker {i + 1}: {type(r).__name__}")
        else:
            print(f"Worker {i + 1}: {r}")

    # Shield from timeout
    print("\n--- Shield from Timeout ---")

    async def critical_operation() -> str:
        await asyncio.sleep(0.2)
        return "critical done"

    try:
        async with asyncio.timeout(0.1):
            result = await asyncio.shield(critical_operation())
            print(f"Shielded result: {result}")
    except TimeoutError:
        print("Timeout but shielded operation continues in background")

    # Wait a bit for shielded to complete
    await asyncio.sleep(0.3)


if __name__ == "__main__":
    asyncio.run(demonstrate_timeouts())
