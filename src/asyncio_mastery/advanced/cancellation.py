"""
Cancellation - Cooperative task cancellation in asyncio.

WHAT IS CANCELLATION
--------------------
Cancellation is how you ask a running task to stop early. The key word is
"ask". Asyncio cancellation is COOPERATIVE. A task is only cancelled at an
``await`` point, where it receives an ``asyncio.CancelledError`` exception.

MENTAL MODEL
------------
Think of cancellation like pulling the plug on a washing machine mid-cycle.
The machine gets one brief moment (the ``CancelledError``) to drain the water
and close the door (your cleanup code in ``except``/``finally`` blocks)
before it truly stops. You must let that moment happen. Never swallow
``CancelledError`` without re-raising it. If you swallow it, the task
"refuses to stop" and breaks asyncio's bookkeeping.

THE CATCH -> CLEANUP -> RE-RAISE PATTERN
----------------------------------------
The correct pattern inside a cancellable coroutine is::

    try:
        await do_work()
    except asyncio.CancelledError:
        # 1. CATCH: the task has been asked to stop.
        # 2. CLEANUP: release resources, close connections, etc.
        await cleanup()
        # 3. RE-RAISE: always re-raise so the task actually ends as
        #    "cancelled". Swallowing it makes the task look successful!
        raise

WHEN YOU NEED THIS
------------------
- Stopping work that is no longer needed (user navigated away, a newer
  request superseded an old one).
- Shutting down a server gracefully (cancel all background tasks).
- Implementing timeouts (a timeout IS a scheduled cancellation).

MINIMAL EXAMPLE
---------------
>>> import asyncio
>>> async def main():
...     task = asyncio.create_task(asyncio.sleep(10))
...     task.cancel()                     # request cancellation
...     try:
...         await task
...     except asyncio.CancelledError:
...         print("Task was cancelled!")

Run the full demo with: ``python -m asyncio_mastery.advanced.cancellation``
"""

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

T = TypeVar("T")


class CancellationReason(Enum):
    """Reason why a cancellation was requested.

    Tagging a cancellation with a reason makes debugging easier. Instead of
    a bare "cancelled", you can tell whether the task stopped because of a
    timeout, a shutdown, a user clicking "cancel", and so on.
    """

    TIMEOUT = "timeout"
    SHUTDOWN = "shutdown"
    USER_REQUEST = "user_request"
    SUPERSEDED = "superseded"
    ERROR = "error"


@dataclass
class CancellationToken:
    """A cooperative cancellation signal that tasks can poll.

    Unlike ``task.cancel()`` (which forces an exception into the task), a
    token is a *polite* flag. Workers check ``throw_if_cancelled()`` at
    convenient points and stop themselves. This is a common pattern borrowed
    from C#/.NET.

    Example:
        >>> token = CancellationToken()
        >>> async def worker():
        ...     while True:
        ...         token.throw_if_cancelled()  # raises CancelledError if set
        ...         await asyncio.sleep(0.1)
        >>> token.cancel(CancellationReason.TIMEOUT)
    """

    _cancelled: bool = False
    _reason: CancellationReason | None = None
    _callbacks: list[Callable[[], None]] = field(default_factory=list)

    @property
    def cancelled(self) -> bool:
        """True once :meth:`cancel` has been called (one-way switch)."""
        return self._cancelled

    @property
    def reason(self) -> CancellationReason | None:
        """Why cancellation was requested, or None if not yet cancelled."""
        return self._reason

    def cancel(self, reason: CancellationReason = CancellationReason.USER_REQUEST) -> None:
        """Request cancellation: flip the flag, record why, fire callbacks.

        Args:
            reason: Why cancellation is happening (defaults to USER_REQUEST).
        """
        if not self._cancelled:
            self._cancelled = True
            self._reason = reason
            for cb in self._callbacks:
                # A misbehaving callback must not break cancellation itself,
                # so we swallow any exception it raises.
                with suppress(Exception):
                    cb()

    def add_callback(self, callback: Callable[[], None]) -> None:
        """Register a zero-argument function to run when cancelled.

        Args:
            callback: Called synchronously inside :meth:`cancel`.
        """
        self._callbacks.append(callback)

    def throw_if_cancelled(self) -> None:
        """Raise ``asyncio.CancelledError`` if the token has been cancelled.

        Call this at points in your worker where stopping is safe.

        Raises:
            asyncio.CancelledError: If cancellation was requested.
        """
        if self._cancelled:
            raise asyncio.CancelledError(self._reason.value if self._reason else "Cancelled")


@asynccontextmanager
async def cancel_on(
    *conditions: asyncio.Event | asyncio.Future | CancellationToken,
    reason: CancellationReason = CancellationReason.USER_REQUEST,
) -> AsyncIterator[None]:
    """Async context manager: run a background watcher while inside the block.

    A watcher task monitors the given conditions (an ``asyncio.Event`` being
    set, a ``Future`` completing, or a ``CancellationToken`` being cancelled)
    and records the first one that fires. On exit, the watcher is always
    cleaned up so no stray task is left running.

    Args:
        *conditions: Events, Futures, or CancellationTokens to watch.
        reason: The cancellation reason associated with this block.

    Yields:
        Nothing; the block simply runs with the watcher active.

    Example:
        >>> stop = asyncio.Event()
        >>> async with cancel_on(stop, reason=CancellationReason.SHUTDOWN):
        ...     await do_work()  # watcher runs concurrently
    """
    done_event = asyncio.Event()

    async def watcher() -> None:
        # Poll/wait on each condition in turn; stop at the first that fires.
        for cond in conditions:
            if isinstance(cond, asyncio.Event):
                await cond.wait()
            elif isinstance(cond, asyncio.Future):
                await cond
            elif isinstance(cond, CancellationToken):
                # Tokens have no "wait", so poll with a small sleep to stay
                # cooperative (yield to the event loop between checks).
                while not cond.cancelled:
                    await asyncio.sleep(0.1)
            else:
                continue
            done_event.set()
            break

    watcher_task = asyncio.create_task(watcher())
    try:
        yield
    finally:
        # Whatever happens inside the block (success, error, cancellation),
        # the watcher must not outlive it. This is the catch -> cleanup ->
        # re-raise pattern in miniature: cancel the watcher, then await it
        # while suppressing the CancelledError we just caused ourselves.
        watcher_task.cancel()
        with suppress(asyncio.CancelledError):
            await watcher_task
        watcher_task.cancel()
        with suppress(asyncio.CancelledError):
            await watcher_task


async def shield(coro: asyncio.coroutines.Coroutine[Any, Any, T]) -> T:
    """Protect ``coro`` from cancellation of the surrounding task.

    If the outer task is cancelled, ``await shield(...)`` raises
    ``CancelledError`` in the outer task, but ``coro`` keeps running in the
    background -- like an umbrella: YOU get wet (the await raises) while the
    work underneath stays dry (keeps executing).

    Args:
        coro: The coroutine to protect.

    Returns:
        Whatever ``coro`` returns, if it finishes before cancellation.

    Raises:
        asyncio.CancelledError: If the outer task is cancelled (the inner
            coroutine still continues running).
    """
    return await asyncio.shield(coro)


class CancellationScope:
    """A group of tasks that can be cancelled together as one unit.

    Mental model: a scope is like a circuit breaker for a whole room --
    flip one switch and every appliance (task) in the room turns off.

    Example:
        >>> scope = CancellationScope()
        >>> scope.create_task(job_a())
        >>> scope.create_task(job_b())
        >>> scope.cancel_all()          # stops both tasks
        >>> results = await scope.wait_all()
    """

    def __init__(self, *, reason: CancellationReason = CancellationReason.SHUTDOWN) -> None:
        """Create a scope.

        Args:
            reason: Default reason used by :meth:`cancel_all`.
        """
        self._reason = reason
        self._tasks: set[asyncio.Task] = set()
        self._cancelled = False

    def create_task(self, coro: asyncio.coroutines.Coroutine[Any, Any, T]) -> asyncio.Task[T]:
        """Create a task owned by this scope.

        Args:
            coro: The coroutine to schedule.

        Returns:
            The new ``asyncio.Task``.

        Raises:
            RuntimeError: If the scope has already been cancelled.
        """
        if self._cancelled:
            raise RuntimeError("Scope already cancelled")
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        # Auto-remove from the set when finished so the set only ever holds
        # live tasks (prevents unbounded memory growth).
        task.add_done_callback(self._tasks.discard)
        return task

    def cancel_all(self, reason: CancellationReason | None = None) -> None:
        """Cancel every still-running task in this scope.

        Args:
            reason: Optional override; defaults to the scope's reason. The
                reason string is attached to the CancelledError message.
        """
        self._cancelled = True
        r = reason or self._reason
        for task in self._tasks:
            task.cancel(r.value if isinstance(r, CancellationReason) else str(r))

    async def wait_all(self) -> list[Any]:
        """Wait for all tasks and collect results, never raising.

        Both successful return values AND exceptions (including
        ``CancelledError``) are collected into the returned list, in set
        iteration order. Note: we catch ``CancelledError`` here as a *result*,
        which is the one place catching without re-raising is acceptable --
        this task is not being cancelled, we are just observing others.

        Returns:
            A list mixing return values and exception objects.
        """
        results = []
        for task in self._tasks:
            try:
                results.append(await task)
            except asyncio.CancelledError as e:
                results.append(e)
            except Exception as e:
                results.append(e)
        return results


async def demonstrate_cancellation() -> None:
    """Run a narrated tour of every cancellation pattern in this module.

    Covers: basic ``task.cancel()``, cleanup with try/except/finally,
    shielding, CancellationToken, cancel_on, and CancellationScope.
    Run with: ``python -m asyncio_mastery.advanced.cancellation``.
    """
    print("=== Cancellation Demo ===")

    # Basic cancellation
    print("\n--- Basic Cancellation ---")

    async def long_task() -> str:
        try:
            for i in range(10):
                print(f"  Working... {i}")
                await asyncio.sleep(0.1)
            return "done"
        except asyncio.CancelledError:
            # Catch for cleanup/logging, then RE-RAISE so the task still
            # ends in the "cancelled" state. Forgetting `raise` here is the
            # most common asyncio cancellation bug.
            print("  Task cancelled!")
            raise

    task = asyncio.create_task(long_task())
    await asyncio.sleep(0.25)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        print("Cancellation caught")

    # Cancellation with cleanup
    print("\n--- Cancellation with Cleanup ---")

    async def task_with_cleanup() -> None:
        resource = "acquired"
        print(f"  Resource: {resource}")
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            print("  Cleaning up resource...")
            resource = "released"
            print(f"  Resource: {resource}")
            raise
        finally:
            print("  Finally block")

    task = asyncio.create_task(task_with_cleanup())
    await asyncio.sleep(0.1)
    try:
        task.cancel()
    finally:
        # finally guarantees the cancelled task is always awaited, even if
        # something unexpected interrupts us. gather(..., return_exceptions=True)
        # hands us the CancelledError as a plain result instead of raising it here.
        await asyncio.gather(task, return_exceptions=True)

    # Shield
    print("\n--- Shield ---")

    async def critical() -> str:
        await asyncio.sleep(0.2)
        return "critical complete"

    async def cancel_after_shield() -> None:
        try:
            async with asyncio.timeout(0.1):
                result = await asyncio.shield(critical())
                print(f"Shielded: {result}")
        except TimeoutError:
            print("Outer timeout, but shielded continues...")

    await cancel_after_shield()
    await asyncio.sleep(0.3)  # Wait for shielded to finish

    # CancellationToken
    print("\n--- CancellationToken ---")
    token = CancellationToken()

    async def token_worker(token: CancellationToken) -> None:
        for i in range(5):
            token.throw_if_cancelled()
            print(f"  Working {i}")
            await asyncio.sleep(0.1)

    task = asyncio.create_task(token_worker(token))
    await asyncio.sleep(0.25)
    token.cancel(CancellationReason.TIMEOUT)
    try:
        await task
    except asyncio.CancelledError as e:
        print(f"Cancelled with reason: {e}")

    # cancel_on
    print("\n--- cancel_on ---")
    stop_event = asyncio.Event()

    async def cancellable_work() -> None:
        async with cancel_on(stop_event, reason=CancellationReason.SHUTDOWN):
            for i in range(10):
                print(f"  Step {i}")
                await asyncio.sleep(0.1)

    task = asyncio.create_task(cancellable_work())
    await asyncio.sleep(0.25)
    stop_event.set()
    try:
        await task
    except asyncio.CancelledError:
        print("Cancelled via event")

    # CancellationScope
    print("\n--- CancellationScope ---")
    scope = CancellationScope()

    async def scope_worker(name: str) -> str:
        for i in range(5):
            print(f"  {name}: {i}")
            await asyncio.sleep(0.1)
        return f"{name} done"

    scope.create_task(scope_worker("A"))
    scope.create_task(scope_worker("B"))

    await asyncio.sleep(0.25)
    scope.cancel_all()
    results = await scope.wait_all()
    print(f"Results: {results}")


if __name__ == "__main__":
    asyncio.run(demonstrate_cancellation())
