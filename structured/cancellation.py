"""
Cancellation Patterns

Advanced cancellation handling for async operations.
"""

import asyncio
import contextvars
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine
from contextlib import asynccontextmanager, suppress
from typing import Any, Generic, Optional, TypeVar

T = TypeVar("T")


class CancellableError(Exception):
    """Base exception for cancellation-related errors."""

    pass


class CancellationRequested(CancellableError):
    """Raised when cancellation is requested."""

    def __init__(
        self, message: str = "Cancellation requested", token: Optional["CancellationToken"] = None
    ) -> None:
        super().__init__(message)
        self.token = token


class CancellationScope:
    """Scope for cooperative cancellation."""

    def __init__(
        self,
        *,
        deadline: float | None = None,
        shield: bool = False,
        parent: Optional["CancellationScope"] = None,
    ) -> None:
        self._deadline = deadline
        self._shield = shield
        self._parent = parent
        self._cancelled = False
        self._cancel_msg: str | None = None
        self._children: set[CancellationScope] = set()
        self._callbacks: list[Callable[[], None]] = []
        self._cancelled_at: float | None = None

        if parent:
            parent._children.add(self)

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    @property
    def deadline(self) -> float | None:
        return self._deadline

    @property
    def shield(self) -> bool:
        return self._shield

    @property
    def cancel_message(self) -> str | None:
        return self._cancel_msg

    def cancel(self, msg: str | None = None) -> None:
        """Cancel this scope and all children."""
        if self._cancelled:
            return

        self._cancelled = True
        self._cancel_msg = msg
        self._cancelled_at = time.monotonic()

        # Cancel children
        for child in self._children:
            child.cancel(msg)

        # Fire callbacks
        for cb in self._callbacks:
            with suppress(Exception):
                cb()

    def check_cancelled(self) -> None:
        """Check if cancelled and raise if so."""
        if self._cancelled:
            raise CancellationRequested(self._cancel_msg or "Cancellation requested", token=None)

        if self._deadline and time.monotonic() > self._deadline:
            self.cancel("Deadline exceeded")
            raise CancellationRequested("Deadline exceeded")

    def add_callback(self, callback: Callable[[], None]) -> None:
        """Add callback to run on cancellation."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[], None]) -> bool:
        """Remove a cancellation callback."""
        try:
            self._callbacks.remove(callback)
            return True
        except ValueError:
            return False

    def time_remaining(self) -> float | None:
        """Get time remaining until deadline."""
        if self._deadline is None:
            return None
        remaining = self._deadline - time.monotonic()
        return max(0, remaining)

    def __enter__(self) -> "CancellationScope":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self._parent:
            self._parent._children.discard(self)
        return False


class CancellationToken:
    """Token for explicit cancellation handling."""

    def __init__(self, *, initial: bool = False) -> None:
        self._cancelled = initial
        self._cancel_msg: str | None = None
        self._callbacks: list[Callable[[], Awaitable[None]]] = []
        self._event = asyncio.Event()
        if initial:
            self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    @property
    def cancel_message(self) -> str | None:
        return self._cancel_msg

    def cancel(self, msg: str | None = None) -> None:
        """Cancel the token."""
        if self._cancelled:
            return

        self._cancelled = True
        self._cancel_msg = msg
        self._event.set()

        # Fire async callbacks
        for cb in self._callbacks:
            with suppress(Exception):
                asyncio.create_task(cb())

    def register_callback(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Register async callback for cancellation."""
        self._callbacks.append(callback)

    def unregister_callback(self, callback: Callable[[], Awaitable[None]]) -> bool:
        """Unregister callback."""
        try:
            self._callbacks.remove(callback)
            return True
        except ValueError:
            return False

    async def wait(self) -> None:
        """Wait for cancellation."""
        await self._event.wait()

    def throw_if_cancelled(self) -> None:
        """Raise if cancelled."""
        if self._cancelled:
            raise CancellationRequested(self._cancel_msg or "Cancelled")

    def __bool__(self) -> bool:
        return self._cancelled


@asynccontextmanager
async def cancel_after(delay: float, *, msg: str | None = None):
    """Context manager that cancels after delay."""
    token = CancellationToken()

    async def cancel_task() -> None:
        await asyncio.sleep(delay)
        token.cancel(msg or f"Cancelled after {delay}s")

    cancel_future = asyncio.create_task(cancel_task())

    try:
        yield token
    finally:
        cancel_future.cancel()
        with suppress(asyncio.CancelledError):
            await cancel_future


def shield(awaitable: Awaitable[T]) -> Awaitable[T]:
    """Shield awaitable from cancellation."""
    return asyncio.shield(awaitable)


class CancellableTask(Generic[T]):
    """Task that can be cancelled cooperatively."""

    def __init__(
        self,
        coro: Coroutine[Any, Any, T],
        *,
        name: str | None = None,
        token: CancellationToken | None = None,
    ) -> None:
        self._coro = coro
        self._name = name or getattr(coro, "__qualname__", "task")
        self._token = token or CancellationToken()
        self._task: asyncio.Task | None = None
        self._result: T | None = None
        self._exception: BaseException | None = None
        self._done = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def token(self) -> CancellationToken:
        return self._token

    @property
    def done(self) -> bool:
        return self._done

    @property
    def cancelled(self) -> bool:
        return self._token.cancelled

    def start(self) -> asyncio.Task[T]:
        """Start the task."""
        if self._task is not None:
            raise RuntimeError("Task already started")

        async def wrapped():
            try:
                # Check token periodically
                while not self._token.cancelled:
                    try:
                        # Run with short timeout to check cancellation
                        self._result = await asyncio.wait_for(self._coro, timeout=0.1)
                        break
                    except TimeoutError:
                        # Check cancellation and continue
                        self._token.throw_if_cancelled()
                        continue

                if self._token.cancelled:
                    raise CancellationRequested("Task cancelled")

            except CancellationRequested:
                raise
            except BaseException as e:
                self._exception = e
                raise
            finally:
                self._done = True

            return self._result

        self._task = asyncio.create_task(wrapped(), name=self._name)
        return self._task

    async def wait(self) -> T:
        """Wait for task completion."""
        if self._task is None:
            self.start()

        try:
            return await self._task
        except CancellationRequested:
            self._exception = CancellationRequested("Task cancelled")
            raise

    def cancel(self, msg: str | None = None) -> bool:
        """Cancel the task."""
        self._token.cancel(msg)
        if self._task and not self._task.done():
            self._task.cancel(msg)
            return True
        return False

    def result(self) -> T:
        """Get task result."""
        if not self._done:
            raise RuntimeError("Task not done")
        if self._exception:
            raise self._exception
        return self._result

    def exception(self) -> BaseException | None:
        return self._exception


async def with_cancellation(
    coro: Awaitable[T], token: CancellationToken | None = None, *, timeout: float | None = None
) -> T:
    """Run coroutine with cancellation token."""
    token = token or CancellationToken()

    async def run_with_token():
        # Wrap coroutine to check token
        async def wrapped():
            try:
                return await coro
            except asyncio.CancelledError:
                if token.cancelled:
                    raise CancellationRequested(token.cancel_message)
                raise

        if timeout:
            return await asyncio.wait_for(wrapped(), timeout=timeout)
        return await wrapped()

    # Start cancellation watcher
    async def watch_token() -> None:
        await token.wait()
        # Token cancelled, cancel the task
        pass

    watcher = asyncio.create_task(watch_token())

    try:
        return await run_with_token()
    finally:
        watcher.cancel()
        with suppress(asyncio.CancelledError):
            await watcher


class GracefulCancellation:
    """Graceful cancellation with cleanup phases."""

    def __init__(self, timeout: float = 30.0, *, phases: list[str] | None = None) -> None:
        self._timeout = timeout
        self._phases = phases or ["shutdown", "cleanup", "finalize"]
        self._callbacks: dict[str, list[Callable[[], Awaitable[None]]]] = {
            phase: [] for phase in self._phases
        }
        self._cancelled = False
        self._phase_results: dict[str, Any] = {}

    def add_callback(self, phase: str, callback: Callable[[], Awaitable[None]]) -> None:
        """Add callback for a cancellation phase."""
        if phase not in self._callbacks:
            raise ValueError(f"Unknown phase: {phase}")
        self._callbacks[phase].append(callback)

    async def cancel(self, msg: str | None = None) -> None:
        """Execute graceful cancellation."""
        if self._cancelled:
            return

        self._cancelled = True

        for phase in self._phases:
            phase_start = time.monotonic()
            callbacks = self._callbacks[phase]

            if callbacks:
                with suppress(TimeoutError):
                    await asyncio.wait_for(
                        asyncio.gather(*[cb() for cb in callbacks], return_exceptions=True),
                        timeout=self._timeout / len(self._phases),
                    )

            self._phase_results[phase] = {
                "duration": time.monotonic() - phase_start,
                "callbacks": len(callbacks),
            }

    def is_cancelled(self) -> bool:
        return self._cancelled

    def get_results(self) -> dict[str, Any]:
        return self._phase_results.copy()


class CancellationGroup:
    """Group of cancellation tokens that cancel together."""

    def __init__(self) -> None:
        self._tokens: list[CancellationToken] = []
        self._lock = asyncio.Lock()

    def add_token(self, token: CancellationToken) -> None:
        self._tokens.append(token)

    def remove_token(self, token: CancellationToken) -> bool:
        try:
            self._tokens.remove(token)
            return True
        except ValueError:
            return False

    def cancel_all(self, msg: str | None = None) -> int:
        count = 0
        for token in self._tokens:
            if not token.cancelled:
                token.cancel(msg)
                count += 1
        return count

    @asynccontextmanager
    async def create_token(self) -> AsyncGenerator[CancellationToken, None]:
        token = CancellationToken()
        self.add_token(token)
        try:
            yield token
        finally:
            self.remove_token(token)


# Context variable for current cancellation scope
_current_scope: contextvars.ContextVar[CancellationScope | None] = contextvars.ContextVar(
    "cancellation_scope", default=None
)


def get_current_scope() -> CancellationScope | None:
    """Get current cancellation scope."""
    return _current_scope.get()


@asynccontextmanager
async def cancellation_scope(
    *, deadline: float | None = None, shield: bool = False
) -> AsyncGenerator[CancellationScope, None]:
    """Create a new cancellation scope."""
    parent = _current_scope.get()
    scope = CancellationScope(deadline=deadline, shield=shield, parent=parent)

    token = _current_scope.set(scope)
    try:
        yield scope
    finally:
        _current_scope.reset(token)
        if parent:
            parent._children.discard(scope)


def check_cancellation() -> None:
    """Check current cancellation scope."""
    scope = _current_scope.get()
    if scope:
        scope.check_cancelled()
