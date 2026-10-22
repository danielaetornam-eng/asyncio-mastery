"""
Context Variables - Task-local storage for async code.

WHAT ARE CONTEXT VARIABLES (contextvars)
----------------------------------------
A ``contextvars.ContextVar`` is a variable whose value depends on *which
task* is looking at it. Each asyncio task automatically carries its own copy
of the context, so values set inside one task are invisible to sibling tasks
-- yet they flow automatically into any coroutine or child task that task
awaits.

MENTAL MODEL
------------
Think of each task as a person carrying a keychain (the context). When a
task is created, it gets a *copy* of its parent's keychain. Whatever you
hang on your keychain (a request ID, a user ID, a trace ID) is visible to
everything you do -- and everything you delegate -- but nobody else's
keychain changes.

WHY NOT GLOBALS OR THREAD-LOCALS
--------------------------------
- Globals are shared by ALL tasks: one request would overwrite another's
  data.
- Thread-locals do not work because many tasks share a single OS thread;
  they would all see the same "thread-local" value. ContextVars switch with the
  task, not the thread.

WHEN YOU NEED THIS
------------------
- Attaching a request/trace/correlation ID to every log line, across many
  nested function calls, without passing it as an argument everywhere.
- Storing per-request auth info, database transactions, locale, etc.

MINIMAL EXAMPLE
---------------
>>> import contextvars
>>> request_id = contextvars.ContextVar("request_id", default=None)
>>> token = request_id.set("req-123")   # set returns a Token
>>> request_id.get()
'req-123'
>>> request_id.reset(token)             # restore the previous value

Run the full demo with: ``python -m asyncio_mastery.advanced.context_vars``
"""

import asyncio
import contextvars
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from typing import Any, TypeVar

# Define context variables. The "default" is returned by .get() when the
# variable was never set in the current context (avoids LookupError).
request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)
user_id_var: contextvars.ContextVar[int | None] = contextvars.ContextVar("user_id", default=None)
trace_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("trace_id", default=None)
correlation_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_id", default=None
)

T = TypeVar("T")


@dataclass
class ContextSnapshot:
    """A frozen picture of several context variables at one moment.

    Useful for "time travel": capture the current values, change them, and
    later put everything back with :meth:`restore`.

    Attributes:
        values: Mapping of ContextVar -> the value it held at snapshot time.
    """

    values: dict[contextvars.ContextVar[Any], Any] = field(default_factory=dict)

    def restore(self) -> list[contextvars.Token]:
        """Write the saved values back into the live context.

        Returns:
            The tokens produced by each ``var.set()``. A Token is a receipt
            that remembers the *previous* value, usable with ``var.reset()``.
        """
        tokens = []
        for var, value in self.values.items():
            tokens.append(var.set(value))
        return tokens


class ContextVarManager:
    """Helper that manages the request-scoped context vars as a group.

    Setting four variables by hand (and remembering to reset each one) is
    tedious and error-prone. This manager tracks every ``set`` token so a
    single ``clear()`` or ``with manager.context(...)`` restores everything.
    """

    def __init__(self) -> None:
        # Receipts from every set() we performed; needed to undo them later.
        self._tokens: list[contextvars.Token] = []

    def set_request_context(
        self,
        request_id: str,
        user_id: int | None = None,
        trace_id: str | None = None,
        correlation_id: str | None = None,
    ) -> None:
        """Set all request-scoped variables in one call.

        Args:
            request_id: Unique ID for the current request (required).
            user_id: Optional authenticated user ID.
            trace_id: Optional distributed-tracing ID.
            correlation_id: Optional ID linking related requests.
        """
        self._tokens.append(request_id_var.set(request_id))
        if user_id is not None:
            self._tokens.append(user_id_var.set(user_id))
        if trace_id is not None:
            self._tokens.append(trace_id_var.set(trace_id))
        if correlation_id is not None:
            self._tokens.append(correlation_id_var.set(correlation_id))

    def clear(self) -> None:
        """Undo every set performed through this manager.

        Tokens must be reset in REVERSE order (like unstacking plates), which
        is why we iterate ``reversed``. A token from the wrong context raises
        ``ValueError``; we suppress it so cleanup never fails loudly.
        """
        for token in reversed(self._tokens):
            with suppress(ValueError):
                request_id_var.reset(token)
        self._tokens.clear()

    def snapshot(self) -> ContextSnapshot:
        """Capture the current values of all managed variables.

        Returns:
            A ContextSnapshot that can later be ``restore()``d.
        """
        return ContextSnapshot(
            values={
                request_id_var: request_id_var.get(),
                user_id_var: user_id_var.get(),
                trace_id_var: trace_id_var.get(),
                correlation_id_var: correlation_id_var.get(),
            }
        )

    @contextmanager
    def context(self, **kwargs: Any) -> Any:
        """Temporarily set variables, automatically undoing on exit.

        Args:
            **kwargs: Mapping of ContextVar objects to values, e.g.
                ``manager.context(**{request_id_var: "r-1"})``.

        Yields:
            None. On block exit (even via exception) every set is reset,
            in reverse order.
        """
        tokens = []
        try:
            for var, value in kwargs.items():
                tokens.append(var.set(value))
            yield
        finally:
            # Reset in reverse: each token must be undone in the opposite
            # order it was created, like popping a stack.
            for token in reversed(tokens):
                var = token.var
                var.reset(token)


def get_request_id() -> str | None:
    """Return the current task's request ID (or None if unset)."""
    return request_id_var.get()


def get_user_id() -> int | None:
    """Return the current task's user ID (or None if unset)."""
    return user_id_var.get()


async def log_with_context(message: str) -> None:
    """Print a log line automatically prefixed with request/user context.

    Args:
        message: The log message; no need to pass IDs by hand -- they come
            from the context of whichever task calls this.
    """
    req_id = get_request_id()
    user_id = get_user_id()
    print(f"[req={req_id} user={user_id}] {message}")


async def demonstrate_context_vars() -> None:
    """Narrated tour of context variables.

    Shows set/get, propagation into awaited calls, isolation between
    concurrent tasks, the context manager, and snapshot/restore.
    Run with: ``python -m asyncio_mastery.advanced.context_vars``.
    """
    print("=== Context Variables Demo ===")

    # Basic usage
    print(f"Initial request_id: {get_request_id()}")

    token = request_id_var.set("req-123")
    print(f"After set: {get_request_id()}")

    request_id_var.reset(token)
    print(f"After reset: {get_request_id()}")

    # Context propagation
    print("\n--- Context Propagation ---")

    async def child_task() -> None:
        await log_with_context("Inside child task")

    async def parent_task() -> None:
        request_id_var.set("req-parent")
        await log_with_context("Inside parent task")
        await child_task()  # Context flows to child: the awaited coroutine
        # sees "req-parent" without it being passed as an argument.

    await parent_task()
    print(f"After parent: {get_request_id()}")

    # Concurrent tasks with different contexts
    print("\n--- Concurrent Contexts ---")

    async def worker(id: int) -> None:
        request_id_var.set(f"req-worker-{id}")
        await log_with_context(f"Worker {id} started")
        await asyncio.sleep(0.01)
        await log_with_context(f"Worker {id} finished")

    await asyncio.gather(worker(1), worker(2), worker(3))

    # Context manager
    print("\n--- Context Manager ---")
    manager = ContextVarManager()

    with manager.context(request_id_var="req-cm", user_id_var=42):
        await log_with_context("Inside context manager")

    await log_with_context("Outside context manager")

    # Snapshot/restore
    print("\n--- Snapshot/Restore ---")
    request_id_var.set("req-original")
    snapshot = manager.snapshot()

    request_id_var.set("req-modified")
    print(f"Modified: {get_request_id()}")

    snapshot.restore()
    print(f"Restored: {get_request_id()}")

    # Practical example: request tracing
    print("\n--- Request Tracing ---")

    async def handle_request(req_id: str) -> None:
        request_id_var.set(req_id)
        await process_request()

    async def process_request() -> None:
        await log_with_context("Processing request")
        await step1()
        await step2()

    async def step1() -> None:
        await log_with_context("Step 1")

    async def step2() -> None:
        await log_with_context("Step 2")

    await handle_request("req-trace-1")


if __name__ == "__main__":
    asyncio.run(demonstrate_context_vars())
