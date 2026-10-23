"""
Logging - Async-friendly logging with context that follows your task.

WHY LOGGING ASYNC CODE NEEDS SPECIAL CARE
-----------------------------------------
In a threaded program, "who is doing this?" is easy: each thread has
its own stack and thread-local storage, so a log line from thread-3 was
produced by thread-3's request. In asyncio, HUNDREDS of tasks share ONE
thread. If two HTTP requests interleave on that thread, a plain global
variable like ``current_request_id = ...`` would leak across tasks —
request A's id would appear on request B's log lines. Garbage.

The fix is ``contextvars.ContextVar``: variables whose value is stored
per-task (asyncio copies the current context into every new task, so
each task sees ITS OWN values). Setting ``request_id_var.set("req-42")``
inside one task is invisible to others — exactly what we need to answer
"which task produced this log line?"

The second gotcha: handlers that write to disk/network can BLOCK the
event loop, freezing every other task. :meth:`AsyncLogger.alog` shows
the workaround — offload the logging call to a thread executor.

JARGON
------
* **structured logging**: emitting log records as machine-parseable data
  (usually JSON) instead of free-text, so log systems can
  filter/aggregate by fields like ``status=500``.
* **context propagation**: carrying metadata (request id, user id,
  trace id) implicitly through a chain of awaits without threading it
  through every function signature.
* **trace id**: an id shared by an entire distributed operation (your
  request → service A → service B), letting you line up logs across
  services.
* **LogRecord / Formatter / Handler**: the standard library ``logging``
  pipeline — a record is created, a formatter renders it, a handler
  ships it somewhere.

what is IN HERE
--------------
* ``request_id_var`` / ``user_id_var`` / ``trace_id_var`` — ContextVars
  holding per-task identity.
* :class:`LogContext` + :func:`log_context` — read/set those vars.
* :class:`StructuredFormatter` — render records as JSON lines.
* :class:`AsyncLogger` / :class:`StructuredLogger` + :func:`get_logger`
  — loggers that auto-attach the context.
* :func:`setup_logging` — one-call configuration.

MINIMAL EXAMPLE
---------------
>>> import asyncio
>>> async def main() -> None:
...     setup_logging(json_format=True)
...     log = get_logger("myapp")
...     async with log_context(request_id="req-42"):
...         log.info("handling request")  # JSON includes request_id
>>> asyncio.run(main())
"""

import asyncio
import contextlib
import json
import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

# Context variables for logging: each asyncio.Task carries its OWN copy
# of these values, so two interleaved requests never mix up their ids.
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
user_id_var: ContextVar[str | None] = ContextVar("user_id", default=None)
trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)


@dataclass
class LogContext:
    """Snapshot of the per-task logging identity (request/user/trace ids).

    This is the PAYLOAD carried by the ContextVars above: whichever task
    is currently running has its own values here, set usually at a
    request boundary (web handler, worker) via :func:`log_context`.

    Attributes:
        request_id: Id of the incoming request being handled.
        user_id: Id of the user on whose behalf we are acting.
        trace_id: Distributed-tracing id spanning multiple services.
        extra: Any additional fields you want attached to every line.
    """

    request_id: str | None = None
    user_id: str | None = None
    trace_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_context(cls) -> "LogContext":
        """Read the CURRENT task's ContextVars into a LogContext.

        This is the magic step: because the vars are per-task, this call
        returns request A's ids when executed inside task A, and request
        B's when executed inside task B — same thread, different answers.

        Returns:
            A new LogContext populated from the current context; fields
            stay None if nothing was set.
        """
        return cls(
            request_id=request_id_var.get(), user_id=user_id_var.get(), trace_id=trace_id_var.get()
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict, dropping None fields for tidy output.

        Returns:
            E.g. ``{"request_id": "req-42"}`` — only what was set.
        """
        return {k: v for k, v in asdict(self).items() if v is not None}


class StructuredFormatter(logging.Formatter):
    """Render each log record as one line of JSON (structured logging).

    Standard log lines like ``"2024-01-01 INFO user logged in"`` are
    hard for machines to query. This formatter instead emits
    ``{"timestamp": ..., "level": "INFO", "message": "...", ...}``,
    including the current task's :class:`LogContext` and any ``extra=``
    fields the caller passed — so you can later run queries like
    "show every 500 for request_id req-42".
    """

    def format(self, record: logging.LogRecord) -> str:
        """Build the JSON representation of one log record.

        Args:
            record: The standard-library LogRecord (created by every
                ``logger.info(...)`` call behind the scenes).

        Returns:
            A JSON string — one line per record — with core fields,
            the current task's LogContext (if any), and any custom
            ``extra=`` fields attached to the record.
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Attach the CURRENT TASK's identity — the whole point of
        # contextvars: interleaved tasks get their own ids here.
        context = LogContext.from_context().to_dict()
        if context:
            log_data["context"] = context

        # Fold in caller-supplied extra= fields, skipping the standard
        # LogRecord attribute names (already covered or internal).
        for key, value in record.__dict__.items():
            if key not in (
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "getMessage",
                "exc_info",
                "exc_text",
                "stack_info",
            ):
                log_data[key] = value

        return json.dumps(log_data, default=str)


class AsyncLogger:
    """Wrapper around the standard logging.Logger that auto-attaches context.

    JARGON: a "wrapper" is an object that holds another object and adds
    behavior around the same calls. Here every level method (info, error,
    ...) first reads the CURRENT task's :class:`LogContext` from the
    ContextVars and injects it into the log record — you never have to
    pass ``request_id=...`` by hand on each call.

    Example:
        >>> log = get_logger("myapp")
        >>> async with log_context(request_id="req-1"):
        ...     log.info("hello")  # record carries {"context": {...}}
    """

    def __init__(self, name: str, logger: logging.Logger | None = None) -> None:
        """Create an AsyncLogger.

        Args:
            name: Logger name (convention: your module or app name, used
                for filtering). Ignored if ``logger`` is given.
            logger: An existing stdlib Logger to wrap; if None, one is
                fetched (or created) via ``logging.getLogger(name)``.
        """
        self._logger = logger or logging.getLogger(name)
        self._loop: asyncio.AbstractEventLoop | None
        try:
            # Remember the running loop now so alog() can hand work to it
            # later. Raises RuntimeError if no loop exists in this thread yet
            # (e.g. constructed at import time) — that's fine, we fall back
            # to None and alog() would need a loop by call time.
            self._loop = asyncio.get_event_loop()
        except RuntimeError:
            self._loop = None

    def _log(self, level: int, msg: str, *args: object, **kwargs: Any) -> None:
        """Shared guts of every level method: attach context, then emit.

        Args:
            level: A stdlib level constant like ``logging.INFO``.
            msg: Message template, optionally with %-style placeholders.
            *args: Values interpolated into the placeholders.
            **kwargs: Forwarded to ``Logger.log``; ``extra={...}`` adds
                custom fields to the record (picked up by StructuredFormatter).
        """
        extra_any = kwargs.pop("extra", {})
        extra: dict[str, object] = dict(extra_any) if isinstance(extra_any, dict) else {}        # Contextvars lookup: from_context() reads THIS task's copy of the
        # request/user/trace ids, so interleaved tasks can never mix them up.
        context = LogContext.from_context().to_dict()
        if context:
            extra["context"] = context
        self._logger.log(level, msg, *args, extra=extra, **kwargs)

    def debug(self, msg: str, *args: object, **kwargs: object) -> None:
        """Log at DEBUG level — verbose detail, usually off in production.

        Args:
            msg: Message template (e.g. ``"fetched %d rows"``).
            *args: Values for any placeholders in ``msg``.
            **kwargs: Forwarded to the stdlib logger; use ``extra={...}``
                to attach custom fields.

        Example:
            >>> log.debug("cache hit for key %s", key)
        """
        self._log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args: object, **kwargs: object) -> None:
        """Log at INFO level — normal, noteworthy events (request handled, ...).

        Args:
            msg: Message template.
            *args: Values for placeholders.
            **kwargs: Forwarded to the stdlib logger; ``extra={...}`` adds fields.

        Example:
            >>> log.info("order created", extra={"order_id": 42})
        """
        self._log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args: object, **kwargs: object) -> None:
        """Log at WARNING — something odd happened but work continues.

        Args:
            msg: Message template.
            *args: Values for placeholders.
            **kwargs: Forwarded to the stdlib logger.

        Example:
            >>> log.warning("retrying (%d/%d)", attempt, max_attempts)
        """
        self._log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args: object, **kwargs: object) -> None:
        """Log at ERROR — an operation failed; a human may need to look.

        Args:
            msg: Message template.
            *args: Values for placeholders.
            **kwargs: Forwarded to the stdlib logger.

        Example:
            >>> log.error("payment failed", extra={"user_id": uid})
        """
        self._log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg: str, *args: object, **kwargs: object) -> None:
        """Log at CRITICAL — the app itself is in serious trouble.

        Args:
            msg: Message template.
            *args: Values for placeholders.
            **kwargs: Forwarded to the stdlib logger.

        Example:
            >>> log.critical("database unreachable, shutting down")
        """
        self._log(logging.CRITICAL, msg, *args, **kwargs)

    def exception(self, msg: str, *args: object, **kwargs: object) -> None:
        """Log an ERROR plus the traceback of the exception being handled.

        Call this INSIDE an ``except`` block; it passes ``exc_info=True``
        so the stdlib automatically attaches the current exception's
        traceback text to the record.

        Args:
            msg: Message template describing what was attempted.
            *args: Values for placeholders.
            **kwargs: Forwarded to the stdlib logger.

        Example:
            >>> try:
            ...     await fetch()
            ... except Exception:
            ...     log.exception("fetch failed")
        """
        # setdefault so a caller-supplied exc_info is never clobbered.
        kwargs.setdefault("exc_info", True)
        self._log(logging.ERROR, msg, *args, **kwargs)

    async def alog(self, level: int, msg: str, *args: object, **kwargs: object) -> None:
        """Log without blocking the event loop — the key async trick.

        WHY: if a handler writes to a slow disk or network, a normal
        ``log()`` call would FREEZE every task in your program until the
        write finishes. Here the whole logging call is handed to a thread
        executor (a small pool of spare threads), so the event loop keeps
        running while the write happens in the background.

        Args:
            level: Stdlib level constant (``logging.INFO``, etc.).
            msg: Message template.
            *args: Values for placeholders.
            **kwargs: Forwarded like the sync methods.

        Example:
            >>> await log.alog(logging.INFO, "slow write offloaded")

        Note:
            Requires the logger to have captured an event loop at
            construction (or asyncio.run_in_executor will fail).
        """
        # Executor offload: run_in_executor schedules the lambda on a
        # thread-pool thread and returns a Future we await. The lambda
        # wrapper defers the actual _log call until the thread runs it.
        if self._loop is not None:
            await self._loop.run_in_executor(None, lambda: self._log(level, msg, *args, **kwargs))
        else:
            self._log(level, msg, *args, **kwargs)

    def with_context(self, **_context: object) -> "AsyncLogger":
        """Return a sibling AsyncLogger sharing the same underlying logger.

        TEACHING NOTE: this is a placeholder API — stdlib loggers have no
        built-in per-logger context store. In real code you achieve "with
        context" via :func:`log_context`, which sets the ContextVars that
        :meth:`_log` reads on every call.

        Args:
            **_context: Field names/values you would like attached (currently
                accepted for API compatibility, not stored on the logger).

        Returns:
            A new AsyncLogger wrapping the SAME stdlib logger.

        Example:
            >>> request_log = log.with_context(request_id="r-1")  # cosmetic
            >>> async with log_context(request_id="r-1"):  # the real way
            ...     log.info("handled")
        """
        logger = AsyncLogger(self._logger.name, self._logger)
        # We can't easily add context to an existing stdlib logger object;
        # in practice, you'd set context variables (see log_context()).
        return logger


class StructuredLogger(AsyncLogger):
    """AsyncLogger with typed convenience methods for common events.

    WHY TYPED METHODS: instead of remembering to pass
    ``extra={"method": ..., "status": ...}`` by hand (and risking typos or
    inconsistent field names), you call ``log_request("GET", "/api", 200, 0.04)``
    and every request log line has the SAME fields — which is exactly what
    makes structured logs queryable later.

    Example:
        >>> slog = StructuredLogger("api")
        >>> slog.log_request("GET", "/users", 200, 0.045)
    """

    def log_request(self, method: str, path: str, status: int, duration: float) -> None:
        """Log one handled HTTP request at INFO level.

        Args:
            method: HTTP verb, e.g. ``"GET"``.
            path: Request path, e.g. ``"/api/users"``.
            status: HTTP status code returned, e.g. ``200``.
            duration: Seconds the request took; logged as milliseconds
                (``duration_ms``) since ms is conventional in dashboards.

        Example:
            >>> slog.log_request("POST", "/orders", 201, 0.120)
        """
        self.info(
            "HTTP request",
            extra={
                "method": method,
                "path": path,
                "status": status,
                "duration_ms": duration * 1000,
            },
        )

    def log_db_query(self, query: str, duration: float, rows: int | None = None) -> None:
        """Log one database query at DEBUG level.

        Args:
            query: SQL text. TRUNCATED to 100 characters (``query[:100]``)
                so huge queries cannot bloat your logs or leak long payloads.
            duration: Seconds the query took; logged as ``duration_ms``.
            rows: Row count returned/affected, if known.

        Example:
            >>> slog.log_db_query("SELECT * FROM users", 0.012, rows=42)
        """
        self.debug(
            "Database query",
            extra={"query": query[:100], "duration_ms": duration * 1000, "rows": rows},
        )

    def log_external_call(self, service: str, endpoint: str, status: int, duration: float) -> None:
        """Log one call to an external (third-party) API at INFO level.

        Args:
            service: Name of the remote service, e.g. ``"stripe"``.
            endpoint: Path/URL hit, e.g. ``"/v1/charges"``.
            status: HTTP status the remote service returned.
            duration: Seconds the call took; logged as ``duration_ms``.

        Example:
            >>> slog.log_external_call("stripe", "/v1/charges", 200, 0.350)
        """
        self.info(
            "External API call",
            extra={
                "service": service,
                "endpoint": endpoint,
                "status": status,
                "duration_ms": duration * 1000,
            },
        )


def get_logger(name: str) -> AsyncLogger:
    """Get (or create) an AsyncLogger by name — the standard entry point.

    Mirrors ``logging.getLogger(name)``: same name always gives the same
    underlying stdlib logger, so modules can share configuration.

    Args:
        name: Convention is ``__name__`` (your module) or your app name;
            it appears on every log line and is used for level filtering.

    Returns:
        A ready-to-use AsyncLogger that auto-attaches LogContext values.

    Example:
        >>> log = get_logger(__name__)
        >>> log.info("starting up")
    """
    return AsyncLogger(name)


def setup_logging(
    level: int = logging.INFO, json_format: bool = False, output: str = "stdout"
) -> None:
    """Configure the whole application's logging in one call.

    Replaces the ROOT logger's handlers (the catch-all logger every
    named logger ultimately reports to) with either a text or a JSON
    pipeline. Call ONCE at program start, before any logging happens.

    Args:
        level: Minimum severity to emit, e.g. ``logging.DEBUG`` shows
            everything, ``logging.WARNING`` hides debug/info noise.
        json_format: If True, use :class:`StructuredFormatter` (one JSON
            object per line — great for log aggregators); if False, use a
            human-friendly ``time | LEVEL | name | message`` layout.
        output: ``"stdout"`` (default; plays well with containers and
            ``1>>file`` redirection) or ``"stderr"``.

    Example:
        >>> setup_logging(level=logging.DEBUG, json_format=True)

    Warning:
        This CLEARS any existing root handlers — later calls overwrite
        earlier configuration.
    """
    handlers = []

    if output == "stdout":
        handlers.append(logging.StreamHandler(sys.stdout))
    elif output == "stderr":
        handlers.append(logging.StreamHandler(sys.stderr))

    formatter: logging.Formatter
    if json_format:
        formatter = StructuredFormatter()
    else:
        # %-8s left-aligns and pads the level name to 8 chars so columns
        # line up ("INFO    " vs "ERROR   ") — purely cosmetic.
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

    for handler in handlers:
        handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    # Clear BEFORE adding: without this, calling setup_logging twice would
    # duplicate every handler and print each line twice.
    root.handlers.clear()
    for handler in handlers:
        root.addHandler(handler)


@asynccontextmanager
async def log_context(**context: object) -> AsyncIterator[None]:
    """Set request/user/trace ids for the body of an ``async with`` block.

    This is CONTEXT PROPAGATION made ergonomic: everything logged inside
    the block — including logs from helper functions you call — carries
    these ids automatically, because every log call reads the same
    per-task ContextVars. On exit, the previous values are restored
    (so nested or sequential requests cannot leak ids into each other).

    Args:
        **context: Any of ``request_id=``, ``user_id=``, ``trace_id=``.
            Unknown keys are ignored; you cannot set ``extra`` here.

    Returns:
        An async context manager; use with ``async with``.

    Example:
        >>> async with log_context(request_id="req-42", user_id="u-7"):
        ...     log.info("handling request")  # ids attached automatically
    """
    tokens: list[Token[str | None]] = []
    for var, value in context.items():
        # ContextVar values are str | None; coerce anything else to str.
        coerced: str | None = None if value is None else str(value)
        if var == "request_id":
            tokens.append(request_id_var.set(coerced))
        elif var == "user_id":
            tokens.append(user_id_var.set(coerced))
        elif var == "trace_id":
            tokens.append(trace_id_var.set(coerced))

    try:
        yield
    finally:
        # Reset in REVERSE order (like unwinding a stack). suppress(ValueError)
        # guards against the edge case where a token is somehow already reset
        # (e.g. context copied across tasks), which would otherwise crash
        # cleanup and mask the original exception.
        for token in reversed(tokens):
            with contextlib.suppress(ValueError):
                token.var.reset(token)


# Example usage
if __name__ == "__main__":

    async def demo() -> None:
        setup_logging(level=logging.DEBUG, json_format=True)

        logger = get_logger("demo")

        # Basic logging
        logger.info("Application started")

        # With context
        async with log_context(request_id="req-123", user_id="user-456"):
            logger.info("Processing request")
            logger.debug("Debug info", extra={"key": "value"})

            # Structured logging
            struct = StructuredLogger("demo")
            struct.log_request("GET", "/api/users", 200, 0.045)
            struct.log_db_query("SELECT * FROM users", 0.012, 42)

    asyncio.run(demo())
