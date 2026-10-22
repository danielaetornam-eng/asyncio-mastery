"""
Async Logging Utilities

Structured logging for async applications.
"""

import asyncio
import contextvars
import json
import logging
import sys
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, TypeVar

T = TypeVar("T")


class LogLevel(Enum):
    """Log levels."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@dataclass
class LogRecord:
    """Structured log record."""

    level: LogLevel
    message: str
    timestamp: float = field(default_factory=time.time)
    logger_name: str = ""
    module: str = ""
    function: str = ""
    line: int = 0
    extra: dict[str, Any] = field(default_factory=dict)
    exception: BaseException | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.name,
            "message": self.message,
            "timestamp": self.timestamp,
            "logger_name": self.logger_name,
            "module": self.module,
            "function": self.function,
            "line": self.line,
            "extra": self.extra,
            "exception": str(self.exception) if self.exception else None,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class AsyncLogger:
    """Async logger with structured logging support."""

    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.INFO,
        format_json: bool = False,
        handler: logging.Handler | None = None,
    ):
        self._name = name
        self._level = level
        self._format_json = format_json

        # Create underlying logger
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level.value)

        if not self._logger.handlers:
            if handler:
                self._logger.addHandler(handler)
            else:
                # Default console handler
                console_handler = logging.StreamHandler(sys.stdout)
                if format_json:
                    console_handler.setFormatter(JsonFormatter())
                else:
                    console_handler.setFormatter(
                        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
                    )
                self._logger.addHandler(console_handler)

    @property
    def name(self) -> str:
        return self._name

    @property
    def level(self) -> LogLevel:
        return self._level

    def set_level(self, level: LogLevel) -> None:
        self._level = level
        self._logger.setLevel(level.value)

    def _log(
        self,
        level: LogLevel,
        message: str,
        extra: dict[str, Any] | None = None,
        exception: BaseException | None = None,
    ) -> None:
        if level.value < self._level.value:
            return

        # Get caller info
        frame = sys._getframe(2)
        module = frame.f_globals.get("__name__", "")
        function = frame.f_code.co_name
        line = frame.f_lineno

        record = LogRecord(
            level=level,
            message=message,
            logger_name=self._name,
            module=module,
            function=function,
            line=line,
            extra=extra or {},
            exception=exception,
        )

        if self._format_json:
            self._logger.log(level.value, record.to_json())
        else:
            extra_msg = f" | {json.dumps(record.extra)}" if record.extra else ""
            exc_msg = f" | {record.exception}" if record.exception else ""
            self._logger.log(
                level.value,
                f"{record.message}{extra_msg}{exc_msg}",
                extra={"structured": record.to_dict()},
            )

    def debug(self, message: str, extra: dict[str, Any] | None = None) -> None:
        self._log(LogLevel.DEBUG, message, extra)

    def info(self, message: str, extra: dict[str, Any] | None = None) -> None:
        self._log(LogLevel.INFO, message, extra)

    def warning(self, message: str, extra: dict[str, Any] | None = None) -> None:
        self._log(LogLevel.WARNING, message, extra)

    def error(
        self,
        message: str,
        extra: dict[str, Any] | None = None,
        exception: BaseException | None = None,
    ) -> None:
        self._log(LogLevel.ERROR, message, extra, exception)

    def critical(
        self,
        message: str,
        extra: dict[str, Any] | None = None,
        exception: BaseException | None = None,
    ) -> None:
        self._log(LogLevel.CRITICAL, message, extra, exception)

    def exception(self, message: str, extra: dict[str, Any] | None = None) -> None:
        self._log(LogLevel.ERROR, message, extra, sys.exc_info()[1])

    @asynccontextmanager
    async def context(self, **context_data) -> AsyncGenerator[None, None]:
        """Context manager for adding context to logs."""
        old_extra = getattr(self, "_context_extra", {})
        self._context_extra = {**old_extra, **context_data}
        try:
            yield
        finally:
            self._context_extra = old_extra

    def _log_with_context(self, level: LogLevel, message: str, extra: dict | None = None) -> None:
        combined_extra = {**getattr(self, "_context_extra", {}), **(extra or {})}
        self._log(level, message, combined_extra)

    def get_logger(self) -> logging.Logger:
        return self._logger


class JsonFormatter(logging.Formatter):
    """JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if hasattr(record, "structured"):
            log_data.update(record.structured)

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


# Global logger registry
_loggers: dict[str, AsyncLogger] = {}
_logger_lock = asyncio.Lock()


def get_logger(
    name: str, level: LogLevel = LogLevel.INFO, format_json: bool = False
) -> AsyncLogger:
    """Get or create logger."""
    if name not in _loggers:
        _loggers[name] = AsyncLogger(name, level, format_json)
    return _loggers[name]


def configure_logging(
    level: LogLevel = LogLevel.INFO,
    format_json: bool = False,
    handlers: list[logging.Handler] | None = None,
) -> None:
    """Configure root logger."""
    root_logger = logging.getLogger()
    root_logger.setLevel(level.value)

    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    if handlers:
        for handler in handlers:
            if format_json:
                handler.setFormatter(JsonFormatter())
            root_logger.addHandler(handler)
    else:
        # Default console handler
        console_handler = logging.StreamHandler(sys.stdout)
        if format_json:
            console_handler.setFormatter(JsonFormatter())
        else:
            console_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
        root_logger.addHandler(console_handler)


@asynccontextmanager
async def log_context(**context) -> AsyncGenerator[None, None]:
    """Context manager for adding context to all logs."""
    # This would need integration with logging context
    yield


def logged(
    logger: AsyncLogger | None = None,
    level: LogLevel = LogLevel.INFO,
    log_args: bool = True,
    log_result: bool = False,
    log_exceptions: bool = True,
) -> Callable:
    """Decorator to log function calls."""
    if logger is None:
        logger = get_logger("async_logger")

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            if log_args:
                logger.info(
                    f"Calling {func.__name__}", extra={"args": str(args), "kwargs": str(kwargs)}
                )
            else:
                logger.info(f"Calling {func.__name__}")

            start = time.time()

            try:
                result = await func(*args, **kwargs)
                elapsed = time.time() - start

                if log_result:
                    logger.info(
                        f"{func.__name__} returned",
                        extra={"result": str(result), "elapsed": elapsed},
                    )
                else:
                    logger.info(f"{func.__name__} completed in {elapsed:.3f}s")

                return result
            except Exception as e:
                elapsed = time.time() - start
                if log_exceptions:
                    logger.exception(
                        f"{func.__name__} failed",
                        extra={"elapsed": elapsed, "error": str(e)},
                        exception=e,
                    )
                raise

        return wrapper

    return decorator


@asynccontextmanager
async def log_operation(
    logger: AsyncLogger, operation: str, level: LogLevel = LogLevel.INFO, **context
) -> AsyncGenerator[None, None]:
    """Context manager for logging an operation."""
    start = time.time()
    logger.log(level, f"Starting {operation}", extra=context)

    try:
        yield
        elapsed = time.time() - start
        logger.log(level, f"Completed {operation} in {elapsed:.3f}s", extra=context)
    except Exception as e:
        elapsed = time.time() - start
        logger.exception(f"Failed {operation} after {elapsed:.3f}s", extra=context, exception=e)
        raise


class LogBuffer:
    """Buffer for collecting logs."""

    def __init__(self, max_size: int = 1000):
        self._buffer: list[LogRecord] = []
        self._max_size = max_size
        self._lock = asyncio.Lock()

    async def add(self, record: LogRecord) -> None:
        async with self._lock:
            self._buffer.append(record)
            if len(self._buffer) > self._max_size:
                self._buffer.pop(0)

    async def get_all(self) -> list[LogRecord]:
        async with self._lock:
            return self._buffer.copy()

    async def get_by_level(self, level: LogLevel) -> list[LogRecord]:
        async with self._lock:
            return [r for r in self._buffer if r.level == level]

    async def clear(self) -> None:
        async with self._lock:
            self._buffer.clear()

    async def __len__(self) -> int:
        async with self._lock:
            return len(self._buffer)


class AsyncLogHandler(logging.Handler):
    """Async log handler that buffers records."""

    def __init__(self, buffer: LogBuffer):
        super().__init__()
        self._buffer = buffer

    def emit(self, record: logging.LogRecord) -> None:
        try:
            log_record = LogRecord(
                level=LogLevel(record.levelno),
                message=record.getMessage(),
                timestamp=record.created,
                logger_name=record.name,
                module=record.module,
                function=record.funcName,
                line=record.lineno,
            )
            asyncio.create_task(self._buffer.add(log_record))
        except Exception:
            pass


def create_logger(
    name: str,
    level: LogLevel = LogLevel.INFO,
    format_json: bool = False,
    log_file: str | None = None,
) -> AsyncLogger:
    """Create configured logger."""
    logger = AsyncLogger(name, level, format_json)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        if format_json:
            file_handler.setFormatter(JsonFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
        logger.get_logger().addHandler(file_handler)

    return logger


# Context variables for request tracing
request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)
trace_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("trace_id", default=None)
span_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("span_id", default=None)


def get_request_id() -> str | None:
    return request_id_var.get()


def get_trace_id() -> str | None:
    return trace_id_var.get()


def get_span_id() -> str | None:
    return span_id_var.get()


@asynccontextmanager
async def trace_context(
    request_id: str | None = None, trace_id: str | None = None, span_id: str | None = None
) -> AsyncGenerator[None, None]:
    """Context manager for distributed tracing."""

    tokens = []

    if request_id:
        tokens.append(request_id_var.set(request_id))
    if trace_id:
        tokens.append(trace_id_var.set(trace_id))
    if span_id:
        tokens.append(span_id_var.set(span_id))

    try:
        yield
    finally:
        for token in tokens:
            if token:
                token.reset()


class StructuredLogger:
    """Logger with structured fields."""

    def __init__(self, logger: AsyncLogger, **default_fields):
        self._logger = logger
        self._default_fields = default_fields

    def _log(self, level: LogLevel, message: str, **fields) -> None:
        extra = {**self._default_fields, **fields}
        self._logger._log(level, message, extra)

    def debug(self, message: str, **fields) -> None:
        self._log(LogLevel.DEBUG, message, **fields)

    def info(self, message: str, **fields) -> None:
        self._log(LogLevel.INFO, message, **fields)

    def warning(self, message: str, **fields) -> None:
        self._log(LogLevel.WARNING, message, **fields)

    def error(self, message: str, **fields) -> None:
        self._log(LogLevel.ERROR, message, **fields)

    def critical(self, message: str, **fields) -> None:
        self._log(LogLevel.CRITICAL, message, **fields)

    def with_fields(self, **fields) -> "StructuredLogger":
        return StructuredLogger(self._logger, **{**self._default_fields, **fields})


def get_structured_logger(
    name: str, level: LogLevel = LogLevel.INFO, format_json: bool = False, **default_fields
) -> StructuredLogger:
    """Get structured logger."""
    logger = get_logger(name, level, format_json)
    return StructuredLogger(logger, **default_fields)
