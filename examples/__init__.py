"""
Real-World Examples

Production-ready implementations of common async patterns.
"""

from .api_gateway import (
    APIGateway,
    AuthMiddleware,
    Middleware,
    RateLimitMiddleware,
    Route,
    create_gateway,
)
from .database_pool import DatabaseConfig, DatabasePool, PoolConfig, create_db_pool
from .event_bus import Event, EventBus, EventHandler, create_event_bus
from .http_client import (
    AsyncHTTPClient,
    HTTPClientConfig,
    HTTPError,
    HTTPResponse,
    create_http_client,
)
from .task_scheduler import (
    CronTrigger,
    IntervalTrigger,
    ScheduledTask,
    TaskScheduler,
    create_scheduler,
)
from .websocket import (
    WebSocketClient,
    WebSocketHandler,
    WebSocketMessage,
    WebSocketServer,
    create_ws_server,
)

__all__ = [
    "APIGateway",
    "AsyncHTTPClient",
    "AuthMiddleware",
    "CronTrigger",
    "DatabaseConfig",
    "DatabasePool",
    "Event",
    "EventBus",
    "EventHandler",
    "HTTPClientConfig",
    "HTTPError",
    "HTTPResponse",
    "IntervalTrigger",
    "Middleware",
    "PoolConfig",
    "RateLimitMiddleware",
    "Route",
    "ScheduledTask",
    "TaskScheduler",
    "WebSocketClient",
    "WebSocketHandler",
    "WebSocketMessage",
    "WebSocketServer",
    "create_db_pool",
    "create_event_bus",
    "create_gateway",
    "create_http_client",
    "create_scheduler",
    "create_ws_server",
]
