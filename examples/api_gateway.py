"""
API Gateway

Production-ready API gateway with routing, middleware, rate limiting, and authentication.
"""

import asyncio
import re
import time
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from re import Pattern
from typing import Any

try:
    from aiohttp import ClientSession, ClientTimeout, web

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None
    ClientSession = None
    ClientTimeout = None


class HTTPMethod(Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"
    ANY = "ANY"


@dataclass
class Route:
    """API route definition."""

    path: str
    method: HTTPMethod
    handler: Callable[..., Awaitable[Any]]
    middleware: list["Middleware"] = field(default_factory=list)
    auth_required: bool = False
    rate_limit: tuple[int, float] | None = None  # (requests, window_seconds)
    timeout: float | None = None
    name: str | None = None


@dataclass
class GatewayConfig:
    """API gateway configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    max_request_size: int = 1024 * 1024  # 1MB
    request_timeout: float = 30.0
    keepalive_timeout: float = 75.0
    max_connections: int = 1000
    enable_cors: bool = True
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    cors_methods: list[str] = field(default_factory=lambda: ["*"])
    cors_headers: list[str] = field(default_factory=lambda: ["*"])


class Middleware:
    """Base middleware class."""

    async def before_request(self, request: "GatewayRequest") -> web.Response | None:
        """Process request before routing. Return response to short-circuit."""
        return None

    async def after_request(
        self, request: "GatewayRequest", response: web.Response
    ) -> web.Response:
        """Process response after handler."""
        return response

    async def on_error(self, request: "GatewayRequest", error: Exception) -> web.Response | None:
        """Handle errors. Return response to handle error."""
        return None


class GatewayRequest:
    """Wrapper for aiohttp request with gateway utilities."""

    def __init__(self, request: web.Request, route: Route | None = None) -> None:
        self._request = request
        self._route = route
        self._matched_params: dict[str, str] = {}
        self._context: dict[str, Any] = {}
        self._start_time = time.time()

    @property
    def request(self) -> web.Request:
        return self._request

    @property
    def method(self) -> str:
        return self._request.method

    @property
    def path(self) -> str:
        return self._request.path

    @property
    def query(self) -> dict[str, str]:
        return dict(self._request.query)

    @property
    def headers(self) -> dict[str, str]:
        return dict(self._request.headers)

    @property
    def matched_params(self) -> dict[str, str]:
        return self._matched_params

    @property
    def context(self) -> dict[str, Any]:
        return self._context

    @property
    def route(self) -> Route | None:
        return self._route

    @property
    def elapsed(self) -> float:
        return time.time() - self._start_time

    async def json(self) -> Any:
        return await self._request.json()

    async def text(self) -> str:
        return await self._request.text()

    async def read(self) -> bytes:
        return await self._request.read()

    def set_param(self, key: str, value: str) -> None:
        self._matched_params[key] = value

    def set_context(self, key: str, value: Any) -> None:
        self._context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        return self._context.get(key, default)


class APIGateway:
    """API Gateway with routing, middleware, and monitoring."""

    def __init__(self, config: GatewayConfig | None = None) -> None:
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp is required. Install with: pip install aiohttp")

        self._config = config or GatewayConfig()
        self._routes: list[Route] = []
        self._global_middleware: list[Middleware] = []
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._running = False

        # Rate limiting
        self._rate_limit_buckets: dict[str, list[float]] = defaultdict(list)
        self._rate_limit_lock = asyncio.Lock()

        # Stats
        self._stats = {
            "requests": 0,
            "errors": 0,
            "total_time": 0.0,
            "by_route": defaultdict(lambda: {"count": 0, "errors": 0, "total_time": 0.0}),
        }
        self._stats_lock = asyncio.Lock()

    @property
    def config(self) -> GatewayConfig:
        return self._config

    @property
    def running(self) -> bool:
        return self._running

    def add_route(
        self, path: str, method: HTTPMethod | str, handler: Callable[..., Awaitable[Any]], **kwargs
    ) -> Route:
        """Add a route to the gateway."""
        if isinstance(method, str):
            method = HTTPMethod(method.upper())

        route = Route(path=path, method=method, handler=handler, **kwargs)

        self._routes.append(route)
        return route

    def add_middleware(self, middleware: Middleware) -> None:
        """Add global middleware."""
        self._global_middleware.append(middleware)

    def route(self, path: str, method: HTTPMethod | str = HTTPMethod.ANY, **kwargs) -> Callable:
        """Decorator for adding routes."""

        def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
            self.add_route(path, method, func, **kwargs)
            return func

        return decorator

    def get(self, path: str, **kwargs) -> Callable:
        return self.route(path, HTTPMethod.GET, **kwargs)

    def post(self, path: str, **kwargs) -> Callable:
        return self.route(path, HTTPMethod.POST, **kwargs)

    def put(self, path: str, **kwargs) -> Callable:
        return self.route(path, HTTPMethod.PUT, **kwargs)

    def delete(self, path: str, **kwargs) -> Callable:
        return self.route(path, HTTPMethod.DELETE, **kwargs)

    async def start(self) -> None:
        """Start the gateway."""
        if self._running:
            return

        self._app = web.Application(
            middlewares=[self._create_aiohttp_middleware()],
            client_max_size=self._config.max_request_size,
        )

        # Add routes
        for route in self._routes:
            self._add_aiohttp_route(route)

        # Add CORS if enabled
        if self._config.enable_cors:
            self._setup_cors()

        # Start server
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()

        self._site = web.TCPSite(
            self._runner,
            self._config.host,
            self._config.port,
            shutdown_timeout=self._config.request_timeout,
        )
        await self._site.start()

        self._running = True

    async def stop(self) -> None:
        """Stop the gateway."""
        self._running = False

        if self._site:
            await self._site.stop()
            self._site = None

        if self._runner:
            await self._runner.cleanup()
            self._runner = None

        self._app = None

    def _create_aiohttp_middleware(self):
        """Create aiohttp middleware for gateway processing."""

        @web.middleware
        async def gateway_middleware(request: web.Request, handler):
            return await self._handle_request(request, handler)

        return gateway_middleware

    async def _handle_request(self, request: web.Request, handler) -> web.Response:
        """Handle incoming request through gateway pipeline."""
        start_time = time.time()

        # Find matching route
        route, params = self._match_route(request.method, request.path)

        # Create gateway request
        gw_request = GatewayRequest(request, route)
        gw_request._matched_params = params

        try:
            # Global middleware - before request
            for middleware in self._global_middleware:
                response = await middleware.before_request(gw_request)
                if response:
                    return response

            # Route middleware - before request
            if route:
                for middleware in route.middleware:
                    response = await middleware.before_request(gw_request)
                    if response:
                        return response

            # Rate limiting
            if route and route.rate_limit:
                await self._check_rate_limit(gw_request, route.rate_limit)

            # Authentication
            if route and route.auth_required:
                await self._check_auth(gw_request)

            # Execute handler
            if route:
                response = await self._execute_handler(gw_request, route)
            else:
                response = web.Response(status=404, text="Not Found")

            # Route middleware - after request
            if route:
                for middleware in reversed(route.middleware):
                    response = await middleware.after_request(gw_request, response)

            # Global middleware - after request
            for middleware in reversed(self._global_middleware):
                response = await middleware.after_request(gw_request, response)

            return response

        except web.HTTPException as e:
            return e
        except Exception as e:
            # Error handling
            error_response = None

            if route:
                for middleware in route.middleware:
                    error_response = await middleware.on_error(gw_request, e)
                    if error_response:
                        break

            if not error_response:
                for middleware in self._global_middleware:
                    error_response = await middleware.on_error(gw_request, e)
                    if error_response:
                        break

            if error_response:
                return error_response

            # Default error response
            return web.Response(
                status=500, text=f"Internal Server Error: {e!s}", content_type="text/plain"
            )
        finally:
            # Record stats
            elapsed = time.time() - start_time
            async with self._stats_lock:
                self._stats["requests"] += 1
                self._stats["total_time"] += elapsed

                if route:
                    route_stats = self._stats["by_route"][route.path]
                    route_stats["count"] += 1
                    route_stats["total_time"] += elapsed

    async def _execute_handler(self, gw_request: GatewayRequest, route: Route) -> web.Response:
        """Execute route handler."""
        timeout = route.timeout or self._config.request_timeout

        try:
            if timeout:
                result = await asyncio.wait_for(route.handler(gw_request), timeout=timeout)
            else:
                result = await route.handler(gw_request)

            # Convert result to response
            if isinstance(result, web.Response):
                return result
            elif isinstance(result, dict):
                return web.json_response(result)
            elif isinstance(result, str):
                return web.Response(text=result, content_type="text/plain")
            elif isinstance(result, bytes):
                return web.Response(body=result, content_type="application/octet-stream")
            else:
                return web.json_response({"data": result})

        except TimeoutError:
            raise web.HTTPGatewayTimeout(text="Request timeout")

    def _match_route(self, method: str, path: str) -> tuple[Route | None, dict[str, str]]:
        """Match request to route."""
        for route in self._routes:
            if route.method != HTTPMethod.ANY and route.method.value != method:
                continue

            # Convert path pattern to regex
            pattern = self._path_to_pattern(route.path)
            match = pattern.match(path)

            if match:
                params = match.groupdict()
                return route, params

        return None, {}

    def _path_to_pattern(self, path: str) -> Pattern:
        """Convert path pattern to regex."""
        # Replace {param} with named capture groups
        pattern = re.sub(r"\{(\w+)\}", r"(?P<\1>[^/]+)", path)
        pattern = f"^{pattern}$"
        return re.compile(pattern)

    def _add_aiohttp_route(self, route: Route) -> None:
        """Add route to aiohttp application."""
        # Routes are handled by middleware, but we add a catch-all
        pass

    def _setup_cors(self) -> None:
        """Setup CORS headers."""

        @web.middleware
        async def cors_middleware(request, handler):
            if request.method == "OPTIONS":
                response = web.Response()
            else:
                response = await handler(request)

            origin = request.headers.get("Origin", "*")
            if "*" in self._config.cors_origins or origin in self._config.cors_origins:
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Access-Control-Allow-Methods"] = ", ".join(
                    self._config.cors_methods
                )
                response.headers["Access-Control-Allow-Headers"] = ", ".join(
                    self._config.cors_headers
                )
                response.headers["Access-Control-Allow-Credentials"] = "true"

            return response

        self._app.middlewares.insert(0, cors_middleware)

    async def _check_rate_limit(self, request: GatewayRequest, limit: tuple[int, float]) -> None:
        """Check rate limit for request."""
        max_requests, window = limit
        key = f"{request.method}:{request.path}:{request.headers.get('X-Forwarded-For', request.headers.get('X-Real-IP', 'unknown'))}"

        async with self._rate_limit_lock:
            now = time.time()
            cutoff = now - window

            # Clean old requests
            self._rate_limit_buckets[key] = [t for t in self._rate_limit_buckets[key] if t > cutoff]

            if len(self._rate_limit_buckets[key]) >= max_requests:
                raise web.HTTPTooManyRequests(text="Rate limit exceeded")

            self._rate_limit_buckets[key].append(now)

    async def _check_auth(self, request: GatewayRequest) -> None:
        """Check authentication. Override in subclass or use middleware."""
        # Default: check for Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            raise web.HTTPUnauthorized(text="Authentication required")

    def get_stats(self) -> dict[str, Any]:
        stats = self._stats.copy()
        stats["by_route"] = dict(stats["by_route"])

        if stats["requests"] > 0:
            stats["avg_time"] = stats["total_time"] / stats["requests"]
            stats["error_rate"] = stats["errors"] / stats["requests"]

        return stats

    async def __aenter__(self) -> "APIGateway":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()


class RateLimitMiddleware(Middleware):
    """Rate limiting middleware."""

    def __init__(self, max_requests: int, window: float, key_func: Callable | None = None) -> None:
        self._max_requests = max_requests
        self._window = window
        self._key_func = key_func or (
            lambda r: f"{r.method}:{r.path}:{r.headers.get('X-Forwarded-For', 'unknown')}"
        )
        self._buckets: dict[str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def before_request(self, request: GatewayRequest) -> web.Response | None:
        key = self._key_func(request)

        async with self._lock:
            now = time.time()
            cutoff = now - self._window

            self._buckets[key] = [t for t in self._buckets[key] if t > cutoff]

            if len(self._buckets[key]) >= self._max_requests:
                return web.Response(
                    status=429,
                    text="Rate limit exceeded",
                    headers={"Retry-After": str(int(self._window))},
                )

            self._buckets[key].append(now)

        return None


class AuthMiddleware(Middleware):
    """Authentication middleware."""

    def __init__(
        self,
        validator: Callable[[str], Awaitable[dict | None]],
        header: str = "Authorization",
        scheme: str = "Bearer",
    ) -> None:
        self._validator = validator
        self._header = header
        self._scheme = scheme

    async def before_request(self, request: GatewayRequest) -> web.Response | None:
        auth_header = request.headers.get(self._header)

        if not auth_header:
            return web.Response(status=401, text="Missing authorization header")

        if not auth_header.startswith(f"{self._scheme} "):
            return web.Response(status=401, text="Invalid authorization scheme")

        token = auth_header[len(self._scheme) + 1 :]

        try:
            user = await self._validator(token)
            if not user:
                return web.Response(status=401, text="Invalid token")

            request.set_context("user", user)
            request.set_context("token", token)

        except Exception as e:
            return web.Response(status=401, text=f"Authentication error: {e!s}")

        return None


class LoggingMiddleware(Middleware):
    """Request/response logging middleware."""

    def __init__(self, logger: Callable[[str], None] | None = None) -> None:
        self._logger = logger or (print)

    async def before_request(self, request: GatewayRequest) -> None:
        self._logger(
            f"→ {request.method} {request.path} from {request.headers.get('X-Forwarded-For', 'unknown')}"
        )

    async def after_request(self, request: GatewayRequest, response: web.Response) -> web.Response:
        self._logger(
            f"← {request.method} {request.path} {response.status} ({request.elapsed:.3f}s)"
        )
        return response

    async def on_error(self, request: GatewayRequest, error: Exception) -> web.Response | None:
        self._logger(f"✗ {request.method} {request.path} ERROR: {error}")
        return None


def create_gateway(host: str = "0.0.0.0", port: int = 8000, **kwargs) -> APIGateway:
    """Factory for creating API gateway."""
    config = GatewayConfig(host=host, port=port, **kwargs)
    return APIGateway(config)


# Example usage
async def example_usage() -> None:
    """Example API gateway usage."""
    gateway = create_gateway(port=8000)

    # Add global middleware
    gateway.add_middleware(LoggingMiddleware())
    gateway.add_middleware(RateLimitMiddleware(100, 60.0))  # 100 req/min

    # Add routes
    @gateway.get("/health")
    async def health(request: GatewayRequest):
        return {"status": "ok", "timestamp": time.time()}

    @gateway.get("/users/{user_id}")
    async def get_user(request: GatewayRequest):
        user_id = request.matched_params["user_id"]
        return {"id": user_id, "name": f"User {user_id}"}

    @gateway.post("/users")
    async def create_user(request: GatewayRequest):
        data = await request.json()
        return {"id": str(uuid.uuid4()), **data}, 201

    # Protected route
    @gateway.get("/profile", auth_required=True)
    async def get_profile(request: GatewayRequest):
        user = request.get_context("user")
        return {"user": user}

    # Custom middleware for specific route
    class AdminMiddleware(Middleware):
        async def before_request(self, request):
            user = request.get_context("user")
            if not user or not user.get("admin"):
                return web.Response(status=403, text="Admin required")

    @gateway.get("/admin/stats", auth_required=True)
    async def admin_stats(request: GatewayRequest):
        return gateway.get_stats()

    # Add route-specific middleware
    gateway.add_route(
        "/admin/users",
        HTTPMethod.GET,
        lambda r: {"users": []},
        middleware=[AdminMiddleware()],
        auth_required=True,
    )

    async with gateway:
        print("API Gateway running on http://0.0.0.0:8000")
        await asyncio.Future()  # Run forever
