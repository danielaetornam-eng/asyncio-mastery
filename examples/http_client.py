"""
Async HTTP Client

Production-ready HTTP client with connection pooling, retries, and middleware.
"""

import asyncio
import ssl
import time
import uuid
from collections import defaultdict
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin, urlparse

try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None


@dataclass
class HTTPClientConfig:
    """Configuration for HTTP client."""

    base_url: str = ""
    timeout: float = 30.0
    connect_timeout: float = 10.0
    max_connections: int = 100
    max_keepalive: int = 20
    keepalive_timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0
    retry_on_status: tuple[int, ...] = (500, 502, 503, 504)
    default_headers: dict[str, str] = field(default_factory=dict)
    follow_redirects: bool = True
    max_redirects: int = 10
    verify_ssl: bool = True
    user_agent: str = "asyncio-mastery-http/1.0"


@dataclass
class HTTPResponse:
    """HTTP response wrapper."""

    status: int
    headers: dict[str, str]
    body: bytes
    url: str
    elapsed: float
    request_id: str

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def json(self) -> Any:
        import json

        return json.loads(self.body.decode())

    def text(self) -> str:
        return self.body.decode()

    def raise_for_status(self) -> None:
        if not self.ok:
            raise HTTPError(self.status, self.text())


class HTTPError(Exception):
    """HTTP error with status code."""

    def __init__(self, status: int, message: str, response: HTTPResponse | None = None) -> None:
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.message = message
        self.response = response


class Middleware:
    """Base middleware class."""

    async def process_request(self, request: "HTTPRequest") -> "HTTPRequest":
        return request

    async def process_response(self, response: HTTPResponse) -> HTTPResponse:
        return response

    async def process_exception(self, exception: Exception) -> HTTPResponse | None:
        return None


@dataclass
class HTTPRequest:
    """HTTP request builder."""

    method: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    data: bytes | str | dict | None = None
    json: Any | None = None
    timeout: float | None = None
    allow_redirects: bool | None = None


class AsyncHTTPClient:
    """Production-ready async HTTP client."""

    def __init__(self, config: HTTPClientConfig | None = None) -> None:
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError(
                "aiohttp is required for AsyncHTTPClient. Install with: pip install aiohttp"
            )

        self._config = config or HTTPClientConfig()
        self._session: aiohttp.ClientSession | None = None
        self._connector: aiohttp.TCPConnector | None = None
        self._middlewares: list[Middleware] = []
        self._running = False
        self._stats = {"requests": 0, "errors": 0, "retries": 0, "total_time": 0.0}
        self._lock = asyncio.Lock()

    @property
    def config(self) -> HTTPClientConfig:
        return self._config

    @property
    def running(self) -> bool:
        return self._running

    def add_middleware(self, middleware: Middleware) -> None:
        """Add middleware to the client."""
        self._middlewares.append(middleware)

    async def start(self) -> None:
        """Start the client and create connection pool."""
        if self._running:
            return

        # Create connector with connection pooling
        self._connector = aiohttp.TCPConnector(
            limit=self._config.max_connections,
            limit_per_host=self._config.max_keepalive,
            keepalive_timeout=self._config.keepalive_timeout,
            enable_cleanup_closed=True,
            ssl=None if not self._config.verify_ssl else ssl.create_default_context(),
        )

        # Create session with timeout
        timeout = aiohttp.ClientTimeout(
            total=self._config.timeout, connect=self._config.connect_timeout
        )

        self._session = aiohttp.ClientSession(
            connector=self._connector,
            timeout=timeout,
            headers={"User-Agent": self._config.user_agent, **self._config.default_headers},
            raise_for_status=False,
        )

        self._running = True

    async def stop(self) -> None:
        """Stop the client and close connections."""
        self._running = False

        if self._session:
            await self._session.close()
            self._session = None

        if self._connector:
            await self._connector.close()
            self._connector = None

    async def __aenter__(self) -> "AsyncHTTPClient":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()

    def _build_url(self, url: str) -> str:
        """Build full URL from base URL and path."""
        if self._config.base_url and not urlparse(url).netloc:
            return urljoin(self._config.base_url, url)
        return url

    async def _execute_request(self, request: HTTPRequest, attempt: int = 0) -> HTTPResponse:
        """Execute HTTP request with retries."""
        if not self._session:
            raise RuntimeError("Client not started. Call start() first.")

        url = self._build_url(request.url)
        request_id = str(uuid.uuid4())[:8]
        start_time = time.monotonic()

        # Process request through middlewares
        for middleware in self._middlewares:
            request = await middleware.process_request(request)

        # Prepare request arguments
        kwargs = {
            "method": request.method,
            "url": url,
            "headers": request.headers,
            "params": request.params,
            "allow_redirects": request.allow_redirects
            if request.allow_redirects is not None
            else self._config.follow_redirects,
            "max_redirects": self._config.max_redirects,
        }

        if request.json is not None:
            kwargs["json"] = request.json
        elif request.data is not None:
            if isinstance(request.data, (dict, str)):
                kwargs["data"] = request.data
            else:
                kwargs["data"] = request.data

        if request.timeout:
            kwargs["timeout"] = aiohttp.ClientTimeout(total=request.timeout)

        try:
            async with self._session.request(**kwargs) as resp:
                body = await resp.read()
                elapsed = time.monotonic() - start_time

                response = HTTPResponse(
                    status=resp.status,
                    headers=dict(resp.headers),
                    body=body,
                    url=str(resp.url),
                    elapsed=elapsed,
                    request_id=request_id,
                )

                # Process response through middlewares
                for middleware in self._middlewares:
                    response = await middleware.process_response(response)

                # Check for retry
                if (
                    resp.status in self._config.retry_on_status
                    and attempt < self._config.retry_attempts
                ):
                    delay = self._config.retry_delay * (self._config.retry_backoff**attempt)
                    await asyncio.sleep(delay)
                    self._stats["retries"] += 1
                    return await self._execute_request(request, attempt + 1)

                return response

        except TimeoutError:
            elapsed = time.monotonic() - start_time
            if attempt < self._config.retry_attempts:
                delay = self._config.retry_delay * (self._config.retry_backoff**attempt)
                await asyncio.sleep(delay)
                self._stats["retries"] += 1
                return await self._execute_request(request, attempt + 1)
            raise HTTPError(0, f"Request timeout after {elapsed:.2f}s")

        except aiohttp.ClientError as e:
            elapsed = time.monotonic() - start_time
            if attempt < self._config.retry_attempts:
                delay = self._config.retry_delay * (self._config.retry_backoff**attempt)
                await asyncio.sleep(delay)
                self._stats["retries"] += 1
                return await self._execute_request(request, attempt + 1)

            # Process exception through middlewares
            for middleware in self._middlewares:
                result = await middleware.process_exception(e)
                if result:
                    return result

            raise HTTPError(0, f"Client error: {e}")

    async def request(self, method: str, url: str, **kwargs) -> HTTPResponse:
        """Make HTTP request."""
        async with self._lock:
            self._stats["requests"] += 1

        request = HTTPRequest(method=method, url=url, **kwargs)

        try:
            response = await self._execute_request(request)
            async with self._lock:
                self._stats["total_time"] += response.elapsed
            return response
        except Exception as _e:
            async with self._lock:
                self._stats["errors"] += 1
            raise

    async def get(self, url: str, **kwargs) -> HTTPResponse:
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs) -> HTTPResponse:
        return await self.request("POST", url, **kwargs)

    async def put(self, url: str, **kwargs) -> HTTPResponse:
        return await self.request("PUT", url, **kwargs)

    async def patch(self, url: str, **kwargs) -> HTTPResponse:
        return await self.request("PATCH", url, **kwargs)

    async def delete(self, url: str, **kwargs) -> HTTPResponse:
        return await self.request("DELETE", url, **kwargs)

    async def head(self, url: str, **kwargs) -> HTTPResponse:
        return await self.request("HEAD", url, **kwargs)

    async def options(self, url: str, **kwargs) -> HTTPResponse:
        return await self.request("OPTIONS", url, **kwargs)

    def get_stats(self) -> dict[str, Any]:
        stats = self._stats.copy()
        if stats["requests"] > 0:
            stats["avg_time"] = stats["total_time"] / stats["requests"]
            stats["error_rate"] = stats["errors"] / stats["requests"]
        return stats


class RateLimitMiddleware(Middleware):
    """Rate limiting middleware."""

    def __init__(self, max_requests: int, window: float) -> None:
        self._max_requests = max_requests
        self._window = window
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def process_request(self, request: HTTPRequest) -> HTTPRequest:
        key = f"{request.method}:{urlparse(request.url).netloc}"

        async with self._lock:
            now = time.monotonic()
            cutoff = now - self._window

            # Clean old requests
            self._requests[key] = [t for t in self._requests[key] if t > cutoff]

            if len(self._requests[key]) >= self._max_requests:
                # Calculate wait time
                oldest = self._requests[key][0]
                wait_time = oldest + self._window - now
                raise HTTPError(429, f"Rate limited, retry after {wait_time:.1f}s")

            self._requests[key].append(now)

        return request


class AuthMiddleware(Middleware):
    """Authentication middleware."""

    def __init__(
        self, token_provider: Callable[[], Awaitable[str]], header: str = "Authorization"
    ) -> None:
        self._token_provider = token_provider
        self._header = header

    async def process_request(self, request: HTTPRequest) -> HTTPRequest:
        token = await self._token_provider()
        request.headers[self._header] = f"Bearer {token}"
        return request


class LoggingMiddleware(Middleware):
    """Request/response logging middleware."""

    def __init__(self, logger: Callable[[str], None] | None = None) -> None:
        self._logger = logger or (print)

    async def process_request(self, request: HTTPRequest) -> HTTPRequest:
        self._logger(f"REQUEST {request.method} {request.url}")
        return request

    async def process_response(self, response: HTTPResponse) -> HTTPResponse:
        self._logger(f"RESPONSE {response.status} {response.url} ({response.elapsed:.3f}s)")
        return response

    async def process_exception(self, exception: Exception) -> HTTPResponse | None:
        self._logger(f"ERROR: {exception}")
        return None


class RetryMiddleware(Middleware):
    """Retry middleware with custom logic."""

    def __init__(
        self, max_retries: int = 3, should_retry: Callable[[HTTPResponse], bool] | None = None
    ) -> None:
        self._max_retries = max_retries
        self._should_retry = should_retry or (lambda r: r.status >= 500)

    async def process_response(self, response: HTTPResponse) -> HTTPResponse:
        # This would need integration with client retry logic
        return response


def create_http_client(
    base_url: str = "", timeout: float = 30.0, max_connections: int = 100, **kwargs
) -> AsyncHTTPClient:
    """Factory for creating HTTP client."""
    config = HTTPClientConfig(
        base_url=base_url, timeout=timeout, max_connections=max_connections, **kwargs
    )
    return AsyncHTTPClient(config)


@asynccontextmanager
async def http_client_context(
    base_url: str = "", **kwargs
) -> AsyncGenerator[AsyncHTTPClient, None]:
    """Context manager for HTTP client."""
    client = create_http_client(base_url, **kwargs)
    await client.start()
    try:
        yield client
    finally:
        await client.stop()


# Example usage
async def example_usage() -> None:
    """Example of using the HTTP client."""
    async with http_client_context("https://api.example.com") as client:
        # Add middleware
        client.add_middleware(LoggingMiddleware())
        client.add_middleware(RateLimitMiddleware(10, 1.0))

        # Make requests
        response = await client.get("/users")
        print(f"Status: {response.status}")
        _users = response.json()

        # POST with JSON
        response = await client.post("/users", json={"name": "John"})
        print(f"Created: {response.status}")
