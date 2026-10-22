"""
Aiohttp Integration - Async HTTP client and server built on asyncio.

WHAT IS THIS, FOR A BEGINNER?
-----------------------------
HTTP is the protocol of the web: your program sends a request to a URL and
gets a response back. The catch is that network requests are SLOW (tens of
milliseconds to seconds). A classic blocking HTTP library (like ``urllib``
or ``requests``) freezes your whole program while it waits. ``aiohttp`` is
an asyncio-native HTTP library: while one request waits on the network, the
event loop runs other tasks -- so you can fire off hundreds of requests
"at once" from a single thread.

This module wraps aiohttp in two beginner-friendly classes:

- :class:`AiohttpClient`: an HTTP client with a *connection pool*. A pool
  keeps a set of already-open TCP connections warm and reuses them for new
  requests, avoiding the expensive "dial and handshake" every time.
- :class:`AiohttpServer`: a tiny web server you can spin up in code. It is
  perfect for tests and demos: your client can talk to a real HTTP server
  without leaving your machine (it listens on ``127.0.0.1``, i.e.
  "localhost", your own computer).

JARGON
------
- ClientSession: aiohttp's client handle. Creating one per request is
  wasteful; you create ONE and reuse it, then ``await session.close()``.
- connector: the object that manages the pool of TCP connections.
- keep-alive: leaving a TCP connection open after a request so the next
  request to the same server skips reconnection.
- route / handler: a mapping from a URL path (like ``/api/hello``) to the
  async function that answers requests to it.
- graceful degradation: this module still imports fine when aiohttp is not
  installed; the classes then raise a clear ``RuntimeError`` if used.

MINIMAL EXAMPLE
---------------
>>> async with AiohttpClient().managed() as client:
...     data = await client.get_json("https://api.example.com/status")
"""

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

try:
    import aiohttp
    from aiohttp import ClientSession, ClientTimeout, web

    AIOHTTP_AVAILABLE = True
except ImportError:
    # aiohttp is an optional dependency; keep the module importable and
    # remember that the real classes are unavailable.
    AIOHTTP_AVAILABLE = False
    aiohttp = None
    web = None
    ClientSession = None
    ClientTimeout = None


@dataclass
class HttpClientConfig:
    """Tuning knobs for :class:`AiohttpClient`.

    A *dataclass* is a compact way to define a class whose main job is to
    hold data; Python writes the ``__init__`` for you.

    Attributes:
        timeout: Max seconds for one entire HTTP request before failing.
        max_connections: Total TCP connections held open in the pool.
        max_keepalive: Connections kept warm per individual host.
        keepalive_timeout: Seconds an idle pooled connection stays open.
    """

    timeout: float = 30.0
    max_connections: int = 100
    max_keepalive: int = 20
    keepalive_timeout: float = 30.0


class AiohttpClient:
    """Async HTTP client with connection pooling and a friendly API.

    Wraps ``aiohttp.ClientSession``. You may: (a) call :meth:`start` /
    :meth:`close` yourself, (b) use ``async with client.managed():`` so
    setup and cleanup happen automatically, or (c) just fire requests --
    the session is created lazily on first use.

    Args:
        config: Optional :class:`HttpClientConfig`; defaults are sensible.

    Raises:
        RuntimeError: On construction if aiohttp is not installed.

    Example:
        >>> async with AiohttpClient().managed() as client:
        ...     resp = await client.get("https://example.com")
        ...     text = await resp.text()
    """

    def __init__(self, config: HttpClientConfig | None = None) -> None:
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp not installed")
        self._config = config or HttpClientConfig()
        self._session: ClientSession | None = None
        self._connector: aiohttp.TCPConnector | None = None

    async def start(self) -> None:
        """Create the connector and session (call once before heavy use).

        Builds a :class:`aiohttp.TCPConnector` honoring the pool limits in
        the config, then a :class:`aiohttp.ClientSession` using it.
        """
        self._connector = aiohttp.TCPConnector(
            limit=self._config.max_connections,
            limit_per_host=self._config.max_keepalive,
            keepalive_timeout=self._config.keepalive_timeout,
        )
        timeout = ClientTimeout(total=self._config.timeout)
        self._session = ClientSession(connector=self._connector, timeout=timeout)

    async def close(self) -> None:
        """Close the session and connector, releasing pooled connections."""
        if self._session:
            await self._session.close()
        if self._connector:
            await self._connector.close()

    @asynccontextmanager
    async def managed(self) -> Any:
        """Async context manager: start on entry, close on exit.

        Yields:
            This client, ready to use.

        Example:
            >>> async with client.managed() as c:
            ...     await c.get_json(url)
        """
        await self.start()
        try:
            yield self
        finally:
            await self.close()

    async def request(self, method: str, url: str, **kwargs: Any) -> aiohttp.ClientResponse:
        """Perform an HTTP request of any method.

        Args:
            method: HTTP verb, e.g. ``"GET"`` or ``"POST"``.
            url: The full URL to request.
            **kwargs: Forwarded to aiohttp (e.g. ``params=``, ``json=``,
                ``headers=``).

        Returns:
            The aiohttp response object. Remember to read its body
            (``await resp.json()`` / ``await resp.text()``), preferably
            inside ``async with ... as resp:`` to release the connection.

        Example:
            >>> resp = await client.request("GET", "https://example.com")
        """
        if not self._session:
            # Lazily start so one-off requests work without start()/managed().
            await self.start()
        return await self._session.request(method, url, **kwargs)

    async def get(self, url: str, **kwargs: Any) -> aiohttp.ClientResponse:
        """Perform an HTTP GET request. See :meth:`request` for details."""
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs: Any) -> aiohttp.ClientResponse:
        """Perform an HTTP POST request. See :meth:`request` for details."""
        return await self.request("POST", url, **kwargs)

    async def put(self, url: str, **kwargs: Any) -> aiohttp.ClientResponse:
        """Perform an HTTP PUT request. See :meth:`request` for details."""
        return await self.request("PUT", url, **kwargs)

    async def delete(self, url: str, **kwargs: Any) -> aiohttp.ClientResponse:
        """Perform an HTTP DELETE request. See :meth:`request` for details."""
        return await self.request("DELETE", url, **kwargs)

    async def get_json(self, url: str, **kwargs: Any) -> Any:
        """GET a URL and return the response body parsed as JSON.

        Args:
            url: The URL to fetch.
            **kwargs: Forwarded to the request (headers, params, ...).

        Returns:
            The decoded JSON (usually a dict or list).
        """
        # `async with` on the response releases the connection back to the
        # pool as soon as the body has been read.
        async with await self.get(url, **kwargs) as resp:
            return await resp.json()

    async def post_json(self, url: str, data: Any, **kwargs: Any) -> Any:
        """POST a JSON body and return the response body parsed as JSON.

        Args:
            url: The URL to post to.
            data: A JSON-serializable object (dict, list, ...) to send.
            **kwargs: Forwarded to the request.

        Returns:
            The decoded JSON response.
        """
        async with await self.post(url, json=data, **kwargs) as resp:
            return await resp.json()


class AiohttpServer:
    """A small async HTTP server, ideal for demos and tests.

    Wraps ``aiohttp.web``. Use port ``0`` to ask the operating system for a
    free port (avoids "address already in use" collisions); the actual port
    is then available via :attr:`url` after :meth:`start`.

    Args:
        host: Interface to bind. ``"127.0.0.1"`` = only this machine.
        port: TCP port to listen on; 0 = pick any free port.

    Raises:
        RuntimeError: On construction if aiohttp is not installed.

    Example:
        >>> server = AiohttpServer(port=0)
        >>> server.add_route("GET", "/hello", my_handler)
        >>> async with server.managed():
        ...     print(server.url)
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp not installed")
        self._host = host
        self._port = port
        self._app = web.Application()
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._routes = []

    def add_route(self, method: str, path: str, handler: Callable[..., Awaitable[Any]]) -> None:
        """Register one URL route and its async handler function.

        Args:
            method: HTTP verb to match, e.g. ``"GET"``.
            path: URL path to match, e.g. ``"/api/hello"``.
            handler: ``async def handler(request) -> web.Response``.
        """
        self._app.router.add_route(method, path, handler)
        self._routes.append((method, path))

    def add_routes(self, routes: list[web.RouteDef]) -> None:
        """Register several routes at once from aiohttp ``RouteDef`` objects.

        Args:
            routes: e.g. the list produced by ``aiohttp.web.RouteTableDef``.
        """
        self._app.add_routes(routes)

    async def start(self) -> None:
        """Bind the socket and begin accepting HTTP requests."""
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()
        if self._site._server:
            # If port 0 was requested, read back which port the OS picked.
            self._port = self._site._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        """Stop accepting requests and release the socket."""
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()

    @property
    def url(self) -> str:
        """Base URL of the running server, e.g. ``http://127.0.0.1:51023``."""
        return f"http://{self._host}:{self._port}"

    @asynccontextmanager
    async def managed(self) -> Any:
        """Async context manager: start on entry, stop on exit.

        Yields:
            This server.

        Example:
            >>> async with server.managed():
            ...     await client.get(server.url + "/api/hello")
        """
        await self.start()
        try:
            yield self
        finally:
            await self.stop()


# Example handlers
async def handle_get(request: web.Request) -> web.Response:
    """Example GET handler: reply with a JSON greeting and the request path.

    Args:
        request: The incoming HTTP request object from aiohttp.

    Returns:
        A JSON response like ``{"message": "Hello!", "path": "/api/hello"}``.
    """
    return web.json_response({"message": "Hello!", "path": request.path})


async def handle_post(request: web.Request) -> web.Response:
    """Example POST handler: echo back the JSON body that was sent.

    Args:
        request: The incoming request; its JSON body is read and echoed.

    Returns:
        A JSON response containing the received data.
    """
    data = await request.json()
    return web.json_response({"received": data, "method": "POST"})


async def demonstrate_aiohttp() -> None:
    """Narrated end-to-end demo: run a server and hit it with a client.

    Shows single GET/POST calls and, most importantly, ``asyncio.gather``
    running ten requests concurrently -- the superpower of async HTTP.
    Skips gracefully if aiohttp is not installed.
    """
    if not AIOHTTP_AVAILABLE:
        print("aiohttp not installed, skipping demo")
        return

    print("=== Aiohttp Integration Demo ===")

    # Server: bind to localhost on any free port and register two routes.
    print("\n--- Server ---")
    server = AiohttpServer("127.0.0.1", 0)
    server.add_route("GET", "/api/hello", handle_get)
    server.add_route("POST", "/api/echo", handle_post)

    async with server.managed():
        print(f"Server started at {server.url}")

        # Client
        print("\n--- Client ---")
        async with AiohttpClient().managed() as client:
            # GET
            resp = await client.get(f"{server.url}/api/hello")
            data = await resp.json()
            print(f"GET: {data}")

            # POST
            resp = await client.post_json(f"{server.url}/api/echo", {"key": "value"})
            data = await resp.json()
            print(f"POST: {data}")

            # Concurrent requests: gather() starts all 10 at once and waits
            # for every response -- total time ~= time of the slowest one.
            print("\n--- Concurrent Requests ---")
            urls = [f"{server.url}/api/hello" for _ in range(10)]
            responses = await asyncio.gather(*[client.get_json(u) for u in urls])
            print(f"Completed {len(responses)} concurrent requests")


if __name__ == "__main__":
    asyncio.run(demonstrate_aiohttp())
