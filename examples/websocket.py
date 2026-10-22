"""
WebSocket Server and Client

Production-ready WebSocket implementation with rooms, broadcasting, and middleware.
"""

import asyncio
import contextlib
import json
import time
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

try:
    import websockets

    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    websockets = None


class MessageType(Enum):
    TEXT = "text"
    BINARY = "binary"
    CLOSE = "close"
    PING = "ping"
    PONG = "pong"


@dataclass
class WebSocketMessage:
    """WebSocket message."""

    type: MessageType
    data: str | bytes
    timestamp: float = field(default_factory=time.time)
    client_id: str | None = None


@dataclass
class WebSocketConfig:
    """WebSocket server configuration."""

    host: str = "0.0.0.0"
    port: int = 8080
    max_connections: int = 1000
    ping_interval: float = 20.0
    ping_timeout: float = 10.0
    close_timeout: float = 10.0
    max_message_size: int = 1024 * 1024  # 1MB
    compression: bool = True


class WebSocketHandler:
    """Base WebSocket handler."""

    async def on_connect(self, client_id: str, websocket: Any) -> None:
        """Called when client connects."""
        pass

    async def on_disconnect(self, client_id: str) -> None:
        """Called when client disconnects."""
        pass

    async def on_message(self, client_id: str, message: WebSocketMessage) -> None:
        """Called when message received."""
        pass

    async def on_error(self, client_id: str, error: Exception) -> None:
        """Called when error occurs."""
        pass


class WebSocketServer:
    """WebSocket server with room support."""

    def __init__(
        self, config: WebSocketConfig | None = None, handler: WebSocketHandler | None = None
    ) -> None:
        if not WEBSOCKETS_AVAILABLE:
            raise RuntimeError("websockets is required. Install with: pip install websockets")

        self._config = config or WebSocketConfig()
        self._handler = handler or WebSocketHandler()
        self._server: Any | None = None
        self._running = False

        # Client management
        self._clients: dict[str, Any] = {}  # client_id -> websocket
        self._client_info: dict[str, dict] = {}  # client_id -> info
        self._rooms: dict[str, set[str]] = defaultdict(set)  # room -> client_ids
        self._client_rooms: dict[str, set[str]] = defaultdict(set)  # client_id -> rooms

        # Message handlers
        self._message_handlers: dict[str, Callable] = {}
        self._lock = asyncio.Lock()

        # Stats
        self._stats = {
            "connections": 0,
            "disconnections": 0,
            "messages_received": 0,
            "messages_sent": 0,
            "errors": 0,
        }

    @property
    def config(self) -> WebSocketConfig:
        return self._config

    @property
    def running(self) -> bool:
        return self._running

    @property
    def connected_clients(self) -> int:
        return len(self._clients)

    def stats(self) -> dict[str, Any]:
        return {
            **self._stats.copy(),
            "connected_clients": self.connected_clients,
            "rooms": len(self._rooms),
        }

    async def start(self) -> None:
        """Start the WebSocket server."""
        if self._running:
            return

        self._running = True

        self._server = await websockets.serve(
            self._handle_connection,
            self._config.host,
            self._config.port,
            ping_interval=self._config.ping_interval,
            ping_timeout=self._config.ping_timeout,
            close_timeout=self._config.close_timeout,
            max_size=self._config.max_message_size,
            compression=self._config.compression,
        )

    async def stop(self) -> None:
        """Stop the WebSocket server."""
        self._running = False

        # Close all client connections
        for client_id in list(self._clients.keys()):
            await self._disconnect_client(client_id, code=1001, reason="Server shutting down")

        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def _handle_connection(self, websocket: Any, path: str) -> None:
        """Handle new WebSocket connection."""
        client_id = str(uuid.uuid4())[:8]

        async with self._lock:
            if len(self._clients) >= self._config.max_connections:
                await websocket.close(code=1013, reason="Server full")
                return

            self._clients[client_id] = websocket
            self._client_info[client_id] = {
                "path": path,
                "connected_at": time.time(),
                "remote_addr": (
                    websocket.remote_address[0] if websocket.remote_address else "unknown"
                ),
            }
            self._stats["connections"] += 1

        try:
            await self._handler.on_connect(client_id, websocket)

            async for message in websocket:
                await self._handle_message(client_id, message)

        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            self._stats["errors"] += 1
            await self._handler.on_error(client_id, e)
        finally:
            await self._disconnect_client(client_id)

    async def _handle_message(self, client_id: str, raw_message: str | bytes) -> None:
        """Handle incoming message."""
        self._stats["messages_received"] += 1

        if isinstance(raw_message, str):
            msg = WebSocketMessage(type=MessageType.TEXT, data=raw_message, client_id=client_id)
        else:
            msg = WebSocketMessage(type=MessageType.BINARY, data=raw_message, client_id=client_id)

        # Parse JSON messages for routing
        if msg.type == MessageType.TEXT:
            try:
                data = json.loads(msg.data)
                if isinstance(data, dict) and "type" in data:
                    msg_type = data["type"]
                    handler = self._message_handlers.get(msg_type)
                    if handler:
                        await handler(client_id, data)
                        return
            except json.JSONDecodeError:
                pass

        await self._handler.on_message(client_id, msg)

    async def _disconnect_client(self, client_id: str, code: int = 1000, reason: str = "") -> None:
        """Disconnect a client."""
        async with self._lock:
            websocket = self._clients.pop(client_id, None)
            info = self._client_info.pop(client_id, None)

            if websocket:
                with contextlib.suppress(Exception):
                    await websocket.close(code=code, reason=reason)

            # Remove from rooms
            rooms = self._client_rooms.pop(client_id, set())
            for room in rooms:
                self._rooms[room].discard(client_id)
                if not self._rooms[room]:
                    del self._rooms[room]

            self._stats["disconnections"] += 1

        if info:
            await self._handler.on_disconnect(client_id)

    def on_message_type(self, msg_type: str):
        """Decorator for registering message type handlers."""

        def decorator(func: Callable):
            self._message_handlers[msg_type] = func
            return func

        return decorator

    async def send(self, client_id: str, message: str | bytes | dict) -> bool:
        """Send message to a specific client."""
        async with self._lock:
            websocket = self._clients.get(client_id)

        if not websocket:
            return False

        try:
            if isinstance(message, dict):
                await websocket.send(json.dumps(message))
            elif isinstance(message, str):
                await websocket.send(message)
            else:
                await websocket.send(message)

            self._stats["messages_sent"] += 1
            return True
        except Exception:
            await self._disconnect_client(client_id)
            return False

    async def broadcast(
        self, message: str | bytes | dict, room: str | None = None, exclude: set[str] | None = None
    ) -> int:
        """Broadcast message to all clients or room."""
        exclude = exclude or set()

        async with self._lock:
            if room:
                targets = self._rooms.get(room, set()) - exclude
            else:
                targets = set(self._clients.keys()) - exclude

        sent = 0
        for client_id in targets:
            if await self.send(client_id, message):
                sent += 1

        return sent

    async def join_room(self, client_id: str, room: str) -> bool:
        """Add client to room."""
        async with self._lock:
            if client_id not in self._clients:
                return False

            self._rooms[room].add(client_id)
            self._client_rooms[client_id].add(room)
            return True

    async def leave_room(self, client_id: str, room: str) -> bool:
        """Remove client from room."""
        async with self._lock:
            if room in self._rooms:
                self._rooms[room].discard(client_id)
                if not self._rooms[room]:
                    del self._rooms[room]

            self._client_rooms[client_id].discard(room)
            return True

    def get_room_clients(self, room: str) -> set[str]:
        """Get clients in a room."""
        return self._rooms.get(room, set()).copy()

    def get_client_rooms(self, client_id: str) -> set[str]:
        """Get rooms for a client."""
        return self._client_rooms.get(client_id, set()).copy()

    async def __aenter__(self) -> "WebSocketServer":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()


class WebSocketClient:
    """WebSocket client with auto-reconnect."""

    def __init__(
        self,
        url: str,
        *,
        reconnect: bool = True,
        reconnect_interval: float = 5.0,
        max_reconnect_attempts: int = 10,
        ping_interval: float = 20.0,
        ping_timeout: float = 10.0,
    ) -> None:
        if not WEBSOCKETS_AVAILABLE:
            raise RuntimeError("websockets is required. Install with: pip install websockets")

        self._url = url
        self._reconnect = reconnect
        self._reconnect_interval = reconnect_interval
        self._max_reconnect_attempts = max_reconnect_attempts
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout

        self._websocket: Any | None = None
        self._connected = False
        self._running = False
        self._reconnect_task: asyncio.Task | None = None
        self._message_queue: asyncio.Queue = asyncio.Queue()

        self._on_message_callbacks: list[Callable[[WebSocketMessage], Awaitable[None]]] = []
        self._on_connect_callbacks: list[Callable[[], Awaitable[None]]] = []
        self._on_disconnect_callbacks: list[Callable[[Exception | None], Awaitable[None]]] = []

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def url(self) -> str:
        return self._url

    def on_message(self, callback: Callable[[WebSocketMessage], Awaitable[None]]) -> None:
        self._on_message_callbacks.append(callback)

    def on_connect(self, callback: Callable[[], Awaitable[None]]) -> None:
        self._on_connect_callbacks.append(callback)

    def on_disconnect(self, callback: Callable[[Exception | None], Awaitable[None]]) -> None:
        self._on_disconnect_callbacks.append(callback)

    async def connect(self) -> None:
        """Connect to WebSocket server."""
        if self._connected:
            return

        self._running = True
        await self._connect_internal()

    async def _connect_internal(self) -> None:
        """Internal connection logic with reconnect."""
        attempts = 0

        while self._running:
            try:
                self._websocket = await websockets.connect(
                    self._url, ping_interval=self._ping_interval, ping_timeout=self._ping_timeout
                )

                self._connected = True

                # Fire connect callbacks
                for cb in self._on_connect_callbacks:
                    with contextlib.suppress(Exception):
                        await cb()

                # Start message handler
                await self._handle_messages()

            except Exception as e:
                self._connected = False

                # Fire disconnect callbacks
                for cb in self._on_disconnect_callbacks:
                    with contextlib.suppress(Exception):
                        await cb(e)

                if not self._reconnect or attempts >= self._max_reconnect_attempts:
                    raise

                attempts += 1
                await asyncio.sleep(self._reconnect_interval)

    async def _handle_messages(self) -> None:
        """Handle incoming messages."""
        try:
            async for message in self._websocket:
                if isinstance(message, str):
                    msg = WebSocketMessage(type=MessageType.TEXT, data=message)
                else:
                    msg = WebSocketMessage(type=MessageType.BINARY, data=message)

                for cb in self._on_message_callbacks:
                    with contextlib.suppress(Exception):
                        await cb(msg)
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception:
            pass
        finally:
            self._connected = False

    async def send(self, message: str | bytes | dict) -> bool:
        """Send message."""
        if not self._connected or not self._websocket:
            return False

        try:
            if isinstance(message, dict):
                await self._websocket.send(json.dumps(message))
            elif isinstance(message, str):
                await self._websocket.send(message)
            else:
                await self._websocket.send(message)
            return True
        except Exception:
            return False

    async def disconnect(self) -> None:
        """Disconnect from server."""
        self._running = False
        self._connected = False

        if self._websocket:
            await self._websocket.close()
            self._websocket = None

    async def __aenter__(self) -> "WebSocketClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()


def create_ws_server(
    host: str = "0.0.0.0", port: int = 8080, handler: WebSocketHandler | None = None, **kwargs
) -> WebSocketServer:
    """Factory for creating WebSocket server."""
    config = WebSocketConfig(host=host, port=port, **kwargs)
    return WebSocketServer(config, handler)


# Example handler
class EchoHandler(WebSocketHandler):
    """Echo handler that sends back received messages."""

    async def on_message(self, client_id: str, message: WebSocketMessage) -> None:
        if message.type == MessageType.TEXT:
            await self._server.send(client_id, f"Echo: {message.data}")


# Example usage
async def example_server() -> None:
    """Example WebSocket server."""
    server = create_ws_server(host="0.0.0.0", port=8080, handler=EchoHandler())

    # Register custom message handlers
    @server.on_message_type("chat")
    async def handle_chat(client_id: str, data: dict) -> None:
        await server.broadcast(
            {"type": "chat", "from": client_id, "message": data.get("message", "")}
        )

    @server.on_message_type("join")
    async def handle_join(client_id: str, data: dict) -> None:
        room = data.get("room", "general")
        await server.join_room(client_id, room)
        await server.send(client_id, {"type": "joined", "room": room})

    async with server:
        print("WebSocket server running on ws://0.0.0.0:8080")
        await asyncio.Future()  # Run forever


async def example_client() -> None:
    """Example WebSocket client."""
    async with WebSocketClient("ws://localhost:8080") as client:
        client.on_message(lambda msg: print(f"Received: {msg.data}"))

        await client.send({"type": "join", "room": "general"})
        await client.send({"type": "chat", "message": "Hello!"})

        await asyncio.sleep(10)
