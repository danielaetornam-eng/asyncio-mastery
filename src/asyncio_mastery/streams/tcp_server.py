"""
TCP Server provides async TCP server with StreamReader and StreamWriter.

A TCP server is the program that waits for phone calls instead of dialing
them. A server is like a pizza shop with a published phone number (host +
port). It sits by the phone (listening). When a customer calls (a client
connects), the shop hands the call to a free worker (your handler coroutine)
who talks to that one customer. The shop keeps answering new calls. Many
conversations happen at once without blocking each other.

Key concepts:
- Host / port: The phone number clients dial. "127.0.0.1" means only calls
  from this same machine (great for testing). Port 0 means the OS picks a free
  port. The actual port is discoverable afterwards.
- Handler: A coroutine you write: async def handler(reader, writer). Called
  once per incoming connection. Gets that connection's reader/writer pair.
  Your protocol logic lives here.
- Backlog: How many incoming calls may wait on hold (in the OS queue) while
  you are busy accepting others. Like the number of phone lines on hold before
  callers get a busy signal.
- Graceful shutdown: Stopping the server politely. Stop answering new calls.
  Then finish/hang up existing conversations instead of cutting everyone off
  mid-sentence.
- SSL/TLS: Encryption layered on top of TCP. Pass an ssl.SSLContext to make
  connections private.

How concurrency works:
asyncio.start_server runs your handler for many connections concurrently in
one thread. Whenever a handler awaits a read, the event loop serves other
connections. No threads, no locks. Just coroutines taking turns.

Minimal example:
    import asyncio
    from asyncio_mastery.streams.tcp_server import TCPServer

    async def main() -> None:
        server = TCPServer("127.0.0.1", 8888)
        async with server.serving():
            await server.serve_forever()

    asyncio.run(main())
"""

import asyncio
import ssl
from asyncio import Server, StreamReader, StreamWriter
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any


class TCPServer:
    """Production-ready TCP server.

    Wraps asyncio.start_server with operational niceties real servers need:
    tracking of active connections, an overridable per-connection handler,
    SSL support, auto-discovered port, and graceful shutdown via an async
    context manager.

    Attributes:
        host: The interface the server listens on.
        port: The actual bound port (resolved after start() — important when
            you passed 0 to get an auto-assigned port).
        is_serving: True while the server is accepting new connections.
        connections: Number of currently open client connections.

    Example:
        async def my_handler(reader, writer):
            data = await reader.readline()
            writer.write(b"You said: " + data)
            await writer.drain()

        server = TCPServer("127.0.0.1", 8888, handler=my_handler)
        async with server.serving():
            await server.serve_forever()
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        *,
        ssl_context: ssl.SSLContext | None = None,
        backlog: int = 100,
        handler: Callable[[StreamReader, StreamWriter], Awaitable[None]] | None = None,
    ) -> None:
        """Configure the server. Nothing listens until start() is awaited.

        Args:
            host: Interface to bind. "127.0.0.1" = only local connections
                (safe for development). "0.0.0.0" = accept from anywhere on
                the network. Be careful, that exposes your server.
            port: Port to listen on. 0 = let the OS choose a free one. Read
                the chosen value from .port after start().
            ssl_context: An ssl.SSLContext configured with your certificate to
                encrypt connections (TLS), or None for plain text.
            backlog: Max number of pending (not-yet-accepted) connections the
                OS will queue. Like the number of callers that can be on hold.
            handler: Coroutine called once per accepted connection, receiving
                that connection's (reader, writer). Defaults to a line-echo
                handler, handy for tests.
        """
        self._host = host
        self._port = port
        self._ssl_context = ssl_context
        self._backlog = backlog
        self._handler = handler or self._default_handler
        self._server: Server | None = None
        self._running = False
        self._connections: set[StreamWriter] = set()

    @property
    def host(self) -> str:
        """The host/interface address the server is bound to."""
        return self._host

    @property
    def port(self) -> int:
        """The bound port (meaningful after start(), especially if 0 was given)."""
        return self._port

    @property
    def is_serving(self) -> bool:
        """True while the server socket is open and accepting connections."""
        return self._server is not None and self._server.is_serving()

    @property
    def connections(self) -> int:
        """Count of currently connected clients."""
        return len(self._connections)

    async def start(self) -> None:
        """Start the server.

        Binds the socket and begins accepting connections. From this moment,
        every incoming connection spawns your handler as a concurrent coroutine.

        Raises:
            OSError: If the address/port is unavailable (e.g. another process
                already listens on that port).
        """
        self._server = await asyncio.start_server(
            self._handle_connection,
            self._host,
            self._port,
            ssl=self._ssl_context,
            backlog=self._backlog,
        )
        if self._server.sockets:
            self._port = self._server.sockets[0].getsockname()[1]
        self._running = True
        print(f"Server started on {self._host}:{self._port}")

    async def stop(self, timeout: float = 5.0) -> None:
        """Gracefully stop the server.

        Two-phase polite shutdown:
        1. server.close() + wait_closed(): Stop accepting new connections.
           Existing conversations continue.
        2. Close each tracked client connection and wait for it to finish.
           No conversation is silently cut off mid-flight.

        Args:
            timeout: Max seconds to wait for the listening socket to close.
        """
        self._running = False
        if self._server:
            self._server.close()
            await asyncio.wait_for(self._server.wait_closed(), timeout=timeout)
        for writer in self._connections:
            writer.close()
            await writer.wait_closed()
        self._connections.clear()

    async def _handle_connection(self, reader: StreamReader, writer: StreamWriter) -> None:
        """Handle incoming connection.

        Internal wrapper around your handler. Registers the connection for
        tracking. Catches per-connection errors so one buggy client cannot
        crash the whole server. Guarantees the socket is closed afterwards.

        Args:
            reader: Incoming bytes from this one client.
            writer: Outgoing bytes to this one client.
        """
        self._connections.add(writer)
        try:
            await self._handler(reader, writer)
        except Exception as e:
            print(f"Connection error: {e}")
        finally:
            self._connections.discard(writer)
            writer.close()
            await writer.wait_closed()

    async def _default_handler(self, reader: StreamReader, writer: StreamWriter) -> None:
        """Default echo handler.

        Reads one line, sends it straight back. Empty bytes from readline
        means EOF (the client hung up). This ends the loop and closes the
        connection.

        Args:
            reader: Incoming bytes from the client.
            writer: Outgoing bytes to the client.
        """
        while True:
            data = await reader.readline()
            if not data:
                break
            writer.write(data)
            await writer.drain()

    async def serve_forever(self) -> None:
        """Run server forever.

        Convenience that starts the server if needed and then runs until the
        program is cancelled (e.g. Ctrl+C) or stop() is called from another
        task. Pair with signal handling for clean Ctrl+C shutdown.

        Raises:
            asyncio.CancelledError: Propagates when the surrounding task is
                cancelled. The normal way to stop serving.
        """
        if not self._server:
            await self.start()
        await self._server.serve_forever()

    @asynccontextmanager
    async def serving(self) -> AsyncIterator["TCPServer"]:
        """Context manager for server lifecycle.

        Starts the server on entry and gracefully stops it on exit, even if an
        exception occurs. Best practice for tests and demos.

        Yields:
            The running server (self).

        Example:
            async with server.serving():
                ... run clients against server.port ...
        """
        await self.start()
        try:
            yield self
        finally:
            await self.stop()


async def demonstrate_tcp_server() -> None:
    """Demonstrate TCP server.

    Starts the server on a random free port, connects to it with a real
    client, exchanges a couple of messages (including a quit command), and
    shuts everything down. Run this file directly to watch a full
    client/server conversation happen on your own machine.
    """
    print("=== TCP Server Demo ===")

    async def echo_handler(reader: StreamReader, writer: StreamWriter) -> None:
        addr = writer.get_extra_info("peername")
        print(f"New connection from {addr}")

        while True:
            data = await reader.read(1024)
            if not data:
                break
            message = data.decode().strip()
            print(f"Received: {message}")
            if message == "quit":
                break
            writer.write(f"Echo: {message}\n".encode())
            await writer.drain()

        print(f"Connection from {addr} closed")

    server = TCPServer("127.0.0.1", 0, handler=echo_handler)
    await server.start()

    reader, writer = await asyncio.open_connection("127.0.0.1", server.port)

    writer.write(b"Hello\n")
    await writer.drain()
    response = await reader.readline()
    print(f"Server response: {response.decode().strip()}")

    writer.write(b"quit\n")
    await writer.drain()

    writer.close()
    await writer.wait_closed()

    await server.stop()
    print("Server stopped")


class ChatServer(TCPServer):
    """Simple chat server example.

    Extends TCPServer into a tiny multi-user chat. Every connected client
    joins the general room. Any line one user types is broadcast to everyone
    else in the room. Demonstrates the classic server pattern of keeping a
    registry of connected writers (the rooms dict) so the server can push
    messages to clients without them asking first.

    Example usage is the same as TCPServer. Connect with multiple clients and
    type lines. Each client sees the others' messages. /quit disconnects.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Create a chat server. All args are forwarded to TCPServer.

        Args:
            *args: Positional arguments passed through to TCPServer.
            **kwargs: Keyword arguments passed through to TCPServer.
        """
        super().__init__(*args, **kwargs)
        self._rooms: dict[str, set[StreamWriter]] = {}

    async def _handle_connection(self, reader: StreamReader, writer: StreamWriter) -> None:
        """Greet a new chatter, drop them into general, and clean up after.

        Overrides the parent's connection bookkeeping so room membership is
        always tidied in finally. A user who crashes or disconnects never
        haunts a room.

        Args:
            reader: Incoming bytes from this user.
            writer: Outgoing bytes to this user.
        """
        addr = writer.get_extra_info("peername")
        print(f"Chat connection from {addr}")

        current_room = "general"
        self._join_room(current_room, writer)

        try:
            await self._handle_chat(reader, writer, current_room)
        finally:
            self._leave_room(current_room, writer)

    def _join_room(self, room: str, writer: StreamWriter) -> None:
        """Add a user's writer to a room, creating the room if needed.

        Args:
            room: Room name (e.g. "general").
            writer: The user's connection writer, used as their handle.
        """
        if room not in self._rooms:
            self._rooms[room] = set()
        self._rooms[room].add(writer)
        self._broadcast(room, f"User joined {room}\n", exclude=writer)

    def _leave_room(self, room: str, writer: StreamWriter) -> None:
        """Remove a user from a room and tell the others they left.

        Args:
            room: Room name.
            writer: The departing user's writer.
        """
        if room in self._rooms:
            self._rooms[room].discard(writer)
            self._broadcast(room, f"User left {room}\n")

    def _broadcast(self, room: str, message: str, exclude: StreamWriter | None = None) -> None:
        """Send a message to every member of a room (optionally skipping one).

        These write() calls are not drained here. Broadcasting is fire-and-
        forget per write. Each writer's buffering/backpressure is managed on
        its own drain cycles.

        Args:
            room: Room name.
            message: Text to send to room members.
            exclude: A writer to skip. Typically the sender, so people do not
                receive an echo of their own line.
        """
        if room in self._rooms:
            for writer in self._rooms[room]:
                if writer != exclude:
                    writer.write(message.encode())

    async def _handle_chat(self, reader: StreamReader, writer: StreamWriter, room: str) -> None:
        """Run one user's chat session: welcome them, then relay their lines.

        The loop ends on EOF (client hung up) or when the user types /quit.

        Args:
            reader: Incoming bytes from this user.
            writer: Outgoing bytes to this user.
            room: The room this session is attached to.
        """
        writer.write(b"Welcome to chat! Type /quit to exit.\n")
        await writer.drain()

        while True:
            data = await reader.readline()
            if not data:
                break
            message = data.decode().strip()
            if message == "/quit":
                break
            self._broadcast(room, f"{message}\n", exclude=writer)
            await writer.drain()


if __name__ == "__main__":
    asyncio.run(demonstrate_tcp_server())
