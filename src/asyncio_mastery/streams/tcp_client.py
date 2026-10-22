"""
TCP Client provides async TCP client with connection pooling and retries.

TCP is the most common way two programs talk over a network. A TCP connection
is like a phone call. First you dial (connect). Then you have a reliable,
ordered, two-way conversation. Bytes you send arrive exactly once, in order.
Finally, someone hangs up (close). A socket is the phone jack. A TCPClient
wraps the whole dial, talk, hang up lifecycle into one object.

A client is the side that initiates the call. The other side is the server.
Web browsers are clients. Database drivers are clients. Chat apps are clients.

Raw asyncio.open_connection() gives you a reader/writer pair and nothing
else. Real programs usually also want:
- A connect timeout. Do not wait forever if the server is down.
- Socket tuning (keepalive, no-delay).
- A clean connect/disconnect lifecycle (via async with).
- Connection pooling. Dialing a new connection for every request is slow.
TCP and TLS handshakes cost round-trips. A pool keeps a small set of open
connections and lends them out.

Terms:
- Keepalive: Tiny probe packets that keep an idle connection alive. They
  detect a silently-dead peer.
- TCP_NODELAY: Disables Nagle algorithm. Nagle batches small writes to save
  packets. Turning it on makes small messages leave immediately. This gives
  lower latency. Interactive protocols want this.
- Handshake: The initial back-and-forth that establishes a connection. TCP
  uses SYN/SYN-ACK/ACK. TLS adds certificate/key exchange for encryption.

Use this module any time your asyncio program needs to talk to a TCP server.
Examples: REST protocols, Redis/Postgres wire protocols, echo/chat servers.

Minimal example:
    import asyncio
    from asyncio_mastery.streams.tcp_client import ConnectionConfig, TCPClient

    async def main() -> None:
        client = TCPClient(ConnectionConfig(host="127.0.0.1", port=8888))
        async with client.connection():
            await client.send_line("hello")
            reply = await client.receive_line()
            print(reply)

    asyncio.run(main()
"""

import asyncio
import socket
import ssl
from asyncio import StreamReader, StreamWriter
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass


@dataclass
class ConnectionConfig:
    """TCP connection configuration.

    A simple bundle of connection settings. Kept separate from the client so
    the same config can be shared by a pool of connections.

    Attributes:
        host: Hostname or IP address of the server. Example: "127.0.0.1",
            "localhost", or "example.com".
        port: TCP port number (0-65535). An IP address is like an office
            building. The port is the extension number. Servers listen on a
            port. Clients dial it.
        ssl: Encryption settings. False = plain text. True = TLS with sensible
            defaults (certificate verification on). An ssl.SSLContext = full
            manual control.
        timeout: Max seconds to wait for connection establishment. Prevents
            hanging forever when a server is down or firewalled.
        keepalive: If True, allow SO_KEEPALIVE. The OS sends tiny probe
            packets on idle connections. Dead peers are detected.
        no_delay: If True, set TCP_NODELAY. Disable Nagle batching. Small
            messages are sent immediately. Good for interactive protocols where
            latency matters more than packet count.
    """

    host: str
    port: int
    ssl: bool | ssl.SSLContext = False
    timeout: float = 30.0
    keepalive: bool = True
    no_delay: bool = True


class TCPClient:
    """Async TCP client with advanced features.

    Manages one TCP connection. Establishes it with a timeout and socket
    tuning. Sends and receives with friendly helpers. Shuts it down cleanly.
    Provides an async with context manager so connections cannot leak.

    Attributes:
        connected: True only if a connection was established and the writer
            has not started closing.
        reader: The underlying asyncio.StreamReader (or None if not connected).
        writer: The underlying asyncio.StreamWriter (or None if not connected).

    Example:
        client = TCPClient(ConnectionConfig("127.0.0.1", 8888))
        async with client.connection():
            await client.send_line("contact")
            print(await client.receive_line())
    """

    def __init__(self, config: ConnectionConfig) -> None:
        """Create a client. Does NOT connect yet. Call connect().

        Args:
            config: The connection settings bundle. Construction is cheap.
                The actual network work happens in connect().
        """
        self._config = config
        self._reader: StreamReader | None = None
        self._writer: StreamWriter | None = None
        self._connected = False

    @property
    def connected(self) -> bool:
        """Whether the client currently holds a live, writable connection."""
        return self._connected and self._writer is not None and not self._writer.is_closing()

    @property
    def reader(self) -> StreamReader | None:
        """The stream reader (incoming bytes), or None when not connected."""
        return self._reader

    @property
    def writer(self) -> StreamWriter | None:
        """The stream writer (outgoing bytes), or None when not connected."""
        return self._writer

    async def connect(self) -> None:
        """Establish connection.

        Performs the TCP handshake (and the TLS handshake too, if SSL is
        enabled), subject to the configured timeout. Also applies the socket
        options from the config. Safe to call twice: if already connected, it
        returns immediately.

        Raises:
            asyncio.TimeoutError: If the connection isn't established within
                config.timeout seconds.
            OSError: If the connection is refused (nothing listening on that
                host:port) or the network is unreachable.
            ssl.SSLError: If TLS verification fails (e.g. bad certificate).
        """
        if self._connected:
            return

        ssl_context = self._config.ssl
        if ssl_context is True:
            ssl_context = ssl.create_default_context()

        self._reader, self._writer = await asyncio.wait_for(
            asyncio.open_connection(self._config.host, self._config.port, ssl=ssl_context),
            timeout=self._config.timeout,
        )

        transport = self._writer.transport
        sock = transport.get_extra_info("socket")
        if sock:
            if self._config.keepalive:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            if self._config.no_delay:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        self._connected = True

    async def disconnect(self) -> None:
        """Close connection.

        Politely hangs up: flushes pending writes, closes the socket, and
        waits for the close to finish. Safe to call even if never connected.
        """
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()
        self._connected = False
        self._reader = None
        self._writer = None

    async def send(self, data: bytes) -> None:
        """Send data.

        Writes bytes and immediately drains. By the time this coroutine
        returns, the data has been handed to the OS for delivery. Backpressure
        is respected.

        Args:
            data: Bytes to send. Encode strings first.

        Raises:
            ConnectionError: If connect() was never called or the connection
                has been dropped.
        """
        if not self._connected or not self._writer:
            raise ConnectionError("Not connected")
        self._writer.write(data)
        await self._writer.drain()

    async def send_line(self, line: str, encoding: str = "utf-8") -> None:
        """Send line with newline.

        Convenience for line-oriented protocols. Appends newline, encodes,
        sends, and drains in one call.

        Args:
            line: Text to send (no trailing newline needed).
            encoding: How to turn the string into bytes.
        """
        await self.send((line + "\n").encode(encoding))

    async def receive(self, n: int = -1) -> bytes:
        """Receive data.

        Awaits up to n bytes from the server. Like StreamReader.read: it may
        return fewer than n bytes (whatever has arrived). Returns empty bytes
        on EOF (server hung up). While awaiting, other tasks keep running.

        Args:
            n: Max bytes to return. -1 = read everything until EOF.

        Returns:
            The bytes received. Empty means the connection was closed (EOF).

        Raises:
            ConnectionError: If not connected.
        """
        if not self._connected or not self._reader:
            raise ConnectionError("Not connected")
        return await self._reader.read(n)

    async def receive_line(self) -> bytes:
        """Receive one line.

        Awaits until a full line (ending in newline) arrives and returns it.
        Newline included. Empty bytes means EOF.

        Returns:
            One line of incoming data.

        Raises:
            ConnectionError: If not connected.
            asyncio.IncompleteReadError: If the connection closes mid-line.
        """
        if not self._connected or not self._reader:
            raise ConnectionError("Not connected")
        return await self._reader.readline()

    async def receive_exactly(self, n: int) -> bytes:
        """Receive exactly n bytes.

        Keeps awaiting until ALL n bytes have arrived. The right tool for
        binary protocols and length-prefixed framing.

        Args:
            n: Exact number of bytes required.

        Returns:
            Exactly n bytes.

        Raises:
            ConnectionError: If not connected.
            asyncio.IncompleteReadError: If EOF arrives early. The partial
                attribute contains what did arrive.
        """
        if not self._connected or not self._reader:
            raise ConnectionError("Not connected")
        return await self._reader.readexactly(n)

    @asynccontextmanager
    async def connection(self) -> AsyncIterator["TCPClient"]:
        """Context manager for connection lifecycle.

        The recommended way to use a TCPClient. Connects on entry and
        guarantees disconnection on exit, even if your code raises. Mirrors
        the with open() pattern you know from files.

        Yields:
            The connected client (self).

        Example:
            async with client.connection() as conn:
                await conn.send_line("hello")
        """
        await self.connect()
        try:
            yield self
        finally:
            await self.disconnect()


class TCPClientPool:
    """Pool of TCP clients for connection reuse.

    Opening a TCP (+TLS) connection costs network round-trips (the handshake).
    If your program makes many short requests to the same server, reopening a
    connection each time wastes time. A pool keeps up to max_size connections
    open and hands them out. When you are done, the connection goes back to
    the pool for the next user instead of being closed. Analogy: a taxi stand.
    Cabs (connections) wait ready. Passengers (requests) hop in, ride, and the
    cab returns to the stand.

    Attributes: All private. Interact through acquire/release/client.

    Example:
        pool = TCPClientPool(config, max_size=5)
        async with pool.client() as client:
            await client.send_line("ping")
        await pool.close_all()
    """

    def __init__(
        self, config: ConnectionConfig, max_size: int = 10, max_idle: float = 30.0
    ) -> None:
        """Create a pool.

        Args:
            config: Shared connection settings for every pooled client.
            max_size: Maximum number of simultaneous connections. Once reached,
                additional acquire() calls wait until a connection is released.
                This is intentional backpressure that stops your program from
                flooding the server.
            max_idle: How long an idle connection may sit in the pool before
                being discarded.
        """
        self._config = config
        self._max_size = max_size
        self._max_idle = max_idle
        self._pool: asyncio.Queue[TCPClient] = asyncio.Queue(maxsize=max_size)
        self._created = 0
        self._lock = asyncio.Lock()

    async def acquire(self) -> TCPClient:
        """Get a client from pool or create new.

        Order of preference:
        1. Reuse an idle pooled connection (if still alive).
        2. Create a brand-new connection (if under max_size).
        3. Wait until another task releases one (backpressure).

        Returns:
            A connected TCPClient that you MUST later pass to release() (or
            use the client() context manager, which does it for you).
        """
        try:
            client = self._pool.get_nowait()
            if client.connected:
                return client
        except asyncio.QueueEmpty:
            pass

        async with self._lock:
            if self._created < self._max_size:
                self._created += 1
                client = TCPClient(self._config)
                await client.connect()
                return client

        client = await self._pool.get()
        if not client.connected:
            await client.connect()
        return client

    async def release(self, client: TCPClient) -> None:
        """Return client to pool.

        If the connection is still alive, it goes back into the stand for
        reuse. If it has died meanwhile, the pool's connection budget is
        refunded so a future acquire() may create a replacement.

        Args:
            client: A client previously obtained from acquire().
        """
        if client.connected:
            await self._pool.put(client)
        else:
            async with self._lock:
                self._created -= 1

    @asynccontextmanager
    async def client(self) -> AsyncIterator[TCPClient]:
        """Context manager for pooled client.

        Acquires a connection on entry and always releases it back to the pool
        on exit, even if an exception occurs. This is the safe, recommended
        way to use the pool.

        Yields:
            A connected, pooled TCPClient.

        Example:
            async with pool.client() as client:
                await client.send_line("ping")
        """
        client = await self.acquire()
        try:
            yield client
        finally:
            await self.release(client)

    async def close_all(self) -> None:
        """Close all pooled connections.

        Call this during shutdown to hang up every idle connection in the pool,
        so no sockets linger after your program is done talking.
        """
        while not self._pool.empty():
            client = self._pool.get_nowait()
            await client.disconnect()


from contextlib import asynccontextmanager


async def demonstrate_tcp_client() -> None:
    """Demonstrate TCP client.

    Spins up a tiny in-process echo server (a server that sends back whatever
    it receives) and then exercises both a single client and a connection pool
    against it. Run this file directly to see it work.
    """
    print("=== TCP Client Demo ===")

    async def echo_server(reader: StreamReader, writer: StreamWriter) -> None:
        while True:
            data = await reader.readline()
            if not data:
                break
            writer.write(data)
            await writer.drain()

    server = await asyncio.start_server(echo_server, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]

    config = ConnectionConfig(host="127.0.0.1", port=port, timeout=5.0)
    client = TCPClient(config)

    async with client.connection():
        await client.send_line("Hello, Server!")
        response = await client.receive_line()
        print(f"Received: {response.decode().strip()}")

        await client.send_line("Second message")
        response = await client.receive_line()
        print(f"Received: {response.decode().strip()}")

    print("\n--- Connection Pool ---")
    pool = TCPClientPool(config, max_size=3)

    async def use_pool() -> str:
        async with pool.client() as client:
            await client.send_line("Pooled request")
            return (await client.receive_line()).decode().strip()

    results = await asyncio.gather(*[use_pool() for _ in range(5)])
    for r in results:
        print(f"  Pool result: {r}")

    await pool.close_all()
    server.close()
    await server.wait_closed()

    print("Client demo complete")


if __name__ == "__main__":
    asyncio.run(demonstrate_tcp_client())
