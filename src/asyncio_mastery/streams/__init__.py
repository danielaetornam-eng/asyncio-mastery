"""
Streams package provides high-level async networking with readers and writers.

This subpackage teaches the asyncio streams API. The streams API is the
high-level way to do network I/O.

A stream connection has two ends. On your end you get:
- StreamReader: Receives incoming bytes.
- StreamWriter: Sends outgoing bytes. The drain method applies backpressure.

This package provides:
- TCPClient: Connects to a server with timeouts and socket tuning.
- TCPClientPool: Reuses connections.
- TCPServer: Listens for connections and runs a handler per connection.
- SSLContextManager: Wraps connections in TLS encryption.

Module map:
1. stream_reader.py: Reading operations.
2. stream_writer.py: Writing operations.
3. tcp_client.py: Client connections and pooling.
4. tcp_server.py: Server connections and shutdown.
5. ssl_context.py: TLS encryption and certificates.

Minimal example:
    import asyncio
    from asyncio_mastery.streams import TCPServer, TCPClient, ConnectionConfig

    async def main() -> None:
        server = TCPServer("127.0.0.1", 0)
        await server.start()

        client = TCPClient(ConnectionConfig("127.0.0.1", server.port))
        async with client.connection():
            await client.send_line("hello")
            print(await client.receive_line())

        await server.stop()

    asyncio.run(main()
"""

from asyncio_mastery.streams.ssl_context import SSLContextManager, demonstrate_ssl
from asyncio_mastery.streams.stream_reader import AsyncStreamReader, demonstrate_stream_reader
from asyncio_mastery.streams.stream_writer import AsyncStreamWriter, demonstrate_stream_writer
from asyncio_mastery.streams.tcp_client import TCPClient, demonstrate_tcp_client
from asyncio_mastery.streams.tcp_server import TCPServer, demonstrate_tcp_server

__all__ = [
    "AsyncStreamReader",
    "AsyncStreamWriter",
    "SSLContextManager",
    "TCPClient",
    "TCPServer",
    "demonstrate_ssl",
    "demonstrate_stream_reader",
    "demonstrate_stream_writer",
    "demonstrate_tcp_client",
    "demonstrate_tcp_server",
]
