#!/usr/bin/env python3
"""
TCP Echo Server Example - Demonstrates streams and networking.

Run with: python -m examples.tcp_echo_server
"""

import asyncio
import sys

sys.path.insert(0, "src")

from asyncio_mastery.patterns import RateLimiter
from asyncio_mastery.streams import TCPClient, TCPServer


async def run_echo_server() -> None:
    """Run a simple echo server."""
    print("=" * 60)
    print("TCP ECHO SERVER")
    print("=" * 60)

    # Server with rate limiting
    rate_limiter = RateLimiter(rate=100, burst=10)

    async def echo_handler(reader, writer) -> None:
        async with rate_limiter:
            addr = writer.get_extra_info("peername")
            print(f"Connection from {addr}")

            while True:
                data = await reader.readline()
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

    print(f"Server started on {server.host}:{server.port}")

    # Test client
    print("\n--- Testing Client ---")
    client = TCPClient(host="127.0.0.1", port=server.port)
    await client.connect()

    for msg in ["Hello", "World", "Test", "quit"]:
        await client.send_line(msg)
        if msg != "quit":
            response = await client.receive_line()
            print(f"Server: {response.decode().strip()}")

    await client.disconnect()
    await server.stop()
    print("Server stopped")


async def run_chat_server() -> None:
    """Run a chat server."""
    print("\n" + "=" * 60)
    print("CHAT SERVER")
    print("=" * 60)

    from asyncio_mastery.streams import ChatServer

    server = ChatServer("127.0.0.1", 0)
    await server.start()

    print(f"Chat server started on {server.host}:{server.port}")

    # Simulate multiple clients
    async def chat_client(name: str) -> None:
        client = TCPClient(host="127.0.0.1", port=server.port)
        await client.connect()
        await client.send_line(name)

        # Send a few messages
        for i in range(3):
            await client.send_line(f"Message {i} from {name}")
            response = await client.receive_line()
            print(f"  {name} received: {response.decode().strip()}")
            await asyncio.sleep(0.1)

        await client.disconnect()

    await asyncio.gather(chat_client("Alice"), chat_client("Bob"), chat_client("Charlie"))

    await server.stop()
    print("Chat server stopped")


async def main() -> None:
    await run_echo_server()
    await run_chat_server()


if __name__ == "__main__":
    asyncio.run(main())
