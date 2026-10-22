"""
StreamReader provides async reading from network streams (the listening end of
a call).

A stream is an ongoing flow of bytes between two programs. Think of it like a
phone call with two ends. Bytes you send arrive, in order, on the other end.
A socket is the operating-system phone jack that carries the call. A
StreamReader is asyncio's friendly helper object that lets you listen (read)
from your end of that call.

Real-world analogy: Imagine a conveyor belt delivering parts to your
workstation. You do not control when parts arrive. Instead of standing there
doing nothing (blocking), you tell asyncio "wake me when a part arrives" and
go do other work. That "wake me" is exactly what await reader.read(...) does.

Why reads must await:
Data arrives from the network at times you cannot predict. If your program
waited for data in a plain (blocking) loop, your entire program would freeze
until bytes arrived. By awaiting a read, you hand control back to the event
loop. The event loop can run other tasks while your data is still traveling
over the wire. When data shows up, the event loop resumes your coroutine with
the bytes.

What is EOF:
EOF (End Of File) means the other side hung up and no more data will ever
arrive. Reading methods signal EOF by returning an empty bytes object (b""),
or, for readexactly/readuntil, by raising an exception. Always check for EOF,
or your loop will wait forever.

Buffer limits and backpressure:
Internally, a StreamReader keeps a buffer (a waiting room) for bytes that
arrived faster than your code consumes them. This buffer has a size limit
(default 64 KiB). If the buffer fills up, asyncio tells the operating system
to stop sending more data until you catch up. This is called backpressure. It
prevents a fast sender from exhausting your memory.

When to use this module:
Use these helpers whenever you already have an asyncio.StreamReader (for
example from asyncio.open_connection() or from a server's connection handler)
and you want convenient, higher-level reading: lines, exact byte counts, JSON
messages, or chunked iteration.

Minimal example:
    import asyncio
    from asyncio_mastery.streams.stream_reader import AsyncStreamReader

    async def main() -> None:
        raw_reader, writer = await asyncio.open_connection("example.com", 80)
        reader = AsyncStreamReader(raw_reader)

        writer.write(b"GET / HTTP/1.0\\r\\n\\r\\n")
        await writer.drain()
        async for line in reader.read_lines():
            print(line.decode(errors="replace").rstrip())

        writer.close()
        await writer.wait_closed()

    asyncio.run(main())
"""

import asyncio
from asyncio import StreamReader
from collections.abc import AsyncIterator
from typing import Any


class AsyncStreamReader:
    """Enhanced stream reader with additional utilities.

    A thin, beginner-friendly wrapper around asyncio.StreamReader. Keeps all
    the familiar methods (read, readline, readexactly, readuntil) and adds
    convenient helpers for common patterns: iterating over chunks or lines,
    reading a whole JSON document, and reading length-prefixed messages.

    All reading methods are coroutines. They await until data is available,
    letting the event loop run other tasks meanwhile. This is the whole point
    of asyncio: waiting without blocking.

    Attributes:
        reader: The wrapped underlying asyncio.StreamReader.
        at_eof: True when the other side has closed the connection and every
            buffered byte has been consumed. EOF = End Of File = no more data
            will ever arrive.

    Example:
        reader = AsyncStreamReader(raw_reader)
        line = await reader.readline()
        async for chunk in reader.read_chunks(4096):
            process(chunk)
    """

    def __init__(self, reader: StreamReader) -> None:
        """Wrap an existing asyncio StreamReader.

        Args:
            reader: The low-level reader you got from asyncio.open_connection()
                or from a server connection handler.
        """
        self._reader = reader

    @property
    def reader(self) -> StreamReader:
        """Return the wrapped underlying asyncio.StreamReader.

        Useful when you need a feature this wrapper does not expose. Everything
        the wrapper does is ultimately delegated to this object.

        Returns:
            The raw asyncio.StreamReader passed to the constructor.
        """
        return self._reader

    @property
    def at_eof(self) -> bool:
        """Check whether the stream has reached EOF (End Of File).

        Returns True only when both of these are true: the peer has closed the
        connection (or sent an EOF), and you have already read every byte that
        was buffered. Before that point there may still be unread data.

        Returns:
            True if no more data will ever be available to read.
        """
        return self._reader.at_eof()

    async def read(self, n: int = -1) -> bytes:
        """Read up to n bytes.

        Args:
            n: Maximum number of bytes to return. -1 (the default) means read
                everything until EOF. Convenient but dangerous with an
                untrusted peer, because all the data is held in memory at once.

        Returns:
            The bytes read. May be shorter than n if that is all that has
            arrived so far. Use readexactly if you must have exactly n. An
            empty b"" means EOF: the other side hung up.

        Note:
            This awaits until at least one byte is available (or EOF). A slow
            network does not freeze your program. It just suspends this
            coroutine while other tasks run.
        """
        return await self._reader.read(n)

    async def readline(self) -> bytes:
        """Read one line (ends with newline).

        Returns everything up to and including the next newline character
        (\\n). Great for text protocols like HTTP headers, SMTP, or a simple
        line-based chat.

        Returns:
            One line of data, including the trailing \\n. An empty b"" means
            EOF was reached with no more data.

        Raises:
            ValueError: If the line is longer than the stream's internal buffer
                limit (default 64 KiB). This protects you from a peer sending a
                never-ending line to exhaust your memory.
            asyncio.IncompleteReadError: If EOF arrives mid-line. Partial data
                is available in the exception's partial attribute.

        Note:
            The await is key. The coroutine pauses here (without blocking the
            program) until a full line has arrived or the buffer limit is hit.
        """
        return await self._reader.readline()

    async def readexactly(self, n: int) -> bytes:
        """Read exactly n bytes.

        Unlike read(n), which may return fewer bytes than asked for, this keeps
        waiting until it has collected ALL n bytes. Essential for binary
        protocols where every byte position has meaning.

        Args:
            n: The exact number of bytes required.

        Returns:
            Exactly n bytes.

        Raises:
            asyncio.IncompleteReadError: If EOF arrives before n bytes. The
                bytes that did arrive are stored in the exception's partial
                attribute, so you can inspect or salvage them.
        """
        return await self._reader.readexactly(n)

    async def readuntil(self, separator: bytes) -> bytes:
        """Read until separator found.

        Reads until the given separator byte sequence is found, and returns
        everything including the separator. Think of readline() as
        readuntil(b"\\n") with extra safety checks.

        Args:
            separator: The byte sequence marking the end of the message, e.g.
                b"\\r\\n\\r\\n" for the end of HTTP headers.

        Returns:
            All bytes up to and including the first occurrence of separator.

        Raises:
            asyncio.IncompleteReadError: If EOF arrives before the separator.
                Partial data in the exception's partial attribute.
            asyncio.LimitOverrunError: If the separator was not found within
                the stream's buffer limit. A guard against peers sending
                unbounded data.
        """
        return await self._reader.readuntil(separator)

    async def read_chunks(self, chunk_size: int = 1024) -> AsyncIterator[bytes]:
        """Iterate over chunks.

        An async generator that yields the stream piece by piece until EOF.
        Ideal for large transfers (file downloads, log forwarding) where
        loading everything into memory at once would be wasteful or impossible.

        Args:
            chunk_size: Maximum bytes per yielded chunk. Smaller values give
                you more frequent progress updates. Larger values make fewer
                round-trips through the event loop.

        Yields:
            bytes: Successive chunks of the stream, in arrival order.

        Example:
            async for chunk in reader.read_chunks(65536):
                outfile.write(chunk)
        """
        while not self._reader.at_eof():
            chunk = await self._reader.read(chunk_size)
            if not chunk:
                break
            yield chunk

    async def read_lines(self) -> AsyncIterator[bytes]:
        """Iterate over lines.

        An async generator yielding one line at a time, until EOF. Perfect for
        line-oriented protocols (logs, chat messages, tail -f-style feeds).

        Yields:
            bytes: One line at a time, each including its trailing newline
                (except possibly the very last line if the peer closes without
                a final \\n).

        Example:
            async for line in reader.read_lines():
                print("server said:", line.decode().rstrip())
        """
        while not self._reader.at_eof():
            line = await self._reader.readline()
            if not line:
                break
            yield line

    async def read_json(self) -> Any:
        """Read and parse JSON (assumes single JSON object).

        Reads until EOF and then parses the entire body as one JSON document.
        Only suitable for short-lived connections where the peer sends one JSON
        value and then closes (for example, a one-shot HTTP-like exchange).
        For ongoing conversations, prefer read_message so each message is
        framed by its length.

        Returns:
            The parsed Python object (dict, list, str, int, ...).

        Raises:
            json.JSONDecodeError: If the received bytes are not valid JSON.
            UnicodeDecodeError: If the bytes are not valid UTF-8 text.
        """
        import json

        data = await self._reader.read()
        return json.loads(data.decode())

    async def read_message(self, length_prefix: int = 4) -> bytes:
        """Read length-prefixed message.

        Implements the classic length-prefixed framing pattern. Every message
        starts with a fixed-size header telling you how many bytes the message
        body has, followed by the body itself. This solves the #1 beginner
        problem with TCP: TCP has no built-in message boundaries. It is a
        continuous stream, so one send does not mean one read.

        The protocol used here:
            [ length_prefix bytes: body length, big-endian ][ body bytes ]

        Args:
            length_prefix: How many bytes encode the length (default 4, which
                allows messages up to ~4 GB). Must match what the sender uses.

        Returns:
            The message body, exactly length bytes long.

        Raises:
            asyncio.IncompleteReadError: If EOF arrives in the middle of the
                header or body (i.e. the peer disconnected mid-message).
        """
        length_bytes = await self.readexactly(length_prefix)
        length = int.from_bytes(length_bytes, "big")
        return await self.readexactly(length)


async def demonstrate_stream_reader() -> None:
    """Demonstrate StreamReader usage.

    Builds an in-memory pipe (no network needed!) so you can run this file
    directly (python stream_reader.py) and watch the reader in action.
    """
    print("=== StreamReader Demo ===")

    from asyncio import streams

    reader = StreamReader()
    protocol = streams.StreamReaderProtocol(reader)
    transport, _ = await asyncio.get_event_loop().create_pipe_transport(None, protocol, None)

    print("StreamReader methods:")
    print("  read(n=-1) - read up to n bytes")
    print("  readline() - read one line")
    print("  readexactly(n) - read exactly n bytes")
    print("  readuntil(separator) - read until separator")
    print("  read_chunks(chunk_size) - async iterator of chunks")
    print("  read_lines() - async iterator of lines")
    print("  read_json() - read and parse JSON")
    print("  read_message(length_prefix) - read length-prefixed message")

    import os

    read_fd, write_fd = os.pipe()

    loop = asyncio.get_event_loop()
    reader = StreamReader()
    protocol = streams.StreamReaderProtocol(reader)

    transport, _ = await loop.connect_read_pipe(lambda: protocol, os.fdopen(read_fd, "rb"))

    writer = os.fdopen(write_fd, "wb")
    writer.write(b"line 1\nline 2\nline 3\n")
    writer.flush()

    async_reader = AsyncStreamReader(reader)

    print("\nReading lines:")
    async for line in async_reader.read_lines():
        print(f"  Got: {line.decode().strip()}")

    transport.close()


async def demo_with_tcp() -> None:
    """Demo with actual TCP connection.

    Intentionally empty: a real TCP demo needs a reachable server on the
    network. See tcp_server.py / tcp_client.py for a full working pair.
    """
    pass


if __name__ == "__main__":
    asyncio.run(demonstrate_stream_reader())
