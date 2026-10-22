"""
StreamWriter provides async writing to network streams (the speaking end of a
call).

If a StreamReader is your ear in the phone call, a StreamWriter is your mouth.
It is how you put bytes out onto the wire so the program on the other end
receives them, in order.

Real-world analogy: Writing to a network is like handing letters to a postal
worker. write() just drops a letter in the outbox (fast, returns instantly).
But the outbox has limited size. drain() is like waiting for the postal truck
to actually pick up the mail. You pause until the outbox has room again. That
pause is called backpressure. Like a waiter telling the busy kitchen "slow
down, I cannot carry any more plates," it stops you from overwhelming a slow
receiver.

Why write() is sync but drain() is async:
writer.write(data) does not touch the network directly. It copies your bytes
into an in-memory buffer and returns immediately. That is why it is a plain
method, not a coroutine. The operating system sends the buffered bytes in the
background. But if you write faster than the network can carry data, the
buffer grows forever... unless you periodically await writer.drain(). drain()
suspends your coroutine (letting other tasks run) until the buffer has drained
below a safe level. The golden rule: after a write() that must actually arrive
soon, always await drain().

Buffer limits:
Asyncio keeps a high-water mark on the write buffer (default 64 KiB). Below
it, drain() returns immediately. Above it, drain() waits until the buffer
falls below the low-water mark (16 KiB). This hysteresis (two thresholds)
avoids waking up your coroutine for every tiny bit of progress.

What is EOF here:
EOF (End Of File) is how you tell the other side "I am done talking."
write_eof() sends a half-close: your side will send no more data, but you can
still read what the other side sends back. Not all transports support this
(ssl connections do not), so check can_write_eof() first.

When to use this module:
Wrap any asyncio.StreamWriter (from open_connection or a server handler) to
get handy helpers: write_line, write_json, write_message (length-prefixed
framing), and write_all (chunked big writes).

Minimal example:
    import asyncio
    from asyncio_mastery.streams.stream_writer import AsyncStreamWriter

    async def main() -> None:
        raw_reader, raw_writer = await asyncio.open_connection("127.0.0.1", 8888)
        writer = AsyncStreamWriter(raw_writer)

        await writer.write_line("Hello, server!")
        await writer.write_json({"action": "contact"})

        writer.close()
        await writer.wait_closed()

    asyncio.run(main())
"""

import asyncio
from asyncio import StreamWriter
from collections.abc import Sequence
from typing import Any


class AsyncStreamWriter:
    """Enhanced stream writer with additional utilities.

    A beginner-friendly wrapper around asyncio.StreamWriter. The underlying
    rules to remember:
    1. write() is non-blocking: it only copies bytes into a memory buffer.
    2. await drain() is the async commit point: it pauses until the buffer has
       actually been flushed toward the network. This applies backpressure so a
       fast sender cannot drown a slow receiver.
    3. close() + await wait_closed() is the polite way to hang up: close
       starts the shutdown, and waiting ensures all buffered bytes were sent.

    Attributes:
        writer: The wrapped underlying asyncio.StreamWriter.
        transport: The low-level transport object (an asyncio abstraction over
            the raw socket). Rarely needed directly, but useful for
            get_extra_info(...) such as the peer's address.

    Example:
        writer = AsyncStreamWriter(raw_writer)
        await writer.write_line("hello")
        await writer.write_message(b"binary payload")
        writer.close()
        await writer.wait_closed()
    """

    def __init__(self, writer: StreamWriter) -> None:
        """Wrap an existing asyncio StreamWriter.

        Args:
            writer: The low-level writer from asyncio.open_connection() or
                from a server connection handler.
        """
        self._writer = writer

    @property
    def writer(self) -> StreamWriter:
        """Return the wrapped underlying asyncio.StreamWriter.

        Returns:
            The raw writer, for any feature this wrapper does not expose.
        """
        return self._writer

    @property
    def transport(self) -> asyncio.Transport | None:
        """Return the low-level transport (the object managing the socket).

        The transport is asyncio's internal layer between the StreamWriter and
        the operating system's socket. You mostly need it for
        transport.get_extra_info("peername") (who is on the other end) or
        "socket" (the raw socket object, to tweak options).

        Returns:
            The asyncio.Transport, or None if not available.
        """
        return self._writer.transport

    def write(self, data: bytes) -> None:
        """Write bytes to stream (buffered).

        Important: this does not immediately send anything over the network.
        It appends data to an in-memory buffer and returns instantly. That is
        why it is a plain method and not a coroutine. Pair it with await
        drain() when you need the data to actually go out now.

        Args:
            data: The bytes to send. Must be bytes. Encode strings first with
                .encode().
        """
        self._writer.write(data)

    def writelines(self, data: Sequence[bytes]) -> None:
        """Write sequence of bytes.

        Convenience for writing many byte chunks in one call. Equivalent to
        calling write() on each element, but slightly faster because asyncio
        can hand the whole batch to the OS at once. Still buffered: follow
        with await drain() to flush.

        Args:
            data: An iterable of bytes objects, sent in order, with no
                separators added between them.
        """
        self._writer.writelines(data)

    async def drain(self) -> None:
        """Flush write buffer.

        The heart of backpressure. Behavior:
        - If the buffered (not-yet-sent) data is below the high-water mark,
          this returns immediately.
        - If the buffer has overflowed (you are producing faster than the
          network can send), this coroutine suspends until the OS has sent
          enough that the buffer falls below the low-water mark.

        This is like a waiter telling the kitchen "slow down until I clear the
        counter." While suspended here, the event loop runs your other tasks,
        so draining costs your program nothing. Call it after every logical
        message you write.
        """
        await self._writer.drain()

    def close(self) -> None:
        """Close the stream.

        Begins an orderly shutdown: no more writes are accepted, any buffered
        bytes are flushed out, and then the connection is closed. As with
        write(), this only starts the process. Always follow with await
        wait_closed() to be sure the flushed bytes really left and the socket
        was released.
        """
        self._writer.close()

    async def wait_closed(self) -> None:
        """Wait until stream is closed.

        Suspends until the close initiated by close() has fully completed:
        buffered data flushed, connection terminated, resources freed. Without
        awaiting this, your program may exit (or reuse resources) before the
        last bytes were actually delivered.
        """
        await self._writer.wait_closed()

    def is_closing(self) -> bool:
        """Check whether the stream is closing (or already closed).

        Returns:
            True once close() has been called or the connection failed. Never
            write to a closing writer. The bytes would be silently dropped.
        """
        return self._writer.is_closing()

    def can_write_eof(self) -> bool:
        """Check whether this transport supports half-closing (write_eof).

        Half-close means: "I am done sending, but I am still listening." Plain
        TCP supports it. TLS/SSL transports do not (closing is always full
        there). Always call this before write_eof() to be safe.

        Returns:
            True if write_eof() may be used on this stream.
        """
        return self._writer.can_write_eof()

    def write_eof(self) -> None:
        """Send EOF (half-close): announce "no more data from my side."

        The peer will read an empty b"" (EOF) once it has consumed everything
        you sent. Your side can still read. Raises if the transport does not
        support it. Check can_write_eof() first.
        """
        self._writer.write_eof()

    async def write_line(self, line: str, encoding: str = "utf-8") -> None:
        """Write a line with newline.

        Perfect for line-oriented protocols. Encodes the string, appends \\n,
        writes it, and drains immediately so the line goes out right away.

        Args:
            line: The text to send (without a trailing newline. It is added
                for you).
            encoding: Text encoding used to turn the string into bytes.
                "utf-8" is the safe modern default.

        Example:
            await writer.write_line("USER alice")
        """
        self.write((line + "\n").encode(encoding))
        await self.drain()

    async def write_json(self, obj: Any, encoding: str = "utf-8") -> None:
        """Write JSON object.

        Serializes a Python object (dict, list, number, ...) to JSON text and
        sends it, draining immediately. Note: no delimiter or length prefix is
        added, so the receiver must know where the JSON ends. Typically by
        closing the connection afterwards (see read_json) or by combining this
        with your own framing.

        Args:
            obj: Any JSON-serializable Python object.
            encoding: Text encoding for the JSON payload.

        Raises:
            TypeError: If obj contains values JSON cannot represent (e.g.
                sets, custom objects).
        """
        import json

        data = json.dumps(obj).encode(encoding)
        self.write(data)
        await self.drain()

    async def write_message(self, data: bytes, length_prefix: int = 4) -> None:
        """Write length-prefixed message.

        The sending half of length-prefixed framing. The standard cure for
        TCP's lack of message boundaries. Wire format:
            [ length_prefix bytes: body length in big-endian ][ body bytes ]

        The receiver can then readexactly(length_prefix) to learn the size and
        readexactly(length) to get the body. Pair with
        AsyncStreamReader.read_message, which uses the same format.

        Args:
            data: The message body to send.
            length_prefix: Number of bytes used for the length header
                (default 4 -> max message size ~4 GB). Must match the reader.

        Example:
            await writer.write_message(b"hello")
        """
        length = len(data).to_bytes(length_prefix, "big")
        self.write(length + data)
        await self.drain()

    async def write_all(self, data: bytes, chunk_size: int = 65536) -> None:
        """Write large data in chunks.

        Sends a big payload piece by piece, draining after every chunk. Why
        not one giant write()? Because the unsent remainder would sit in memory
        as one huge buffer. Draining between chunks keeps memory bounded and
        honors backpressure. You only generate the next chunk once the network
        has caught up with the previous ones.

        Args:
            data: The complete payload to send.
            chunk_size: Max bytes sent per drain cycle (64 KiB default aligns
                with asyncio's own buffer sizes).
        """
        for i in range(0, len(data), chunk_size):
            self.write(data[i : i + chunk_size])
            await self.drain()


async def demonstrate_stream_writer() -> None:
    """Demonstrate StreamWriter usage.

    Uses a socket.socketpair() -- two sockets connected directly to each other
    inside your own machine -- so no server or network is needed to see the
    writer in action. Run this file directly to try it.
    """
    print("=== StreamWriter Demo ===")

    import socket

    sock1, sock2 = socket.socketpair()

    loop = asyncio.get_event_loop()
    _reader1, writer1 = await loop.connect_accepted_socket(asyncio.StreamReaderProtocol, sock1)

    reader2 = asyncio.StreamReader()
    protocol2 = asyncio.StreamReaderProtocol(reader2)
    transport2, _ = await loop.connect_accepted_socket(lambda: protocol2, sock2)

    async_writer = AsyncStreamWriter(writer1)

    print("\nWriting data...")
    await async_writer.write_line("Hello, World!")
    await async_writer.write_line("Second line")
    await async_writer.write_json({"key": "value", "numbers": [1, 2, 3]})

    line1 = await reader2.readline()
    line2 = await reader2.readline()
    json_data = await reader2.read()

    print(f"Read line 1: {line1.decode().strip()}")
    print(f"Read line 2: {line2.decode().strip()}")
    print(f"Read JSON: {json_data.decode()}")

    print("\nMessage framing:")
    await async_writer.write_message(b"Message 1")
    await async_writer.write_message(b"Message 2 is longer")

    await asyncio.sleep(0.01)

    writer1.close()
    await writer1.wait_closed()
    transport2.close()

    print("\nStreamWriter methods:")
    print("  write(data) - write bytes (buffered)")
    print("  writelines(seq) - write sequence of bytes")
    print("  drain() - flush write buffer")
    print("  close() - close stream")
    print("  wait_closed() - wait for close")
    print("  write_line(str) - write line with newline")
    print("  write_json(obj) - write JSON")
    print("  write_message(bytes) - write length-prefixed")
    print("  write_all(bytes) - write large data in chunks")


if __name__ == "__main__":
    asyncio.run(demonstrate_stream_writer())
