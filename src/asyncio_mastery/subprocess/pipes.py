"""
Pipes. Advanced pipe management for subprocesses.

A pipe is a one-way garden hose for bytes connecting two programs.
When you launch a subprocess with stdout=asyncio.subprocess.PIPE,
the OS builds a hose. Whatever the child prints flows through it.
Your asyncio program holds the other end as a StreamReader — the
very same reader you met in the streams package. Similarly
stdin=PIPE gives you a StreamWriter hose into the child.

Real-world analogies used in this module:

* piping: connecting the out-hose of one program to the in-hose of
  another. Like chaining garden hoses. Exactly what the shell pipe
  operator does in echo hi | tr a-z A-Z.
* tee: a Y-splitter on the hose. One stream of water in. Identical
  streams out to several buckets. Named after the shell tee command.
* buffering: collecting water in a bucket and only pouring when the
  bucket is full. Fewer, bigger pours (writes) instead of many tiny
  ones.

Backpressure, again:

Every hose has limited capacity (the pipe buffer plus the reader
limit, by default 64 KiB). If you write faster than the receiver
reads, await drain() pauses you until there is room. A waiter telling
the kitchen to slow down. All helpers in this module drain after each
chunk. So fast producers can never blow up a slow consumer memory.
Conversely, EOF (empty b"" from a read) means the other end closed
their hose. This is the universal loop-termination signal.

When to use this module:

When simple communicate() (read everything at the end) is not enough:
streaming huge outputs chunk-by-chunk, processing lines as they are
produced, duplicating output to several sinks, or smoothing bursty
writes through a buffer.

Minimal example:

import asyncio
from asyncio_mastery.subprocess.pipes import PipeManager

async def main() -> None:
    proc = await asyncio.create_subprocess_exec(
        "cat", stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE
    )
    proc.stdin.write(b"hello pipe\n")
    await proc.stdin.drain()
    proc.stdin.close()              # EOF: tell cat input done
    data = await PipeManager.read_all(proc.stdout)
    print(data)                     # b"hello pipe\\n"
    await proc.wait()

asyncio.run(main())
"""

import asyncio
from asyncio import StreamReader, StreamWriter
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable


class PipeManager:
    """Manage pipes between processes.

    A toolbox of static helpers for moving bytes between StreamReaders
    and StreamWriters (the ends of your subprocess hoses). All
    helpers share the same core loop pattern:

        chunk = await source.read(n)   # wait for data (other tasks run!)
        if not chunk: break            # b"" = EOF = the other end hung up
        dest.write(chunk); await dest.drain()   # send plus respect backpressure

    Because everything is static and stateless, these helpers compose
    freely with asyncio.gather to build pipelines.
    """

    def __init__(self) -> None:
        """Create a PipeManager (kept for symmetry; all helpers are static)."""
        self._pipes: list[tuple[StreamReader, StreamWriter]] = []

    @staticmethod
    async def pipe_data(
        source: StreamReader,
        dest: StreamWriter,
        *,
        chunk_size: int = 8192,
        transform: Callable[[bytes], bytes] | None = None,
    ) -> int:
        """Pipe data from source to dest.

        Copies bytes from one hose into another. Optionally transforms
        each chunk on the way (uppercasing, filtering, encrypting).
        Ends when the source signals EOF.

        Args:
            source: Where bytes come from (e.g. a child stdout).
            dest: Where bytes go (e.g. another child stdin).
            chunk_size: Max bytes moved per iteration. Larger = fewer
                event-loop round trips. Smaller = finer-grained flow
                control.
            transform: Optional function applied to each chunk before
                writing.

        Returns:
            Total number of bytes transferred. Handy for progress
            logs.

        Note:
            await dest.drain() after every chunk applies backpressure.
            A slow destination makes this loop slow down instead of
            piling unsent bytes into memory.
        """
        total = 0
        while True:
            chunk = await source.read(chunk_size)
            if not chunk:
                # EOF: the source closed its end of the hose.
                break
            if transform:
                chunk = transform(chunk)
            dest.write(chunk)
            await dest.drain()
            total += len(chunk)
        return total

    @staticmethod
    async def tee(source: StreamReader, *destinations: StreamWriter, chunk_size: int = 8192) -> int:
        """Copy source to multiple destinations.

        A Y-splitter. Every chunk read from the source is written to
        every destination. Example: show a child output on screen and
        save it to a log. Like the shell tee command.

        Args:
            source: The reader to copy from.
            *destinations: One or more writers receiving identical
                copies.
            chunk_size: Max bytes read per iteration.

        Returns:
            Total bytes read (each destination receives this many).
        """
        total = 0
        while True:
            chunk = await source.read(chunk_size)
            if not chunk:
                # EOF reached: all destinations have received the full stream.
                break
            for dest in destinations:
                dest.write(chunk)
                # Draining inside the loop keeps the slowest destination
                # from being overrun. Tee inherits all destinations
                # backpressure.
                await dest.drain()
            total += len(chunk)
        return total

    @staticmethod
    async def read_all(reader: StreamReader, chunk_size: int = 8192) -> bytes:
        """Read all data from reader.

        Drains the entire hose into one bytes object. Reads until EOF.

        Args:
            reader: The stream to empty.
            chunk_size: Read granularity. The chunks are concatenated
                for you.

        Returns:
            Everything the stream produced, joined in order.
        """
        chunks = []
        while True:
            chunk = await reader.read(chunk_size)
            if not chunk:
                # EOF: nothing more will ever arrive.
                break
            chunks.append(chunk)
        return b"".join(chunks)

    @staticmethod
    async def read_lines(reader: StreamReader) -> AsyncIterator[bytes]:
        """Read lines from reader.

        Async generator yielding one line at a time as it arrives.
        Perfect for reacting to a child output live (progress bars,
        logs) instead of waiting for the whole process to finish.

        Args:
            reader: The stream to read lines from.

        Yields:
            bytes: Successive lines, trailing newline included.
        """
        while True:
            line = await reader.readline()
            if not line:
                # b"" means EOF: child closed stdout (typically by exiting).
                break
            yield line

    @staticmethod
    async def write_lines(writer: StreamWriter, lines: AsyncIterable[bytes]) -> int:
        """Write lines to writer.

        Consumes an async iterable of byte-lines. Feeds them down the
        hose one by one. Handy for streaming generated input into a
        child (e.g. feeding an async database cursor into sort).

        Args:
            writer: The hose to write into (e.g. a child stdin).
            lines: Async iterable of bytes to send, in order.

        Returns:
            Total bytes written.
        """
        total = 0
        async for line in lines:
            writer.write(line)
            # Drain per line: keeps the child input buffer from
            # overflowing when the producer is fast (backpressure).
            await writer.drain()
            total += len(line)
        return total


class LineProcessor:
    """Process output line by line.

    A manual line-splitter built on raw read() chunks. Shows how line
    framing actually works under the hood. Bytes arrive in
    arbitrary-sized chunks that rarely align with newlines. So we keep
    a carry-over buffer of partial line bytes until a newline finally
    shows up. readline() does this for you internally. This class
    exists to teach the pattern and to allow custom handling.

    Attributes: (private; interact via read_line and process_lines)

    Example:
        proc = await asyncio.create_subprocess_exec(..., stdout=PIPE)
        lp = LineProcessor(proc.stdout)
        await lp.process_lines(lambda line: print("got", line))
    """

    def __init__(self, reader: StreamReader) -> None:
        """Create a processor for one reader.

        Args:
            reader: The byte stream to split into lines (e.g. child
                stdout).
        """
        self._reader = reader
        # Leftover bytes from the previous chunk that did not yet form
        # a complete line. Chunks almost never align with line
        # boundaries!
        self._buffer = bytearray()

    async def read_line(self) -> bytes | None:
        """Read a single line.

        First searches the carry-over buffer for a newline. A previous
        chunk may contain several lines. If none is found, awaits more
        data from the stream. At EOF, any final unterminated line is
        returned as-is.

        Returns:
            One line including its trailing newline, or None at EOF
            with no leftover data. None is the loop-termination
            signal.
        """
        while True:
            # Check buffer for newline
            try:
                idx = self._buffer.index(ord("\n"))
                # Found: hand out everything through the newline
                # inclusive, and keep the rest for the next call.
                line = bytes(self._buffer[: idx + 1])
                self._buffer = self._buffer[idx + 1 :]
                return line
            except ValueError:
                pass

            # Read more data: no complete line yet, so wait for the
            # next chunk.
            chunk = await self._reader.read(8192)
            if not chunk:
                if self._buffer:
                    # EOF with a partial line left: flush it as the
                    # last line.
                    line = bytes(self._buffer)
                    self._buffer.clear()
                    return line
                return None
            self._buffer.extend(chunk)

    async def process_lines(self, handler: Callable[[bytes], Awaitable[None]]) -> None:
        """Process each line with handler.

        Repeatedly reads a line and awaits your handler on it, until
        EOF. Because the handler is awaited, a slow handler naturally
        throttles the reading pace (another form of backpressure).

        Args:
            handler: Async callable invoked once per line, in order.
        """
        while True:
            line = await self.read_line()
            if line is None:
                break
            await handler(line)


class BufferedWriter:
    """Buffered writer with automatic flush.

    Batches small writes into a bucket. Only pours (writes to the real
    StreamWriter) when the bucket reaches buffer_size. Why? Every real
    write plus drain is a trip through the event loop and possibly a
    syscall. Batching amortizes that cost when producing lots of tiny
    pieces. Remember to await flush() (or close()) at the end. Or the
    last partial bucket is never poured!

    Example:
        buf = BufferedWriter(proc.stdin, buffer_size=4096)
        for item in items:
            buf.write(encode(item))
        await buf.close()   # flushes the remainder, then closes with EOF
    """

    def __init__(
        self, writer: StreamWriter, *, buffer_size: int = 8192, auto_flush: bool = True
    ) -> None:
        """Wrap a writer with a byte bucket.

        Args:
            writer: The real stream to eventually write to.
            buffer_size: Flush threshold in bytes. Bigger buckets =
                fewer, larger writes. Smaller = data reaches the child
                sooner.
            auto_flush: If True (default), write() flushes
                automatically once the bucket is full. If False, you
                control all flushing.
        """
        self._writer = writer
        self._buffer = bytearray()
        self._buffer_size = buffer_size
        self._auto_flush = auto_flush

    def write(self, data: bytes) -> int:
        """Write data to buffer.

        Adds bytes to the bucket. With auto-flush on, a full bucket is
        immediately poured into the real writer. Note the sync flush
        only moves bytes into the StreamWriter own buffer. Call
        flush() to guarantee an actual network or pipe push plus
        drain.

        Args:
            data: Bytes to append.

        Returns:
            Number of bytes accepted (always len(data)).
        """
        self._buffer.extend(data)
        if self._auto_flush and len(self._buffer) >= self._buffer_size:
            self._flush_sync()
        return len(data)

    def _flush_sync(self) -> None:
        """Synchronous flush (call from event loop).

        Pours the bucket into the wrapped StreamWriter without
        draining. This is sync because StreamWriter.write() itself is
        sync (buffered). The actually send it half of the flush lives
        in the async flush().
        """
        if self._buffer:
            self._writer.write(bytes(self._buffer))
            self._buffer.clear()

    async def flush(self) -> None:
        """Async flush.

        Pours any pending bytes and then await drain()s. Guarantees
        the data has left your program and the receiver buffer was not
        overrun.
        """
        self._flush_sync()
        await self._writer.drain()

    async def close(self) -> None:
        """Close writer.

        Flushes the final partial bucket and closes the underlying
        stream. Closing sends EOF down the hose. This is how most
        filter-style children (cat, sort) learn that input is finished
        and exit.
        """
        await self.flush()
        self._writer.close()
        await self._writer.wait_closed()


async def demonstrate_pipes() -> None:
    """Demonstrate pipe usage.

    Chains echo into tr exactly like the shell pipe operator.
    Processes another child output line-by-line as it streams. Feeds a
    cat child through a BufferedWriter. Uses Unix-style commands. Run
    directly to try it.
    """
    print("=== Pipe Demo ===")

    # Pipe between two processes: proc1 stdout hose is plugged straight
    # into proc2 stdin. Just like echo ... | tr a-z A-Z in a shell.
    proc1 = await asyncio.create_subprocess_exec(
        "echo", "hello\nworld\nfrom\npipe", stdout=asyncio.subprocess.PIPE
    )
    proc2 = await asyncio.create_subprocess_exec(
        "tr", "a-z", "A-Z", stdin=proc1.stdout, stdout=asyncio.subprocess.PIPE
    )

    stdout, _ = await proc2.communicate()
    print(f"Piped output: {stdout.decode().strip()}")

    await proc1.wait()

    # Tee output to multiple destinations
    print("\n--- Tee Demo ---")
    _proc3 = await asyncio.create_subprocess_exec(
        "echo", "tee test", stdout=asyncio.subprocess.PIPE
    )

    # Create two output pipes
    _reader1, _writer1 = await asyncio.open_connection("127.0.0.1", 0)
    _reader2, _writer2 = await asyncio.open_connection("127.0.0.1", 0)
    # This is complex without actual servers

    # Simple line processing
    print("\n--- Line Processor ---")
    proc4 = await asyncio.create_subprocess_exec(
        "python3", "-c", "import sys; [print(i) for i in range(5)]", stdout=asyncio.subprocess.PIPE
    )

    processor = LineProcessor(proc4.stdout)

    async def handle_line(line: bytes) -> None:
        print(f"  Processed: {line.decode().strip()}")

    await processor.process_lines(handle_line)
    await proc4.wait()

    # Buffered writer
    print("\n--- Buffered Writer ---")
    proc5 = await asyncio.create_subprocess_exec(
        "cat", stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE
    )

    buffered = BufferedWriter(proc5.stdin, buffer_size=10)
    for i in range(5):
        buffered.write(f"line {i}\n".encode())
    await buffered.flush()
    # close() sends EOF to cat stdin. That is the signal that makes
    # cat finish and lets communicate() below return.
    await buffered.close()

    stdout, _ = await proc5.communicate()
    print(f"Buffered output: {stdout.decode().strip()}")


if __name__ == "__main__":
    asyncio.run(demonstrate_pipes())
