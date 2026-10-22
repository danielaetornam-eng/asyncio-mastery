"""
Locks - Mutual exclusion for async code.

WHAT IS A LOCK
A lock (short for "mutual exclusion lock", or "mutex") guarantees that only
ONE task at a time can enter a protected section of code. The protected
section is called a "critical section" — code that must not be run by two
tasks simultaneously because it touches shared data.

REAL-WORLD ANALOGY: THE BATHROOM KEY
Imagine a coffee shop with a single bathroom and a key hanging by the
counter. Only one person can hold the key at a time:

- To use the bathroom, you must take the key ("acquire" the lock).
- While you have the key, everyone else who wants to go must WAIT in line.
- When you are done, you hang the key back up ("release" the lock), and the
  next person in line gets it.

That is exactly what a lock does for code: async with lock: takes the
key, runs the critical section, and hands the key back — even if an
exception occurs.

WHY DO I NEED THIS IN ASYNCIO? IT IS SINGLE-THREADED!
True, asyncio runs on ONE thread. But every await is a point where your
task PAUSES and the event loop switches to another task. Consider this
innocent-looking code WITHOUT a lock::

    current = counter
    await asyncio.sleep(0.01)      # <-- task yields control HERE
    counter = current + 1

If two tasks both read counter = 5 before either one writes, they both
compute 5 + 1 and write 6 — one increment is LOST. This is called a
"race condition": the result depends on the timing of which task runs when.
Race conditions are among the hardest bugs to find because they depend on
timing and do not happen consistently.

With a lock, the read-modify-write sequence happens atomically (no other
task can interleave into the middle), so 100 tasks doing 100 increments
reliably produce 100.

TERMS DEFINED (jargon buster):
- Race condition: a bug where the outcome depends on unpredictable timing
  of concurrent operations.
- Critical section: a stretch of code that must run without interruption
  by other tasks touching the same data.
- Mutual exclusion: the property that at most one task can hold the lock.
- Contention: when many tasks want the same lock and must queue up.

MINIMAL EXAMPLE
::

    import asyncio
    from asyncio_mastery.synchronization import AsyncLock

    lock = AsyncLock()
    counter = 0

    async def increment():
        global counter
        async with lock:              # take the bathroom key
            current = counter
            await asyncio.sleep(0)    # safe to yield inside the lock
            counter = current + 1     # nobody else can interleave here

    asyncio.run(asyncio.gather(*[increment() for _ in range(100)]))
    assert counter == 100             # without the lock this would fail!

Run this file directly to see lock patterns in action::

    python -m asyncio_mastery.synchronization.locks
"""

import asyncio
from asyncio import Lock
from contextlib import asynccontextmanager


class AsyncLock:
    """An asyncio lock with extra instrumentation for learning.

    Wraps asyncio.Lock and adds a human-readable name, tracking of
    which task currently holds the lock, and a counter of total acquisitions.
    These extras make debugging ("who holds my lock?!") and teaching easier
    than with the bare stdlib primitive.

    Attributes:
        name: Human-readable name of the lock (useful in logs).
        locked: True while some task holds the lock.
        holder: The asyncio.Task currently holding the lock,
            or None if free.

    Example:
        ::

            lock = AsyncLock(name="bank-account")

            async with lock:
                # Only one task at a time executes this block.
                balance += 100
    """

    def __init__(self, *, name: str | None = None) -> None:
        """Create a new AsyncLock.

        Args:
            name: Optional human-readable name for debugging and logging.
                Defaults to the generic string "lock".
        """
        self._lock = Lock()
        self._name = name or "lock"
        self._holder: asyncio.Task | None = None
        self._acquire_count = 0

    @property
    def name(self) -> str:
        """The human-readable name of this lock."""
        return self._name

    @property
    def locked(self) -> bool:
        """True if some task currently holds the lock, False if it is free."""
        return self._lock.locked()

    @property
    def holder(self) -> asyncio.Task | None:
        """The task currently holding the lock, or None if it is free.

        Handy when debugging deadlock-like stalls: if a lock stays locked
        forever, inspect lock.holder to see which task forgot to release.
        """
        return self._holder

    async def acquire(self) -> bool:
        """Acquire the lock, waiting if another task holds it.

        If the lock is free, this returns immediately. If another task holds
        it, this coroutine SUSPENDS at the await below and the event loop
        runs other tasks until the lock is released; waiters are woken in
        roughly first-come-first-served order.

        Prefer async with lock: over calling this directly — the context
        manager guarantees release even on exceptions.

        Returns:
            True once the lock has been acquired.
        """
        # If the lock is held, this await is where we park in the wait queue
        # and hand control back to the event loop.
        await self._lock.acquire()
        # We now own the lock. No other task can get past its own acquire()
        # until release() is called.
        self._holder = asyncio.current_task()
        self._acquire_count += 1
        return True

    def release(self) -> None:
        """Release the lock, waking one waiting task (if any).

        Raises:
            RuntimeError: If the lock is not currently held. Releasing a
                lock you do not hold is a logic bug — this check surfaces it
                loudly instead of silently corrupting state.
        """
        if not self._lock.locked():
            raise RuntimeError("Lock not acquired")
        self._holder = None
        self._lock.release()

    async def __aenter__(self) -> "AsyncLock":
        """Enter async with — acquires the lock (see acquire)."""
        await self.acquire()
        return self

    async def __aexit__(self, *args: object) -> None:
        """Exit async with — releases the lock even if the body raised."""
        # This runs whether the with-block finished normally or exploded with
        # an exception, which is exactly why async with is the safe pattern.
        self.release()


@asynccontextmanager
async def lock_context(lock: AsyncLock) -> None:
    """Acquire lock as an async context manager (building-block version).

    This is functionally the same as async with lock: but demonstrates
    how to build such helpers yourself with @asynccontextmanager:
    everything before yield runs on entry (acquire), everything in the
    finally runs on exit (release) — including on exceptions.

    Args:
        lock: The AsyncLock to hold for the duration of the block.

    Yields:
        None; the body of the async with runs with the lock held.

    Example:
        ::

            async with lock_context(lock):
                shared_value += 1
    """
    await lock.acquire()
    try:
        yield
    finally:
        # 'finally' guarantees the key goes back on the hook even if the
        # code inside the with-block raised an exception.
        lock.release()


async def demonstrate_locks() -> None:
    """Run a narrated tour of common lock usage patterns.

    Demonstrates:
      1. Protecting a shared counter from a race condition.
      2. Acquiring a lock with a timeout (avoid waiting forever).
      3. Observing contention: 10 workers queue up and run one at a time.
      4. Manual acquire()/release() (the low-level pattern).

    Returns:
        None. Prints demonstration output to stdout.
    """
    print("=== Basic Lock ===")
    lock = AsyncLock(name="basic")
    counter = 0

    async def increment() -> None:
        nonlocal counter
        # Without the lock, two tasks could both read 'counter' before either
        # writes, and one increment would be silently lost (a race condition).
        async with lock:
            current = counter
            # This sleep forces a context switch INSIDE the critical section.
            # The lock is what makes this safe: while we sleep, other tasks
            # wanting the lock simply queue up instead of reading a stale
            # counter value.
            await asyncio.sleep(0.01)  # Simulate work
            counter = current + 1

    # Run many concurrent increments
    await asyncio.gather(*[increment() for _ in range(100)])
    print(f"Counter: {counter} (expected 100)")

    print("\n=== Lock with Timeout ===")
    lock2 = AsyncLock(name="timeout")

    async def holder() -> None:
        async with lock2:
            print("Holder acquired lock")
            # Hold the lock long enough that 'waiter' below gives up waiting.
            await asyncio.sleep(0.5)
            print("Holder releasing")

    async def waiter() -> None:
        try:
            # asyncio.timeout cancels the inner block if it runs too long —
            # here, acquire() will be cancelled because the holder is slow.
            async with asyncio.timeout(0.1):
                async with lock2:
                    print("Waiter got lock")
        except TimeoutError:
            print("Waiter timed out!")

    await asyncio.gather(holder(), waiter())

    print("\n=== Lock Contention ===")
    lock3 = AsyncLock(name="contention")
    results = []

    async def worker(id: int) -> None:
        # Ten workers, one lock: they execute this block strictly one at a
        # time, in the order they arrived (FIFO-ish fairness of asyncio.Lock).
        async with lock3:
            results.append(id)
            await asyncio.sleep(0.01)

    await asyncio.gather(*[worker(i) for i in range(10)])
    print(f"Execution order: {results}")

    print("\n=== Manual Acquire/Release ===")
    lock4 = AsyncLock(name="manual")
    # Low-level pattern. WARNING: if an exception happens between acquire()
    # and release(), the lock stays held forever — use async with instead.
    await lock4.acquire()
    print(f"Locked: {lock4.locked()}")
    lock4.release()
    print(f"Locked after release: {lock4.locked()}")


class LockExample:
    """Practical lock example: protecting a shared in-memory store.

    Models a tiny key/value store that many tasks read and write
    concurrently. Every method takes the same lock so that compound
    operations (like read-modify-write in increment) are atomic and
    the _version counter always matches the number of writes.

    Example:
        ::

            store = LockExample()
            await store.write("apples", 1)
            value = await store.increment("apples")   # -> 2
            version = await store.get_version()       # -> 2
    """

    def __init__(self) -> None:
        """Create the store with an empty dict and a version counter."""
        self._lock = AsyncLock(name="resource")
        self._data: dict[str, int] = {}
        self._version = 0

    async def read(self, key: str) -> int | None:
        """Read a value under the lock.

        Args:
            key: The key to look up.

        Returns:
            The stored integer, or None if the key has never been written.
        """
        # Even a plain read takes the lock so it cannot observe a half-written
        # state in the middle of another task's compound operation.
        async with self._lock:
            return self._data.get(key)

    async def write(self, key: str, value: int) -> None:
        """Write a value and bump the version counter, atomically.

        Args:
            key: The key to store under.
            value: The integer value to store.
        """
        async with self._lock:
            self._data[key] = value
            self._version += 1

    async def increment(self, key: str) -> int:
        """Atomically increment a counter-style key.

        This is the classic read-modify-write that NEEDS a lock: between
        reading the current value and writing the new one there is an
        await (simulating processing), which is exactly where a race
        condition would strike without the lock.

        Args:
            key: The key to increment (starts at 0 if missing).

        Returns:
            The new (incremented) value.
        """
        async with self._lock:
            current = self._data.get(key, 0)
            # Yield point on purpose: with the lock held this is harmless;
            # without it, another task could interleave and lose an update.
            await asyncio.sleep(0.001)  # Simulate processing
            self._data[key] = current + 1
            self._version += 1
            return self._data[key]

    async def get_version(self) -> int:
        """Return how many writes have ever happened.

        Returns:
            The monotonically increasing version number of the store.
        """
        async with self._lock:
            return self._version


if __name__ == "__main__":
    asyncio.run(demonstrate_locks())
