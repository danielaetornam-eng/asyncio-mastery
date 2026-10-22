"""
Advanced Lock Primitives

Enhanced synchronization primitives beyond standard asyncio locks.
"""

import asyncio
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Any, TypeVar

T = TypeVar("T")


class Lock:
    """Enhanced asyncio Lock with additional features."""

    def __init__(self, *, name: str | None = None, fair: bool = False) -> None:
        self._lock = asyncio.Lock()
        self._name = name or f"Lock-{id(self)}"
        self._fair = fair
        self._waiters: deque[asyncio.Future] = deque()
        self._holder: asyncio.Task | None = None
        self._acquire_count = 0
        self._lock_stats = {"acquisitions": 0, "contentions": 0, "total_wait_time": 0.0}

    @property
    def name(self) -> str:
        return self._name

    @property
    def locked(self) -> bool:
        return self._lock.locked()

    @property
    def holder(self) -> asyncio.Task | None:
        return self._holder

    def stats(self) -> dict[str, Any]:
        return self._lock_stats.copy()

    async def acquire(self, timeout: float | None = None) -> bool:
        start_time = time.monotonic()
        try:
            if timeout is not None:
                await asyncio.wait_for(self._lock.acquire(), timeout=timeout)
            else:
                await self._lock.acquire()

            self._holder = asyncio.current_task()
            self._acquire_count += 1
            self._lock_stats["acquisitions"] += 1
            self._lock_stats["total_wait_time"] += time.monotonic() - start_time
            return True
        except TimeoutError:
            self._lock_stats["contentions"] += 1
            return False

    def release(self) -> None:
        if self._holder != asyncio.current_task():
            raise RuntimeError("Lock not held by current task")
        self._holder = None
        self._lock.release()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()

    @asynccontextmanager
    async def acquire_timeout(self, timeout: float):
        """Context manager with timeout."""
        acquired = await self.acquire(timeout)
        try:
            yield acquired
        finally:
            if acquired:
                self.release()

    def locked_by_current_task(self) -> bool:
        return self._holder == asyncio.current_task()


class RLock:
    """Reentrant async lock."""

    def __init__(self, *, name: str | None = None) -> None:
        self._lock = asyncio.Lock()
        self._name = name or f"RLock-{id(self)}"
        self._owner: asyncio.Task | None = None
        self._count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def locked(self) -> bool:
        return self._count > 0

    async def acquire(self, timeout: float | None = None) -> bool:
        current = asyncio.current_task()
        if self._owner == current:
            self._count += 1
            return True

        try:
            if timeout is not None:
                await asyncio.wait_for(self._lock.acquire(), timeout=timeout)
            else:
                await self._lock.acquire()
        except TimeoutError:
            return False

        self._owner = current
        self._count = 1
        return True

    def release(self) -> None:
        current = asyncio.current_task()
        if self._owner != current:
            raise RuntimeError("Lock not held by current task")

        self._count -= 1
        if self._count == 0:
            self._owner = None
            self._lock.release()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()


class Semaphore:
    """Enhanced Semaphore with monitoring."""

    def __init__(self, value: int = 1, *, name: str | None = None) -> None:
        if value < 0:
            raise ValueError("Semaphore value must be >= 0")
        self._semaphore = asyncio.Semaphore(value)
        self._name = name or f"Semaphore-{id(self)}"
        self._initial_value = value
        self._waiters_count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def value(self) -> int:
        return self._semaphore._value

    @property
    def initial_value(self) -> int:
        return self._initial_value

    async def acquire(self, timeout: float | None = None) -> bool:
        self._waiters_count += 1
        try:
            if timeout is not None:
                await asyncio.wait_for(self._semaphore.acquire(), timeout=timeout)
            else:
                await self._semaphore.acquire()
            return True
        except TimeoutError:
            return False
        finally:
            self._waiters_count -= 1

    def release(self) -> None:
        self._semaphore.release()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()


class BoundedSemaphore(Semaphore):
    """Bounded Semaphore that prevents over-release."""

    def __init__(self, value: int = 1, *, name: str | None = None) -> None:
        super().__init__(value, name=name)
        self._bound = value

    def release(self) -> None:
        if self.value >= self._bound:
            raise ValueError("Semaphore released too many times")
        super().release()


class Condition:
    """Enhanced Condition variable with broadcasting."""

    def __init__(self, lock: asyncio.Lock | None = None, *, name: str | None = None) -> None:
        self._lock = lock or asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)
        self._name = name or f"Condition-{id(self)}"
        self._waiters: set[asyncio.Task] = set()

    @property
    def name(self) -> str:
        return self._name

    @property
    def lock(self) -> asyncio.Lock:
        return self._lock

    async def acquire(self) -> None:
        await self._lock.acquire()

    def release(self) -> None:
        self._lock.release()

    async def wait(self, timeout: float | None = None) -> bool:
        self._waiters.add(asyncio.current_task())
        try:
            if timeout is not None:
                await asyncio.wait_for(self._condition.wait(), timeout=timeout)
            else:
                await self._condition.wait()
            return True
        except TimeoutError:
            return False
        finally:
            self._waiters.discard(asyncio.current_task())

    def notify(self, n: int = 1) -> None:
        self._condition.notify(n)

    def notify_all(self) -> None:
        self._condition.notify_all()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()


class Event:
    """Enhanced Event with waiters tracking."""

    def __init__(self, *, name: str | None = None) -> None:
        self._event = asyncio.Event()
        self._name = name or f"Event-{id(self)}"
        self._waiters: set[asyncio.Task] = set()
        self._set_count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_set(self) -> bool:
        return self._event.is_set()

    def set(self) -> None:
        self._set_count += 1
        self._event.set()

    def clear(self) -> None:
        self._event.clear()

    async def wait(self, timeout: float | None = None) -> bool:
        self._waiters.add(asyncio.current_task())
        try:
            if timeout is not None:
                return await asyncio.wait_for(self._event.wait(), timeout=timeout)
            await self._event.wait()
            return True
        except TimeoutError:
            return False
        finally:
            self._waiters.discard(asyncio.current_task())

    def waiters_count(self) -> int:
        return len(self._waiters)


class Barrier:
    """Enhanced Barrier with phase tracking."""

    def __init__(self, parties: int, *, name: str | None = None) -> None:
        if parties <= 0:
            raise ValueError("parties must be > 0")
        self._barrier = asyncio.Barrier(parties)
        self._name = name or f"Barrier-{id(self)}"
        self._parties = parties
        self._phase = 0
        self._waiting = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def parties(self) -> int:
        return self._parties

    @property
    def phase(self) -> int:
        return self._phase

    @property
    def n_waiting(self) -> int:
        return self._waiting

    async def wait(self, timeout: float | None = None) -> int:
        self._waiting += 1
        try:
            if timeout is not None:
                result = await asyncio.wait_for(self._barrier.wait(), timeout=timeout)
            else:
                result = await self._barrier.wait()
            return result
        except TimeoutError:
            raise
        finally:
            self._waiting -= 1
            if self._waiting == 0:
                self._phase += 1

    async def abort(self) -> None:
        await self._barrier.abort()

    async def reset(self) -> None:
        await self._barrier.reset()
        self._phase = 0


class Mutex:
    """Mutex with priority inheritance support."""

    def __init__(self, *, name: str | None = None) -> None:
        self._lock = asyncio.Lock()
        self._name = name or f"Mutex-{id(self)}"
        self._owner: asyncio.Task | None = None
        self._priority_boost = 0

    @property
    def name(self) -> str:
        return self._name

    async def acquire(self, timeout: float | None = None, priority: int = 0) -> bool:
        try:
            if timeout is not None:
                await asyncio.wait_for(self._lock.acquire(), timeout=timeout)
            else:
                await self._lock.acquire()
        except TimeoutError:
            return False

        self._owner = asyncio.current_task()
        self._priority_boost = priority
        return True

    def release(self) -> None:
        if self._owner != asyncio.current_task():
            raise RuntimeError("Mutex not held by current task")
        self._owner = None
        self._priority_boost = 0
        self._lock.release()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()


class RWLock:
    """Reader-Writer Lock for concurrent reads, exclusive writes."""

    def __init__(self, *, name: str | None = None, prefer_writer: bool = False) -> None:
        self._name = name or f"RWLock-{id(self)}"
        self._read_ready = asyncio.Condition(asyncio.Lock())
        self._readers = 0
        self._writers_waiting = 0
        self._writer_active = False
        self._prefer_writer = prefer_writer
        self._read_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return self._name

    async def acquire_read(self, timeout: float | None = None) -> bool:
        async with self._read_ready:
            while self._writer_active or (self._prefer_writer and self._writers_waiting > 0):
                try:
                    if timeout is not None:
                        await asyncio.wait_for(self._read_ready.wait(), timeout=timeout)
                    else:
                        await self._read_ready.wait()
                except TimeoutError:
                    return False
            self._readers += 1
        return True

    async def release_read(self) -> None:
        async with self._read_ready:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()

    async def acquire_write(self, timeout: float | None = None) -> bool:
        self._writers_waiting += 1
        try:
            async with self._write_lock, self._read_ready:
                while self._readers > 0 or self._writer_active:
                    try:
                        if timeout is not None:
                            await asyncio.wait_for(self._read_ready.wait(), timeout=timeout)
                        else:
                            await self._read_ready.wait()
                    except TimeoutError:
                        return False
                self._writer_active = True
            return True
        finally:
            self._writers_waiting -= 1

    async def release_write(self) -> None:
        self._writer_active = False
        self._write_lock.release()
        async with self._read_ready:
            self._read_ready.notify_all()

    @asynccontextmanager
    async def read_lock(self, timeout: float | None = None):
        acquired = await self.acquire_read(timeout)
        try:
            yield acquired
        finally:
            if acquired:
                await self.release_read()

    @asynccontextmanager
    async def write_lock(self, timeout: float | None = None):
        acquired = await self.acquire_write(timeout)
        try:
            yield acquired
        finally:
            if acquired:
                await self.release_write()


class SpinLock:
    """Spin lock for very short critical sections."""

    def __init__(self, *, name: str | None = None, max_spins: int = 1000) -> None:
        self._name = name or f"SpinLock-{id(self)}"
        self._locked = False
        self._max_spins = max_spins
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def locked(self) -> bool:
        return self._locked

    async def acquire(self, timeout: float | None = None) -> bool:
        start = time.monotonic()
        spins = 0

        while self._locked:
            if spins < self._max_spins:
                spins += 1
                await asyncio.sleep(0)
            else:
                # Fall back to regular lock
                async with self._lock:
                    while self._locked:
                        if timeout is not None:
                            remaining = timeout - (time.monotonic() - start)
                            if remaining <= 0:
                                return False
                            await asyncio.wait_for(self._lock.acquire(), timeout=remaining)
                        else:
                            await self._lock.acquire()
                return True

            if timeout is not None and (time.monotonic() - start) >= timeout:
                return False

        self._locked = True
        return True

    def release(self) -> None:
        self._locked = False

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()


@asynccontextmanager
async def lock_ctx(lock: Lock | RLock | Semaphore | BoundedSemaphore | Mutex | RWLock):
    """Generic lock context manager."""
    await lock.acquire()
    try:
        yield
    finally:
        lock.release()


class LockManager:
    """Manage multiple named locks."""

    def __init__(self) -> None:
        self._locks: dict[str, Lock] = {}
        self._lock = asyncio.Lock()

    async def get_lock(self, name: str) -> Lock:
        async with self._lock:
            if name not in self._locks:
                self._locks[name] = Lock(name=name)
            return self._locks[name]

    async def acquire(self, name: str, timeout: float | None = None) -> Lock:
        lock = await self.get_lock(name)
        await lock.acquire(timeout)
        return lock

    async def release(self, name: str) -> bool:
        async with self._lock:
            lock = self._locks.get(name)
            if lock and lock.locked:
                lock.release()
                return True
            return False

    async def release_all(self) -> None:
        async with self._lock:
            for lock in self._locks.values():
                if lock.locked:
                    lock.release()
