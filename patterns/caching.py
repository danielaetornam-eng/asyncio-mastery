"""
Caching Patterns

Async caching implementations with various eviction policies.
"""

import asyncio
import hashlib
import pickle
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Generic, TypeVar

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


@dataclass
class CacheEntry(Generic[V]):
    """Cache entry with metadata."""

    key: Any
    value: V
    created_at: float = field(default_factory=time.monotonic)
    accessed_at: float = field(default_factory=time.monotonic)
    access_count: int = 0
    ttl: float | None = None
    size: int = 0

    @property
    def expired(self) -> bool:
        if self.ttl is None:
            return False
        return time.monotonic() - self.created_at > self.ttl

    @property
    def age(self) -> float:
        return time.monotonic() - self.created_at

    @property
    def idle_time(self) -> float:
        return time.monotonic() - self.accessed_at

    def touch(self) -> None:
        self.accessed_at = time.monotonic()
        self.access_count += 1


class CacheBackend(ABC, Generic[K, V]):
    """Abstract cache backend."""

    @abstractmethod
    async def get(self, key: K) -> V | None:
        pass

    @abstractmethod
    async def set(self, key: K, value: V, ttl: float | None = None) -> None:
        pass

    @abstractmethod
    async def delete(self, key: K) -> bool:
        pass

    @abstractmethod
    async def clear(self) -> None:
        pass

    @abstractmethod
    async def keys(self) -> list[K]:
        pass


class AsyncCache(Generic[K, V], CacheBackend[K, V]):
    """Base async cache with common functionality."""

    def __init__(
        self, maxsize: int = 128, *, name: str | None = None, default_ttl: float | None = None
    ) -> None:
        self._maxsize = maxsize
        self._name = name or f"Cache-{id(self)}"
        self._default_ttl = default_ttl
        self._cache: dict[K, CacheEntry[V]] = {}
        self._lock = asyncio.Lock()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0, "sets": 0, "deletes": 0}

    @property
    def name(self) -> str:
        return self._name

    @property
    def maxsize(self) -> int:
        return self._maxsize

    @property
    def size(self) -> int:
        return len(self._cache)

    def stats(self) -> dict[str, Any]:
        return {
            **self._stats.copy(),
            "size": self.size,
            "maxsize": self.maxsize,
            "hit_rate": self._stats["hits"] / max(1, self._stats["hits"] + self._stats["misses"]),
        }

    async def get(self, key: K) -> V | None:
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._stats["misses"] += 1
                return None

            if entry.expired:
                del self._cache[key]
                self._stats["misses"] += 1
                return None

            entry.touch()
            self._stats["hits"] += 1
            return entry.value

    async def set(self, key: K, value: V, ttl: float | None = None) -> None:
        async with self._lock:
            # Check if we need to evict
            if len(self._cache) >= self._maxsize and key not in self._cache:
                await self._evict()

            effective_ttl = ttl if ttl is not None else self._default_ttl
            entry = CacheEntry(key=key, value=value, ttl=effective_ttl)
            self._cache[key] = entry
            self._stats["sets"] += 1

    async def delete(self, key: K) -> bool:
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats["deletes"] += 1
                return True
            return False

    async def clear(self) -> None:
        async with self._lock:
            self._cache.clear()

    async def keys(self) -> list[K]:
        async with self._lock:
            return list(self._cache.keys())

    async def _evict(self) -> None:
        """Evict one entry - override in subclasses."""
        if self._cache:
            # Default: remove oldest
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].created_at)
            del self._cache[oldest_key]
            self._stats["evictions"] += 1

    def __contains__(self, key: K) -> bool:
        # Non-async check for compatibility
        entry = self._cache.get(key)
        return entry is not None and not entry.expired


class LRUCache(AsyncCache[K, V]):
    """Least Recently Used cache."""

    def __init__(self, maxsize: int = 128, **kwargs) -> None:
        super().__init__(maxsize, **kwargs)
        self._order = OrderedDict()

    async def get(self, key: K) -> V | None:
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._stats["misses"] += 1
                return None

            if entry.expired:
                del self._cache[key]
                self._order.pop(key, None)
                self._stats["misses"] += 1
                return None

            entry.touch()
            self._order.move_to_end(key)
            self._stats["hits"] += 1
            return entry.value

    async def set(self, key: K, value: V, ttl: float | None = None) -> None:
        async with self._lock:
            if key in self._cache:
                # Update existing
                entry = self._cache[key]
                entry.value = value
                entry.ttl = ttl if ttl is not None else self._default_ttl
                entry.touch()
                self._order.move_to_end(key)
            else:
                # Check eviction
                if len(self._cache) >= self._maxsize:
                    await self._evict()

                effective_ttl = ttl if ttl is not None else self._default_ttl
                entry = CacheEntry(key=key, value=value, ttl=effective_ttl)
                self._cache[key] = entry
                self._order[key] = True

            self._stats["sets"] += 1

    async def delete(self, key: K) -> bool:
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._order.pop(key, None)
                self._stats["deletes"] += 1
                return True
            return False

    async def clear(self) -> None:
        async with self._lock:
            self._cache.clear()
            self._order.clear()

    async def _evict(self) -> None:
        if self._order:
            oldest_key = next(iter(self._order))
            del self._cache[oldest_key]
            del self._order[oldest_key]
            self._stats["evictions"] += 1


class TTLCache(AsyncCache[K, V]):
    """Time-To-Live cache with automatic expiration."""

    def __init__(
        self,
        maxsize: int = 128,
        default_ttl: float = 300.0,
        *,
        name: str | None = None,
        cleanup_interval: float = 60.0,
    ) -> None:
        super().__init__(maxsize, name=name, default_ttl=default_ttl)
        self._cleanup_interval = cleanup_interval
        self._cleanup_task: asyncio.Task | None = None

    async def start_cleanup(self) -> None:
        """Start background cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop_cleanup(self) -> None:
        """Stop background cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._cleanup_task
            self._cleanup_task = None

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(self._cleanup_interval)
            await self._cleanup_expired()

    async def _cleanup_expired(self) -> int:
        """Remove expired entries."""
        async with self._lock:
            now = time.monotonic()
            expired_keys = [
                key
                for key, entry in self._cache.items()
                if entry.ttl and (now - entry.created_at) > entry.ttl
            ]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)

    async def _evict(self) -> None:
        # First try to evict expired
        cleaned = await self._cleanup_expired()
        if cleaned > 0:
            return

        # Then evict oldest
        if self._cache:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].created_at)
            del self._cache[oldest_key]
            self._stats["evictions"] += 1


class LFUCache(AsyncCache[K, V]):
    """Least Frequently Used cache."""

    def __init__(self, maxsize: int = 128, **kwargs) -> None:
        super().__init__(maxsize, **kwargs)
        self._freq: dict[int, OrderedDict] = {}
        self._min_freq = 0

    async def get(self, key: K) -> V | None:
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._stats["misses"] += 1
                return None

            if entry.expired:
                await self._remove_entry(key)
                self._stats["misses"] += 1
                return None

            self._update_freq(key, entry)
            self._stats["hits"] += 1
            return entry.value

    async def set(self, key: K, value: V, ttl: float | None = None) -> None:
        async with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                entry.value = value
                entry.ttl = ttl if ttl is not None else self._default_ttl
                self._update_freq(key, entry)
            else:
                if len(self._cache) >= self._maxsize:
                    await self._evict()

                effective_ttl = ttl if ttl is not None else self._default_ttl
                entry = CacheEntry(key=key, value=value, ttl=effective_ttl)
                self._cache[key] = entry
                self._add_to_freq(1, key)
                self._min_freq = 1

            self._stats["sets"] += 1

    async def _remove_entry(self, key: K) -> None:
        entry = self._cache.pop(key, None)
        if entry:
            freq = entry.access_count
            if freq in self._freq:
                self._freq[freq].pop(key, None)
                if not self._freq[freq]:
                    del self._freq[freq]
                    if freq == self._min_freq:
                        self._min_freq += 1

    def _add_to_freq(self, freq: int, key: K) -> None:
        if freq not in self._freq:
            self._freq[freq] = OrderedDict()
        self._freq[freq][key] = True

    def _update_freq(self, key: K, entry: CacheEntry) -> None:
        old_freq = entry.access_count
        entry.touch()
        new_freq = entry.access_count

        if old_freq in self._freq:
            self._freq[old_freq].pop(key, None)
            if not self._freq[old_freq]:
                del self._freq[old_freq]
                if old_freq == self._min_freq:
                    self._min_freq = new_freq

        self._add_to_freq(new_freq, key)

    async def _evict(self) -> None:
        if self._freq.get(self._min_freq):
            key = next(iter(self._freq[self._min_freq]))
            await self._remove_entry(key)
            self._stats["evictions"] += 1


class CacheDecorator(Generic[T]):
    """Decorator for caching async function results."""

    def __init__(
        self,
        cache: AsyncCache,
        key_func: Callable[..., Any] | None = None,
        ttl: float | None = None,
    ) -> None:
        self._cache = cache
        self._key_func = key_func or self._default_key
        self._ttl = ttl

    def _default_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = pickle.dumps((args, kwargs))
        return hashlib.sha256(key_data).hexdigest()

    def __call__(self, func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            key = self._key_func(*args, **kwargs)

            # Try cache
            cached = await self._cache.get(key)
            if cached is not None:
                return cached

            # Compute and cache
            result = await func(*args, **kwargs)
            await self._cache.set(key, result, self._ttl)
            return result

        return wrapper


class MultiLevelCache(Generic[K, V]):
    """Multi-level cache (L1 + L2)."""

    def __init__(
        self, l1: AsyncCache[K, V], l2: AsyncCache[K, V], *, name: str | None = None
    ) -> None:
        self._l1 = l1
        self._l2 = l2
        self._name = name or f"MultiLevelCache-{id(self)}"

    @property
    def name(self) -> str:
        return self._name

    async def get(self, key: K) -> V | None:
        # Try L1
        value = await self._l1.get(key)
        if value is not None:
            return value

        # Try L2
        value = await self._l2.get(key)
        if value is not None:
            # Promote to L1
            await self._l1.set(key, value)
            return value

        return None

    async def set(self, key: K, value: V, ttl: float | None = None) -> None:
        await self._l1.set(key, value, ttl)
        await self._l2.set(key, value, ttl)

    async def delete(self, key: K) -> bool:
        l1_result = await self._l1.delete(key)
        l2_result = await self._l2.delete(key)
        return l1_result or l2_result

    async def clear(self) -> None:
        await self._l1.clear()
        await self._l2.clear()

    def stats(self) -> dict[str, Any]:
        return {"l1": self._l1.stats(), "l2": self._l2.stats()}


class WriteThroughCache(AsyncCache[K, V]):
    """Cache with write-through to backend."""

    def __init__(self, backend: CacheBackend[K, V], maxsize: int = 128, **kwargs) -> None:
        super().__init__(maxsize, **kwargs)
        self._backend = backend

    async def set(self, key: K, value: V, ttl: float | None = None) -> None:
        await super().set(key, value, ttl)
        await self._backend.set(key, value, ttl)

    async def delete(self, key: K) -> bool:
        result = await super().delete(key)
        await self._backend.delete(key)
        return result


class WriteBehindCache(AsyncCache[K, V]):
    """Cache with async write-behind to backend."""

    def __init__(
        self,
        backend: CacheBackend[K, V],
        maxsize: int = 128,
        write_delay: float = 1.0,
        batch_size: int = 10,
        **kwargs,
    ) -> None:
        super().__init__(maxsize, **kwargs)
        self._backend = backend
        self._write_delay = write_delay
        self._batch_size = batch_size
        self._write_queue: asyncio.Queue = asyncio.Queue()
        self._writer_task: asyncio.Task | None = None

    async def start(self) -> None:
        self._writer_task = asyncio.create_task(self._writer_loop())

    async def stop(self) -> None:
        if self._writer_task:
            self._writer_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._writer_task

    async def set(self, key: K, value: V, ttl: float | None = None) -> None:
        await super().set(key, value, ttl)
        await self._write_queue.put(("set", key, value, ttl))

    async def delete(self, key: K) -> bool:
        result = await super().delete(key)
        await self._write_queue.put(("delete", key, None, None))
        return result

    async def _writer_loop(self) -> None:
        batch = []
        last_flush = time.monotonic()

        while True:
            try:
                # Wait for items
                item = await asyncio.wait_for(self._write_queue.get(), timeout=self._write_delay)
                batch.append(item)

                # Check if we should flush
                now = time.monotonic()
                if len(batch) >= self._batch_size or (now - last_flush) >= self._write_delay:
                    await self._flush_batch(batch)
                    batch.clear()
                    last_flush = now

            except TimeoutError:
                if batch:
                    await self._flush_batch(batch)
                    batch.clear()
                    last_flush = time.monotonic()
            except asyncio.CancelledError:
                if batch:
                    await self._flush_batch(batch)
                break

    async def _flush_batch(self, batch: list[tuple]) -> None:
        for op, key, value, ttl in batch:
            try:
                if op == "set":
                    await self._backend.set(key, value, ttl)
                elif op == "delete":
                    await self._backend.delete(key)
            except Exception:
                pass  # Log error in production


@asynccontextmanager
async def cached(
    cache: AsyncCache, key: Any, ttl: float | None = None
) -> AsyncGenerator[Any, None]:
    """Context manager for cached value."""
    value = await cache.get(key)
    if value is not None:
        yield value
        return

    # Value not cached, caller should set it
    yield None


def cached_property(cache: AsyncCache, ttl: float | None = None):
    """Decorator for cached property."""

    def decorator(func: Callable[..., Awaitable[T]]) -> property:
        @wraps(func)
        async def wrapper(self) -> T:
            key = f"{self.__class__.__name__}.{func.__name__}.{id(self)}"
            cached = await cache.get(key)
            if cached is not None:
                return cached

            result = await func(self)
            await cache.set(key, result, ttl)
            return result

        return property(wrapper)

    return decorator


def make_cache_key(*args, **kwargs) -> str:
    """Create a cache key from arguments."""
    key_data = pickle.dumps((args, kwargs))
    return hashlib.sha256(key_data).hexdigest()


async def memoize(
    func: Callable[..., Awaitable[T]],
    cache: AsyncCache | None = None,
    ttl: float | None = None,
    key_func: Callable[..., Any] | None = None,
) -> Callable[..., Awaitable[T]]:
    """Create a memoized version of an async function."""
    if cache is None:
        cache = LRUCache(maxsize=128)

    if key_func is None:
        key_func = make_cache_key

    @wraps(func)
    async def wrapper(*args, **kwargs) -> T:
        key = key_func(*args, **kwargs)
        cached = await cache.get(key)
        if cached is not None:
            return cached

        result = await func(*args, **kwargs)
        await cache.set(key, result, ttl)
        return result

    return wrapper
