#!/usr/bin/env python3
"""
Tests for utils module.
"""

import asyncio
import logging
import sys

import pytest

sys.path.insert(0, "src")

from asyncio_mastery.utils import (
    AsyncCache,
    AsyncRateLimiter,
    async_cached,
    async_retry,
    async_singleton,
    async_timeout,
    background_task,
    debounce,
    gather_with_concurrency,
    throttle,
)
from asyncio_mastery.utils.logging import AsyncLogger, setup_logging


class TestDecorators:
    """Test async decorators."""

    async def test_async_retry(self) -> None:
        attempt = 0

        @async_retry(max_attempts=3, base_delay=0.01)
        async def flaky() -> str:
            nonlocal attempt
            attempt += 1
            if attempt < 3:
                raise ConnectionError("fail")
            return "success"

        result = await flaky()
        assert result == "success"
        assert attempt == 3

    async def test_async_timeout(self) -> None:
        @async_timeout(0.1)
        async def fast() -> str:
            await asyncio.sleep(0.05)
            return "done"

        result = await fast()
        assert result == "done"

    async def test_async_timeout_exceeded(self) -> None:
        @async_timeout(0.05)
        async def slow() -> str:
            await asyncio.sleep(0.2)
            return "done"

        with pytest.raises(TimeoutError):
            await slow()

    async def test_async_cached(self) -> None:
        call_count = 0

        @async_cached(ttl=1.0)
        async def cached_func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        r1 = await cached_func(5)
        r2 = await cached_func(5)
        assert r1 == r2 == 10
        assert call_count == 1

    async def test_async_singleton(self) -> None:
        call_count = 0

        @async_singleton
        async def singleton_func() -> int:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return 42

        r1 = await singleton_func()
        r2 = await singleton_func()
        assert r1 == r2 == 42
        assert call_count == 1

    async def test_debounce(self) -> None:
        calls = []

        @debounce(0.1)
        async def debounced(msg: str) -> None:
            calls.append(msg)

        await debounced("first")
        await debounced("second")
        await debounced("third")
        await asyncio.sleep(0.15)

        assert calls == ["third"]

    async def test_throttle(self) -> None:
        calls = []

        @throttle(0.1)
        async def throttled(msg: str) -> None:
            calls.append(msg)

        await throttled("1")
        await throttled("2")
        await asyncio.sleep(0.15)
        await throttled("3")

        assert calls == ["1", "3"]

    async def test_background_task(self) -> None:
        @background_task()
        async def bg_task(x: int) -> int:
            await asyncio.sleep(0.05)
            return x * 2

        task = bg_task(21)
        result = await task
        assert result == 42


class TestHelpers:
    """Test helper utilities."""

    async def test_gather_with_concurrency(self) -> None:
        sem = asyncio.Semaphore(2)
        running = 0
        max_running = 0

        async def task(i: int) -> int:
            nonlocal running, max_running
            async with sem:
                running += 1
                max_running = max(max_running, running)
                await asyncio.sleep(0.05)
                running -= 1
            return i

        results = await gather_with_concurrency(2, *[task(i) for i in range(5)])
        assert results == [0, 1, 2, 3, 4]
        assert max_running <= 2

    async def test_async_cache(self) -> None:
        cache = AsyncCache(ttl=1.0)

        async def factory() -> int:
            await asyncio.sleep(0.01)
            return 42

        r1 = await cache.get_or_set("key", factory)
        r2 = await cache.get_or_set("key", factory)
        assert r1 == r2 == 42

    async def test_async_rate_limiter(self) -> None:
        limiter = AsyncRateLimiter(rate=10, burst=3)

        start = asyncio.get_event_loop().time()
        for _ in range(5):
            await limiter.acquire()
        elapsed = asyncio.get_event_loop().time() - start

        assert elapsed >= 0.2


class TestLogging:
    """Test logging utilities."""

    def test_logger_creation(self) -> None:
        logger = AsyncLogger("test")
        assert logger._logger.name == "test"

    def test_setup_logging(self) -> None:
        setup_logging(level=10, json_format=True)
        root_logger = logging.getLogger()
        assert root_logger.level == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
