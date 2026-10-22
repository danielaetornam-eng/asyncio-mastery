#!/usr/bin/env python3
"""
Tests for patterns module.
"""

import asyncio
import sys

import pytest

sys.path.insert(0, "src")

from typing import Never

from asyncio_mastery.patterns import (
    BatchConfig,
    Batcher,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitState,
    ConnectionPool,
    PoolConfig,
    PubSub,
    RateLimiter,
    RetryConfig,
    RetryPolicy,
    SemaphorePool,
    TaskGroup,
    WorkerPool,
)
from asyncio_mastery.patterns.connection_pool import ConnectionFactory


class MockConnectionFactory(ConnectionFactory[str]):
    def __init__(self) -> None:
        self._count = 0

    async def create(self) -> str:
        self._count += 1
        await asyncio.sleep(0.001)
        return f"conn-{self._count}"

    async def close(self, conn: str) -> None:
        pass

    async def health_check(self, conn: str) -> bool:
        return True


class TestConnectionPool:
    """Test connection pool."""

    async def test_basic_pool(self) -> None:
        factory = MockConnectionFactory()
        config = PoolConfig(min_size=2, max_size=5)
        pool = ConnectionPool(factory, config)
        await pool.start()

        async with pool.connection() as conn:
            assert conn.startswith("conn-")

        await pool.close()

    async def test_concurrent_access(self) -> None:
        factory = MockConnectionFactory()
        config = PoolConfig(min_size=2, max_size=10)
        pool = ConnectionPool(factory, config)
        await pool.start()

        async def use_conn(i: int) -> str:
            async with pool.connection() as conn:
                await asyncio.sleep(0.01)
                return f"{conn}-{i}"

        results = await asyncio.gather(*[use_conn(i) for i in range(10)])
        assert len(results) == 10

        await pool.close()


class TestRateLimiter:
    """Test rate limiter."""

    async def test_token_bucket(self) -> None:
        limiter = RateLimiter(rate=10, burst=3)

        start = asyncio.get_event_loop().time()
        for _ in range(5):
            await limiter.acquire()
        elapsed = asyncio.get_event_loop().time() - start

        # Should take some time due to rate limiting
        assert elapsed >= 0.2

    async def test_try_acquire(self) -> None:
        limiter = RateLimiter(rate=10, burst=2)

        assert limiter.try_acquire() is True
        assert limiter.try_acquire() is True
        assert limiter.try_acquire() is False  # Burst exhausted


class TestRetryPolicy:
    """Test retry policy."""

    async def test_basic_retry(self) -> None:
        attempt = 0

        async def flaky() -> str:
            nonlocal attempt
            attempt += 1
            if attempt < 3:
                raise ConnectionError("fail")
            return "success"

        config = RetryConfig(max_attempts=5, base_delay=0.01)
        policy = RetryPolicy(config)
        result = await policy.execute(flaky)
        assert result == "success"
        assert attempt == 3

    async def test_stop_on_exception(self) -> None:
        attempt = 0

        async def stop_on_value_error() -> str:
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                raise ValueError("stop")
            raise ConnectionError("retry")

        config = RetryConfig(max_attempts=5, retry_on=(ConnectionError,), stop_on=(ValueError,))
        policy = RetryPolicy(config)

        with pytest.raises(ValueError):
            await policy.execute(stop_on_value_error)


class TestCircuitBreaker:
    """Test circuit breaker."""

    async def test_circuit_opens(self) -> None:
        config = CircuitBreakerConfig(failure_threshold=3, timeout=0.5)
        breaker = CircuitBreaker("test", config)

        async def failing() -> str:
            raise ConnectionError("fail")

        for _ in range(3):
            with pytest.raises(ConnectionError):
                await breaker.call(failing)

        assert breaker.state == CircuitState.OPEN

        # Should reject calls
        with pytest.raises(CircuitBreakerOpenError):
            await breaker.call(failing)

    async def test_half_open_recovery(self) -> None:
        config = CircuitBreakerConfig(failure_threshold=2, success_threshold=2, timeout=0.1)
        breaker = CircuitBreaker("test", config)

        call_count = 0

        async def recovering() -> str:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ConnectionError("fail")
            return "ok"

        # Trigger open
        for _ in range(2):
            with pytest.raises(ConnectionError):
                await breaker.call(recovering)

        await asyncio.sleep(0.15)  # Wait for half-open

        # Should succeed and close - need 2 successes for success_threshold=2
        result = await breaker.call(recovering)
        assert result == "ok"
        result = await breaker.call(recovering)
        assert result == "ok"
        assert breaker.state == CircuitState.CLOSED


class TestTaskGroup:
    """Test task group."""

    async def test_basic_task_group(self) -> None:
        async with TaskGroup() as group:

            async def worker(i: int) -> int:
                await asyncio.sleep(0.01)
                return i * 2

            for i in range(5):
                group.create_task(worker(i))

        assert len(group.result.results) == 5
        assert group.result.success

    async def test_task_group_exception(self) -> None:
        async with TaskGroup() as group:

            async def fail() -> Never:
                raise ValueError("fail")

            group.create_task(fail())

        assert not group.result.success
        assert len(group.result.exceptions) == 1


class TestWorkerPool:
    """Test worker pool."""

    async def test_worker_pool(self) -> None:
        async def process(item: int) -> str:
            await asyncio.sleep(0.01)
            return f"done-{item}"

        pool = WorkerPool(process, max_workers=3)
        await pool.start()
        try:
            jobs = [await pool.submit(i) for i in range(10)]
            results = await asyncio.gather(*[job.execute() for job in jobs])
            assert len(results) == 10
        finally:
            await pool.stop()

    async def test_worker_pool_map(self) -> None:
        async def process(item: int) -> int:
            return item * 2

        pool = WorkerPool(process, max_workers=2)
        await pool.start()
        try:
            results = await pool.map(list(range(5)))
            assert results == [0, 2, 4, 6, 8]
        finally:
            await pool.stop()


class TestPubSub:
    """Test pub/sub."""

    async def test_pubsub(self) -> None:
        pubsub = PubSub()
        received = []

        async def handler(msg) -> None:
            received.append(msg.payload)

        await pubsub.subscribe("topic", handler)
        await pubsub.publish("topic", "message1")
        await pubsub.publish("topic", "message2")

        await asyncio.sleep(0.01)  # Allow delivery

        assert received == ["message1", "message2"]


class TestBatcher:
    """Test batcher."""

    async def test_batcher(self) -> None:
        batches = []

        async def processor(items: list[int]) -> list[str]:
            batches.append(items.copy())
            return [f"r-{x}" for x in items]

        config = BatchConfig(max_size=3, max_wait=0.1)
        batcher = Batcher(processor, config)

        tasks = [batcher.add(i) for i in range(7)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 7
        assert len(batches) >= 2  # At least 2 batches for 7 items


class TestSemaphorePool:
    """Test semaphore pool."""

    async def test_semaphore_pool(self) -> None:
        pool = SemaphorePool()
        pool.create("resource", 2)

        async def use_resource() -> str:
            async with pool.acquire("resource"):
                await asyncio.sleep(0.01)
                return "done"

        results = await asyncio.gather(*[use_resource() for _ in range(4)])
        assert len(results) == 4

    async def test_multi_semaphore(self) -> None:
        pool = SemaphorePool()
        pool.create("a", 1)
        pool.create("b", 1)

        async def use_both() -> str:
            async with pool.acquire_many("a", "b"):
                await asyncio.sleep(0.01)
                return "done"

        # Only one can run at a time
        results = await asyncio.gather(*[use_both() for _ in range(3)])
        assert len(results) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
