#!/usr/bin/env python3
"""
Patterns Demo - Demonstrates production-ready async patterns.

Run with: python -m examples.patterns_demo
"""

import asyncio
import sys

sys.path.insert(0, "src")

import logging

from asyncio_mastery.patterns import (
    AsyncContextManager,
    CircuitBreaker,
    CircuitBreakerConfig,
    ConnectionPool,
    PubSub,
    RateLimiter,
    RetryPolicy,
    SemaphorePool,
    TaskGroup,
)
from asyncio_mastery.utils import setup_logging


async def connection_pool_demo() -> None:
    """Connection pool pattern."""
    print("=" * 60)
    print("CONNECTION POOL")
    print("=" * 60)

    from asyncio_mastery.patterns.connection_pool import ConnectionFactory

    class MockConnectionFactory(ConnectionFactory[str]):
        def __init__(self) -> None:
            self._count = 0

        async def create(self) -> str:
            self._count += 1
            await asyncio.sleep(0.01)
            return f"conn-{self._count}"

        async def close(self, conn: str) -> None:
            await asyncio.sleep(0.001)

        async def health_check(self, conn: str) -> bool:
            return True

    factory = MockConnectionFactory()
    pool = ConnectionPool(factory, min_size=2, max_size=5)

    await pool.start()

    async def use_db(query: str) -> str:
        async with pool.connection() as conn:
            await asyncio.sleep(0.02)
            return f"{conn}: {query}"

    # Concurrent requests
    tasks = [use_db(f"SELECT {i}") for i in range(15)]
    results = await asyncio.gather(*tasks)
    for r in results[:3]:
        print(f"  {r}")
    print(f"  ... {len(results)} total")

    print(f"Pool stats: {pool.stats}")
    await pool.close()


async def rate_limiter_demo() -> None:
    """Rate limiter pattern."""
    print("\n" + "=" * 60)
    print("RATE LIMITER")
    print("=" * 60)

    limiter = RateLimiter(rate=20, burst=5)

    async def make_request(i: int) -> str:
        async with limiter:
            await asyncio.sleep(0.01)
            return f"req-{i}"

    # Burst of requests
    start = asyncio.get_event_loop().time()
    _results = await asyncio.gather(*[make_request(i) for i in range(10)])
    elapsed = asyncio.get_event_loop().time() - start

    print(f"10 requests took {elapsed:.3f}s (rate: 20/sec)")
    print(f"Available tokens: {limiter.available}")


async def retry_demo() -> None:
    """Retry policy pattern."""
    print("\n" + "=" * 60)
    print("RETRY POLICY")
    print("=" * 60)

    attempt = 0

    async def flaky_service() -> str:
        nonlocal attempt
        attempt += 1
        if attempt < 3:
            raise ConnectionError(f"Attempt {attempt} failed")
        return "success"

    policy = RetryPolicy(max_attempts=5, base_delay=0.05)
    result = await policy.execute(flaky_service)
    print(f"Result: {result} (attempts: {attempt})")


async def circuit_breaker_demo() -> None:
    """Circuit breaker pattern."""
    print("\n" + "=" * 60)
    print("CIRCUIT BREAKER")
    print("=" * 60)

    config = CircuitBreakerConfig(failure_threshold=3, success_threshold=2, timeout=0.5)
    breaker = CircuitBreaker("demo-service", config)

    call_count = 0

    async def unreliable() -> str:
        nonlocal call_count
        call_count += 1
        if call_count <= 4:
            raise ConnectionError("Service down")
        return "recovered"

    # Trigger failures
    for i in range(5):
        try:
            await breaker.call(unreliable)
            print(f"Call {i + 1}: Success")
        except Exception as e:
            print(f"Call {i + 1}: {type(e).__name__}")

    print(f"State: {breaker.state}")

    # Wait for half-open
    await asyncio.sleep(0.6)
    print(f"After timeout: {breaker.state}")

    # Test recovery
    for i in range(3):
        try:
            await breaker.call(unreliable)
            print(f"Recovery {i + 1}: Success")
        except Exception as e:
            print(f"Recovery {i + 1}: {type(e).__name__}")


async def task_group_demo() -> None:
    """Task group pattern."""
    print("\n" + "=" * 60)
    print("TASK GROUP")
    print("=" * 60)

    async with TaskGroup() as group:

        async def worker(name: str) -> str:
            await asyncio.sleep(0.05)
            return f"{name} done"

        for i in range(5):
            group.create_task(worker(f"worker-{i}"))

    print(f"Results: {group.result.results}")


async def pubsub_demo() -> None:
    """Pub/Sub pattern."""
    print("\n" + "=" * 60)
    print("PUB/SUB")
    print("=" * 60)

    pubsub = PubSub()
    received = []

    async def handler(msg) -> None:
        received.append(msg.payload)
        print(f"  Received: {msg.payload}")

    await pubsub.subscribe("events", handler)
    await pubsub.publish("events", "message-1")
    await pubsub.publish("events", "message-2")
    await pubsub.publish("notifications", "notify-1")

    print(f"Total received: {len(received)}")


async def semaphore_pool_demo() -> None:
    """Semaphore pool pattern."""
    print("\n" + "=" * 60)
    print("SEMAPHORE POOL")
    print("=" * 60)

    pool = SemaphorePool()
    pool.create("database", 3)
    pool.create("api", 5)

    async def use_resources() -> str:
        async with pool.acquire_many("database", "api"):
            await asyncio.sleep(0.01)
            return "done"

    results = await asyncio.gather(*[use_resources() for _ in range(6)])
    print(f"Completed: {len(results)}")

    print(f"Stats: {pool.stats()}")


async def async_context_manager_demo() -> None:
    """Async context manager pattern."""
    print("\n" + "=" * 60)
    print("ASYNC CONTEXT MANAGER")
    print("=" * 60)

    class Database(AsyncContextManager[str]):
        async def _initialize(self) -> str:
            print("  Connecting to DB...")
            await asyncio.sleep(0.02)
            return "db-connection"

        async def _close(self) -> None:
            print("  Closing DB connection")
            await asyncio.sleep(0.01)

    async with Database("main-db") as conn:
        print(f"  Using: {conn}")

    print("  Connection closed")


async def main() -> None:
    setup_logging(level=logging.INFO)

    await connection_pool_demo()
    await rate_limiter_demo()
    await retry_demo()
    await circuit_breaker_demo()
    await task_group_demo()
    await pubsub_demo()
    await semaphore_pool_demo()
    await async_context_manager_demo()

    print("\n" + "=" * 60)
    print("ALL PATTERN DEMOS COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    import logging

    asyncio.run(main())
