#!/usr/bin/env python3
"""
Tests for advanced concepts.
"""

import asyncio
import sys

import pytest

sys.path.insert(0, "src")

from asyncio_mastery.advanced import (
    CancellationToken,
    async_filter,
    async_map,
    async_range,
    cancel_on,
    run_with_timeout,
    shield,
    timeout_after,
)
from asyncio_mastery.advanced.context_vars import request_id_var


class TestTimeouts:
    """Test timeout utilities."""

    async def test_run_with_timeout_success(self) -> None:
        async def fast() -> str:
            await asyncio.sleep(0.01)
            return "done"

        result = await run_with_timeout(fast(), 1.0)
        assert result.success
        assert result.value == "done"

    async def test_run_with_timeout_failure(self) -> None:
        async def slow() -> str:
            await asyncio.sleep(1)
            return "done"

        result = await run_with_timeout(slow(), 0.1)
        assert not result.success
        assert result.timed_out

    async def test_run_with_timeout_default(self) -> None:
        async def slow() -> str:
            await asyncio.sleep(1)
            return "done"

        result = await run_with_timeout(slow(), 0.1, default="fallback")
        assert not result.success
        assert result.value == "fallback"

    async def test_timeout_after(self) -> None:
        async def fast() -> str:
            await asyncio.sleep(0.01)
            return "done"

        async with timeout_after(1.0):
            result = await fast()
            assert result == "done"

    async def test_timeout_after_expired(self) -> None:
        async def slow() -> str:
            await asyncio.sleep(1)
            return "done"

        with pytest.raises(TimeoutError):
            async with timeout_after(0.1):
                await slow()


class TestCancellation:
    """Test cancellation utilities."""

    async def test_cancellation_token(self) -> None:
        token = CancellationToken()
        cancelled = []

        async def worker() -> None:
            for i in range(10):
                if token.cancelled:
                    cancelled.append(True)
                    return
                await asyncio.sleep(0.01)

        task = asyncio.create_task(worker())
        await asyncio.sleep(0.05)
        token.cancel()

        await asyncio.gather(task, return_exceptions=True)
        assert len(cancelled) == 1

    async def test_cancel_on(self) -> None:
        event = asyncio.Event()

        async def worker() -> None:
            async with cancel_on(event):
                for i in range(10):
                    await asyncio.sleep(0.1)

        task = asyncio.create_task(worker())
        await asyncio.sleep(0.05)
        event.set()

        await asyncio.gather(task, return_exceptions=True)

    async def test_shield(self) -> None:
        async def critical() -> str:
            await asyncio.sleep(0.2)
            return "critical done"

        try:
            async with asyncio.timeout(0.1):
                _result = await shield(critical())
        except TimeoutError:
            pass

        # Shielded task should still complete
        await asyncio.sleep(0.3)


class TestAsyncGenerators:
    """Test async generators."""

    async def test_async_range(self) -> None:
        result = []
        async for i in async_range(5, delay=0.01):
            result.append(i)
        assert result == [0, 1, 2, 3, 4]

    async def test_async_map(self) -> None:
        async def double(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 2

        result = []
        async for item in async_map(double, async_range(5)):
            result.append(item)
        assert result == [0, 2, 4, 6, 8]

    async def test_async_filter(self) -> None:
        async def is_even(x: int) -> bool:
            await asyncio.sleep(0.01)
            return x % 2 == 0

        result = []
        async for item in async_filter(is_even, async_range(10)):
            result.append(item)
        assert result == [0, 2, 4, 6, 8]


class TestContextVars:
    """Test context variables."""

    async def test_context_propagation(self) -> None:
        request_id_var.set("req-123")

        async def child():
            return request_id_var.get()

        result = await child()
        assert result == "req-123"

    async def test_concurrent_contexts(self) -> None:
        async def worker(id: int):
            request_id_var.set(f"req-{id}")
            await asyncio.sleep(0.01)
            return request_id_var.get()

        results = await asyncio.gather(*[worker(i) for i in range(5)])
        assert set(results) == {f"req-{i}" for i in range(5)}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
