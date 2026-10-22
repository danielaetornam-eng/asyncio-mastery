#!/usr/bin/env python3
"""
Tests for basics module.
"""

import asyncio
import sys

import pytest

sys.path.insert(0, "src")

from typing import Never

from asyncio_mastery.basics.coroutines import (
    CoroutineExample,
    coroutine_raising_exception,
    coroutine_returning_value,
    coroutine_with_args,
    simple_coroutine,
)
from asyncio_mastery.basics.event_loop import EventLoopManager
from asyncio_mastery.basics.futures import FutureWrapper
from asyncio_mastery.basics.tasks import TaskManager


class TestCoroutines:
    """Test coroutine basics."""

    async def test_simple_coroutine(self) -> None:
        result = await simple_coroutine()
        assert result == "Hello from coroutine"

    async def test_coroutine_with_args(self) -> None:
        result = await coroutine_with_args("World", 0.01)
        assert result == "Hello, World!"

    async def test_coroutine_returning_value(self) -> None:
        result = await coroutine_returning_value()
        assert result["status"] == "success"
        assert result["data"] == [1, 2, 3]

    async def test_coroutine_raising_exception(self) -> None:
        with pytest.raises(ValueError):
            await coroutine_raising_exception()

    async def test_coroutine_example(self) -> None:
        example = CoroutineExample("test")
        result = await example.method_coroutine(21)
        assert result == 42

        result = await CoroutineExample.static_coroutine(10, 20)
        assert result == 30

        result = await example
        assert result == "Awaited test"


class TestTasks:
    """Test task management."""

    async def test_create_task(self) -> None:
        async def worker() -> str:
            await asyncio.sleep(0.01)
            return "done"

        task = asyncio.create_task(worker())
        result = await task
        assert result == "done"

    async def test_task_cancellation(self) -> None:
        async def long_running() -> str:
            await asyncio.sleep(10)
            return "done"

        task = asyncio.create_task(long_running())
        await asyncio.sleep(0.01)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_task_exception_handling(self) -> None:
        async def failing() -> Never:
            await asyncio.sleep(0.01)
            raise ValueError("fail")

        task = asyncio.create_task(failing())
        with pytest.raises(ValueError):
            await task

    async def test_task_manager(self) -> None:
        manager = TaskManager()

        async def worker(name: str) -> str:
            await asyncio.sleep(0.01)
            return name

        for i in range(5):
            manager.create_task(worker(f"worker-{i}"))

        results = await manager.wait_all()
        assert len(results) == 5

    async def test_gather_with_exceptions(self) -> None:
        async def succeed() -> str:
            return "ok"

        async def fail() -> str:
            raise ValueError("fail")

        results = await asyncio.gather(succeed(), fail(), succeed(), return_exceptions=True)
        assert results[0] == "ok"
        assert isinstance(results[1], ValueError)
        assert results[2] == "ok"


class TestFutures:
    """Test futures."""

    async def test_future_basics(self) -> None:
        loop = asyncio.get_event_loop()
        future = loop.create_future()

        def set_result() -> None:
            future.set_result("done")

        loop.call_later(0.05, set_result)
        result = await future
        assert result == "done"

    async def test_future_wrapper(self) -> None:
        wrapper = FutureWrapper()
        future = wrapper.start_operation()

        assert not wrapper.is_done()
        result = await future
        assert result == "Operation complete"
        assert wrapper.is_done()

    async def test_future_cancellation(self) -> None:
        wrapper = FutureWrapper()
        future = wrapper.start_operation()
        await asyncio.sleep(0.01)

        cancelled = wrapper.cancel()
        assert cancelled

        with pytest.raises(asyncio.CancelledError):
            await future


class TestEventLoop:
    """Test event loop management."""

    async def test_event_loop_manager(self) -> None:
        manager = EventLoopManager(debug=True)
        with manager as loop:

            async def test() -> str:
                await asyncio.sleep(0.01)
                return "works"

            # Use the loop directly since we're already in an async context
            result = await test()
            assert result == "works"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
