"""
Testing package for asyncio_mastery.

Provides utilities for testing asynchronous code.
Includes test case base class, async mock objects, and time control tools.
"""

from asyncio_mastery.testing.async_mock import AsyncMagicMock, AsyncMock, patch_async
from asyncio_mastery.testing.mock_time import MockClock, TimeTravel, patch_time
from asyncio_mastery.testing.test_helpers import (
    AsyncTestCase,
    assert_eventually,
    run_async_test,
    wait_for_condition,
)

__all__ = [
    "AsyncMagicMock",
    "AsyncMock",
    "AsyncTestCase",
    "MockClock",
    "TimeTravel",
    "assert_eventually",
    "patch_async",
    "patch_time",
    "run_async_test",
    "wait_for_condition",
]
