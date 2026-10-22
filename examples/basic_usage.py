#!/usr/bin/env python3
"""
Basic Usage Examples - Demonstrates core asyncio concepts.

Run with: python -m examples.basic_usage
"""

import asyncio
import sys

sys.path.insert(0, "src")

from asyncio_mastery.advanced import (
    demonstrate_async_generators,
    demonstrate_context_vars,
    demonstrate_timeouts,
)
from asyncio_mastery.basics import (
    demonstrate_coroutines,
    demonstrate_event_loop,
    demonstrate_futures,
    demonstrate_tasks,
)
from asyncio_mastery.synchronization import (
    demonstrate_events,
    demonstrate_locks,
    demonstrate_semaphores,
)


async def main() -> None:
    print("=" * 60)
    print("ASYNCIO MASTERY - BASIC USAGE EXAMPLES")
    print("=" * 60)

    # Basics
    print("\n" + "=" * 60)
    print("BASICS")
    print("=" * 60)

    print("\n--- Coroutines ---")
    await demonstrate_coroutines()

    print("\n--- Tasks ---")
    await demonstrate_tasks()

    print("\n--- Futures ---")
    await demonstrate_futures()

    print("\n--- Event Loop ---")
    await demonstrate_event_loop()

    # Synchronization
    print("\n" + "=" * 60)
    print("SYNCHRONIZATION")
    print("=" * 60)

    print("\n--- Locks ---")
    await demonstrate_locks()

    print("\n--- Semaphores ---")
    await demonstrate_semaphores()

    print("\n--- Events ---")
    await demonstrate_events()

    # Advanced
    print("\n" + "=" * 60)
    print("ADVANCED CONCEPTS")
    print("=" * 60)

    print("\n--- Context Variables ---")
    await demonstrate_context_vars()

    print("\n--- Async Generators ---")
    await demonstrate_async_generators()

    print("\n--- Timeouts ---")
    await demonstrate_timeouts()

    print("\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
