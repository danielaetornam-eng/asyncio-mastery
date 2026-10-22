#!/usr/bin/env python3
"""
Worker Pool Demo - Demonstrates worker pool pattern.

Run with: python -m examples.worker_pool_demo
"""

import asyncio
import sys

sys.path.insert(0, "src")

from asyncio_mastery.patterns import Batcher, WorkerPool, producer_consumer


async def run_worker_pool() -> None:
    """Demonstrate worker pool."""
    print("=" * 60)
    print("WORKER POOL DEMO")
    print("=" * 60)

    # Simulate CPU-bound work
    async def process_item(item: int) -> str:
        await asyncio.sleep(0.05)  # Simulate async work
        return f"processed-{item * 2}"

    async with WorkerPool(process_item, max_workers=4) as pool:
        # Submit jobs
        jobs = [await pool.submit(i) for i in range(20)]
        print(f"Submitted {len(jobs)} jobs")

        # Wait for results
        results = await asyncio.gather(*[job.execute() for job in jobs])
        for r in results[:5]:
            print(f"  {r}")
        print(f"  ... and {len(results) - 5} more")

        print(f"Stats: {pool.stats}")

    # Using map
    print("\n--- Map ---")
    async with WorkerPool(process_item, max_workers=2) as pool:
        results = await pool.map(list(range(10)))
        print(f"Results: {results}")


async def run_producer_consumer() -> None:
    """Demonstrate producer-consumer."""
    print("\n" + "=" * 60)
    print("PRODUCER-CONSUMER DEMO")
    print("=" * 60)

    processed: list[int] = []

    async def consumer(item: int) -> None:
        await asyncio.sleep(0.01)
        processed.append(item)

    async with producer_consumer(consumer, max_queue_size=10, num_consumers=3) as pc:
        # Produce items
        await pc.produce_many(range(30))

        # Wait for queue to drain
        while pc.queue_size > 0:
            await asyncio.sleep(0.01)

    print(f"Processed: {len(processed)} items")
    print(f"Stats: {pc.stats}")


async def run_batcher() -> None:
    """Demonstrate batcher."""
    print("\n" + "=" * 60)
    print("BATCHER DEMO")
    print("=" * 60)

    batches: list[list[int]] = []

    async def batch_processor(items: list[int]) -> list[str]:
        await asyncio.sleep(0.01)
        batches.append(items.copy())
        return [f"result-{x}" for x in items]

    batcher = Batcher(batch_processor, max_size=5, max_wait=0.05)

    # Add items
    tasks = [batcher.add(i) for i in range(20)]
    results = await asyncio.gather(*tasks)

    print(f"Processed {len(results)} items")
    print(f"Batches: {batches}")

    # Wait for any remaining
    await asyncio.sleep(0.1)
    print(f"Final batches: {batches}")


async def main() -> None:
    await run_worker_pool()
    await run_producer_consumer()
    await run_batcher()


if __name__ == "__main__":
    asyncio.run(main())
