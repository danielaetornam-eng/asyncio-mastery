"""
Conditions - Waiting for a state to become true, not just a signal.

WHAT IS A CONDITION
-------------------
A Condition combines TWO things in one object:

1. A Lock, which protects shared state (mutual exclusion), and
2. A wait/notify channel, so tasks can sleep INSIDE the lock until another task changes the shared state and says check again.

The magic of wait is that it ATOMICALLY releases the lock and goes to sleep. While you are waiting, other tasks can acquire the lock, change the state, and notify. When you are woken, you automatically re-acquire the lock before continuing. The classic pattern is::

    async with condition:
        await condition.wait_for(lambda: state_is_ready)
        # now lock held AND state_is_ready is guaranteed true
        use_the_state()

ANALOGY: THE BAKERY
-------------------
You want fresh bread, but you only buy if (a) the shop is OPEN and (b) a fresh batch is OUT OF THE OVEN. That is a compound condition (open AND fresh). A plain Event cannot express it.

So you wait in the shop waiting area (wait). Politely stepping aside so the baker can work (releasing the lock). Each time the baker pulls out a batch or flips the sign, they call out check again (notify). You wake up, re-check BOTH parts of your condition (wait_for re-tests the predicate), and either buy or go back to waiting. The lock guarantees that only one person inspects/modifies the bread shelf at a time.

WHY DO I NEED THIS IN ASYNCIO
-----------------------------
Event alone says something happened. It cannot say the buffer now has an item for ME. In producer/consumer scenarios, consumers must wait for len(buffer) greater-than 0 and producers for len(buffer) less-than max. Predicates on shared state that many tasks mutate. Every await between checking the state and acting on it would be a race condition without the lock. And without wait/notify, tasks would have to busy-poll. Conditions give you both halves of the solution in one primitive.

TERMS DEFINED
-------------
- Predicate: a boolean function tested against the shared state. For example lambda: len(buffer) > 0.
- Producer/consumer: the classic pattern where some tasks make items and others use them, coordinated through a bounded buffer.
- notify vs notify_all: wake one waiter vs wake every waiter. Woken waiters re-check the predicate before proceeding. So waking too many is safe (just slightly less efficient).

MINIMAL EXAMPLE
---------------
::

    import asyncio
    from asyncio_mastery.synchronization import AsyncCondition

    cond = AsyncCondition()
    items = []

    async def consumer():
        async with cond:
            await cond.wait_for(lambda: len(items) > 0)  # sleep until item
            return items.pop()

    async def producer():
        async with cond:
            items.append("bread")
            cond.notify()        # tell one waiter check again!

    asyncio.run(asyncio.gather(consumer(), producer()))

Run this file directly to see condition patterns in action:

    python -m asyncio_mastery.synchronization.conditions
"""

import asyncio
from asyncio import Condition
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class AsyncCondition:
    """An asyncio Condition with monitoring for learning.

    Wraps asyncio.Condition (a lock + wait/notify channel) and adds a name plus counters of waits and notifications.

    IMPORTANT: you must hold the condition lock before calling wait, notify, or notify_all. The easiest way is async with cond.

    Example:
        ::

            cond = AsyncCondition(name="buffer")

            async with cond:
                await cond.wait_for(lambda: len(buffer) > 0)
                item = buffer.pop()
    """

    def __init__(self, lock: asyncio.Lock | None = None, *, name: str | None = None) -> None:
        """Create a condition, optionally over an existing lock.

        Args:
            lock: The lock guarding the shared state. If None, a fresh asyncio.Lock is created (the common case).
            name: Optional human-readable name for debugging and logging.
        """
        self._condition = Condition(lock)
        self._name = name or "condition"
        self._wait_count = 0
        self._notify_count = 0

    @property
    def name(self) -> str:
        """The human-readable name of this condition."""
        return self._name

    @property
    def lock(self) -> asyncio.Lock:
        """The underlying lock that guards the shared state."""
        return self._condition.lock

    async def acquire(self) -> None:
        """Acquire the underlying lock (must be held before wait/notify)."""
        await self._condition.acquire()

    def release(self) -> None:
        """Release the underlying lock."""
        self._condition.release()

    async def __aenter__(self) -> "AsyncCondition":
        """Enter async with. Acquire the underlying lock."""
        await self.acquire()
        return self

    async def __aexit__(self, *args: object) -> None:
        """Exit async with. Release the lock even if the body raised."""
        self.release()

    async def wait(self) -> bool:
        """Sleep until another task calls notify/notify_all.

        While waiting, this ATOMICALLY releases the lock so other tasks can enter and change state. Then re-acquires it before returning. You almost always want wait_for (with a predicate) instead of a bare wait. Because being woken does NOT guarantee your condition is true. Another waiter may have consumed the item first.

        Returns:
            True when woken and the lock re-acquired.

        Raises:
            RuntimeError: If called without holding the lock.
        """
        self._wait_count += 1
        return await self._condition.wait()

    async def wait_for(self, predicate: Callable[[], bool]) -> bool:
        """Sleep until predicate() returns True (checked with lock held).

        This is the recommended pattern. It loops wait internally, re-testing the predicate after every wakeup. So you can never proceed on false hope.

        Args:
            predicate: Zero-argument callable evaluated under the lock. For example lambda: len(buffer) > 0.

        Returns:
            True (the predicate is guaranteed true on return, lock held).

        Example:
            ::

                async with cond:
                    await cond.wait_for(lambda: shop_open and bread_fresh)
                    buy_bread()
        """
        self._wait_count += 1
        return await self._condition.wait_for(predicate)

    def notify(self, n: int = 1) -> None:
        """Wake up to n waiting tasks (check again!).

        Woken tasks re-acquire the lock before continuing. So they run one at a time. Notify does not break mutual exclusion. You must hold the lock to call this.

        Args:
            n: How many waiters to wake (default 1). Choose greater-than 1 when the state change can satisfy several waiters.

        Raises:
            RuntimeError: If called without holding the lock.
        """
        self._condition.notify(n)
        self._notify_count += n

    def notify_all(self) -> None:
        """Wake every waiting task.

        Safe default when unsure how many waiters the change can satisfy. Each woken task re-checks its predicate under the lock. So extra wakeups cost a little CPU but never corrupt state.

        Raises:
            RuntimeError: If called without holding the lock.
        """
        self._condition.notify_all()
        self._notify_count += 1  # Approximate


class ConditionExample:
    """Producer/consumer over a bounded buffer, coordinated by a Condition.

    The canonical condition-variable problem:

    - Producers must wait while the buffer is FULL.
    - Consumers must wait while the buffer is EMPTY.
    - Each side notifies the other after changing the buffer.

    Example:
        ::

            pc = ConditionExample(max_size=3)
            await asyncio.gather(pc.produce_many(5), pc.consume_many(5))
    """

    def __init__(self, max_size: int = 5) -> None:
        """Create a bounded buffer.

        Args:
            max_size: Maximum items the buffer can hold. Producers wait once the buffer reaches this size.
        """
        self._buffer: list[object] = []
        self._max_size = max_size
        self._condition = AsyncCondition(name="buffer")
        self._produced = 0
        self._consumed = 0

    async def produce(self, item: object) -> None:
        """Put an item in the buffer, waiting if it is full.

        Args:
            item: Anything to enqueue for consumers.
        """
        async with self._condition:
            # Sleep (releasing the lock) until there is space. The lock is re-acquired automatically each time the predicate is checked.
            await self._condition.wait_for(lambda: len(self._buffer) < self._max_size)
            self._buffer.append(item)
            self._produced += 1
            print(f"Produced: {item} (buffer: {len(self._buffer)})")
            # Tell a potentially sleeping consumer: there is an item now.
            self._condition.notify()

    async def consume(self) -> object:
        """Take an item from the buffer, waiting if it is empty.

        Returns:
            The oldest item in the buffer.
        """
        async with self._condition:
            # Sleep (releasing the lock) until there is Something to eat.
            # Without wait_for, two consumers could both wake for one item and the second would pop from an empty buffer. A classic race.
            await self._condition.wait_for(lambda: len(self._buffer) > 0)
            item = self._buffer.pop(0)
            self._consumed += 1
            print(f"Consumed: {item} (buffer: {len(self._buffer)})")
            # Tell a potentially sleeping producer: there is space now.
            self._condition.notify()
            return item

    async def produce_many(self, count: int) -> None:
        """Produce count items with a small delay between each.

        Args:
            count: Number of items to produce.
        """
        for i in range(count):
            await self.produce(f"item-{i}")
            await asyncio.sleep(0.01)

    async def consume_many(self, count: int) -> list[object]:
        """Consume count items with a small delay between each.

        Args:
            count: Number of items to consume.

        Returns:
            The consumed items, in order.
        """
        results = []
        for _ in range(count):
            results.append(await self.consume())
            await asyncio.sleep(0.01)
        return results


async def demonstrate_conditions() -> None:
    """Run a narrated tour of common condition patterns.

    Demonstrates:
      1. wait_for with a simple boolean predicate (the bakery analogy).
      2. The full bounded-buffer producer/consumer dance.
      3. notify_all waking several workers competing for data.
      4. Waiting with a timeout.

    Returns:
        None. Prints demonstration output to stdout.
    """
    print("=== Basic Condition ===")
    condition = AsyncCondition(name="basic")
    ready = False

    async def waiter() -> None:
        async with condition:
            print("Waiter: waiting for condition")
            # Suspends, releasing the lock so notifier can enter and set ready=True. Re-checks ready on every wakeup.
            await condition.wait_for(lambda: ready)
            print("Waiter: condition met!")

    async def notifier() -> None:
        await asyncio.sleep(0.1)
        nonlocal ready
        # Change the shared state FIRST, then notify under the lock. Flipping ready without the lock could be missed between a waiter predicate check and its sleep (the lost wakeup race).
        ready = True
        async with condition:
            print("Notifier: notifying")
            condition.notify_all()

    await asyncio.gather(waiter(), notifier())

    print("\n=== Producer-Consumer ===")
    pc = ConditionExample(max_size=3)

    # Producer and consumer run concurrently. The condition keeps the buffer bounded at 3 and ensures neither side pops/appends blindly.
    await asyncio.gather(pc.produce_many(5), pc.consume_many(5))
    print(f"Produced: {pc._produced}, Consumed: {pc._consumed}")

    print("\n=== Condition with Multiple Waiters ===")
    cond = AsyncCondition(name="multi")
    data: list[int] = []

    async def worker(id: int) -> None:
        async with cond:
            await cond.wait_for(lambda: len(data) > 0)
            value = data.pop()
            print(f"Worker {id} got {value}")

    async def filler() -> None:
        await asyncio.sleep(0.05)
        async with cond:
            data.extend(range(5))
            print("Filler: added data, notifying all")
            # notify_all because 5 waiters can all be satisfied (5 items).
            cond.notify_all()

    await asyncio.gather(*[worker(i) for i in range(5)], filler())

    print("\n=== Wait with Timeout ===")
    cond2 = AsyncCondition(name="timeout")

    async def timeout_waiter() -> None:
        async with cond2:
            try:
                # Nobody notifies cond2, so wait_for cancels the wait.
                await asyncio.wait_for(cond2.wait(), timeout=0.1)
                print("Waiter: notified")
            except TimeoutError:
                print("Waiter: timeout!")

    await timeout_waiter()


if __name__ == "__main__":
    asyncio.run(demonstrate_conditions())
