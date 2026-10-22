"""
Events - Simple one-way signaling between tasks.

WHAT IS AN EVENT
----------------
An Event is a shared boolean flag (set / not set) that tasks can wait on. One or more tasks call await event.wait() and suspend until some other task calls event.set(). Unlike a Lock or Semaphore, an Event carries no ownership and no count. It is just has it happened yet.

KEY OPERATIONS
--------------
- set: raise the flag. Every currently waiting task wakes up. Any task that calls wait afterwards returns immediately.
- clear: lower the flag. New calls to wait will block again.
- wait: suspend until the flag is up.

ANALOGY: THE STARTING GUN
-------------------------
Runners at a race stand at the line (waiters calling wait). They cannot start until the starter fires the gun (set). The moment the gun fires, ALL runners start at the same time. Events wake ALL waiters, not just one. If another runner arrives late after the gun already fired, they can also start immediately. Because the flag stays up until someone calls clear (resetting the race for the next round).

WHY DO I NEED THIS IN ASYNCIO
-----------------------------
Async tasks often need to say X is ready now to other tasks. A config file is loaded. A connection is established. A shutdown is requested. Without an event, waiting tasks would have to poll in a loop with sleeps (is it ready yet? how about now?). This wastes CPU and adds latency. With an event, waiters sleep efficiently inside await. The event loop does not schedule them at all until set wakes them instantly and simultaneously.

Remember: this matters even on one thread. Because the task that waits and the task that signals only interleave via the event loop. Without a signaling primitive there is no clean way for one to pause until the other acts.

TERMS DEFINED
-------------
- Signaling: one task notifying others that something happened.
- Polling: repeatedly checking a condition in a loop. The wasteful alternative to events.
- Spurious wakeup protection: not needed here. Once set, wait() always returns. The flag is the source of truth. (Contrast with Conditions.)

MINIMAL EXAMPLE
---------------
::

    import asyncio
    from asyncio_mastery.synchronization import AsyncEvent

    ready = AsyncEvent()

    async def worker():
        await ready.wait()        # stand at the starting line
        print("go!")

    async def starter():
        await asyncio.sleep(1)
        ready.set()               # fire the gun

    asyncio.run(asyncio.gather(worker(), worker(), starter()))

Run this file directly to see event patterns in action:

    python -m asyncio_mastery.synchronization.events
"""

import asyncio
from asyncio import Event
from collections.abc import Awaitable
from typing import TypeVar

T = TypeVar("T")


class AsyncEvent:
    """An asyncio event with instrumentation for learning.

    Wraps asyncio.Event and adds a name, counters of set/clear calls, and a live set of currently waiting tasks. Useful for debugging is anyone actually waiting on this.

    Example:
        ::

            event = AsyncEvent(name="startup-done")

            async def service():
                await event.wait()   # sleep until someone calls set()
                ...
    """

    def __init__(self, *, name: str | None = None) -> None:
        """Create a new (initially unset/cleared) event.

        Args:
            name: Optional human-readable name for debugging and logging.
        """
        self._event = Event()
        self._name = name or "event"
        self._set_count = 0
        self._clear_count = 0
        self._waiters: set[asyncio.Task] = set()

    @property
    def name(self) -> str:
        """The human-readable name of this event."""
        return self._name

    @property
    def is_set(self) -> bool:
        """True if the flag is currently raised (wait returns immediately)."""
        return self._event.is_set()

    def set(self) -> None:
        """Raise the flag, waking ALL current and future waiters.

        Unlike Lock.release (which wakes one waiter), set wakes every task currently blocked in wait. Any task calling wait afterwards proceeds immediately. Until clear lowers the flag again.
        """
        self._event.set()
        self._set_count += 1

    def clear(self) -> None:
        """Lower the flag so future wait calls block again.

        Tasks already waiting are NOT re-suspended (they were woken when the flag was raised). This only affects subsequent wait calls. Use it to reset the race for the next round.
        """
        self._event.clear()
        self._clear_count += 1

    async def wait(self) -> bool:
        """Suspend until the flag is raised.

        If the flag is already set, returns immediately without yielding. Otherwise this coroutine parks at the await and the event loop runs other tasks until someone calls set.

        Returns:
            True once the event has been set.
        """
        task = asyncio.current_task()
        if task:
            self._waiters.add(task)
        try:
            # If the flag is down, THIS is the line where the task goes to sleep. Consuming zero CPU until set wakes it.
            return await self._event.wait()
        finally:
            if task:
                self._waiters.discard(task)

    def wait_timeout(self, timeout: float) -> Awaitable[bool]:
        """Build a wait that gives up after timeout seconds.

        Args:
            timeout: Maximum seconds to wait for the flag.

        Returns:
            An awaitable that resolves to True if the event was set, or raises TimeoutError if the deadline passes first.

        Example:
            ::

                try:
                    await event.wait_timeout(5.0)
                except TimeoutError:
                    print("nobody fired the gun in 5 seconds")
        """
        return asyncio.wait_for(self.wait(), timeout=timeout)


class EventNotifier:
    """A small publish/subscribe hub built on named events.

    Lets producers signal named topics and consumers wait for them without either side knowing about the other::

        notifier = EventNotifier()
        await notifier.wait_for("data-ready")    # consumer
        await notifier.notify("data-ready")      # producer (in another task)

    Events are created lazily by name the first time they are used.

    Note: an event is a flag, not a queue. It tells waiters THAT something happened, not WHAT. To pass data per notification, pair an event with an asyncio.Queue.
    """

    def __init__(self) -> None:
        """Create an empty notifier with no topics yet."""
        self._events: dict[str, AsyncEvent] = {}
        self._lock = asyncio.Lock()

    def get_event(self, name: str) -> AsyncEvent:
        """Get (creating if needed) the named event/topic.

        Args:
            name: Topic name, for example data-ready.

        Returns:
            The AsyncEvent for that topic.
        """
        if name not in self._events:
            self._events[name] = AsyncEvent(name=name)
        return self._events[name]

    async def notify(self, name: str, data: object = None) -> int:
        """Fire a topic, waking every task waiting on it.

        Args:
            name: Topic name to signal.
            data: Optional payload. NOTE: events are flags and do not deliver data. This parameter is a placeholder for a real queue-based implementation.

        Returns:
            The number of tasks that were waiting on the topic.
        """
        event = self.get_event(name)
        # set() wakes ALL subscribers for this topic simultaneously.
        event.set()
        if data is not None:
            # Store data for waiters (in real impl, use queue)
            pass
        return len(event._waiters)

    async def wait_for(self, name: str, timeout: float | None = None) -> bool:
        """Wait until a topic is notified.

        Args:
            name: Topic name to subscribe to.
            timeout: Optional maximum seconds to wait. On expiry, returns False instead of raising.

        Returns:
            True if the topic was notified, False on timeout.
        """
        event = self.get_event(name)
        if timeout:
            try:
                return await asyncio.wait_for(event.wait(), timeout=timeout)
            except TimeoutError:
                # Convert the exception into a plain False so callers can write if await notifier.wait_for(...): naturally.
                return False
        return await event.wait()

    def clear(self, name: str) -> None:
        """Lower a topic flag so future subscribers wait again.

        Args:
            name: Topic name to reset (no-op if it was never used).
        """
        if name in self._events:
            self._events[name].clear()


async def demonstrate_events() -> None:
    """Run a narrated tour of common event patterns.

    Demonstrates:
      1. Basic wait/set: two waiters woken by one setter (the starting gun).
      2. set/clear cycles to reuse one event repeatedly.
      3. Waiting with a timeout.
      4. A named-topic publish/subscribe hub (EventNotifier).
      5. One set() waking many waiters simultaneously.

    Returns:
        None. Prints demonstration output to stdout.
    """
    print("=== Basic Event ===")
    event = AsyncEvent(name="basic")

    async def waiter(id: int) -> None:
        print(f"Waiter {id} waiting...")
        # Task suspends HERE until setter() calls event.set().
        await event.wait()
        print(f"Waiter {id} got signal!")

    async def setter() -> None:
        await asyncio.sleep(0.1)
        print("Setter: setting event!")
        # Fires the starting gun: both waiters wake in the same moment.
        event.set()

    await asyncio.gather(waiter(1), waiter(2), setter())

    print("\n=== Event Clear ===")
    event2 = AsyncEvent(name="clear")

    async def cyclic_waiter() -> None:
        for i in range(3):
            await event2.wait()
            print(f"Cycle {i}: got signal")
            # Lower the flag so the NEXT iteration waits again. Resetting the race for the next round.
            event2.clear()
            await asyncio.sleep(0.01)

    async def cyclic_setter() -> None:
        for i in range(3):
            await asyncio.sleep(0.05)
            print(f"Cycle {i}: setting")
            event2.set()

    await asyncio.gather(cyclic_waiter(), cyclic_setter())

    print("\n=== Event with Timeout ===")
    event3 = AsyncEvent(name="timeout")

    async def timeout_waiter() -> None:
        try:
            # Nobody ever sets event3, so wait_for cancels us after 0.1s.
            await asyncio.wait_for(event3.wait(), timeout=0.1)
            print("Got event in time")
        except TimeoutError:
            print("Timed out waiting!")

    await timeout_waiter()

    print("\n=== EventNotifier ===")
    notifier = EventNotifier()

    async def subscriber(topic: str) -> None:
        print(f"Subscriber waiting for {topic}")
        await notifier.wait_for(topic, timeout=1.0)
        print(f"Subscriber got {topic}")

    async def publisher() -> None:
        await asyncio.sleep(0.1)
        await notifier.notify("news", "Breaking news!")
        await asyncio.sleep(0.05)
        await notifier.notify("updates", "Update available!")

    await asyncio.gather(subscriber("news"), subscriber("updates"), publisher())

    print("\n=== Multiple Waiters ===")
    event4 = AsyncEvent(name="multi")
    results = []

    async def multi_waiter(id: int) -> None:
        await event4.wait()
        results.append(id)

    waiters = [multi_waiter(i) for i in range(10)]
    # One set() releases all ten waiters at once. Broadcast signaling.
    await asyncio.gather(*waiters, event4.set())
    print(f"All waiters notified: {results}")


if __name__ == "__main__":
    asyncio.run(demonstrate_events())
