"""
Pub/Sub - Publish-subscribe pattern for async messaging.

THE PROBLEM
-----------
In a growing application, components need to react to events ("user
signed up", "file uploaded") without the *producer* of the event knowing
who cares. Hard-wiring callers to handlers creates a tangled web of
imports and fragile coupling: add a new listener and you must edit the
caller's code.

REAL-WORLD ANALOGY
------------------
A **mailing list / neighborhood newsletter**. The editor (publisher)
prints one issue and drops a copy in every subscriber's mailbox. The
editor does not know or care how many subscribers there are; subscribers
join and leave freely. The "topic" here is which newsletter you
subscribed to.

WHY ASYNC PRIMITIVES ARE NEEDED
-------------------------------
Subscribers are async handlers (send an email, write to a DB), and there
may be many on one topic. ``asyncio.gather`` delivers one message to all
subscribers *concurrently* so a slow handler does not stall the others.
An :class:`asyncio.Lock` guards the subscriber registry because
subscriptions can be added/removed while publishes are in flight;
we take a *copy* of the handler list before delivering so iteration
is not affected by mid-delivery changes.

WHEN TO USE IT
--------------
* Event-driven architectures: decouple emitters from reactors.
* Fan-out notifications: one event → email + metrics + audit log.
* In-process messaging between independently developed components.

Caveats: this broker is in-memory and best-effort — no persistence,
no delivery retry, no ordering guarantees across topics. For
production cross-process messaging look at Redis Streams, NATS,
Kafka, etc.

Minimal example
---------------
>>> import asyncio
>>> async def on_event(m) -> None:
...     print(f"{m.topic}: {m.payload}")
>>> async def main() -> None:
...     bus = PubSub()
...     unsub = await bus.subscribe("users", on_event)
...     await bus.publish("users", "alice signed up")
...     await unsub()
>>> asyncio.run(main())
"""

import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass
class Message(Generic[T]):
    """Message envelope.

    Carries the payload plus routing info (topic) and metadata
    (timestamp, headers) — think of it as the stamped envelope around
    your letter.

    Type Parameters:
        T: The payload type.

    Attributes:
        topic: Which channel this message was published on.
        payload: The actual data.
        timestamp: ``time.monotonic()`` when the message was created
            (monotonic clocks never jump backward, so they are ideal for
            ordering/durations — though not for wall-clock display).
        headers: Free-form string metadata (e.g. sender id, trace id).
    """

    topic: str
    payload: T
    timestamp: float = field(default_factory=__import__("time").monotonic)
    headers: dict[str, str] = field(default_factory=dict)


class Subscriber(ABC, Generic[T]):
    """Abstract subscriber.

    Alternative to passing a bare async function: subclass this and
    implement :meth:`on_message` — handy when the handler needs state.

    Example:
        >>> class AuditLog(Subscriber[str]):
        ...     async def on_message(self, message: Message[str]) -> None:
        ...         print("AUDIT:", message.payload)
    """

    @abstractmethod
    async def on_message(self, message: Message[T]) -> None:
        """Handle one message.

        Args:
            message: The full envelope (topic, payload, timestamp,
                headers) — not just the raw payload.
        """
        pass


class PubSub:
    """Publish-subscribe broker.

    Keeps a registry mapping topic names to handler lists. Publishing
    wraps the payload in a :class:`Message`, archives it in a bounded
    history log, then concurrently invokes every handler subscribed to
    that topic.

    Attributes:
        _subscribers: topic → list of handler coroutine callables.
        _lock: serializes registry mutations vs. reads during publish.
        _message_log: Bounded history of every published message.
        _max_log_size: Cap on history length (oldest evicted first).

    Example:
        >>> bus = PubSub()
        >>> await bus.subscribe("orders", handle_order)
        >>> count = await bus.publish("orders", order, source="web")
    """

    def __init__(self) -> None:
        """Create an empty broker with no subscribers and empty history."""
        self._subscribers: dict[str, list[Callable[[Message[Any]], Awaitable[None]]]] = defaultdict(
            list
        )
        self._lock = asyncio.Lock()
        self._message_log: list[Message[Any]] = []
        self._max_log_size = 1000

    async def subscribe(
        self, topic: str, handler: Callable[[Message[Any]], Awaitable[None]]
    ) -> Callable[[], Awaitable[None]]:
        """Subscribe to topic.

        Args:
            topic: Topic name to listen to (e.g. "events", "orders").
            handler: Async callable receiving a :class:`Message` for
                every future publish on that topic.

        Returns:
            Callable: An async ``unsubscribe()`` function — call it to
            stop receiving messages. Using this returned handle (rather
            than re-passing the handler) is the clean way to detach.

        Example:
            >>> unsubscribe = await bus.subscribe("alerts", on_alert)
            >>> await unsubscribe()  # stop listening
        """
        async with self._lock:
            self._subscribers[topic].append(handler)

        async def unsubscribe() -> None:
            async with self._lock:
                if handler in self._subscribers[topic]:
                    self._subscribers[topic].remove(handler)

        return unsubscribe

    async def publish(self, topic: str, payload: Any, **headers: str) -> int:
        """Publish message to topic.

        Wraps the payload in a Message, logs it to history, then calls
        every subscriber handler concurrently. Handler exceptions are
        *contained* via ``gather(..., return_exceptions=True)``: one
        misbehaving subscriber never blocks or kills the others (nor the
        publisher).

        Args:
            topic: Channel to publish on.
            payload: The message data.
            **headers: Optional string metadata (user="alice", etc.).

        Returns:
            int: How many handlers the message was delivered to.
        """
        message = Message(topic=topic, payload=payload, headers=headers)
        self._message_log.append(message)
        if len(self._message_log) > self._max_log_size:
            self._message_log.pop(0)

        async with self._lock:
            # Copy under the lock: a handler might (un)subscribe during
            # delivery, and mutating the list mid-iteration is a classic
            # bug source. The copy snapshots "who gets this message".
            handlers = self._subscribers.get(topic, []).copy()

        if not handlers:
            return 0

        # Deliver to all subscribers concurrently
        await asyncio.gather(*[handler(message) for handler in handlers], return_exceptions=True)
        return len(handlers)

    async def publish_many(self, messages: list[Message[Any]]) -> int:
        """Publish multiple messages.

        Args:
            messages: Messages (each already knows its topic/headers).

        Returns:
            int: Total handler deliveries across all messages.
        """
        return sum(await self.publish(m.topic, m.payload, **m.headers) for m in messages)

    def get_history(self, topic: str | None = None, limit: int = 100) -> list[Message[Any]]:
        """Get message history.

        Args:
            topic: Filter to a single topic; None gives all topics.
            limit: Maximum number of most-recent messages to return.

        Returns:
            list[Message]: The newest ``limit`` matching messages.
        """
        if topic:
            return [m for m in self._message_log if m.topic == topic][-limit:]
        return self._message_log[-limit:]


class Topic:
    """Typed topic for publish/subscribe.

    A thin convenience handle bound to one topic name on a broker —
    saves you from re-typing (and typo-ing) the topic string.

    Example:
        >>> orders = Topic(bus, "orders")
        >>> await orders.subscribe(handle)
        >>> await orders.publish({"id": 42})
    """

    def __init__(self, pubsub: PubSub, name: str) -> None:
        """Bind to a topic on a broker.

        Args:
            pubsub: The broker to route through.
            name: The topic name.
        """
        self._pubsub = pubsub
        self._name = name

    @property
    def name(self) -> str:
        """The topic name.

        Returns:
            str: Name passed at construction.
        """
        return self._name

    async def publish(self, payload: Any, **headers: str) -> int:
        """Publish to this topic (delegates to :meth:`PubSub.publish`).

        Args:
            payload: The message data.
            **headers: Optional string metadata.

        Returns:
            int: Number of handlers reached.
        """
        return await self._pubsub.publish(self._name, payload, **headers)

    async def subscribe(
        self, handler: Callable[[Message[Any]], Awaitable[None]]
    ) -> Callable[[], Awaitable[None]]:
        """Subscribe a handler to this topic.

        Args:
            handler: Async callable receiving each :class:`Message`.

        Returns:
            Callable: Async ``unsubscribe()`` handle.
        """
        return await self._pubsub.subscribe(self._name, handler)


@asynccontextmanager
async def subscription(
    pubsub: PubSub, topic: str, handler: Callable[[Message[Any]], Awaitable[None]]
):
    """Context manager for subscription.

    Guarantees unsubscription on exit — no lingering handlers even if
    the block raises::

        async with subscription(bus, "events", handler):
            ...  # receive messages here
        # automatically unsubscribed here

    Args:
        pubsub: The broker.
        topic: Topic to subscribe to.
        handler: Async handler for each message.

    Yields:
        None: The handler receives messages for the block's duration.
    """
    unsubscribe = await pubsub.subscribe(topic, handler)
    try:
        yield
    finally:
        await unsubscribe()


async def demonstrate_pubsub() -> None:
    """Run a self-contained demo of the pub/sub broker.

    Covers: subscribing, publishing with headers, unsubscription, the
    history log, and the :class:`Topic` convenience wrapper.
    """
    print("=== Pub/Sub Demo ===")

    pubsub = PubSub()
    received: list[str] = []

    async def handler(message: Message[str]) -> None:
        received.append(f"{message.topic}: {message.payload}")
        print(f"  Received: {message.topic} -> {message.payload}")

    # Subscribe
    unsubscribe = await pubsub.subscribe("events", handler)
    await pubsub.subscribe("notifications", handler)

    # Publish
    await pubsub.publish("events", "user_login", user="alice")
    await pubsub.publish("notifications", "welcome_email")
    await pubsub.publish("events", "user_logout", user="bob")

    # Unsubscribe and publish again
    await unsubscribe()
    await pubsub.publish("events", "should_not_receive")

    print(f"\nTotal received: {len(received)}")
    print(f"History: {pubsub.get_history('events')}")

    # Typed topics
    print("\n--- Typed Topics ---")
    events_topic = Topic(pubsub, "typed_events")
    received2: list[str] = []

    await events_topic.subscribe(lambda m: received2.append(m.payload))
    await events_topic.publish("typed message 1")
    await events_topic.publish("typed message 2")

    print(f"Typed received: {received2}")


if __name__ == "__main__":
    asyncio.run(demonstrate_pubsub())
