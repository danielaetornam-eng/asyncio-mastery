"""
Event Bus

Distributed event bus with pub/sub, event sourcing, and replay capabilities.
"""

import asyncio
import json
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

T = TypeVar("T")


class EventPriority(Enum):
    LOW = 0
    NORMAL = 50
    HIGH = 100
    CRITICAL = 200


@dataclass
class Event:
    """Event representation."""

    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    source: str | None = None
    correlation_id: str | None = None
    causation_id: str | None = None
    priority: EventPriority = EventPriority.NORMAL

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "type": self.type,
            "payload": self.payload,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "source": self.source,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "priority": self.priority.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            type=data["type"],
            payload=data.get("payload", {}),
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", time.time()),
            source=data.get("source"),
            correlation_id=data.get("correlation_id"),
            causation_id=data.get("causation_id"),
            priority=EventPriority(data.get("priority", 50)),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "Event":
        return cls.from_dict(json.loads(json_str))


class EventHandler(ABC):
    """Base event handler."""

    @abstractmethod
    async def handle(self, event: Event) -> None:
        pass

    @property
    @abstractmethod
    def event_types(self) -> list[str]:
        pass


class EventFilter:
    """Filter for event subscription."""

    def __init__(
        self,
        event_types: list[str] | None = None,
        source: str | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> None:
        self._event_types = set(event_types) if event_types else None
        self._source = source
        self._metadata_filters = metadata_filters or {}

    def matches(self, event: Event) -> bool:
        if self._event_types and event.type not in self._event_types:
            return False

        if self._source and event.source != self._source:
            return False

        for key, value in self._metadata_filters.items():
            if event.metadata.get(key) != value:
                return False

        return True


class Subscription:
    """Event subscription."""

    def __init__(
        self,
        subscription_id: str,
        handler: Callable[[Event], Awaitable[None]],
        filter: EventFilter | None = None,
        priority: int = 0,
    ) -> None:
        self._subscription_id = subscription_id
        self._handler = handler
        self._filter = filter
        self._priority = priority
        self._active = True
        self._created_at = time.time()
        self._stats = {"events_received": 0, "events_processed": 0, "errors": 0, "total_time": 0.0}

    @property
    def subscription_id(self) -> str:
        return self._subscription_id

    @property
    def active(self) -> bool:
        return self._active

    @property
    def priority(self) -> int:
        return self._priority

    def matches(self, event: Event) -> bool:
        if not self._active:
            return False
        if self._filter:
            return self._filter.matches(event)
        return True

    async def handle(self, event: Event) -> None:
        if not self.matches(event):
            return

        self._stats["events_received"] += 1
        start = time.time()

        try:
            await self._handler(event)
            self._stats["events_processed"] += 1
        except Exception:
            self._stats["errors"] += 1
            raise
        finally:
            self._stats["total_time"] += time.time() - start

    def cancel(self) -> None:
        self._active = False

    def get_stats(self) -> dict[str, Any]:
        return {
            "subscription_id": self._subscription_id,
            "active": self._active,
            "priority": self._priority,
            "created_at": self._created_at,
            **self._stats.copy(),
        }


class EventStore:
    """Event store for persistence and replay."""

    def __init__(self, max_events: int = 100000) -> None:
        self._max_events = max_events
        self._events: list[Event] = []
        self._index: dict[str, list[int]] = defaultdict(list)  # type -> indices
        self._lock = asyncio.Lock()

    async def append(self, event: Event) -> None:
        async with self._lock:
            index = len(self._events)
            self._events.append(event)
            self._index[event.type].append(index)

            # Trim if exceeding max
            if len(self._events) > self._max_events:
                removed = len(self._events) - self._max_events
                self._events = self._events[removed:]

                # Rebuild index
                self._index.clear()
                for i, event in enumerate(self._events):
                    self._index[event.type].append(i)

    async def get_events(
        self,
        event_type: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[Event]:
        async with self._lock:
            if event_type:
                indices = self._index.get(event_type, [])
                events = [self._events[i] for i in indices]
            else:
                events = self._events[:]

            # Apply time filters
            if start_time:
                events = [e for e in events if e.timestamp >= start_time]
            if end_time:
                events = [e for e in events if e.timestamp <= end_time]

            # Apply pagination
            events = events[offset : offset + limit]

            return events

    async def get_latest(self, event_type: str, count: int = 1) -> list[Event]:
        async with self._lock:
            indices = self._index.get(event_type, [])
            if not indices:
                return []

            indices = indices[-count:]
            return [self._events[i] for i in reversed(indices)]

    async def count(self, event_type: str | None = None) -> int:
        async with self._lock:
            if event_type:
                return len(self._index.get(event_type, []))
            return len(self._events)

    async def clear(self) -> None:
        async with self._lock:
            self._events.clear()
            self._index.clear()


class EventBus:
    """Async event bus with pub/sub, filtering, and event store."""

    def __init__(
        self,
        *,
        enable_store: bool = True,
        max_store_events: int = 100000,
        max_concurrent_handlers: int = 100,
    ) -> None:
        self._subscriptions: dict[str, Subscription] = {}
        self._subscriptions_by_type: dict[str, set[str]] = defaultdict(set)
        self._store = EventStore(max_store_events) if enable_store else None
        self._running = False
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrent_handlers)

        # Stats
        self._stats = {"published": 0, "delivered": 0, "failed": 0, "active_subscriptions": 0}

    @property
    def running(self) -> bool:
        return self._running

    @property
    def store(self) -> EventStore | None:
        return self._store

    def stats(self) -> dict[str, Any]:
        return {
            **self._stats.copy(),
            "total_subscriptions": len(self._subscriptions),
            "store_enabled": self._store is not None,
        }

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False
        # Cancel all subscriptions
        for sub in self._subscriptions.values():
            sub.cancel()
        self._subscriptions.clear()
        self._subscriptions_by_type.clear()

    def subscribe(
        self,
        handler: Callable[[Event], Awaitable[None]],
        event_types: list[str] | None = None,
        filter: EventFilter | None = None,
        priority: int = 0,
    ) -> str:
        """Subscribe to events."""
        if filter is None and event_types:
            filter = EventFilter(event_types=event_types)

        sub = Subscription(
            subscription_id=str(uuid.uuid4())[:8], handler=handler, filter=filter, priority=priority
        )

        self._subscriptions[sub.subscription_id] = sub

        # Index by event type
        if filter and filter._event_types:
            for et in filter._event_types:
                self._subscriptions_by_type[et].add(sub.subscription_id)
        elif not filter:
            # Subscribe to all events
            self._subscriptions_by_type["*"].add(sub.subscription_id)

        self._stats["active_subscriptions"] += 1
        return sub.subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events."""
        sub = self._subscriptions.pop(subscription_id, None)
        if sub:
            sub.cancel()

            # Remove from indexes
            for type_set in self._subscriptions_by_type.values():
                type_set.discard(subscription_id)

            self._stats["active_subscriptions"] -= 1
            return True
        return False

    async def publish(self, event: Event, *, persist: bool = True) -> int:
        """Publish event to all matching subscribers."""
        if not self._running:
            raise RuntimeError("Event bus not running")

        self._stats["published"] += 1

        # Store event
        if persist and self._store:
            await self._store.append(event)

        # Find matching subscriptions
        matching = set()

        # Exact type matches
        type_subs = self._subscriptions_by_type.get(event.type, set())
        matching.update(type_subs)

        # Wildcard subscriptions
        wildcard_subs = self._subscriptions_by_type.get("*", set())
        matching.update(wildcard_subs)

        # Execute handlers
        delivered = 0
        tasks = []

        for sub_id in matching:
            sub = self._subscriptions.get(sub_id)
            if sub and sub.matches(event):
                task = asyncio.create_task(self._execute_handler(sub, event))
                tasks.append(task)

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    self._stats["failed"] += 1
                else:
                    delivered += 1
                    self._stats["delivered"] += 1

        return delivered

    async def _execute_handler(self, sub: Subscription, event: Event) -> None:
        async with self._semaphore:
            await sub.handle(event)

    async def publish_and_wait(
        self, event: Event, timeout: float = 30.0, *, persist: bool = True
    ) -> int:
        """Publish event and wait for all handlers to complete."""
        return await asyncio.wait_for(self.publish(event, persist=persist), timeout=timeout)

    def get_subscription(self, subscription_id: str) -> Subscription | None:
        return self._subscriptions.get(subscription_id)

    def list_subscriptions(self) -> list[Subscription]:
        return list(self._subscriptions.values())

    async def replay(
        self,
        event_types: list[str] | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        handler: Callable[[Event], Awaitable[None]] | None = None,
    ) -> int:
        """Replay events from store."""
        if not self._store:
            raise RuntimeError("Event store not enabled")

        events = await self._store.get_events(
            event_type=event_types[0] if event_types and len(event_types) == 1 else None,
            start_time=start_time,
            end_time=end_time,
        )

        if event_types and len(event_types) > 1:
            events = [e for e in events if e.type in event_types]

        if handler:
            for event in events:
                await handler(event)

        return len(events)

    @asynccontextmanager
    async def subscription(
        self,
        handler: Callable[[Event], Awaitable[None]],
        event_types: list[str] | None = None,
        filter: EventFilter | None = None,
        priority: int = 0,
    ) -> AsyncGenerator[str, None]:
        """Context manager for temporary subscription."""
        sub_id = self.subscribe(handler, event_types, filter, priority)
        try:
            yield sub_id
        finally:
            self.unsubscribe(sub_id)

    async def __aenter__(self) -> "EventBus":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()


class DistributedEventBus(EventBus):
    """Event bus with distributed support (Redis, Kafka, etc.)."""

    def __init__(
        self, *, redis_url: str | None = None, kafka_bootstrap: str | None = None, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self._redis_url = redis_url
        self._kafka_bootstrap = kafka_bootstrap
        self._redis = None
        self._kafka_producer = None
        self._kafka_consumer = None

    async def start(self) -> None:
        await super().start()

        if self._redis_url:
            try:
                import aioredis

                self._redis = await aioredis.create_redis_pool(self._redis_url)
            except ImportError:
                pass

        if self._kafka_bootstrap:
            try:
                from aiokafka import AIOKafkaProducer

                self._kafka_producer = AIOKafkaProducer(bootstrap_servers=self._kafka_bootstrap)
                await self._kafka_producer.start()
            except ImportError:
                pass

    async def stop(self) -> None:
        if self._redis:
            self._redis.close()
            await self._redis.wait_closed()

        if self._kafka_producer:
            await self._kafka_producer.stop()

        await super().stop()

    async def publish(self, event: Event, *, persist: bool = True) -> int:
        # Local delivery
        delivered = await super().publish(event, persist=persist)

        # Distributed delivery
        if self._redis:
            with suppress(Exception):
                await self._redis.publish(f"events:{event.type}", event.to_json())

        if self._kafka_producer:
            with suppress(Exception):
                await self._kafka_producer.send_and_wait(
                    f"events.{event.type}", event.to_json().encode()
                )

        return delivered


def create_event_bus(**kwargs) -> EventBus:
    """Factory for creating event bus."""
    return EventBus(**kwargs)


# Example usage
async def example_usage() -> None:
    """Example event bus usage."""
    bus = create_event_bus(enable_store=True)

    # Subscribe to events
    async def handle_user_created(event: Event) -> None:
        print(f"User created: {event.payload}")

    async def handle_order(event: Event) -> None:
        print(f"Order event: {event.type} - {event.payload}")

    bus.subscribe(handle_user_created, event_types=["user.created"])
    bus.subscribe(handle_order, event_types=["order.created", "order.updated"])

    # Subscribe to all events with filter
    def all_events_filter(event: Event) -> bool:
        return event.priority == EventPriority.CRITICAL

    bus.subscribe(
        lambda e: print(f"CRITICAL: {e.type}"),
        filter=EventFilter(metadata_filters={"priority": "critical"}),
    )

    async with bus:
        # Publish events
        await bus.publish(
            Event(
                type="user.created",
                payload={"user_id": "123", "name": "John"},
                source="user-service",
            )
        )

        await bus.publish(
            Event(
                type="order.created",
                payload={"order_id": "456", "total": 100.0},
                source="order-service",
                correlation_id="corr-123",
            )
        )

        # Replay events
        count = await bus.replay(event_types=["user.created"])
        print(f"Replayed {count} events")

        # Wait a bit for handlers
        await asyncio.sleep(1)
