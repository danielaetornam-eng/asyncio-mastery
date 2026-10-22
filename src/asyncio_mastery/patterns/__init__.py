"""
Patterns - production async design patterns, beginner edition.

This package provides one module per classic async pattern. Each module
contains a small, readable class that you can use and learn from. Every
module ends with a demonstrate function that you can run to see the
pattern in action.

The patterns solve recurring problems. Think of them as recipes:

* async_context_manager - Guaranteed setup and teardown for resources.
  Use async with to make sure cleanup even if you abandon the operation
  halfway.
* connection_pool - Keep a set of open connections and lend them out.
  Like a taxi fleet: cheaper than building a new taxi per trip.
* rate_limiter - Cap how fast operations happen. Token bucket or
  sliding window algorithms act like a bouncer letting N people in per
  minute.
* retry - Automatically re-attempt flaky operations with exponential
  backoff. Each retry waits longer, giving the other side time to
  recover.
* circuit_breaker - Stop calling a failing service entirely for a
  while. Like a circuit breaker in your house: it trips to prevent
  damage and resets after a cooldown.
* task_group - Spawn and supervise many tasks as one unit so none are
  orphaned.
* worker_pool - A fixed crew of workers pulling jobs from a queue.
  Like chefs on a line, each grabbing the next order ticket.
* producer_consumer - Decouple makers from takers with a queue buffer
  between them, so each side runs at its own pace.
* semaphore_pool - Limit how many tasks may hold a resource at once.
  N slots; extra tasks wait their turn.
* pubsub - Publish and subscribe. Senders broadcast to a named topic.
  Every subscriber receives a copy. Like a group chat.
* batching - Collect many small items and process them together. Like a
  mail truck: cheaper than one trip per letter.

Suggested order:

Start with async_context_manager for lifecycle thinking. Then
semaphore_pool and rate_limiter for concurrency control. Then retry and
circuit_breaker for resilience. Then the queue-based patterns:
producer_consumer, worker_pool, pubsub, batching, and task_group.
"""

from asyncio_mastery.patterns.async_context_manager import (
    AsyncContextManager,
    async_context_manager,
    demonstrate_async_context_manager,
)
from asyncio_mastery.patterns.batching import (
    BatchConfig,
    Batcher,
    BatchProcessor,
    demonstrate_batching,
)
from asyncio_mastery.patterns.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitState,
    demonstrate_circuit_breaker,
)
from asyncio_mastery.patterns.connection_pool import (
    ConnectionPool,
    PoolConfig,
    PooledConnection,
    demonstrate_connection_pool,
)
from asyncio_mastery.patterns.producer_consumer import (
    ProducerConsumer,
    demonstrate_producer_consumer,
)
from asyncio_mastery.patterns.pubsub import PubSub, Topic, demonstrate_pubsub
from asyncio_mastery.patterns.rate_limiter import (
    RateLimiter,
    SlidingWindow,
    TokenBucket,
    demonstrate_rate_limiter,
)
from asyncio_mastery.patterns.retry import (
    ExponentialBackoff,
    RetryConfig,
    RetryPolicy,
    demonstrate_retry,
)
from asyncio_mastery.patterns.semaphore_pool import SemaphorePool, demonstrate_semaphore_pool
from asyncio_mastery.patterns.task_group import TaskGroup, TaskGroupResult, demonstrate_task_group
from asyncio_mastery.patterns.worker_pool import Job, WorkerPool, demonstrate_worker_pool

__all__ = [
    "AsyncContextManager",
    "BatchConfig",
    "BatchProcessor",
    "Batcher",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerOpenError",
    "CircuitState",
    "ConnectionPool",
    "PoolConfig",
    "ExponentialBackoff",
    "Job",
    "PooledConnection",
    "ProducerConsumer",
    "PubSub",
    "RateLimiter",
    "RetryConfig",
    "RetryPolicy",
    "SemaphorePool",
    "SlidingWindow",
    "TaskGroup",
    "TaskGroupResult",
    "TokenBucket",
    "Topic",
    "WorkerPool",
    "async_context_manager",
    "demonstrate_async_context_manager",
    "demonstrate_batching",
    "demonstrate_circuit_breaker",
    "demonstrate_connection_pool",
    "demonstrate_producer_consumer",
    "demonstrate_pubsub",
    "demonstrate_rate_limiter",
    "demonstrate_retry",
    "demonstrate_semaphore_pool",
    "demonstrate_task_group",
    "demonstrate_worker_pool",
]
