# Asyncio Mastery - Production Patterns Guide

This document details the production patterns available in the codebase.

## How to Read This Guide

Each pattern is a reusable solution to a recurring problem in async programs. For every pattern you will find:

1. **Purpose** — what it does.
2. **The Problem It Solves** — why writing this by hand goes wrong.
3. **Analogy** — a real-world mental picture.
4. **A Runnable Example** using the real API (verified against `src/asyncio_mastery/patterns/`).
5. **Key Features** — what to tweak to fit your system.

If you are new to asyncio, first read `docs/concepts.md` and the beginner-friendly module docstrings under `src/asyncio_mastery/`.

## Pattern Catalog

### 1. ConnectionPool

**Purpose**: Manage reusable connections with lifecycle management.

**The Problem**: Opening a database or TCP connection is expensive. Network round trips and authentication take time. Opening one per request is slow. It overloads the server. Sharing one global connection causes requests to queue on each other.

**Analogy**: A taxi rank at the airport. You do not buy a car per trip. You hire one and return it for the next passenger. If the rank is empty you wait. If it is full, old taxis are sent home.

```python
from asyncio_mastery.patterns import ConnectionPool
from asyncio_mastery.patterns.connection_pool import ConnectionFactory


class MyConnectionFactory(ConnectionFactory[MyConnection]):
    async def create(self) -> MyConnection:
        return await MyConnection.connect()

    async def close(self, conn: MyConnection) -> None:
        await conn.close()

    async def health_check(self, conn: MyConnection) -> bool:
        return await conn.ping()


factory = MyConnectionFactory()
pool = ConnectionPool(factory, min_size=5, max_size=20)
await pool.start()

async with pool.connection() as conn:
    result = await conn.query("SELECT 1")

await pool.close()
```

**Key Features**:
- Automatic connection creation/validation
- Health checks for stale connections
- Configurable min/max pool size
- Acquire timeout and idle timeout
- Statistics tracking

---

### 2. RateLimiter

**Purpose**: Control request rates to prevent overload.

**The Problem**: Third-party APIs ban you if you burst too many requests; your own database melts under a load spike. You need a throttle that smooths traffic.

**Analogy**: A turnstile at a stadium. It lets N people in per minute. Early arrivals may burst through a little (if you stockpile tokens). The long-run rate is capped.

```python
from asyncio_mastery.patterns import RateLimiter

# Token bucket (smooth with burst)
limiter = RateLimiter(rate=100, burst=20)

# Or sliding window (precise)
limiter = RateLimiter(rate=100, algorithm="sliding_window")

async with limiter:
    await api_call()

# Or try without blocking
if limiter.try_acquire():
    await api_call()
else:
    # Handle rejection
    pass
```

**Algorithms**:
- **Token Bucket**: Allows burst, smooths over time
- **Sliding Window**: Precise rate limiting

---

### 3. RetryPolicy

**Purpose**: Automatically retry failed operations with backoff.

**The Problem**: Networks hiccup. Servers hiccup harder. Manually sprinkling `try/except` + `await sleep` loops everywhere is error-prone. You may forget to wait *longer* after each failure. This can hammer a struggling service.

**Analogy**: Knocking on a door with no answer. You knock again. You wait a little longer each time (1s, 2s, 4s...) before giving up. *Jitter* adds random wobble. This prevents hundreds of clients from retrying at the same instant (the "thundering herd" problem).

```python
from asyncio_mastery.patterns import RetryPolicy, RetryConfig, ExponentialBackoff

# Default exponential backoff, up to 3 attempts total.
policy = RetryPolicy(RetryConfig(max_attempts=3, base_delay=1.0))

# Custom backoff (delays: 0.5s, 1s, 2s, 4s... capped at 30s, with jitter)
policy = RetryPolicy(
    RetryConfig(max_attempts=5),
    backoff=ExponentialBackoff(base_delay=0.5, max_delay=30.0, jitter=True),
)

# Retry only specific exceptions; stop immediately on others.
policy = RetryPolicy(RetryConfig(
    max_attempts=3,
    retry_on=(ConnectionError, TimeoutError),
    stop_on=(ValueError,),
))

result = await policy.execute(flaky_api_call)
```

**Backoff Strategies**:
- **ExponentialBackoff**: 1s, 2s, 4s, 8s... with optional jitter
- **LinearBackoff**: 1s, 2s, 3s, 4s...
- **ConstantBackoff**: Fixed delay

---

### 4. CircuitBreaker

**Purpose**: Prevent cascade failures by stopping calls to failing services.

**The Problem**: A downstream service is down. Calling it wastes time. Each request waits for a timeout. Queues fill up. Your service starts failing too. This is a cascade.

**Analogy**: The breaker in your electrical panel. When a circuit keeps shorting, the breaker trips. Power cuts so nothing burns down. After a cooldown, one careful check decides whether to flip it back on.

```python
from asyncio_mastery.patterns import CircuitBreaker, CircuitBreakerConfig

config = CircuitBreakerConfig(
    failure_threshold=5,  # Open after 5 failures
    success_threshold=2,  # Close after 2 successes in half-open
    timeout=30.0,  # Try half-open after 30s
    excluded_exceptions=(ValueError,),  # Do not count these
)

breaker = CircuitBreaker("payment-service", config)

try:
    result = await breaker.call(payment_api.charge, amount=100)
except CircuitBreakerOpenError as e:
    # Fast fail - service likely down
    fallback()
```

**State Transitions**:
```
CLOSED --(5 failures)--> OPEN --(30s)--> HALF_OPEN --(2 successes)--> CLOSED
     ^                                                                      |
     |--(success)----------------------------------------------------------|
```

---

### 5. TaskGroup

**Purpose**: Structured concurrency with automatic cleanup.

**The Problem**: Bare `asyncio.create_task()` can leak tasks. If the parent fails or returns, background tasks float orphaned. They silently run or silently die with unlogged exceptions.

**Analogy**: A team meeting. Nobody leaves until every member completes. If the coach blows the whistle (an error), everyone stops at once.

```python
from asyncio_mastery.patterns import TaskGroup

async with TaskGroup() as group:
    for url in urls:
        group.create_task(fetch_url(url))

# All tasks completed or cancelled
results = group.result.results
exceptions = group.result.exceptions
```

**Benefits**:
- Automatic cancellation on exception
- No orphaned tasks
- Clean error aggregation

---

### 6. WorkerPool

**Purpose**: Process jobs with controlled concurrency.

**The Problem**: You have 10,000 items to process. Firing all as tasks at once exhausts memory and API quotas. Doing them one at a time is slow. You need a fixed team of workers pulling from a queue.

**Analogy**: A call center. Calls (jobs) queue up. A fixed number of operators (workers) pick up the next call as they free up. A full queue tells callers to wait (backpressure).

```python
from asyncio_mastery.patterns import worker_pool


async def process_item(item: dict) -> Result:
    # CPU or I/O work
    return await do_work(item)


async with worker_pool(process_item, max_workers=4) as pool:
    # Submit individual jobs
    job = await pool.submit(item)
    result = await job.execute()

    # Or process many
    results = await pool.map(items)
```

**Features**:
- Bounded queue with backpressure
- Job tracking with status
- Statistics (submitted, completed, failed)

---

### 7. PubSub

**Purpose**: Decoupled event messaging.

**The Problem**: Component A wants to announce "user logged in" without knowing or caring which components listen. Direct function calls couple everything into spaghetti.

**Analogy**: A newsletter subscription. The publisher sends one issue. The mailman delivers a copy to every subscriber listed for that topic. Publisher and subscribers never know each other.

```python
from asyncio_mastery.patterns import PubSub, Topic

pubsub = PubSub()


# Subscribe
async def handler(message):
    print(f"Got: {message.payload}")


unsubscribe = await pubsub.subscribe("events", handler)

# Publish
await pubsub.publish("events", {"type": "user_login", "user": "alice"})

# Typed topics
events = Topic(pubsub, "events")
await events.publish("message")
```

**Features**:
- Multiple subscribers per topic
- Message history for debugging
- Context manager for subscriptions

---

### 8. Batcher

**Purpose**: Accumulate items for efficient batch processing.

**The Problem**: Inserting 1,000 rows one `INSERT` each = 1,000 network round trips. Databases, cloud APIs, and billing systems all reward bulk calls — but collecting items into batches by hand is fiddly.

**Analogy**: A hotel shuttle bus. It leaves when it has N (max_size) passengers or when the schedule (max_wait) expires, whichever comes first. Each passenger is dropped at their own stop (per-item result).

```python
from asyncio_mastery.patterns import Batcher, BatchConfig


async def process_batch(items: list[Item]) -> list[Result]:
    # Single DB query, API call, etc.
    return await bulk_process(items)


batcher = Batcher(process_batch, BatchConfig(max_size=100, max_wait=0.1))

# Items batched automatically
results = await asyncio.gather(*[batcher.add(item) for item in items])

# Or force flush
await batcher.flush()
```

**Configuration**:
- `max_size`: Maximum batch size
- `max_wait`: Maximum time to wait for batch

---

### 9. SemaphorePool

**Purpose**: Manage multiple named semaphores for different resources.

**The Problem**: One app talks to a database (limit 10 concurrent), Redis (50), and a slow third-party API (5). Managing N separate `Semaphore` objects by hand is clutter; acquiring two at once naively risks deadlock.

**Analogy**: A gym with separate racks for each machine type. Only 10 treadmill-users at a time. 50 locker-holders. 5 squash courts. One front desk hands out tokens per section.

```python
from asyncio_mastery.patterns import SemaphorePool

pool = SemaphorePool()
pool.create("database", 10)
pool.create("redis", 50)
pool.create("external_api", 5)

# Individual
async with pool.acquire("database"):
    await db.query()

# Multiple (atomic acquire)
async with pool.acquire_many("database", "redis"):
    await db.query()
    await redis.get()
```

---

### 10. AsyncContextManager

**Purpose**: Base class for async resource management.

**The Problem**: Files, connections, and pools need setup and guaranteed teardown. Both operations may themselves need to await (e.g. closing a connection cleanly). Plain `with` cannot await. Forgetting `close()` leaks resources.

**Analogy**: A library book. You check it out (`__aenter__`). No matter how your reading went (even if a fire drill interrupted you), you check it back in (`__aexit__`).

```python
from asyncio_mastery.patterns import AsyncContextManager


class DatabaseConnection(AsyncContextManager[Connection]):
    async def _initialize(self) -> Connection:
        return await Connection.connect()

    async def _close(self) -> None:
        await self._resource.close()


async with DatabaseConnection() as conn:
    await conn.query()
# Automatically closed
```

---

### 11. ProducerConsumer

**Purpose**: Classic pattern with managed lifecycle.

**The Problem**: One part of your system discovers work (a crawler finds pages). Another part does the work (downloads pages). They work at different speeds. Connecting them directly stalls the faster side.

**Analogy**: A restaurant. Chefs (producers) plate dishes onto an order rail. Waiters (consumers) grab plates as they become free. The rail (a bounded queue) smooths the speed difference: full rail → chefs wait; empty rail → waiters wait. The context manager guarantees all worker tasks are cleanly shut down.

```python
from asyncio_mastery.patterns import producer_consumer


async def process(event: Event) -> None:
    await handle_event(event)


async with producer_consumer(process, max_queue_size=100, num_consumers=4) as pc:
    await pc.produce_many(events)
    # Or non-blocking
    pc.try_produce(event)
```

---

## Combining Patterns

### Resilient HTTP Client

```python
from asyncio_mastery.patterns import (
    CircuitBreaker, CircuitBreakerConfig, RetryPolicy, RetryConfig,
    RateLimiter, ConnectionPool, PoolConfig,
)


class ResilientHttpClient:
    def __init__(self):
        self.limiter = RateLimiter(rate=100)
        self.breaker = CircuitBreaker("api", CircuitBreakerConfig(failure_threshold=5))
        self.retry = RetryPolicy(RetryConfig(max_attempts=3))
        self.pool = ConnectionPool(HttpConnectionFactory(), PoolConfig(max_size=20))

    async def request(self, method: str, url: str) -> Response:
        async with self.limiter:
            return await self.breaker.call(
                lambda: self.retry.execute(lambda: self.pool.connection().request(method, url))
            )
```

### Background Job Processor

```python
from asyncio_mastery.patterns import WorkerPool, Batcher, PubSub


class JobProcessor:
    def __init__(self):
        self.pubsub = PubSub()
        self.batcher = Batcher(self.bulk_save, max_size=100, max_wait=1.0)
        self.workers = WorkerPool(self.process_job, max_workers=8)

    async def submit(self, job: Job) -> None:
        await self.pubsub.publish("jobs", job)

    async def process_job(self, job: Job) -> Result:
        result = await do_work(job)
        await self.batcher.add(result)
        return result

    async def bulk_save(self, results: list[Result]) -> list[Saved]:
        return await db.bulk_insert(results)
```

---

## Best Practices

1. **Always use context managers** for automatic cleanup
2. **Configure timeouts** for all external calls
3. **Monitor statistics** for capacity planning
4. **Use structured logging** with context
5. **Test failure scenarios** with mock time and async mocks
6. **Combine patterns** for defense in depth

---

## Running Examples

```bash
# Basic patterns
python -m examples.patterns_demo

# Worker pool
python -m examples.worker_pool_demo

# TCP networking
python -m examples.tcp_echo_server
```