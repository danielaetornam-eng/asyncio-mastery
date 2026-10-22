# Asyncio Mastery — Concepts Guide (for Beginners)

This guide explains every asyncio concept in this codebase in simple language. No prior concurrency knowledge is necessary. After you read this guide you will understand what the code in each `src/asyncio_mastery/` subpackage does. You will understand why it exists. You will understand how to read it.

**How to use this guide:** Read the "Big Picture" first. Then follow the sections in the order of the learning path at the bottom. Every section ends with the module path. Open the matching source file. The modules have many comments. The modules are lessons.

---

## Table of Contents

1. [The Big Picture](#the-big-picture)
2. [Basics](#basics)
3. [Synchronization](#synchronization)
4. [Streams and Networking](#streams-and-networking)
5. [Subprocesses](#subprocesses)
6. [Advanced Concepts](#advanced-concepts)
7. [Production Patterns](#production-patterns)
8. [Testing](#testing)
9. [Integrations](#integrations)
10. [Utilities](#utilities)
11. [Learning Path](#learning-path)
12. [Glossary](#glossary)

---

## The Big Picture

### What problem does asyncio solve?

Imagine you are a chef. You cook dinner alone. Some steps are active work. Example: chop onions. Other steps are waiting. Example: water boils. Bread bakes. A chef who stares at the pot is synchronous. Only one thing happens at a time. A chef who puts the water on, then chops onions, then checks the pot is asynchronous. While one thing waits, another makes progress.

Programs face the same choice. Code waits for a network response. Code waits for a file on disk. A synchronous program sits there. An asynchronous program says "wake me when the data arrives." The program does other work in the meantime.

### What is the event loop?

The **event loop** manages this cooperation. Think of a waiter. The waiter walks between many tables. The waiter gives attention to whoever is ready to order. One table reads the menu (waiting). The waiter serves drinks elsewhere (makes progress). The waiter is a single thread. Many tables feel served at the same time.

### The three building blocks

A **coroutine** (`async def`) is a recipe. It is instructions that can pause at marked points.

An **await** is a pause point. It means: "I wait for something. You may serve others now."

A **Task** is the recipe given to the waiter. It turns a coroutine into scheduled work. The work runs at the same time as other tasks.

```python
import asyncio

async def make_tea() -> str:
    print("Kettle on...")
    await asyncio.sleep(3)
    return "Tea is ready"

asyncio.run(make_tea())
```

### Why only one thread, yet still "concurrent"?

Asyncio is **cooperative** single-threaded concurrency. Tasks never run at the same instant. They interleave at `await` points. This makes asyncio bugs rare. Between `await` points, your code owns the thread. It also makes bugs subtle. If you do not `await`, nobody else runs. A blocking call like `time.sleep()` freezes everyone.

---

## Basics

### Coroutines (`src/asyncio_mastery/basics/coroutines.py`)

- **What it is**: A function defined with `async def`. Calling it does NOT run it. It returns a "coroutine object" (a recipe, not a meal). Running happens via `await`, `asyncio.run()`, or `asyncio.create_task()`.
- **Mental model**: A coroutine is a videotape. You play it. You pause it (at every `await`). You resume it. The event loop owns the play and pause buttons.

```python
from asyncio_mastery.basics.coroutines import coroutine_with_args

result = await coroutine_with_args("Alice", delay=0.1)
```

**Gotcha for beginners**: Calling `simple_coroutine()` without `await` produces a warning ("coroutine was never awaited"). Nothing runs. If your function "does not do anything," you probably forgot `await`.

### Tasks (`src/asyncio_mastery/basics/tasks.py`)

- **What it is**: `asyncio.create_task(coro)` schedules a coroutine on the event loop. It runs "in the background" (concurrently, not in parallel threads).
- **Why it matters**: Plain `await` is sequential. One coroutine at a time. Tasks let N coroutines make progress during each other's waits.
- **Key tools**: `asyncio.gather(*tasks)` (wait for all). `asyncio.wait(...)` (fine-grained control). `task.cancel()` (abort a task).

```python
task = asyncio.create_task(fetch_page("https://example.com"))
other_work()
page = await task
```

**Gotcha**: `create_task()` only *schedules*. The task does not start until you hit an `await` that lets the event loop run it. A bare `create_task(...)` followed by CPU-heavy synchronous code means the task never gets a turn.

### Futures (`src/asyncio_mastery/basics/futures.py`)

- **What it is**: A low-level "promise" object. It represents a result that does not exist yet. Someone else fills it in later via `future.set_result(value)`.
- **Mental model**: A future is an empty box. You hand out empty boxes. The kitchen puts food in when ready. The person holding the box can `await` it.
- **Relationship**: `Task` is a subclass of `Future`. Tasks drive coroutines. Plain futures are manual signalling or result slots.

```python
future = loop.create_future()
loop.call_later(1.0, future.set_result, "one second passed")
value = await future
```

### Event Loop (`src/asyncio_mastery/basics/event_loop.py`)

- **What it is**: The engine that schedules callbacks and coroutines. One per thread. Normally you never create it yourself. `asyncio.run()` does.
- **Key APIs**: `loop.time()` (the loop's monotonic clock). `loop.call_later(delay, fn)` (schedule a callback). `loop.run_in_executor(None, blocking_fn)` (offload sync work).
- **Mental model**: The loop keeps two lists. "Run now" (ready). "Run later" (timers). It cycles through them forever until `asyncio.run` stops it.

**Gotcha**: `asyncio.get_event_loop()` is legacy. Inside async code always use `asyncio.get_running_loop()`. The difference: "running" fails loudly if you are not in async code. This is a helpful error, not a helpful silence.

---

## Synchronization

Single-threaded does not mean safe! Every `await` is a point where another task may touch your shared data. Synchronization primitives are the traffic rules of this interleaving.

### Lock (`synchronization/locks.py`)

- **Analogy**: A bathroom key. Only one task may hold it. Others wait.
- **Use it for**: Protecting read-modify-write sequences. Example: `value = shared; value += 1; shared = value`. Two tasks doing this concurrently can lose updates.

```python
async with lock:
    counter += 1
```

### Semaphore (`synchronization/semaphores.py`)

- **Analogy**: A parking lot with N spots. Cars (tasks) enter if a spot is free. Otherwise they queue. When a car leaves, the next in line enters.
- **Use it for**: Bounding concurrency. Example: "At most 10 downloads at once."

```python
semaphore = asyncio.Semaphore(10)
async with semaphore:
    await download(url)
```

### Event (`synchronization/events.py`)

- **Analogy**: A starting gun at a race. Runners wait until `set()` fires. `clear()` resets for the next race.
- **Use it for**: One-to-many "you may proceed" signalling.

```python
await event.wait()
```

### Condition (`synchronization/conditions.py`)

- **Analogy**: Waiting for the shop to open AND fresh bread to arrive. You need to check a real condition, not just flip a switch.
- **Use it for**: Waiting *for a predicate*. Example: `await cond.wait_for(lambda: len(buf) > 0)`. Usually paired with `notify()` / `notify_all()` from the changing side.

### Barrier (`synchronization/barriers.py`)

- **Analogy**: Friends agreeing nobody enters the restaurant until everyone has arrived.
- **Use it for**: Phased parallel work. N tasks must all reach a checkpoint before any proceeds to the next phase.

**Important**: Always use `async with lock:` (or `try/finally`). If an exception escapes between `acquire()` and `release()`, the lock leaks. The next waiter hangs forever.

---

## Streams and Networking

Asyncio streams are the high-level way to do socket I/O. A **stream** is like a phone call. Two ends. Each has an ear (`StreamReader`) and a mouth (`StreamWriter`). Bytes flow asynchronously. Your task runs only when bytes are available or buffer space exists.

### StreamReader and StreamWriter (`streams/stream_reader.py`, `stream_writer.py`)

- **Reads are awaits**: `await reader.read(n)` suspends until n bytes arrive (or EOF).
- **Writes are fire-and-check**: `writer.write(data)` only buffers. `await writer.drain()` suspends if the send buffer is full. Skipping `drain()` can balloon memory on a slow connection. This "slow receivers slow you down" effect is **backpressure**.
- **EOF (end of file)**: When the other side hangs up, reads return `b""`. Always handle `b""`. Otherwise your server will loop forever on dead connections.

### TCP client and server (`streams/tcp_client.py`, `streams/tcp_server.py`)

- `asyncio.open_connection(host, port)` gives you a connected `(reader, writer)` pair.
- `asyncio.start_server(handler, host, port)` calls your `handler(reader, writer)` for every new client. Each handler runs as its own task. One server can chat with thousands of clients on one thread.

### SSL/TLS (`streams/ssl_context.py`)

- **Analogy**: TLS is a sealed, tamper-evident envelope. The certificate is the sender's ID card. It is checked against trusted authorities.
- Build an `ssl.SSLContext`. Pass it to `open_connection` / `start_server`.

---

## Subprocesses

Your Python program can launch other programs (example: `ffmpeg`, `git`). It can talk to them without blocking.

- **Analogy**: A kitchen with a second chef (the OS process). `stdin`, `stdout`, `stderr` are three hoses between you. Commands go in. Output and errors come out.
- **Key APIs** (`subprocess/create_subprocess.py`): `asyncio.create_subprocess_exec` / `create_subprocess_shell`. Then `await proc.communicate()` to send input and collect all output.
- **Deadlock warning**: If the child writes more than a screenful of output while you `await proc.wait()` instead of `communicate()`, the pipe buffers fill. The child blocks waiting for you to read. You block waiting for the child to exit. `communicate()` reads continuously and avoids this.
- **Signals** (`subprocess/signals.py`): The OS "knocks on the door" to request things. Examples: SIGINT (Ctrl+C), SIGTERM ("please exit"), SIGKILL (pull the plug). Asyncio can convert signals into task cancellation for graceful shutdown.

---

## Advanced Concepts

### Cancellation (`advanced/cancellation.py`)

- **Analogy**: Pulling the plug on a washing machine. It gets a moment (a `finally` block) to drain the water before going fully dark.
- **How it works**: `task.cancel()` throws `asyncio.CancelledError` into the task at its next `await`. The task may catch it to clean up. It **must re-raise** it. Swallowing cancellation is the async equivalent of ignoring a shutdown command.

```python
try:
    await long_operation()
except asyncio.CancelledError:
    await cleanup()
    raise
```

### Timeouts (`advanced/timeouts.py`)

- **Analogy**: Kitchen timers. "If the cake is not done in 30 minutes, take it out anyway." In Python 3.11+, `asyncio.timeout(seconds)` is a context manager. It cancels the block when time is up. `asyncio.wait_for(coro, seconds)` wraps a single awaitable.

### Context Variables (`advanced/context_vars.py`)

- **Analogy**: Each task carries its own keychain with ID tags. Global variables are one shared whiteboard everyone sees. Context variables are per-task notes. They travel with the task across every `await`.
- **Use it for**: Request IDs, tracing info, user sessions. Data that should follow the call chain implicitly instead of being passed through every function argument.

```python
request_id: ContextVar[str] = ContextVar("request_id")
request_id.set("req-42")
await handler()
```

### Async Generators (`advanced/async_generators.py`)

- **Analogy**: A conveyor belt that delivers items over time. You ask for the next item (`async for item in gen`). The event loop may wait between items.
- **How**: `async def` with `yield`. Under the hood: `__aiter__()` and `__anext__()`. The loop ends with `StopAsyncIteration`.

### Running Blocking Code (`advanced/run_in_executor.py`)

- **Analogy**: Delegating heavy chores to a worker next door. The event loop stays free while a thread (or process) does the blocking call.
- **Rule of thumb**: I/O that is truly async (aiohttp, asyncio streams) needs no executor. Blocking libraries (`time.sleep`, `requests`, many DB drivers) do: wrap them via `loop.run_in_executor(None, blocking_fn)`. Use a *process* pool only for CPU-heavy work. The GIL means threads share one Python core.

---

## Production Patterns

These compositions of primitives appear in every real-world asyncio system. Each lives in `src/asyncio_mastery/patterns/` with runnable demos. `docs/patterns.md` has the full API-reference guide.

- **ConnectionPool** — Reuse expensive connections (DB, HTTP) instead of reopening per call. Analogy: A taxi rank. Instead of buying a car per trip you hire one and return it.
- **RateLimiter** — Cap how many operations happen per second. Analogy: A turnstile at a stadium letting N people in per minute.
- **RetryPolicy** — Automatically retry flaky operations with exponential backoff (wait longer after each failure) plus jitter (random wiggle). This avoids the **thundering herd** of all clients retrying at the same instant.
- **CircuitBreaker** — If a service keeps failing, stop calling it for a while. Fail fast instead of piling up doomed requests. Analogy: An electrical breaker that trips on overload. States: CLOSED (normal) → OPEN (failing, calls rejected) → HALF_OPEN (careful probe) → back to CLOSED.
- **TaskGroup** — Structured concurrency: Start sibling tasks such that if one fails, the others are cancelled. Tasks do not become orphaned. Analogy: A team meeting. Nobody leaves until everyone completes or the coach ends the play.
- **WorkerPool** — N workers pull jobs from a queue with bounded concurrency and backpressure. Analogy: A call center with a fixed number of operators.
- **PubSub** — Decoupled "publish / subscribe" messaging within one process. Analogy: A newsletter. Publishers do not know subscribers. The mailman routes copies.
- **Batcher** — Collect many tiny items. Flush them as one batch (example: one bulk DB insert instead of 100 small ones). Analogy: A hotel shuttle that leaves when full or every N minutes, whichever comes first.
- **SemaphorePool** — Many named semaphores (one per resource). Atomic multi-acquire avoids deadlock.
- **ProducerConsumer** — One side produces work. Another consumes. A queue in the middle smooths speed differences. Analogy: Chefs and waiters with an order rail.
- **AsyncContextManager** — `async with` for resources needing asynchronous open/close. Guaranteed cleanup even on exceptions. Analogy: A library book you *check in* no matter how your reading went.

---

## Testing

Async code is non-deterministic (interleavings vary) and time-dependent. This makes tests flaky. `src/asyncio_mastery/testing/` provides:

- **`test_helpers.py`** — Utilities like async assertions with polling. Provides "wait up to X for condition to hold" instead of fragile `sleep + assert`.
- **`async_mock.py`** — Mocks that can be awaited. Fake a database/API method without hitting the network. (`AsyncMock` records calls and returns canned values.)
- **`mock_time.py`** — Replace the event loop's clock with a fake clock. You advance it manually. Test "wait 30 seconds" logic in microseconds, deterministically.

**Beginner tip**: Never use `time.sleep()` in an async test. It freezes the loop. Use `asyncio.sleep(0)` to yield a turn. Or use the fake clock for longer durations.

---

## Integrations

Asyncio cooperates with the wider Python world (`src/asyncio_mastery/integrations/`):

- **aiohttp** — The standard async HTTP client/server library. Key rule: Create one `ClientSession` per application and reuse it. Connection pooling is built in.
- **Databases** — Async DB drivers (example: `asyncpg`, `aiosqlite`) pair naturally with a `ConnectionPool`. Transactions show why async context managers matter (commit on success, rollback on error).
- **Threading** — A blocking call in a thread + `loop.run_in_executor` bridges the sync world into async land. A thread-safe queue bridges messages back.
- **Multiprocessing** — For CPU-bound work (resizing images, number crunching). The **GIL** (Global Interpreter Lock) means Python threads share one core for Python bytecode. Separate processes escape that limit. Pickling is needed for data crossing the boundary.

---

## Utilities

`src/asyncio_mastery/utils/` collects cross-cutting helpers:

- **`decorators.py`** — Reusable decorators (example: `@retry`, `@memoize`) adapted for async functions. Functools wraps metadata so debugging still shows the original name.
- **`helpers.py`** — Common combos: `gather` with a concurrency limit. Converting async results to sync entry points. Simple task groups. Iterators with `map/filter`.
- **`logging.py`** — Every task shares one thread. Log lines from interleaved tasks interleave too. Contextvars attach a per-task label (request ID) to each line. You can grep one request's story out of the noise.

---

## Learning Path

Read the source files in this order. Each module's docstring is the lesson. The code is the worked examples:

| Step | Read | Source |
|------|------|--------|
| 1 | Coroutines, Tasks, Futures, Event Loop | `src/asyncio_mastery/basics/` |
| 2 | Locks, Semaphores, Events, Conditions, Barriers | `src/asyncio_mastery/synchronization/` |
| 3 | TCP streams, SSL | `src/asyncio_mastery/streams/` |
| 4 | Subprocesses and signals | `src/asyncio_mastery/subprocess/` |
| 5 | Cancellation, Timeouts, ContextVars, Async Generators, Executors | `src/asyncio_mastery/advanced/` |
| 6 | All production patterns | `src/asyncio_mastery/patterns/` + `docs/patterns.md` |
| 7 | Testing async code | `src/asyncio_mastery/testing/` |
| 8 | Real-world integrations | `src/asyncio_mastery/integrations/` |
| 9 | Helpers | `src/asyncio_mastery/utils/` |

After reading each module, run the matching example in `examples/` (example: `python -m examples.basic_usage`).

---

## Glossary

- **Awaitable** — Anything legal after `await`: a coroutine, Task, or Future.
- **Backpressure** — The signal: "slow down, I cannot process as fast as you produce" (bounded queues, `drain()`, semaphores).
- **Backlog**, **TCP_NODELAY**, **keepalive** — TCP socket tuning. See `streams/` module docstrings.
- **CancelledError** — The exception thrown into a task by `task.cancel()`. Catch, clean up, re-raise it.
- **Context variable (ContextVar)** — A variable whose value is per-task. Travels through awaits implicitly.
- **Cooperative concurrency** — Tasks voluntarily yield at `await` points. Nothing preempts them between those points.
- **Coroutine** — An `async def` function. Pausable at `await` expressions.
- **Event loop** — The single-threaded scheduler running all tasks.
- **Future** — A placeholder for a result. Another task fills it in later.
- **GIL (Global Interpreter Lock)** — The lock making Python threads share one core for bytecode. Why CPU-bound scaling needs processes, not threads.
- **Jitter** — Small random variation added to retry delays. Prevents many waiters from colliding ("thundering herd").
- **Pickling** — Converting a Python object to bytes. Required to send data between processes.
- **Structured concurrency** — Tasks form a scoped tree. Children cannot outlive their parent scope (TaskGroup / `async with` style).
- **Task** — A coroutine scheduled on the event loop. The unit of concurrency in asyncio.
- **Thundering herd** — When many waiters are released at once and stampede the resource they were waiting for.

(End of file)