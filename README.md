# Asyncio Mastery

This Python codebase covers asyncio concepts from basics to advanced patterns.

## Overview

This project is a reference for the Python `asyncio` library. It covers:

- **Basics**: Coroutines, Tasks, Futures, Event Loop
- **Synchronization**: Locks, Semaphores, Events, Conditions, Barriers
- **Streams**: TCP, UDP, SSL networking with readers and writers
- **Subprocesses**: Async process management
- **Advanced**: Context Variables, Async Generators, Async Iterators, Timeouts, Cancellation, Shielding
- **Patterns**: Connection Pooling, Rate Limiting, Retry Logic, Circuit Breakers, Pub/Sub, Worker Pools, Batching
- **Testing**: Async test utilities, time mocking, async mocks
- **Integrations**: Threading, Multiprocessing, aiohttp, Databases

## Installation

Run this command:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import asyncio
from asyncio_mastery.basics import demonstrate_coroutines
from asyncio_mastery.patterns import RateLimiter


async def main():
    # Basic coroutine usage (demo from basics/coroutines.py)
    await demonstrate_coroutines()

    # Production pattern: limit traffic to 100 operations per second.
    limiter = RateLimiter(rate=100)
    async with limiter:
        result = await some_operation()  # runs when permit is available

    print(result)


asyncio.run(main())
```

Then run the examples. See `docs/getting_started.md`. Each example is a guided lesson.

## Module Structure

```
src/asyncio_mastery/
├── basics/           # Core asyncio fundamentals
├── synchronization/  # Locks, semaphores, events, conditions
├── streams/          # Network I/O (TCP, UDP, SSL)
├── subprocess/       # Async process management
├── advanced/         # Context vars, generators, cancellation
├── patterns/         # Production patterns
├── testing/          # Test utilities
├── integrations/     # Threading, aiohttp, databases
└── utils/            # Decorators, helpers, logging
```

## Examples

Run the examples to see concepts in action:

```bash
python -m examples.basic_usage
python -m examples.tcp_echo_server
python -m examples.worker_pool_demo
python -m examples.patterns_demo
```

## Testing

Run these commands:

```bash
pytest tests/ -v
pytest tests/ --cov=src/asyncio_mastery
```

## Documentation

If you are new to asyncio, start here in order:

- **`docs/getting_started.md`** — installation, first async program, running examples and tests
- **`docs/concepts.md`** — beginner guide to every concept with analogies and glossary
- **`docs/patterns.md`** — production-patterns API guide

Every module under `src/asyncio_mastery/` has inline documentation. The module docstrings are lessons with analogies and minimal runnable examples.

## Learning Path

1. **Beginner**: Start with `basics/`. Understand coroutines, tasks, and the event loop.
2. **Intermediate**: Move to `synchronization/` and `streams/`. Learn coordination and networking.
3. **Advanced**: Study `advanced/`. Learn context vars, generators, cancellation.
4. **Expert**: Master `patterns/`. Learn production patterns for real applications.
5. **Integration**: See `integrations/`. Combine asyncio with threads, databases, HTTP.

## Key Features

- **Typed**: Full type hints with mypy strict mode
- **Tested**: Complete test suite with pytest-asyncio
- **Production**: Patterns used in large systems
- **Educational**: Docstrings and examples
- **Modern**: Uses Python 3.11+ features (TaskGroups, timeout())

## License

MIT License. See LICENSE file for details.