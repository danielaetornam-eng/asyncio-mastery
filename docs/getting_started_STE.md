# Getting Started with Asyncio Mastery

This guide helps you install the project, run your first asyncio program, and find the next file to read. You must know basic Python.

---

## 1. Installation

Do this step from the project root:

```bash
pip install -e ".[dev]"
```

This command installs the library in editable mode. Changes to `src/` take effect immediately. The command also installs the development tools that tests need (pytest, pytest-asyncio).

**Requirements**: Use Python 3.11 or later. Python 3.11 improved asyncio (`asyncio.timeout`, stronger `TaskGroup` semantics). The codebase uses these improvements.

---

## 2. The Hello World of asyncio

Create a file with this code:

```python
import asyncio


async def hello(name: str) -> None:
    print(f"{name}: starting")
    await asyncio.sleep(1)
    print(f"{name}: done")


async def main() -> None:
    await asyncio.gather(hello("Alice"), hello("Bob"))


asyncio.run(main())
```

Run the file. Both "starting" lines print immediately. Both "done" lines appear together after one second. The interleaving at `await` is the core behavior of asyncio.

See `src/asyncio_mastery/basics/__init__.py` and `asyncio_mastery/basics/coroutines.py` for the fully-commented version.

---

## 3. Running the Examples

The `examples/` folder contains runnable demonstrations. Each demonstration is self-contained.

Run a demonstration with one of these commands:

```bash
python -m examples.basic_usage
python -m examples.patterns_demo
python -m examples.worker_pool_demo
python -m examples.tcp_echo_server
python -m examples.api_gateway
python -m examples.database_pool
python -m examples.event_bus
python -m examples.http_client
python -m examples.task_scheduler
python -m examples.websocket
```

Read the source code of an example while it runs. The docstrings explain what you see.

---

## 4. Running the Tests

The tests are executable specifications of feature behavior.

Run the tests with one of these commands:

```bash
pytest tests/ -v
pytest tests/test_basics.py
pytest tests/ --cov=src/asyncio_mastery
```

---

## 5. What to Read Next

1. Open `docs/concepts.md`. This file is the concept guide with analogies and gotchas.
2. Read the source code in this order. Treat each module docstring as a lesson:
   `src/asyncio_mastery/basics/`, `synchronization/`, `streams/`, `subprocess/`, `advanced/`, `patterns/`, `testing/`, `integrations/`, `utils/`.
3. When you reach patterns, keep `docs/patterns.md` open as the API reference.
4. Re-run the relevant example after each module. Then try to modify the example.

---

## 6. Troubleshooting

**WARNING**: Do not call an `async def` function without `await`. The call only creates the coroutine. It does not run it.

**CAUTION**: Do not use `asyncio.get_running_loop()` or `create_task()` from synchronous code. All async code must start from one `asyncio.run(main())`.

**NOTE**: If tests or examples hang, a task is waiting on a queue or lock forever. Check for a missing `drain()`, `set()`, or `put()`, or a forgotten `await`.