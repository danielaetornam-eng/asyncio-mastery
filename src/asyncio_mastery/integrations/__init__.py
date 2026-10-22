"""
Integrations - Connecting asyncio to the rest of the Python world.

WHY THIS SUBPACKAGE EXISTS (FOR A BEGINNER)
-------------------------------------------
asyncio shines at concurrent *waiting*, but real programs must talk to
things that do not speak ``async``/``await``: blocking libraries, heavy CPU
work, HTTP services, and databases. This subpackage shows the four classic
bridges, so you can keep one clean event loop at the center of your app:

- :mod:`asyncio_mastery.integrations.threading` -- run *blocking* code on
  worker threads and await the results (:class:`ThreadBridge`,
  :func:`run_in_thread`), run coroutines on threads (:class:`AsyncThread`),
  and pass data between the worlds (``AsyncQueue``, ``ThreadSafeEvent``).
- :mod:`asyncio_mastery.integrations.multiprocessing` -- run *CPU-bound*
  code in separate processes so the GIL cannot block you
  (:class:`ProcessBridge`, :func:`run_in_process`).
- :mod:`asyncio_mastery.integrations.aiohttp_example` -- async HTTP client
  and server with connection pooling via ``aiohttp``.
- :mod:`asyncio_mastery.integrations.database` -- async database access
  with connection pools and transactions (``aiosqlite`` / ``asyncpg``).

RULE OF THUMB
-------------
- Waiting on network/disk?  -> asyncio directly, or threads for blocking libs.
- Heavy pure-Python math?   -> multiprocessing.
- Talking HTTP or SQL?      -> aiohttp / async database drivers.

Note: the aiohttp and database modules degrade gracefully -- they import
fine without their optional third-party drivers and raise a clear error
only if you actually try to use them.

Each module also ships a ``demonstrate_*`` coroutine you can run to see the
concepts in action:

>>> import asyncio
>>> asyncio.run(demonstrate_threading())
"""

from asyncio_mastery.integrations.aiohttp_example import (
    AiohttpClient,
    AiohttpServer,
    demonstrate_aiohttp,
)
from asyncio_mastery.integrations.database import DatabasePool, Transaction, demonstrate_database
from asyncio_mastery.integrations.multiprocessing import (
    ProcessBridge,
    demonstrate_multiprocessing,
    run_in_process,
)
from asyncio_mastery.integrations.threading import (
    AsyncThread,
    ThreadBridge,
    demonstrate_threading,
    run_in_thread,
)

__all__ = [
    "AiohttpClient",
    "AiohttpServer",
    "AsyncThread",
    "DatabasePool",
    "ProcessBridge",
    "ThreadBridge",
    "Transaction",
    "demonstrate_aiohttp",
    "demonstrate_database",
    "demonstrate_multiprocessing",
    "demonstrate_threading",
    "run_in_process",
    "run_in_thread",
]
