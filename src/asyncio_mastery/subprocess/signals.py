"""
Signals. Handle OS signals in asyncio.

A signal is a short asynchronous knock on the door. The operating
system delivers it to your process to announce an event. Most
famously: please shut down. Real-world analogy: signals are like the
fire alarm or the manager closing-time announcement in a restaurant.
They can arrive at any moment, interrupting whatever the staff is
doing.

The signals you will meet most:

* SIGINT (Ctrl+C) — the user asking your program to stop.
* SIGTERM — the system politely asking your program to stop. The
  default of the kill command. Sent on shutdown. Polite means your
  program may catch it and clean up (or even ignore it).
* SIGKILL — the unstoppable axe. The OS kills immediately. Programs
  cannot catch or ignore it (that is why no SIGKILL handling appears
  here).
* SIGUSR1 and SIGUSR2 (Unix only) — two spare doorbells with no
  fixed meaning. Free for your app to define. Commonly reload config
  or dump state.

Platform note: signals are a Unix concept. Windows only supports a
subset (SIGINT-ish Ctrl+C handling). SIGUSR1 and friends do not
exist there. This demo refuses to run fully on Windows for that
reason.

Why signals need special handling in asyncio:

OS signal handlers cannot be coroutines. The OS invokes a plain
function at an arbitrary instant, even mid-bytecode. The safe pattern
(used by SignalHandler) is: the tiny sync handler merely schedules
your async handler with asyncio.create_task(...) and returns
immediately. Your real logic then runs as a normal task on the event
loop. Never do slow or awaiting work directly inside a sync signal
handler.

Graceful shutdown, the big idea:

Graceful shutdown = when told to stop, do not just vanish. Stop
taking new work. Cancel tracked tasks. Let them run their cleanup
(finally blocks, closing connections). Only then exit.
GracefulShutdown packages that pattern. An asyncio.Event flips on the
first signal. Registered tasks get cancelled so their cleanup code
runs.

Minimal example:

import asyncio
from asyncio_mastery.subprocess.signals import GracefulShutdown
import signal

async def main() -> None:
    async with GracefulShutdown() as shutdown:
        task = asyncio.create_task(do_work())
        shutdown.track_task(task)
        await shutdown.wait_for_shutdown()   # returns on Ctrl+C / SIGTERM
        # cleanup here...

asyncio.run(main())
"""

import asyncio
import signal
import sys
from collections.abc import Awaitable, Callable, Set
from typing import Any


class SignalHandler:
    """Manage signal handlers for asyncio.

    Bridges the sync and async worlds. You register async handler
    coroutines for a signal. Internally a minimal sync handler is
    installed with the OS that schedules your coroutines onto the
    event loop. It also remembers the previously installed OS handler
    so it can restore it on unregister (good citizenship: put things
    back the way you found them).

    Attributes: (private; use register, unregister, unregister_all)

    Example:
        handler = SignalHandler()

        async def on_stop(signum, frame):
            print("stopping soon...")

        handler.register(signal.SIGINT, on_stop)
        ...later...
        handler.unregister_all()
    """

    def __init__(self) -> None:
        """Create a handler bound to the current event loop."""
        self._handlers: dict[int, list[Callable[..., Awaitable[None]]]] = {}
        self._original_handlers: dict[int, Any] = {}
        self._loop = asyncio.get_event_loop()

    def register(self, sig: int, handler: Callable[..., Awaitable[None]]) -> None:
        """Register async signal handler.

        Multiple handlers may be registered per signal. All of them
        are scheduled (as separate tasks) when the signal arrives. The
        first registration for a signal also installs the sync OS-level
        handler and saves whatever handler was there before.

        Args:
            sig: The signal number, from the signal module (e.g.
                signal.SIGINT, signal.SIGTERM, signal.SIGUSR1).
            handler: Async callable taking (signum, frame). signum is
                the delivered signal number. frame is the Python stack
                frame at interruption time (rarely needed in asyncio
                programs).
        """
        if sig not in self._handlers:
            self._handlers[sig] = []
            # Save original handler so unregister can restore it.
            self._original_handlers[sig] = signal.signal(sig, self._make_sync_handler(sig))

        self._handlers[sig].append(handler)

    def _make_sync_handler(self, sig: int) -> Callable[[int, Any], None]:
        """Create sync handler that schedules async handler.

        This factory builds the tiny function the OS will call at
        interruption time. It must be fast and must not block or
        await. All it does is create one task per registered async
        handler and return. Doing heavy work in a real signal handler
        is unsafe. It can interrupt code in an inconsistent state.
        Deferring to the event loop is the safe pattern.

        Args:
            sig: The signal number this dispatch function will serve.

        Returns:
            A sync handler(signum, frame) suitable for signal.signal().
        """

        def sync_handler(signum: int, frame: Any) -> None:
            for handler in self._handlers.get(sig, []):
                # Hand the real work to the event loop as a normal task.
                asyncio.create_task(handler(signum, frame))

        return sync_handler

    def unregister(self, sig: int, handler: Callable[..., Awaitable[None]] | None = None) -> None:
        """Unregister handler(s).

        Args:
            sig: The signal to modify.
            handler: If given, remove just this one handler. If None,
                remove all handlers for the signal. When the last
                handler is removed, the previously-saved OS handler is
                restored automatically.
        """
        if sig in self._handlers:
            if handler:
                self._handlers[sig].remove(handler)
            else:
                self._handlers[sig].clear()

            if not self._handlers[sig]:
                # Restore original OS handler and forget our bookkeeping.
                if sig in self._original_handlers:
                    signal.signal(sig, self._original_handlers[sig])
                del self._handlers[sig]
                del self._original_handlers[sig]

    def unregister_all(self) -> None:
        """Unregister all handlers.

        Removes every registered handler and restores every original
        OS-level handler. Call this during shutdown or in a finally
        block to leave the process signal configuration as you found
        it.
        """
        for sig in list(self._handlers.keys()):
            self.unregister(sig)


class GracefulShutdown:
    """Handle graceful shutdown on signals.

    The closing time coordinator. When SIGTERM or SIGINT arrives, it
    (1) flips a public asyncio.Event so any code can poll or await
    the shutdown, and (2) cancels every task you have registered with
    track_task, giving each the chance to run its finally and cleanup
    before your program exits. Designed as an async context manager so
    handlers are always removed.

    Attributes:
        shutdown_requested: True once a shutdown signal has been
            received.

    Example:
        async with GracefulShutdown() as shutdown:
            worker = asyncio.create_task(serve())
            shutdown.track_task(worker)
            await shutdown.wait_for_shutdown()
    """

    def __init__(self, *, signals: Set[int] = {signal.SIGTERM, signal.SIGINT}) -> None:
        """Configure which signals trigger shutdown.

        Args:
            signals: The signal numbers to watch. Defaults to SIGTERM
                (system shutdown request) plus SIGINT (Ctrl+C). The
                two standard please stop knocks.
        """
        self._signals = signals
        self._shutdown_event = asyncio.Event()
        self._handler = SignalHandler()
        self._tasks: set[asyncio.Task] = set()

    async def __aenter__(self) -> "GracefulShutdown":
        """Register handlers for all watched signals. Returns self."""
        for sig in self._signals:
            self._handler.register(sig, self._on_signal)
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Remove all signal handlers, restoring original OS behavior."""
        self._handler.unregister_all()

    async def _on_signal(self, signum: int, frame: Any) -> None:
        """React to a shutdown signal: announce it, set the event, cancel tasks.

        Args:
            signum: The delivered signal number.
            frame: Stack frame at interruption time (unused here).
        """
        print(f"Received signal {signum}, initiating shutdown...")
        # Flipping the event wakes every task parked in wait_for_shutdown().
        self._shutdown_event.set()

        # Cancel all tracked tasks. Cancellation raises CancelledError
        # inside them. Their try/finally blocks can use this to clean
        # up gracefully.
        for task in self._tasks:
            task.cancel()

    def track_task(self, task: asyncio.Task) -> None:
        """Track task for cancellation on shutdown.

        Args:
            task: Any running task that should be cancelled when a
                shutdown signal arrives. Finished tasks remove
                themselves from tracking automatically via a
                done-callback, so the set never grows stale entries.
        """
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    @property
    def shutdown_requested(self) -> bool:
        """True once a watched signal has been received (event latched)."""
        return self._shutdown_event.is_set()

    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown signal.

        Suspends (without blocking anything else) until a watched
        signal arrives. Then returns. Typically the main task awaits
        this while workers run. Example: await shutdown.wait_for_shutdown()
        then cleanup.
        """
        await self._shutdown_event.wait()


async def demonstrate_signals() -> None:
    """Demonstrate signal handling.

    Two parts: (1) catch a SIGUSR1 that the process sends to itself
    and record it. (2) Run two worker tasks under GracefulShutdown.
    Trigger shutdown. Watch the workers get cancelled and clean up.
    Unix-only in practice. Run this file directly on Linux or macOS
    to try it.
    """
    print("=== Signal Demo ===")

    # Basic signal handler
    handler = SignalHandler()
    received = []

    async def on_sigusr1(signum: int, frame: Any) -> None:
        received.append(signum)
        print(f"Got signal {signum}")

    handler.register(signal.SIGUSR1, on_sigusr1)

    # Send signal to self: os.kill(pid, sig) is the general deliver signal
    # syscall. Despite the scary name, with SIGUSR1 it just rings our bell.
    import os

    os.kill(os.getpid(), signal.SIGUSR1)
    await asyncio.sleep(0.1)

    print(f"Received signals: {received}")
    handler.unregister_all()

    # Graceful shutdown
    print("\n--- Graceful Shutdown ---")
    shutdown = GracefulShutdown()

    async def worker(name: str, shutdown: GracefulShutdown) -> None:
        try:
            while not shutdown.shutdown_requested:
                print(f"  {name} working...")
                await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            # The graceful-shutdown pattern: on cancellation, do
            # cleanup, then re-raise so the task is properly marked
            # cancelled.
            print(f"  {name} cancelled")
            raise

    async with shutdown:
        task1 = asyncio.create_task(worker("Worker-1", shutdown))
        task2 = asyncio.create_task(worker("Worker-2", shutdown))
        shutdown.track_task(task1)
        shutdown.track_task(task2)

        # Simulate shutdown signal after 0.5s
        async def trigger_shutdown() -> None:
            await asyncio.sleep(0.5)
            print("Triggering shutdown...")
            shutdown._shutdown_event.set()

        await asyncio.gather(trigger_shutdown(), shutdown.wait_for_shutdown())
        # return_exceptions=True swallows the workers CancelledErrors so
        # gather does not raise. Cancellation was the intended outcome here.
        await asyncio.gather(task1, task2, return_exceptions=True)

    print("Shutdown complete")


if __name__ == "__main__":
    if sys.platform == "win32":
        # SIGUSR1 and classic signal semantics do not exist on Windows.
        print("Signal demo limited on Windows")
    else:
        asyncio.run(demonstrate_signals())
