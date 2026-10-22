"""
Subprocess package. Async subprocess management.

This subpackage teaches how asyncio programs launch and supervise other
programs (child processes). Your program is the head chef. Subprocesses
are kitchen assistants you hire. You delegate work. You talk to them
while they work. You never stand idle watching them. Asyncio lets you
keep cooking while they chop. Asyncio taps you on the shoulder when
they are done.

The three lessons, in suggested reading order:

1. create_subprocess.py — spawning children: create_subprocess_exec
   vs shell, the three standard pipes (stdin, stdout, stderr), waiting
   with timeouts, and stopping children politely (SIGTERM) or
   forcefully (SIGKILL). AsyncSubprocess and run_command wrap all this.

2. pipes.py — plumbing between programs: streaming data chunk-by-chunk,
   connecting one child output to another input (like the shell pipe
   operator), duplicating a stream (tee), buffering writes, and
   line-by-line processing — all while respecting backpressure (slow
   readers pause fast writers).

3. signals.py — answering the OS knock on the door: catching SIGINT
   (Ctrl+C) and SIGTERM so servers shut down gracefully (finish cleanup,
   cancel tasks politely) instead of losing work. Mostly a Unix concept.

Minimal example:

import asyncio
from asyncio_mastery.subprocess import AsyncSubprocess

async def main() -> None:
    proc = AsyncSubprocess(["echo", "hello from a child process"])
    result = await proc.run()
    print(result.success, result.stdout.decode().strip())

asyncio.run(main())
"""

from asyncio_mastery.subprocess.create_subprocess import AsyncSubprocess, demonstrate_subprocess
from asyncio_mastery.subprocess.pipes import PipeManager, demonstrate_pipes
from asyncio_mastery.subprocess.signals import SignalHandler, demonstrate_signals

__all__ = [
    "AsyncSubprocess",
    "PipeManager",
    "SignalHandler",
    "demonstrate_pipes",
    "demonstrate_signals",
    "demonstrate_subprocess",
]
