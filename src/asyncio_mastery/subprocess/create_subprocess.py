"""
Create Subprocess. High-level async subprocess API.

A subprocess is another program your program launches and supervises.
Your program is the chef. The child process is the kitchen assistant.
The assistant chops vegetables while the chef keeps cooking. Any
command you could type in a terminal can be a subprocess.

Real-world analogy: the three standard streams (stdin, stdout, stderr)
are garden hoses connected between your program and the child:
* stdin — a hose you pour into the child (its keyboard input)
* stdout — a hose the child pours its normal output into
* stderr — a separate hose for its error and diagnostic output
  (so real output and error chatter never get mixed up)

Why async subprocesses?

Child processes are slow and independent. They run for unpredictable
amounts of time. The blocking subprocess.run() freezes your whole
program until the child finishes. asyncio.create_subprocess_exec and
create_subprocess_shell instead let your program await the child.
While the child works, your event loop keeps handling other tasks.
The event loop resumes you when output arrives or the child exits.

Key terms:

* PIPE: when you pass asyncio.subprocess.PIPE for a stream, asyncio
  builds one of those garden hoses and hands you a StreamReader or
  StreamWriter on your end — the exact same reader and writer you
  learned in the streams package.
* exec vs shell: create_subprocess_exec("git", "status") runs the
  program directly (each argument separate — safe).
  create_subprocess_shell("git status | head") runs through the
  system shell, enabling pipes and globs but inheriting the shell
  injection dangers if you interpolate untrusted text.
* returncode: the integer the child exits with. By convention 0 =
  success. Anything else = some kind of failure.
* buffer limit (limit=): the reader internal buffer size (default
  64 KiB). If the child prints faster than you read and the buffer
  fills, backpressure pauses the child output until you catch up.
  Like a waiter telling the kitchen to slow down.
* EOF on stdin: closing the stdin hose tells the child no more input
  is coming. Many tools (like cat) finish only after seeing EOF.
* SIGTERM and SIGKILL: see terminate() and kill() — please stop
  vs stop now.

When to use:

Shelling out to tools (ffmpeg, git, compilers), running scripts,
glueing existing CLIs into your async app — without freezing your
event loop.

Minimal example:

import asyncio
from asyncio_mastery.subprocess.create_subprocess import run_command

async def main() -> None:
    result = await run_command(["echo", "hello from child"])
    print(result.returncode)            # 0
    print(result.stdout.decode())       # hello from child
    print(result.duration, "seconds")

asyncio.run(main())
"""

import asyncio
import shlex
from asyncio import StreamReader, StreamWriter
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ProcessResult:
    """Result of a completed process.

    Everything you typically want to know after a child process
    finishes, captured in one bundle. Like the assistant report card
    handed back to the chef.

    Attributes:
        returncode: Exit status integer. Convention: 0 = success.
            Non-zero = failure (exact meaning depends on the tool).
        stdout: Everything the child wrote to its normal output hose
            (bytes).
        stderr: Everything written to the error hose (bytes). Kept
            separate so diagnostics never corrupt real output.
        started_at: When the child was spawned.
        finished_at: When its output was fully collected.
        args: The command as it was launched (list or shell string).
    """

    returncode: int
    stdout: bytes
    stderr: bytes
    started_at: datetime
    finished_at: datetime
    args: list[str] | str

    @property
    def duration(self) -> float:
        """Wall-clock seconds the process took from spawn to completion."""
        return (self.finished_at - self.started_at).total_seconds()

    @property
    def success(self) -> bool:
        """True if the process exited with code 0 (the success convention)."""
        return self.returncode == 0


class AsyncSubprocess:
    """Enhanced async subprocess manager.

    Wraps asyncio.create_subprocess_exec and create_subprocess_shell
    with a friendlier object. Start the child. Talk to it through
    pipes. Wait for it (optionally with a timeout that kills runaway
    children). Stop it (politely or forcefully).

    Attributes:
        pid: Operating-system process ID once started (None before).
            Useful for logs and for OS-level tools like Task Manager
            and ps.
        returncode: Exit code once the child has finished (None while
            running).
        stdin: StreamWriter feeding the child input hose (None if
            stdin was not piped or process not started).
        stdout: StreamReader for the child normal output hose.
        stderr: StreamReader for the child error output hose.

    Example:
        proc = AsyncSubprocess(["cat"])
        await proc.start()
        proc.stdin.write(b"contact\n")
        await proc.stdin.drain()
        proc.stdin.close()               # EOF: tell cat no more input
        result = await proc.run()
        print(result.stdout)             # b"contact\n"
    """

    def __init__(
        self,
        args: list[str] | str,
        *,
        shell: bool = False,
        stdin: int | None = asyncio.subprocess.PIPE,
        stdout: int | None = asyncio.subprocess.PIPE,
        stderr: int | None = asyncio.subprocess.PIPE,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        limit: int = 65536,
    ) -> None:
        """Describe the process to launch (nothing starts until start()).

        Args:
            args: What to run. With shell=False, pass a list like
                ["git", "status"] (or a string, which is shlex-split).
                With shell=True, pass a full shell command string.
            shell: Run through the system shell. Enables shell features
                (pipes, wildcards, variables) — but never use with
                untrusted input because special characters would be
                interpreted by the shell (shell injection).
            stdin: asyncio.subprocess.PIPE to get a writable hose into
                the child, or None to share or ignore.
            stdout: PIPE to read the child normal output, or None to
                inherit your terminal.
            stderr: PIPE to read error output separately, None to
                inherit, or asyncio.subprocess.STDOUT to merge errors
                into stdout.
            cwd: Working directory the child starts in, or None for
                yours.
            env: Full environment mapping for the child, or None to
                inherit yours. It replaces rather than extends the
                environment.
            limit: Size of each pipe reader internal buffer in bytes.
                If a chatty child fills it, the child is paused
                (backpressure) until you read some output.
        """
        self._args = args
        self._shell = shell
        self._stdin = stdin
        self._stdout = stdout
        self._stderr = stderr
        self._cwd = cwd
        self._env = env
        self._limit = limit
        self._process: asyncio.subprocess.Process | None = None
        self._started_at: datetime | None = None

    @property
    def pid(self) -> int | None:
        """The OS process ID, or None if the process was never started."""
        return self._process.pid if self._process else None

    @property
    def returncode(self) -> int | None:
        """Exit code once finished. None while running or if never started."""
        return self._process.returncode if self._process else None

    @property
    def stdin(self) -> StreamWriter | None:
        """Writable hose into the child (None unless stdin=PIPE and started).

        Remember to await drain() after writes. Close it to send EOF
        when you are done feeding the child.
        """
        return self._process.stdin if self._process else None

    @property
    def stdout(self) -> StreamReader | None:
        """Readable hose carrying the child normal output."""
        return self._process.stdout if self._process else None

    @property
    def stderr(self) -> StreamReader | None:
        """Readable hose carrying the child error and diagnostic output."""
        return self._process.stderr if self._process else None

    async def start(self) -> "AsyncSubprocess":
        """Start the process.

        Spawns the child and connects the requested pipes. Returns
        immediately after spawning. The child runs concurrently with
        your program. Use wait(), communicate(), or run() to await
        its completion.

        Returns:
            self, so you can chain: proc = await AsyncSubprocess(...).start().

        Raises:
            FileNotFoundError: If the program in args[0] does not
                exist.
            PermissionError: If the file exists but is not executable.
        """
        if self._shell:
            self._process = await asyncio.create_subprocess_shell(
                self._args if isinstance(self._args, str) else " ".join(self._args),
                stdin=self._stdin,
                stdout=self._stdout,
                stderr=self._stderr,
                cwd=self._cwd,
                env=self._env,
                limit=self._limit,
            )
        else:
            # exec form: each list element is one exact argument.
            # No shell reinterprets anything. This is the safe default.
            self._process = await asyncio.create_subprocess_exec(
                *(self._args if isinstance(self._args, list) else shlex.split(self._args)),
                stdin=self._stdin,
                stdout=self._stdout,
                stderr=self._stderr,
                cwd=self._cwd,
                env=self._env,
                limit=self._limit,
            )
        self._started_at = datetime.now()
        return self

    async def wait(self) -> int:
        """Wait for process to complete.

        Suspends (without blocking the event loop) until the child
        exits. Then returns its exit code. Note: if stdout or stderr
        are piped and the child prints a lot, prefer communicate().
        wait() alone can deadlock if the child blocks on a full output
        pipe while you are blocked waiting for it to exit.

        Returns:
            The process exit code.

        Raises:
            RuntimeError: If start() was never called.
        """
        if not self._process:
            raise RuntimeError("Process not started")
        return await self._process.wait()

    async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
        """Send input and get output.

        The classic all-in-one talk to the child operation. Optionally
        sends input to stdin. Closes stdin (sending EOF). Concurrently
        drains stdout and stderr until the child exits. Concurrency
        matters: reading both hoses at once prevents the classic
        deadlock where the child will not write more stdout because
        stderr buffer is full, and vice versa.

        Args:
            input: Bytes to feed the child stdin before closing it, or
                None to close stdin immediately.

        Returns:
            A (stdout_bytes, stderr_bytes) tuple with everything the
            child printed.

        Raises:
            RuntimeError: If start() was never called.
        """
        if not self._process:
            raise RuntimeError("Process not started")
        return await self._process.communicate(input)

    async def run(self, input: bytes | None = None, timeout: float | None = None) -> ProcessResult:
        """Run process to completion with timeout.

        The convenient just run this command for me operation. Starts
        the process if needed. Communicates with it. Enforces an
        optional timeout (killing the child if it overruns — no
        runaway processes left behind). Packages everything into a
        ProcessResult.

        Args:
            input: Optional bytes to feed the child stdin.
            timeout: Max seconds to let the child run. On expiry the
                child is killed (SIGKILL — see module docs) and
                TimeoutError is raised.

        Returns:
            A ProcessResult with output, return code, and timing.

        Raises:
            TimeoutError: If timeout elapses (the child is killed first).
        """
        if not self._process:
            await self.start()

        try:
            if timeout:
                # wait_for = raise TimeoutError if not done in N seconds.
                stdout, stderr = await asyncio.wait_for(
                    self._process.communicate(input), timeout=timeout
                )
            else:
                stdout, stderr = await self._process.communicate(input)
        except TimeoutError:
            # A timed-out child must not be left running. Kill it
            # forcefully. Then reap it (wait) so it does not linger
            # as a zombie process.
            self._process.kill()
            await self._process.wait()
            raise

        finished_at = datetime.now()
        return ProcessResult(
            returncode=self._process.returncode,
            stdout=stdout,
            stderr=stderr,
            started_at=self._started_at or finished_at,
            finished_at=finished_at,
            args=self._args,
        )

    def terminate(self) -> None:
        """Terminate process (SIGTERM).

        SIGTERM is the polite stop request. Like tapping the kitchen
        assistant on the shoulder and asking them to wrap up. The
        process gets a chance to clean up (flush files, close
        connections) before exiting. A process may choose to ignore
        it. Prefer this first. Escalate to kill() only if the child
        will not go away.
        """
        if self._process:
            self._process.terminate()

    def kill(self) -> None:
        """Kill process (SIGKILL).

        SIGKILL is the axe. The OS stops the process immediately. No
        cleanup. No chance to save state. The process cannot block or
        ignore it. On Windows, kill() and terminate() both map to
        TerminateProcess. The polite and forceful distinction is a
        Unix thing.
        """
        if self._process:
            self._process.kill()

    async def send_signal(self, signal: int) -> None:
        """Send signal to process.

        Low-level control. Deliver any OS signal (e.g. signal.SIGINT
        to ask for interruption, signal.SIGHUP for reload config on
        many daemons). Mostly meaningful on Unix. Windows supports
        only a small subset.

        Args:
            signal: A signal number from the signal module.
        """
        if self._process:
            self._process.send_signal(signal)


async def run_command(
    cmd: str | list[str],
    *,
    shell: bool = False,
    timeout: float | None = None,
    input: bytes | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> ProcessResult:
    """Convenience function to run a command.

    One-call helper for the common case: run this, here is the input,
    give me the result. Roughly the async equivalent of the familiar
    subprocess.run(...).

    Args:
        cmd: Command as a list (["git", "status"] — safe/default) or
            a string (parsed with shlex, or run verbatim through the
            shell if shell=True).
        shell: Run through the system shell. Enables pipes and globs
            but beware shell injection with untrusted text.
        timeout: Max seconds before the child is force-killed and
            TimeoutError raised.
        input: Optional bytes fed to the child stdin.
        cwd: Working directory for the child.
        env: Complete environment for the child (None = inherit yours).

    Returns:
        A ProcessResult with returncode, stdout, stderr, and timing.

    Raises:
        TimeoutError: If timeout elapses.
        FileNotFoundError: If the program does not exist.

    Example:
        result = await run_command(["echo", "hi"], timeout=5)
        assert result.success
    """
    proc = AsyncSubprocess(cmd, shell=shell, cwd=cwd, env=env)
    await proc.start()
    return await proc.run(input=input, timeout=timeout)


async def demonstrate_subprocess() -> None:
    """Demonstrate subprocess usage.

    Runs a series of tiny child processes: echo, cat (with piped
    stdin), a shell command, a command that times out, an interactive
    Python child, and a background sleep. Prints what happens at each
    step. Note these demos use Unix-style commands. On Windows, adapt
    commands (e.g. cmd /c ...). Run this file directly to watch your
    program conduct an orchestra of children.
    """
    print("=== Subprocess Demo ===")

    # Simple command
    result = await run_command(["echo", "Hello, World!"])
    print(f"echo stdout: {result.stdout.decode().strip()}")

    # Command with input: cat echoes whatever flows into its stdin hose.
    result = await run_command(["cat"], input=b"stdin data")
    print(f"cat stdout: {result.stdout.decode().strip()}")

    # Shell command: only the shell understands "$HOME", so shell=True is
    # required here — the price is the shell parsing rules.
    result = await run_command("echo $HOME", shell=True)
    print(f"shell stdout: {result.stdout.decode().strip()}")

    # With timeout: sleep 10 would outlive its 0.1s budget, so it is killed.
    try:
        await run_command(["sleep", "10"], timeout=0.1)
    except TimeoutError:
        print("Timeout worked!")

    # Long-running process with interaction
    print("\n--- Interactive Process ---")
    proc = AsyncSubprocess(
        [
            "python3",
            "-c",
            """
import sys
for line in sys.stdin:
    print(f'Got: {line.strip()}')
    sys.stdout.flush()
""",
        ]
    )
    await proc.start()

    # Write to stdin: each drain() actually pushes the bytes down the hose.
    for i in range(3):
        proc.stdin.write(f"line {i}\n".encode())
        await proc.stdin.drain()

    # Close stdin to signal EOF: the child for line in sys.stdin loop only
    # finishes once it sees EOF. Without this close the run() below would
    # wait forever.
    proc.stdin.close()
    await proc.stdin.wait_closed()

    result = await proc.run()
    print(f"Output: {result.stdout.decode().strip()}")

    # Background process
    print("\n--- Background Process ---")
    proc2 = AsyncSubprocess(["sleep", "2"])
    await proc2.start()
    print(f"Started PID: {proc2.pid}")

    # Do other work... (the whole point: the child runs while we do this!)
    await asyncio.sleep(0.5)

    # Wait for completion
    await proc2.wait()
    print(f"Finished with code: {proc2.returncode}")


if __name__ == "__main__":
    asyncio.run(demonstrate_subprocess())
