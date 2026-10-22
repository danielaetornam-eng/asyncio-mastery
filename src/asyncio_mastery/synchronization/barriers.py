"""
Barriers - A meeting point where N tasks wait for each other.

WHAT IS A BARRIER?
------------------
A Barrier is created for a fixed number of "parties" (tasks). Every task
that arrives calls ``await barrier.wait()`` and suspends there. Only when
ALL N parties have arrived does the barrier trip — and then every waiter is
released simultaneously. After tripping, a barrier automatically resets, so
it can be reused for the next "rendezvous".

REAL-WORLD ANALOGY: DINNER WITH FRIENDS
---------------------------------------
A group of friends agree: nobody enters the restaurant until EVERYONE has
arrived. Each friend shows up at a different time and waits by the door
(``await barrier.wait()``). The moment the last friend walks up, the whole
group goes in together. If they plan to do this again next week, the
agreement resets automatically — that is why a barrier is called "reusable".

WHY DO I NEED THIS IN ASYNCIO?
------------------------------
In phased parallel algorithms — simulations, iterative solvers, pipelines —
each task computes a phase and must not start the next phase until ALL
tasks finished the current one (later phases read other tasks' results).
Without a barrier, a fast task would race ahead and read half-updated
state: a race condition created purely by timing. Remember, asyncio tasks
interleave at every ``await``, so "they started together" guarantees
nothing about when they finish. A barrier re-synchronises them at each
phase boundary.

TERMS DEFINED:
- Parties: the number of tasks that must arrive before the barrier trips.
- Rendezvous: a meeting point where tasks wait for one another.
- Broken barrier: a barrier that was aborted (or whose waiting was
  interrupted); waiting on it raises ``BrokenBarrierError`` — a safety
  valve so surviving tasks do not wait forever for a task that died.
- Arrival index: the int returned by ``wait()`` (0..N-1). The task that
  gets 0 is "last to arrive" and often acts as the phase leader.

MINIMAL EXAMPLE
---------------
::

    import asyncio
    from asyncio_mastery.synchronization import AsyncBarrier

    barrier = AsyncBarrier(3)   # 3 friends

    async def friend(i):
        await asyncio.sleep(i * 0.1)     # everyone arrives at a different
        print(f"friend {i} arrived")     # time...
        await barrier.wait()             # ...but waits by the door
        print(f"friend {i} goes in")     # all released together!

    asyncio.run(asyncio.gather(*[friend(i) for i in range(3)]))

Run this file directly to see barrier patterns in action::

    python -m asyncio_mastery.synchronization.barriers
"""

import asyncio
from asyncio import Barrier
from typing import TypeVar

T = TypeVar("T")


class AsyncBarrier:
    """An asyncio Barrier with monitoring for learning.

    Wraps :class:`asyncio.Barrier` (Python 3.11+) and adds a name, a count
    of total waits, and a local broken-state flag.

    Example:
        ::

            barrier = AsyncBarrier(4, name="phase-gate")

            async def worker():
                do_phase_one()
                await barrier.wait()   # nobody continues until all 4 arrive
                do_phase_two()
    """

    def __init__(self, parties: int, *, name: str | None = None) -> None:
        """Create a barrier that trips when ``parties`` tasks have arrived.

        Args:
            parties: How many tasks must call :meth:`wait` before all are
                released. Must be > 0.
            name: Optional human-readable name for debugging/logging.

        Raises:
            ValueError: If ``parties`` is not positive.
        """
        if parties <= 0:
            raise ValueError("Parties must be > 0")
        self._barrier = Barrier(parties)
        self._name = name or "barrier"
        self._parties = parties
        self._wait_count = 0
        self._broken = False

    @property
    def name(self) -> str:
        """The human-readable name of this barrier."""
        return self._name

    @property
    def parties(self) -> int:
        """How many tasks are required to trip the barrier."""
        return self._parties

    @property
    def n_waiting(self) -> int:
        """How many tasks are currently parked at the barrier."""
        return self._barrier.n_waiting

    @property
    def broken(self) -> bool:
        """True if the barrier has been aborted/broken (waiters will fail)."""
        return self._barrier.broken

    async def wait(self) -> int:
        """Arrive at the barrier and suspend until all parties arrive.

        If you are not the last to arrive, this coroutine parks at the
        ``await`` and the event loop runs other tasks. When the final party
        arrives, everyone is released together and the barrier resets for
        the next round.

        Returns:
            Your arrival index: an int in ``range(parties)``. The task that
            receives 0 is the one whose arrival tripped the barrier — a
            common idiom is ``if index == 0: <do per-phase leader work>``.

        Raises:
            asyncio.BrokenBarrierError: If the barrier was aborted or
                broken while you waited.
        """
        self._wait_count += 1
        return await self._barrier.wait()

    async def wait_timeout(self, timeout: float) -> int:
        """Wait at the barrier, giving up after ``timeout`` seconds.

        Useful when you cannot trust every party to arrive (a crashed task
        would otherwise leave everyone parked forever).

        Args:
            timeout: Maximum seconds to wait.

        Returns:
            The arrival index, as in :meth:`wait`.

        Raises:
            TimeoutError: If not all parties arrived in time.
        """
        return await asyncio.wait_for(self.wait(), timeout=timeout)

    def reset(self) -> None:
        """Reset the barrier to its initial (empty) state.

        Any tasks currently waiting are released with
        ``BrokenBarrierError`` — reset is for recovery, not a normal rendezvous.
        """
        self._barrier.reset()
        self._broken = False

    async def abort(self) -> None:
        """Break the barrier: current and future waiters fail fast.

        Every task parked in :meth:`wait` wakes with
        ``BrokenBarrierError``, and later calls to ``wait`` raise it too
        (until :meth:`reset`). Use as an emergency escape hatch, e.g. when
        one party has crashed and the rendezvous can never complete.
        """
        await self._barrier.abort()
        self._broken = True


class BarrierExample:
    """Phased parallel computation synchronised by a barrier.

    Each of ``workers`` tasks runs 3 phases. A worker may finish a phase
    early or late relative to the others, but the barrier guarantees nobody
    starts phase N+1 until ALL workers have reported phase N results. The
    task that arrives last (index 0) acts as the "phase leader" and prints
    the synchronization message.
    """

    def __init__(self, workers: int) -> None:
        """Create the computation.

        Args:
            workers: Number of parallel worker tasks; also the barrier's
                party count (every worker must check in each phase).
        """
        self._workers = workers
        self._barrier = AsyncBarrier(workers, name="compute")
        self._results: list[tuple[int, int, str]] = []
        self._phase = 0

    async def worker(self, id: int) -> None:
        """Run all phases for one worker, rendezvousing between phases.

        Args:
            id: Worker number (0 .. workers-1). Also controls its simulated
                workload, so workers genuinely finish at different times.
        """
        for phase in range(3):
            # Phase computation
            result = await self._compute_phase(id, phase)
            self._results.append((id, phase, result))

            # Wait at the barrier: without this, a fast worker would start
            # the next phase reading results that slow workers haven't
            # produced yet — a race condition born purely of timing.
            index = await self._barrier.wait()
            if index == 0:
                # Arrival index 0 = last one through the gate; it acts as
                # leader and may safely observe everyone's phase output.
                print(f"Phase {phase}: all workers synchronized")
                self._phase = phase + 1

    async def _compute_phase(self, worker_id: int, phase: int) -> str:
        """Simulate variable-duration work for one worker/phase.

        Args:
            worker_id: Which worker is computing.
            phase: Which phase is being computed.

        Returns:
            A string describing the result.
        """
        await asyncio.sleep(0.01 * (worker_id + 1))
        return f"worker-{worker_id}-phase-{phase}"

    async def run(self) -> list[tuple[int, int, str]]:
        """Run all workers to completion.

        Returns:
            A list of ``(worker_id, phase, result)`` triples — one per
            worker per phase, so ``workers * 3`` entries.
        """
        await asyncio.gather(*[self.worker(i) for i in range(self._workers)])
        return self._results


async def demonstrate_barriers() -> None:
    """Run a narrated tour of common barrier patterns.

    Demonstrates:
      1. Three workers waiting for each other, then proceeding together.
      2. Barrier reuse across multiple rounds (auto-reset after tripping).
      3. waiting with a timeout when one party is too slow.
      4. abort(): breaking a barrier so waiters fail fast instead of
         hanging forever.
      5. A phased parallel computation (BarrierExample).

    Returns:
        None. Prints demonstration output to stdout.
    """
    print("=== Basic Barrier ===")
    barrier = AsyncBarrier(3, name="basic")

    async def worker(id: int) -> None:
        print(f"Worker {id} starting")
        # Different arrival times — that's the whole point of the demo.
        await asyncio.sleep(0.05 * (id + 1))
        print(f"Worker {id} at barrier")
        # Suspends here until the other two also arrive; then all proceed.
        index = await barrier.wait()
        print(f"Worker {id} released (index: {index})")

    await asyncio.gather(*[worker(i) for i in range(3)])

    print("\n=== Reusable Barrier ===")
    barrier2 = AsyncBarrier(2, name="reusable")

    for round_num in range(3):
        print(f"\n--- Round {round_num + 1} ---")

        async def round_worker(id: int) -> None:
            await asyncio.sleep(0.01)
            idx = await barrier2.wait()
            print(f"  Worker {id} passed (index: {idx})")

        # Same barrier used each round: it auto-resets after tripping.
        await asyncio.gather(round_worker(1), round_worker(2))

    print("\n=== Barrier with Timeout ===")
    barrier3 = AsyncBarrier(3, name="timeout")

    async def slow_worker(id: int) -> None:
        await asyncio.sleep(0.2)
        try:
            # Misses the quorum: by the time it arrives the others may
            # already be gone, and its own short timeout saves it from
            # hanging.
            await barrier3.wait_timeout(0.1)
        except TimeoutError:
            print(f"Worker {id} timed out at barrier!")

    async def fast_worker(id: int) -> None:
        await asyncio.sleep(0.01)
        try:
            await barrier3.wait_timeout(0.5)
        except TimeoutError:
            print(f"Worker {id} timed out!")

    await asyncio.gather(fast_worker(1), fast_worker(2), slow_worker(3))

    print("\n=== Broken Barrier ===")
    barrier4 = AsyncBarrier(3, name="broken")

    async def abort_waiter(id: int) -> None:
        try:
            await barrier4.wait()
        except asyncio.BrokenBarrierError:
            # Safety valve: instead of waiting forever for a party that
            # will never come, the waiter fails fast and can clean up.
            print(f"Worker {id}: barrier broken!")

    barrier4.abort()
    await asyncio.gather(*[abort_waiter(i) for i in range(3)])

    print("\n=== Parallel Computation ===")
    example = BarrierExample(workers=4)
    results = await example.run()
    print(f"Total results: {len(results)}")


if __name__ == "__main__":
    asyncio.run(demonstrate_barriers())
