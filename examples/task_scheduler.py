"""
Task Scheduler

Cron-like scheduler with support for cron expressions, intervals, and one-time tasks.
"""

import asyncio
import contextlib
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import datetime


class TriggerType(Enum):
    ONCE = "once"
    INTERVAL = "interval"
    CRON = "cron"


@dataclass
class ScheduledTask:
    """Scheduled task definition."""

    id: str
    name: str
    func: Callable[..., Awaitable[Any]]
    trigger_type: TriggerType
    trigger_config: dict[str, Any]
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    enabled: bool = True
    max_runs: int | None = None
    run_count: int = 0
    last_run: float | None = None
    next_run: float | None = None
    created_at: float = field(default_factory=time.time)
    timeout: float | None = None
    retry_attempts: int = 0
    retry_delay: float = 1.0


@dataclass
class TaskExecution:
    """Task execution record."""

    task_id: str
    task_name: str
    started_at: float
    completed_at: float | None = None
    success: bool = False
    result: Any = None
    error: Exception | None = None
    attempt: int = 1


class CronTrigger:
    """Cron expression parser and scheduler."""

    def __init__(self, expression: str) -> None:
        self._expression = expression
        self._parts = expression.split()
        if len(self._parts) != 5:
            raise ValueError(f"Invalid cron expression: {expression}")

        self._minute, self._hour, self._day, self._month, self._weekday = self._parts

    def get_next_run(self, after: float) -> float:
        """Calculate next run time after given timestamp."""
        import datetime

        dt = datetime.datetime.fromtimestamp(after)

        # Simple implementation - in production use croniter
        # This is a basic implementation
        next_dt = dt.replace(second=0, microsecond=0) + datetime.timedelta(minutes=1)

        while True:
            if self._matches(next_dt):
                return next_dt.timestamp()
            next_dt += datetime.timedelta(minutes=1)

    def _matches(self, dt: "datetime.datetime") -> bool:
        """Check if datetime matches cron expression."""

        def match_field(value: int, pattern: str) -> bool:
            if pattern == "*":
                return True
            if "," in pattern:
                return value in [int(x) for x in pattern.split(",")]
            if "-" in pattern:
                start, end = map(int, pattern.split("-"))
                return start <= value <= end
            if "/" in pattern:
                base, step = pattern.split("/")
                if base == "*":
                    return value % int(step) == 0
                return value % int(step) == 0 and value >= int(base)
            return value == int(pattern)

        return (
            match_field(dt.minute, self._minute)
            and match_field(dt.hour, self._hour)
            and match_field(dt.day, self._day)
            and match_field(dt.month, self._month)
            and match_field(dt.weekday(), self._weekday)
        )


class IntervalTrigger:
    """Fixed interval trigger."""

    def __init__(self, seconds: float) -> None:
        if seconds <= 0:
            raise ValueError("Interval must be positive")
        self._interval = seconds

    def get_next_run(self, after: float) -> float:
        return after + self._interval

    @property
    def interval(self) -> float:
        return self._interval


class TaskScheduler:
    """Async task scheduler with cron and interval support."""

    def __init__(
        self,
        *,
        timezone: str = "UTC",
        max_concurrent: int = 10,
        default_timeout: float | None = None,
    ) -> None:
        self._timezone = timezone
        self._max_concurrent = max_concurrent
        self._default_timeout = default_timeout

        self._tasks: dict[str, ScheduledTask] = {}
        self._running = False
        self._scheduler_task: asyncio.Task | None = None
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._lock = asyncio.Lock()

        # Execution history
        self._history: list[TaskExecution] = []
        self._max_history = 10000

        # Callbacks
        self._on_start_callbacks: list[Callable[[str], Awaitable[None]]] = []
        self._on_complete_callbacks: list[Callable[[TaskExecution], Awaitable[None]]] = []
        self._on_error_callbacks: list[Callable[[str, Exception], Awaitable[None]]] = []

    @property
    def running(self) -> bool:
        return self._running

    @property
    def task_count(self) -> int:
        return len(self._tasks)

    def add_task(
        self,
        func: Callable[..., Awaitable[Any]],
        name: str,
        *,
        cron: str | None = None,
        interval: float | None = None,
        once_at: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        enabled: bool = True,
        max_runs: int | None = None,
        timeout: float | None = None,
        retry_attempts: int = 0,
        retry_delay: float = 1.0,
    ) -> str:
        """Add a scheduled task."""
        if sum(x is not None for x in [cron, interval, once_at]) != 1:
            raise ValueError("Exactly one of cron, interval, or once_at must be provided")

        task_id = str(uuid.uuid4())[:8]

        if cron:
            trigger_type = TriggerType.CRON
            trigger = CronTrigger(cron)
            trigger_config = {"expression": cron}
            next_run = trigger.get_next_run(time.time())
        elif interval:
            trigger_type = TriggerType.INTERVAL
            trigger = IntervalTrigger(interval)
            trigger_config = {"interval": interval}
            next_run = time.time() + interval
        else:
            trigger_type = TriggerType.ONCE
            trigger_config = {"at": once_at}
            next_run = once_at

        task = ScheduledTask(
            id=task_id,
            name=name,
            func=func,
            trigger_type=trigger_type,
            trigger_config=trigger_config,
            args=args,
            kwargs=kwargs or {},
            enabled=enabled,
            max_runs=max_runs,
            next_run=next_run,
            timeout=timeout or self._default_timeout,
            retry_attempts=retry_attempts,
            retry_delay=retry_delay,
        )

        self._tasks[task_id] = task

        # If scheduler is running, reschedule
        if self._running:
            asyncio.create_task(self._reschedule_task(task))

        return task_id

    def remove_task(self, task_id: str) -> bool:
        """Remove a task."""
        if task_id in self._tasks:
            del self._tasks[task_id]
            return True
        return False

    def enable_task(self, task_id: str) -> bool:
        """Enable a task."""
        if task_id in self._tasks:
            self._tasks[task_id].enabled = True
            if self._running:
                asyncio.create_task(self._reschedule_task(self._tasks[task_id]))
            return True
        return False

    def disable_task(self, task_id: str) -> bool:
        """Disable a task."""
        if task_id in self._tasks:
            self._tasks[task_id].enabled = False
            return True
        return False

    def get_task(self, task_id: str) -> ScheduledTask | None:
        return self._tasks.get(task_id)

    def list_tasks(self) -> list[ScheduledTask]:
        return list(self._tasks.values())

    async def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            return

        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())

        # Schedule initial runs
        for task in self._tasks.values():
            if task.enabled:
                asyncio.create_task(self._reschedule_task(task))

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False

        if self._scheduler_task:
            self._scheduler_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._scheduler_task

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                now = time.time()

                # Find tasks that should run
                to_run = []
                async with self._lock:
                    for task in self._tasks.values():
                        if (
                            task.enabled
                            and task.next_run
                            and task.next_run <= now
                            and (task.max_runs is None or task.run_count < task.max_runs)
                        ):
                            to_run.append(task)

                # Run tasks
                for task in to_run:
                    asyncio.create_task(self._run_task(task))

                await asyncio.sleep(1)  # Check every second

            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(1)

    async def _reschedule_task(self, task: ScheduledTask) -> None:
        """Recalculate next run time for task."""
        if not task.enabled:
            task.next_run = None
            return

        if task.trigger_type == TriggerType.CRON:
            trigger = CronTrigger(task.trigger_config["expression"])
            task.next_run = trigger.get_next_run(time.time())
        elif task.trigger_type == TriggerType.INTERVAL:
            trigger = IntervalTrigger(task.trigger_config["interval"])
            task.next_run = trigger.get_next_run(time.time())
        elif task.trigger_type == TriggerType.ONCE:
            if task.run_count == 0:
                task.next_run = task.trigger_config["at"]
            else:
                task.next_run = None

    async def _run_task(self, task: ScheduledTask) -> None:
        """Run a scheduled task."""
        async with self._semaphore:
            execution = TaskExecution(task_id=task.id, task_name=task.name, started_at=time.time())

            # Fire start callbacks
            for cb in self._on_start_callbacks:
                with contextlib.suppress(Exception):
                    await cb(task.id)

            attempt = 0

            while attempt <= task.retry_attempts:
                try:
                    if task.timeout:
                        result = await asyncio.wait_for(
                            task.func(*task.args, **task.kwargs), timeout=task.timeout
                        )
                    else:
                        result = await task.func(*task.args, **task.kwargs)

                    execution.success = True
                    execution.result = result
                    execution.completed_at = time.time()

                    # Update task
                    task.run_count += 1
                    task.last_run = execution.started_at
                    await self._reschedule_task(task)

                    break

                except Exception as e:
                    attempt += 1

                    if attempt <= task.retry_attempts:
                        await asyncio.sleep(task.retry_delay * attempt)
                    else:
                        execution.success = False
                        execution.error = e
                        execution.completed_at = time.time()

                        # Fire error callbacks
                        for cb in self._on_error_callbacks:
                            with contextlib.suppress(Exception):
                                await cb(task.id, e)

            # Record execution
            async with self._lock:
                self._history.append(execution)
                if len(self._history) > self._max_history:
                    self._history.pop(0)

            # Fire complete callbacks
            for cb in self._on_complete_callbacks:
                with contextlib.suppress(Exception):
                    await cb(execution)

    def get_history(self, task_id: str | None = None, limit: int = 100) -> list[TaskExecution]:
        history = self._history
        if task_id:
            history = [e for e in history if e.task_id == task_id]
        return history[-limit:]

    def get_stats(self) -> dict[str, Any]:
        total = len(self._history)
        successful = sum(1 for e in self._history if e.success)

        return {
            "running": self._running,
            "total_tasks": len(self._tasks),
            "enabled_tasks": sum(1 for t in self._tasks.values() if t.enabled),
            "total_executions": total,
            "successful_executions": successful,
            "failed_executions": total - successful,
            "success_rate": successful / total if total > 0 else 0,
        }

    def on_task_start(self, callback: Callable[[str], Awaitable[None]]) -> None:
        self._on_start_callbacks.append(callback)

    def on_task_complete(self, callback: Callable[[TaskExecution], Awaitable[None]]) -> None:
        self._on_complete_callbacks.append(callback)

    def on_task_error(self, callback: Callable[[str, Exception], Awaitable[None]]) -> None:
        self._on_error_callbacks.append(callback)

    async def __aenter__(self) -> "TaskScheduler":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()


def create_scheduler(**kwargs) -> TaskScheduler:
    """Factory for creating task scheduler."""
    return TaskScheduler(**kwargs)


# Example usage
async def example_usage() -> None:
    """Example task scheduler usage."""
    scheduler = create_scheduler(max_concurrent=5)

    # Add interval task
    scheduler.add_task(lambda: print("Interval task running"), "interval_task", interval=10.0)

    # Add cron task (every minute)
    scheduler.add_task(lambda: print("Cron task running"), "cron_task", cron="* * * * *")

    # Add one-time task
    scheduler.add_task(lambda: print("One-time task running"), "once_task", once_at=time.time() + 5)

    async with scheduler:
        await asyncio.sleep(30)
