"""
Supervision Patterns

Erlang-style supervision trees for fault tolerance.
"""

import asyncio
import time
from collections import defaultdict
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class SupervisionStrategy(Enum):
    """Supervision restart strategies."""

    ONE_FOR_ONE = "one_for_one"  # Restart only failed child
    ONE_FOR_ALL = "one_for_all"  # Restart all children if one fails
    REST_FOR_ONE = "rest_for_one"  # Restart failed child and those after it


@dataclass
class ChildSpec:
    """Specification for a supervised child."""

    id: str
    start_func: Callable[[], Awaitable[T]]
    restart: str = "permanent"  # permanent, transient, temporary
    shutdown_timeout: float = 5.0
    max_restarts: int = 3
    max_restart_window: float = 60.0  # seconds
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)


@dataclass
class ChildStatus:
    """Status of a supervised child."""

    spec: ChildSpec
    task: asyncio.Task | None = None
    state: str = "stopped"  # stopped, starting, running, stopping, failed
    restarts: int = 0
    last_restart: float = 0
    last_error: BaseException | None = None


@dataclass
class SupervisorConfig:
    """Configuration for supervisor."""

    strategy: SupervisionStrategy = SupervisionStrategy.ONE_FOR_ONE
    max_restarts: int = 3
    max_restart_window: float = 60.0
    auto_start: bool = True


class Supervisor(Generic[T]):
    """Supervisor for managing child processes with restart strategies."""

    def __init__(
        self,
        children: list[ChildSpec],
        config: SupervisorConfig | None = None,
        *,
        name: str | None = None,
    ) -> None:
        self._children_specs = {c.id: c for c in children}
        self._config = config or SupervisorConfig()
        self._name = name or f"Supervisor-{id(self)}"

        self._children: dict[str, ChildStatus] = {}
        self._running = False
        self._supervisor_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._restart_history: dict[str, list[float]] = defaultdict(list)

    @property
    def name(self) -> str:
        return self._name

    @property
    def running(self) -> bool:
        return self._running

    @property
    def children(self) -> dict[str, ChildStatus]:
        return self._children

    async def start(self) -> None:
        """Start the supervisor and all children."""
        if self._running:
            return

        self._running = True

        # Initialize child statuses
        for spec in self._children_specs.values():
            self._children[spec.id] = ChildStatus(spec=spec)

        if self._config.auto_start:
            await self._start_all_children()

    async def stop(self) -> None:
        """Stop the supervisor and all children."""
        self._running = False

        # Stop all children
        for child_id in list(self._children.keys()):
            await self._stop_child(child_id)

        if self._supervisor_task:
            self._supervisor_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._supervisor_task

    async def _start_all_children(self) -> None:
        """Start all children in order."""
        for child_id in self._children_specs:
            await self._start_child(child_id)

    async def _start_child(self, child_id: str) -> bool:
        """Start a single child."""
        async with self._lock:
            status = self._children.get(child_id)
            if not status:
                return False

            if status.state == "running":
                return True

            status.state = "starting"

        spec = status.spec

        async def child_wrapper():
            try:
                async with self._lock:
                    status.state = "running"

                result = await spec.start_func(*spec.args, **spec.kwargs)
                return result
            except asyncio.CancelledError:
                raise
            except BaseException as e:
                async with self._lock:
                    status.state = "failed"
                    status.last_error = e
                raise
            finally:
                async with self._lock:
                    if status.state != "failed":
                        status.state = "stopped"

        try:
            task = asyncio.create_task(child_wrapper(), name=f"{self._name}-{child_id}")

            async with self._lock:
                status.task = task
                status.state = "running"

            # Add done callback
            task.add_done_callback(lambda t, cid=child_id: self._on_child_done(cid, t))

            return True
        except Exception:
            async with self._lock:
                status.state = "failed"
            return False

    def _on_child_done(self, child_id: str, task: asyncio.Task) -> None:
        """Handle child task completion."""
        asyncio.create_task(self._handle_child_done(child_id, task))

    async def _handle_child_done(self, child_id: str, task: asyncio.Task) -> None:
        """Handle child completion with restart logic."""
        if not self._running:
            return

        status = self._children.get(child_id)
        if not status:
            return

        try:
            exc = task.exception()
            if exc is None:
                # Normal completion
                async with self._lock:
                    status.state = "stopped"
                    status.task = None
                return

            if isinstance(exc, asyncio.CancelledError):
                async with self._lock:
                    status.state = "stopped"
                    status.task = None
                return

            # Child failed, check restart policy
            spec = status.spec
            if spec.restart == "permanent" or (
                spec.restart == "transient" and not isinstance(exc, Exception)
            ):
                await self._maybe_restart(child_id, exc)
            else:
                async with self._lock:
                    status.state = "failed"
                    status.task = None

                # Apply supervision strategy
                await self._apply_strategy(child_id, exc)

        except Exception:
            pass

    async def _maybe_restart(self, child_id: str, exception: BaseException) -> None:
        """Check if child should be restarted."""
        status = self._children[child_id]
        spec = status.spec

        now = time.monotonic()
        history = self._restart_history[child_id]

        # Clean old restarts
        cutoff = now - spec.max_restart_window
        self._restart_history[child_id] = [t for t in history if t > cutoff]

        if len(self._restart_history[child_id]) >= spec.max_restarts:
            # Too many restarts, mark as failed
            async with self._lock:
                status.state = "failed"
                status.last_error = exception
                status.task = None

            await self._apply_strategy(child_id, exception)
            return

        # Record restart
        self._restart_history[child_id].append(now)

        async with self._lock:
            status.restarts += 1
            status.last_restart = now
            status.last_error = exception

        # Restart child
        await self._start_child(child_id)

    async def _apply_strategy(self, failed_child_id: str, exception: BaseException) -> None:
        """Apply supervision strategy."""
        strategy = self._config.strategy

        if strategy == SupervisionStrategy.ONE_FOR_ONE:
            # Only the failed child is affected (already handled)
            pass

        elif strategy == SupervisionStrategy.ONE_FOR_ALL:
            # Stop and restart all children
            for child_id in self._children:
                if child_id != failed_child_id:
                    await self._stop_child(child_id)
            await self._start_all_children()

        elif strategy == SupervisionStrategy.REST_FOR_ONE:
            # Restart failed child and all children started after it
            child_order = list(self._children_specs.keys())
            failed_index = child_order.index(failed_child_id)

            for child_id in child_order[failed_index:]:
                await self._stop_child(child_id)

            for child_id in child_order[failed_index:]:
                await self._start_child(child_id)

    async def _stop_child(self, child_id: str) -> None:
        """Stop a child."""
        status = self._children.get(child_id)
        if not status or not status.task:
            return

        async with self._lock:
            status.state = "stopping"

        status.task.cancel()
        with suppress(TimeoutError, asyncio.CancelledError):
            await asyncio.wait_for(status.task, timeout=status.spec.shutdown_timeout)

        async with self._lock:
            status.state = "stopped"
            status.task = None

    async def restart_child(self, child_id: str) -> bool:
        """Manually restart a child."""
        await self._stop_child(child_id)
        return await self._start_child(child_id)

    def get_status(self) -> dict[str, Any]:
        """Get supervisor status."""
        return {
            "name": self._name,
            "running": self._running,
            "strategy": self._config.strategy.value,
            "children": {
                cid: {
                    "state": status.state,
                    "restarts": status.restarts,
                    "last_restart": status.last_restart,
                    "last_error": str(status.last_error) if status.last_error else None,
                }
                for cid, status in self._children.items()
            },
        }


class DynamicSupervisor(Supervisor):
    """Supervisor that can add/remove children dynamically."""

    def __init__(self, config: SupervisorConfig | None = None, **kwargs) -> None:
        super().__init__([], config, **kwargs)

    async def add_child(self, spec: ChildSpec) -> bool:
        """Add a new child."""
        if spec.id in self._children_specs:
            return False

        self._children_specs[spec.id] = spec
        self._children[spec.id] = ChildStatus(spec=spec)

        if self._running:
            return await self._start_child(spec.id)
        return True

    async def remove_child(self, child_id: str) -> bool:
        """Remove a child."""
        if child_id not in self._children_specs:
            return False

        await self._stop_child(child_id)
        del self._children_specs[child_id]
        del self._children[child_id]
        return True

    def get_child_spec(self, child_id: str) -> ChildSpec | None:
        return self._children_specs.get(child_id)


@asynccontextmanager
async def supervised(
    children: list[ChildSpec], config: SupervisorConfig | None = None
) -> AsyncGenerator[Supervisor, None]:
    """Context manager for supervisor."""
    supervisor = Supervisor(children, config)
    await supervisor.start()
    try:
        yield supervisor
    finally:
        await supervisor.stop()


def create_supervisor(
    children: list[ChildSpec],
    strategy: SupervisionStrategy = SupervisionStrategy.ONE_FOR_ONE,
    **kwargs,
) -> Supervisor:
    """Factory for creating supervisor."""
    config = SupervisorConfig(strategy=strategy, **kwargs)
    return Supervisor(children, config)
