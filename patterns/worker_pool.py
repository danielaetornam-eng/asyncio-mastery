"""
Worker Pool Patterns

Flexible worker pool implementations for parallel task execution.
"""

import asyncio
import time
import uuid
from collections.abc import AsyncGenerator, Callable, Coroutine
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class WorkerStatus(Enum):
    IDLE = "idle"
    BUSY = "busy"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass
class WorkerInfo:
    """Information about a worker."""

    worker_id: str
    name: str
    status: WorkerStatus = WorkerStatus.IDLE
    current_task: str | None = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    start_time: float = field(default_factory=time.monotonic)
    last_activity: float = field(default_factory=time.monotonic)


@dataclass
class TaskResult(Generic[R]):
    """Result of a task execution."""

    task_id: str
    success: bool
    result: R | None = None
    exception: BaseException | None = None
    worker_id: str | None = None
    start_time: float = field(default_factory=time.monotonic)
    end_time: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.monotonic() - self.start_time


class WorkerPool(Generic[T, R]):
    """Fixed-size worker pool for executing tasks."""

    def __init__(
        self,
        worker_func: Callable[[T], Coroutine[Any, Any, R]],
        num_workers: int,
        *,
        name: str | None = None,
        queue_size: int = 0,
        max_retries: int = 0,
        retry_delay: float = 1.0,
    ) -> None:
        if num_workers <= 0:
            raise ValueError("num_workers must be > 0")

        self._worker_func = worker_func
        self._num_workers = num_workers
        self._name = name or f"WorkerPool-{id(self)}"
        self._queue = asyncio.Queue(maxsize=queue_size)
        self._max_retries = max_retries
        self._retry_delay = retry_delay

        self._workers: dict[str, WorkerInfo] = {}
        self._worker_tasks: dict[str, asyncio.Task] = {}
        self._results: list[TaskResult[R]] = []
        self._running = False
        self._lock = asyncio.Lock()
        self._task_counter = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def num_workers(self) -> int:
        return self._num_workers

    @property
    def running(self) -> bool:
        return self._running

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    async def start(self) -> None:
        """Start all workers."""
        if self._running:
            return

        self._running = True
        for i in range(self._num_workers):
            worker_id = f"worker-{i}"
            info = WorkerInfo(worker_id=worker_id, name=f"{self._name}-{worker_id}")
            self._workers[worker_id] = info

            task = asyncio.create_task(self._worker_loop(worker_id))
            self._worker_tasks[worker_id] = task

    async def stop(self, wait: bool = True, timeout: float | None = None) -> None:
        """Stop all workers."""
        self._running = False

        # Signal workers to stop
        for _ in range(self._num_workers):
            await self._queue.put(None)

        if wait:
            if timeout:
                await asyncio.wait_for(
                    asyncio.gather(*self._worker_tasks.values(), return_exceptions=True),
                    timeout=timeout,
                )
            else:
                await asyncio.gather(*self._worker_tasks.values(), return_exceptions=True)

        self._worker_tasks.clear()

    async def _worker_loop(self, worker_id: str) -> None:
        info = self._workers[worker_id]

        while self._running:
            try:
                item = await self._queue.get()
                if item is None:
                    break

                task_id, payload = item
                info.status = WorkerStatus.BUSY
                info.current_task = task_id
                info.last_activity = time.monotonic()

                await self._execute_task(worker_id, task_id, payload)

                info.status = WorkerStatus.IDLE
                info.current_task = None
                info.tasks_completed += 1
                info.last_activity = time.monotonic()

            except asyncio.CancelledError:
                break
            except Exception:
                info.tasks_failed += 1
                info.status = WorkerStatus.IDLE
                info.current_task = None

        info.status = WorkerStatus.STOPPED

    async def _execute_task(self, worker_id: str, task_id: str, payload: T) -> TaskResult[R]:
        _start_time = time.monotonic()
        last_exception = None

        for attempt in range(self._max_retries + 1):
            try:
                result = await self._worker_func(payload)
                task_result = TaskResult(
                    task_id=task_id,
                    success=True,
                    result=result,
                    worker_id=worker_id,
                    end_time=time.monotonic(),
                )
                async with self._lock:
                    self._results.append(task_result)
                return task_result
            except BaseException as e:
                last_exception = e
                if attempt < self._max_retries:
                    await asyncio.sleep(self._retry_delay * (attempt + 1))

        task_result = TaskResult(
            task_id=task_id,
            success=False,
            exception=last_exception,
            worker_id=worker_id,
            end_time=time.monotonic(),
        )
        async with self._lock:
            self._results.append(task_result)
        return task_result

    async def submit(self, payload: T, task_id: str | None = None) -> str:
        """Submit a task and return task ID."""
        if not self._running:
            raise RuntimeError("Worker pool not running")

        if task_id is None:
            task_id = str(uuid.uuid4())

        await self._queue.put((task_id, payload))
        return task_id

    async def submit_batch(self, payloads: list[T]) -> list[str]:
        """Submit multiple tasks."""
        return [await self.submit(p) for p in payloads]

    async def get_result(self, task_id: str, timeout: float | None = None) -> TaskResult[R]:
        """Get result for a specific task."""
        start = time.monotonic()

        while True:
            async with self._lock:
                for result in self._results:
                    if result.task_id == task_id:
                        return result

            if timeout and (time.monotonic() - start) >= timeout:
                raise TimeoutError(f"Task {task_id} not completed within timeout")

            await asyncio.sleep(0.01)

    async def wait_for_results(
        self, task_ids: list[str], timeout: float | None = None
    ) -> list[TaskResult[R]]:
        """Wait for multiple tasks to complete."""
        results = []
        for task_id in task_ids:
            result = await self.get_result(task_id, timeout)
            results.append(result)
        return results

    def get_worker_info(self, worker_id: str) -> WorkerInfo | None:
        return self._workers.get(worker_id)

    def get_all_worker_info(self) -> list[WorkerInfo]:
        return list(self._workers.values())

    def get_results(self) -> list[TaskResult[R]]:
        return self._results.copy()

    def clear_results(self) -> None:
        self._results.clear()

    def stats(self) -> dict[str, Any]:
        return {
            "name": self._name,
            "num_workers": self._num_workers,
            "running": self._running,
            "queue_size": self.queue_size,
            "total_submitted": self._task_counter,
            "total_completed": len([r for r in self._results if r.success]),
            "total_failed": len([r for r in self._results if not r.success]),
            "workers": [vars(w) for w in self._workers.values()],
        }

    async def __aenter__(self) -> "WorkerPool[T, R]":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()


class TaskPool(Generic[T, R]):
    """Task pool with dynamic scaling."""

    def __init__(
        self,
        worker_func: Callable[[T], Coroutine[Any, Any, R]],
        min_workers: int = 1,
        max_workers: int = 10,
        *,
        name: str | None = None,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.2,
        scale_check_interval: float = 5.0,
    ) -> None:
        if min_workers < 0 or max_workers < min_workers:
            raise ValueError("Invalid worker limits")

        self._worker_func = worker_func
        self._min_workers = min_workers
        self._max_workers = max_workers
        self._name = name or f"TaskPool-{id(self)}"
        self._scale_up_threshold = scale_up_threshold
        self._scale_down_threshold = scale_down_threshold
        self._scale_check_interval = scale_check_interval

        self._queue = asyncio.Queue()
        self._workers: dict[str, WorkerInfo] = {}
        self._worker_tasks: dict[str, asyncio.Task] = {}
        self._results: list[TaskResult[R]] = []
        self._running = False
        self._lock = asyncio.Lock()
        self._scaler_task: asyncio.Task | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def current_workers(self) -> int:
        return len(self._workers)

    async def start(self) -> None:
        if self._running:
            return

        self._running = True

        # Start minimum workers
        for i in range(self._min_workers):
            await self._add_worker(f"worker-{i}")

        # Start scaler
        self._scaler_task = asyncio.create_task(self._scaler_loop())

    async def stop(self, wait: bool = True) -> None:
        self._running = False

        if self._scaler_task:
            self._scaler_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._scaler_task

        # Signal all workers to stop
        for _ in range(len(self._workers)):
            await self._queue.put(None)

        if wait:
            await asyncio.gather(*self._worker_tasks.values(), return_exceptions=True)

        self._worker_tasks.clear()
        self._workers.clear()

    async def _add_worker(self, worker_id: str) -> None:
        info = WorkerInfo(worker_id=worker_id, name=f"{self._name}-{worker_id}")
        self._workers[worker_id] = info

        task = asyncio.create_task(self._worker_loop(worker_id))
        self._worker_tasks[worker_id] = task

    async def _remove_worker(self, worker_id: str) -> bool:
        if worker_id not in self._workers:
            return False

        if len(self._workers) <= self._min_workers:
            return False

        info = self._workers[worker_id]
        info.status = WorkerStatus.STOPPING

        # Send stop signal
        await self._queue.put(None)

        task = self._worker_tasks.pop(worker_id, None)
        if task:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

        self._workers.pop(worker_id, None)
        return True

    async def _worker_loop(self, worker_id: str) -> None:
        info = self._workers[worker_id]

        while self._running:
            try:
                item = await self._queue.get()
                if item is None:
                    break

                task_id, payload = item
                info.status = WorkerStatus.BUSY
                info.current_task = task_id
                info.last_activity = time.monotonic()

                try:
                    result = await self._worker_func(payload)
                    task_result = TaskResult(
                        task_id=task_id,
                        success=True,
                        result=result,
                        worker_id=worker_id,
                        end_time=time.monotonic(),
                    )
                except BaseException as e:
                    task_result = TaskResult(
                        task_id=task_id,
                        success=False,
                        exception=e,
                        worker_id=worker_id,
                        end_time=time.monotonic(),
                    )
                    info.tasks_failed += 1
                else:
                    info.tasks_completed += 1

                async with self._lock:
                    self._results.append(task_result)

                info.status = WorkerStatus.IDLE
                info.current_task = None
                info.last_activity = time.monotonic()

            except asyncio.CancelledError:
                break
            except Exception:
                info.tasks_failed += 1
                info.status = WorkerStatus.IDLE
                info.current_task = None

        info.status = WorkerStatus.STOPPED

    async def _scaler_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self._scale_check_interval)

            if not self._running:
                break

            await self._check_scaling()

    async def _check_scaling(self) -> None:
        _queue_size = self._queue.qsize()
        worker_count = len(self._workers)

        if worker_count == 0:
            return

        # Calculate utilization
        busy_workers = sum(1 for w in self._workers.values() if w.status == WorkerStatus.BUSY)
        utilization = busy_workers / worker_count

        # Scale up
        if utilization >= self._scale_up_threshold and worker_count < self._max_workers:
            new_id = f"worker-{len(self._workers)}"
            await self._add_worker(new_id)

        # Scale down
        elif utilization <= self._scale_down_threshold and worker_count > self._min_workers:
            # Find idle worker
            for worker_id, info in self._workers.items():
                if info.status == WorkerStatus.IDLE:
                    await self._remove_worker(worker_id)
                    break

    async def submit(self, payload: T) -> str:
        task_id = str(uuid.uuid4())
        await self._queue.put((task_id, payload))
        return task_id

    def get_results(self) -> list[TaskResult[R]]:
        return self._results.copy()

    def stats(self) -> dict[str, Any]:
        return {
            "name": self._name,
            "current_workers": len(self._workers),
            "min_workers": self._min_workers,
            "max_workers": self._max_workers,
            "queue_size": self._queue.qsize(),
            "running": self._running,
            "total_results": len(self._results),
        }


class DynamicWorkerPool(Generic[T, R]):
    """Worker pool that adapts to workload."""

    def __init__(
        self,
        worker_func: Callable[[T], Coroutine[Any, Any, R]],
        *,
        name: str | None = None,
        initial_workers: int = 2,
        max_workers: int = 50,
        idle_timeout: float = 30.0,
        queue_size: int = 0,
    ) -> None:
        self._worker_func = worker_func
        self._name = name or f"DynamicWorkerPool-{id(self)}"
        self._max_workers = max_workers
        self._idle_timeout = idle_timeout
        self._queue = asyncio.Queue(maxsize=queue_size)

        self._workers: dict[str, asyncio.Task] = {}
        self._worker_info: dict[str, WorkerInfo] = {}
        self._results: list[TaskResult[R]] = []
        self._running = False
        self._lock = asyncio.Lock()
        self._worker_counter = 0

        # Start initial workers
        self._initial_workers = initial_workers

    @property
    def name(self) -> str:
        return self._name

    async def start(self) -> None:
        if self._running:
            return

        self._running = True
        for _ in range(self._initial_workers):
            await self._ensure_worker()

    async def stop(self) -> None:
        self._running = False

        # Signal all workers
        for _ in range(len(self._workers)):
            await self._queue.put(None)

        await asyncio.gather(*self._workers.values(), return_exceptions=True)
        self._workers.clear()
        self._worker_info.clear()

    async def _ensure_worker(self) -> str:
        async with self._lock:
            if len(self._workers) >= self._max_workers:
                # Return existing idle worker
                for worker_id, info in self._worker_info.items():
                    if info.status == WorkerStatus.IDLE:
                        return worker_id

            worker_id = f"worker-{self._worker_counter}"
            self._worker_counter += 1

            info = WorkerInfo(worker_id=worker_id, name=f"{self._name}-{worker_id}")
            self._worker_info[worker_id] = info

            task = asyncio.create_task(self._worker_loop(worker_id))
            self._workers[worker_id] = task

            return worker_id

    async def _worker_loop(self, worker_id: str) -> None:
        info = self._worker_info[worker_id]
        last_activity = time.monotonic()

        while self._running:
            try:
                # Wait for work with idle timeout
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                except TimeoutError:
                    # Check idle timeout
                    if time.monotonic() - last_activity > self._idle_timeout:
                        async with self._lock:
                            if worker_id in self._workers and info.status == WorkerStatus.IDLE:
                                # Remove idle worker
                                del self._workers[worker_id]
                                del self._worker_info[worker_id]
                                break
                    continue

                if item is None:
                    break

                task_id, payload = item
                info.status = WorkerStatus.BUSY
                info.current_task = task_id
                last_activity = time.monotonic()

                try:
                    result = await self._worker_func(payload)
                    task_result = TaskResult(
                        task_id=task_id,
                        success=True,
                        result=result,
                        worker_id=worker_id,
                        end_time=time.monotonic(),
                    )
                except BaseException as e:
                    task_result = TaskResult(
                        task_id=task_id,
                        success=False,
                        exception=e,
                        worker_id=worker_id,
                        end_time=time.monotonic(),
                    )
                    info.tasks_failed += 1
                else:
                    info.tasks_completed += 1

                async with self._lock:
                    self._results.append(task_result)

                info.status = WorkerStatus.IDLE
                info.current_task = None
                last_activity = time.monotonic()

            except asyncio.CancelledError:
                break
            except Exception:
                info.tasks_failed += 1
                info.status = WorkerStatus.IDLE
                info.current_task = None

        async with self._lock:
            self._workers.pop(worker_id, None)
            self._worker_info.pop(worker_id, None)

    async def submit(self, payload: T) -> str:
        task_id = str(uuid.uuid4())
        await self._queue.put((task_id, payload))

        # Ensure we have workers
        await self._ensure_worker()

        return task_id

    def get_results(self) -> list[TaskResult[R]]:
        return self._results.copy()

    def stats(self) -> dict[str, Any]:
        active = sum(1 for w in self._worker_info.values() if w.status == WorkerStatus.BUSY)
        idle = sum(1 for w in self._worker_info.values() if w.status == WorkerStatus.IDLE)

        return {
            "name": self._name,
            "active_workers": active,
            "idle_workers": idle,
            "total_workers": len(self._workers),
            "max_workers": self._max_workers,
            "queue_size": self._queue.qsize(),
            "total_tasks": len(self._results),
        }


@asynccontextmanager
async def worker_pool(
    worker_func: Callable[[T], Coroutine[Any, Any, R]], num_workers: int, **kwargs
) -> AsyncGenerator[WorkerPool[T, R], None]:
    """Context manager for worker pool."""
    pool = WorkerPool(worker_func, num_workers, **kwargs)
    await pool.start()
    try:
        yield pool
    finally:
        await pool.stop()


@asynccontextmanager
async def task_pool(
    worker_func: Callable[[T], Coroutine[Any, Any, R]],
    min_workers: int = 1,
    max_workers: int = 10,
    **kwargs,
) -> AsyncGenerator[TaskPool[T, R], None]:
    """Context manager for task pool."""
    pool = TaskPool(worker_func, min_workers, max_workers, **kwargs)
    await pool.start()
    try:
        yield pool
    finally:
        await pool.stop()
