"""
Monitoring Utilities

Health checks, metrics collection, and service status monitoring.
"""

import asyncio
import time
from collections import defaultdict, deque
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

T = TypeVar("T")


class ServiceStatus(Enum):
    """Service health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    name: str
    status: ServiceStatus
    message: str = ""
    timestamp: float = field(default_factory=time.time)
    duration: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "duration": self.duration,
            "metadata": self.metadata,
        }


class HealthCheck:
    """Health check definition."""

    def __init__(
        self,
        name: str,
        check_func: Callable[[], Awaitable[HealthCheckResult]],
        interval: float = 30.0,
        timeout: float = 10.0,
        critical: bool = True,
    ):
        self.name = name
        self._check_func = check_func
        self.interval = interval
        self.timeout = timeout
        self.critical = critical
        self._last_result: HealthCheckResult | None = None
        self._running = False
        self._task: asyncio.Task | None = None

    @property
    def last_result(self) -> HealthCheckResult | None:
        return self._last_result

    async def run(self) -> HealthCheckResult:
        """Run the health check."""
        start = time.time()

        try:
            result = await asyncio.wait_for(self._check_func(), timeout=self.timeout)
            result.duration = time.time() - start
        except TimeoutError:
            result = HealthCheckResult(
                name=self.name,
                status=ServiceStatus.UNHEALTHY,
                message=f"Health check timed out after {self.timeout}s",
                duration=time.time() - start,
            )
        except Exception as e:
            result = HealthCheckResult(
                name=self.name,
                status=ServiceStatus.UNHEALTHY,
                message=f"Health check failed: {e}",
                duration=time.time() - start,
            )

        self._last_result = result
        return result

    async def start(self) -> None:
        """Start periodic health checks."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Stop periodic health checks."""
        self._running = False
        if self._task:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task

    async def _run_loop(self) -> None:
        while self._running:
            await self.run()
            await asyncio.sleep(self.interval)


class HealthChecker:
    """Manage multiple health checks."""

    def __init__(self):
        self._checks: dict[str, HealthCheck] = {}
        self._lock = asyncio.Lock()

    def add_check(self, check: HealthCheck) -> None:
        self._checks[check.name] = check

    def remove_check(self, name: str) -> bool:
        if name in self._checks:
            del self._checks[name]
            return True
        return False

    def get_check(self, name: str) -> HealthCheck | None:
        return self._checks.get(name)

    async def run_all(self) -> dict[str, HealthCheckResult]:
        results = {}
        for name, check in self._checks.items():
            results[name] = await check.run()
        return results

    async def run_critical(self) -> dict[str, HealthCheckResult]:
        results = {}
        for name, check in self._checks.items():
            if check.critical:
                results[name] = await check.run()
        return results

    def get_overall_status(self) -> ServiceStatus:
        """Get overall service status."""
        if not self._checks:
            return ServiceStatus.UNKNOWN

        has_unhealthy = False
        has_degraded = False

        for check in self._checks.values():
            if check.last_result:
                if check.last_result.status == ServiceStatus.UNHEALTHY:
                    if check.critical:
                        return ServiceStatus.UNHEALTHY
                    has_unhealthy = True
                elif check.last_result.status == ServiceStatus.DEGRADED:
                    has_degraded = True

        if has_unhealthy:
            return ServiceStatus.DEGRADED
        if has_degraded:
            return ServiceStatus.DEGRADED

        return ServiceStatus.HEALTHY

    async def start_all(self) -> None:
        for check in self._checks.values():
            await check.start()

    async def stop_all(self) -> None:
        for check in self._checks.values():
            await check.stop()


def http_health_check(
    name: str, url: str, expected_status: int = 200, timeout: float = 5.0
) -> HealthCheck:
    """Create HTTP health check."""

    async def check() -> HealthCheckResult:
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                    if resp.status == expected_status:
                        return HealthCheckResult(
                            name=name, status=ServiceStatus.HEALTHY, message=f"HTTP {resp.status}"
                        )
                    else:
                        return HealthCheckResult(
                            name=name,
                            status=ServiceStatus.UNHEALTHY,
                            message=f"HTTP {resp.status} (expected {expected_status})",
                        )
        except Exception as e:
            return HealthCheckResult(name=name, status=ServiceStatus.UNHEALTHY, message=str(e))

    return HealthCheck(name, check, timeout=timeout)


def database_health_check(name: str, pool: Any, query: str = "SELECT 1") -> HealthCheck:
    """Create database health check."""

    async def check() -> HealthCheckResult:
        try:
            async with pool.acquire() as conn:
                await conn.execute(query)
            return HealthCheckResult(
                name=name, status=ServiceStatus.HEALTHY, message="Database connection OK"
            )
        except Exception as e:
            return HealthCheckResult(name=name, status=ServiceStatus.UNHEALTHY, message=str(e))

    return HealthCheck(name, check)


def memory_health_check(name: str = "memory", max_percent: float = 90.0) -> HealthCheck:
    """Create memory usage health check."""

    async def check() -> HealthCheckResult:
        import psutil

        memory = psutil.virtual_memory()
        percent = memory.percent

        if percent > max_percent:
            return HealthCheckResult(
                name=name,
                status=ServiceStatus.UNHEALTHY,
                message=f"Memory usage {percent:.1f}% exceeds {max_percent}%",
                metadata={"percent": percent, "max_percent": max_percent},
            )
        elif percent > max_percent * 0.8:
            return HealthCheckResult(
                name=name,
                status=ServiceStatus.DEGRADED,
                message=f"Memory usage {percent:.1f}% approaching limit",
                metadata={"percent": percent, "max_percent": max_percent},
            )

        return HealthCheckResult(
            name=name,
            status=ServiceStatus.HEALTHY,
            message=f"Memory usage {percent:.1f}%",
            metadata={"percent": percent},
        )

    return HealthCheck(name, check)


def disk_health_check(
    name: str = "disk", path: str = "/", max_percent: float = 90.0
) -> HealthCheck:
    """Create disk space health check."""

    async def check() -> HealthCheckResult:
        import psutil

        disk = psutil.disk_usage(path)
        percent = (disk.used / disk.total) * 100

        if percent > max_percent:
            return HealthCheckResult(
                name=name,
                status=ServiceStatus.UNHEALTHY,
                message=f"Disk usage {percent:.1f}% exceeds {max_percent}%",
                metadata={"percent": percent, "max_percent": max_percent},
            )
        elif percent > max_percent * 0.8:
            return HealthCheckResult(
                name=name,
                status=ServiceStatus.DEGRADED,
                message=f"Disk usage {percent:.1f}% approaching limit",
                metadata={"percent": percent, "max_percent": max_percent},
            )

        return HealthCheckResult(
            name=name,
            status=ServiceStatus.HEALTHY,
            message=f"Disk usage {percent:.1f}%",
            metadata={"percent": percent},
        )

    return HealthCheck(name, check)


@dataclass
class MetricValue:
    """Single metric value."""

    name: str
    value: int | float
    timestamp: float = field(default_factory=time.time)
    labels: dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Collect and aggregate metrics."""

    def __init__(self, max_points: int = 10000):
        self._max_points = max_points
        self._metrics: dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points))
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    def increment(
        self, name: str, value: float = 1.0, labels: dict[str, str] | None = None
    ) -> None:
        """Increment counter."""
        key = self._make_key(name, labels)
        self._counters[key] += value
        self._record(name, value, labels)

    def decrement(
        self, name: str, value: float = 1.0, labels: dict[str, str] | None = None
    ) -> None:
        """Decrement counter."""
        self.increment(name, -value, labels)

    def gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Set gauge value."""
        key = self._make_key(name, labels)
        self._gauges[key] = value
        self._record(name, value, labels)

    def histogram(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Record histogram value."""
        key = self._make_key(name, labels)
        self._histograms[key].append(value)
        self._record(name, value, labels)

    def _make_key(self, name: str, labels: dict[str, str] | None) -> str:
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def _record(self, name: str, value: float, labels: dict[str, str] | None) -> None:
        metric = MetricValue(name=name, value=value, labels=labels or {})
        self._metrics[name].append(metric)

    def get_counter(self, name: str, labels: dict[str, str] | None = None) -> float:
        key = self._make_key(name, labels)
        return self._counters.get(key, 0.0)

    def get_gauge(self, name: str, labels: dict[str, str] | None = None) -> float | None:
        key = self._make_key(name, labels)
        return self._gauges.get(key)

    def get_histogram_stats(
        self, name: str, labels: dict[str, str] | None = None
    ) -> dict[str, float]:
        key = self._make_key(name, labels)
        values = self._histograms.get(key, [])

        if not values:
            return {}

        sorted_values = sorted(values)
        count = len(values)

        return {
            "count": count,
            "sum": sum(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / count,
            "p50": sorted_values[int(count * 0.5)],
            "p90": sorted_values[int(count * 0.9)],
            "p95": sorted_values[int(count * 0.95)],
            "p99": sorted_values[int(count * 0.99)],
        }

    def get_all_metrics(self) -> dict[str, Any]:
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {name: self.get_histogram_stats(name) for name in self._histograms},
        }

    def reset(self) -> None:
        self._metrics.clear()
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()


# Global metrics collector
_global_metrics = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    return _global_metrics


def counter(name: str, labels: dict[str, str] | None = None) -> Callable[[float], None]:
    """Create counter function."""

    def inc(value: float = 1.0) -> None:
        _global_metrics.increment(name, value, labels)

    return inc


def gauge(name: str, labels: dict[str, str] | None = None) -> Callable[[float], None]:
    """Create gauge function."""

    def set_value(value: float) -> None:
        _global_metrics.gauge(name, value, labels)

    return set_value


def histogram(name: str, labels: dict[str, str] | None = None) -> Callable[[float], None]:
    """Create histogram function."""

    def record(value: float) -> None:
        _global_metrics.histogram(name, value, labels)

    return record


@asynccontextmanager
async def measure_histogram(
    name: str, labels: dict[str, str] | None = None
) -> AsyncGenerator[None, None]:
    """Context manager to measure duration in histogram."""
    start = time.monotonic()
    try:
        yield
    finally:
        elapsed = time.monotonic() - start
        _global_metrics.histogram(name, elapsed, labels)


@asynccontextmanager
async def measure_counter(
    name: str, labels: dict[str, str] | None = None, increment: float = 1.0
) -> AsyncGenerator[None, None]:
    """Context manager to increment counter."""
    try:
        yield
    finally:
        _global_metrics.increment(name, increment, labels)


class ServiceMonitor:
    """Monitor service health and metrics."""

    def __init__(self, service_name: str):
        self._service_name = service_name
        self._health_checker = HealthChecker()
        self._metrics = MetricsCollector()
        self._start_time = time.time()
        self._request_count = 0
        self._error_count = 0
        self._lock = asyncio.Lock()

    @property
    def service_name(self) -> str:
        return self._service_name

    @property
    def uptime(self) -> float:
        return time.time() - self._start_time

    def add_health_check(self, check: HealthCheck) -> None:
        self._health_checker.add_check(check)

    async def start(self) -> None:
        await self._health_checker.start_all()

    async def stop(self) -> None:
        await self._health_checker.stop_all()

    async def health_check(self) -> dict[str, Any]:
        results = await self._health_checker.run_all()
        overall = self._health_checker.get_overall_status()

        return {
            "service": self._service_name,
            "status": overall.value,
            "uptime": self.uptime,
            "checks": {name: result.to_dict() for name, result in results.items()},
        }

    def record_request(self, success: bool = True) -> None:
        self._request_count += 1
        if not success:
            self._error_count += 1

    def get_stats(self) -> dict[str, Any]:
        return {
            "service": self._service_name,
            "uptime": self.uptime,
            "requests_total": self._request_count,
            "errors_total": self._error_count,
            "error_rate": self._error_count / max(1, self._request_count),
            "metrics": self._metrics.get_all_metrics(),
        }


def create_service_monitor(service_name: str) -> ServiceMonitor:
    """Create service monitor with default health checks."""
    monitor = ServiceMonitor(service_name)

    # Add default health checks
    monitor.add_health_check(memory_health_check())
    monitor.add_health_check(disk_health_check())

    return monitor


# Prometheus-style metrics export
def export_prometheus(metrics: MetricsCollector) -> str:
    """Export metrics in Prometheus format."""
    lines = []

    # Counters
    for name, value in metrics._counters.items():
        lines.append(f"# TYPE {name} counter")
        lines.append(f"{name} {value}")

    # Gauges
    for name, value in metrics._gauges.items():
        lines.append(f"# TYPE {name} gauge")
        lines.append(f"{name} {value}")

    # Histograms
    for name in metrics._histograms:
        stats = metrics.get_histogram_stats(name)
        if stats:
            lines.append(f"# TYPE {name} histogram")
            lines.append(f"{name}_count {stats['count']}")
            lines.append(f"{name}_sum {stats['sum']}")
            lines.append(f"{name}_min {stats['min']}")
            lines.append(f"{name}_max {stats['max']}")
            lines.append(f"{name}_mean {stats['mean']}")
            for percentile in ["p50", "p90", "p95", "p99"]:
                if percentile in stats:
                    lines.append(f"{name}_{percentile} {stats[percentile]}")

    return "\n".join(lines) + "\n"
