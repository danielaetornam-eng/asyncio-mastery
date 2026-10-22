"""
Async Test Fixtures

Reusable fixtures for async testing.
"""

import asyncio
import functools
import os
import shutil
import tempfile
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any, Generic, TypeVar

try:
    import pytest

    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    pytest = None


T = TypeVar("T")


class AsyncFixture(Generic[T]):
    """Base class for async fixtures."""

    def __init__(self) -> None:
        self._setup_done = False
        self._teardown_done = False

    async def setup(self) -> T:
        """Set up the fixture."""
        self._setup_done = True
        return await self._setup()

    async def teardown(self) -> None:
        """Tear down the fixture."""
        self._teardown_done = True
        await self._teardown()

    async def _setup(self) -> T:
        raise NotImplementedError

    async def _teardown(self) -> None:
        raise NotImplementedError

    @asynccontextmanager
    async def use(self) -> AsyncGenerator[T, None]:
        """Context manager for using fixture."""
        value = await self.setup()
        try:
            yield value
        finally:
            await self.teardown()


class EventLoopFixture(AsyncFixture[asyncio.AbstractEventLoop]):
    """Event loop fixture."""

    def __init__(self, *, new_loop: bool = True) -> None:
        super().__init__()
        self._new_loop = new_loop
        self._loop: asyncio.AbstractEventLoop | None = None
        self._original_loop: asyncio.AbstractEventLoop | None = None

    async def _setup(self) -> asyncio.AbstractEventLoop:
        if self._new_loop:
            self._loop = asyncio.new_event_loop()
        else:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()

        self._original_loop = asyncio.get_event_loop_policy().get_event_loop()
        asyncio.set_event_loop(self._loop)
        return self._loop

    async def _teardown(self) -> None:
        if self._loop and not self._loop.is_closed():
            # Cancel all pending tasks
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

            self._loop.close()

        if self._original_loop:
            asyncio.set_event_loop(self._original_loop)


class TempDirFixture(AsyncFixture[Path]):
    """Temporary directory fixture."""

    def __init__(self, prefix: str = "async_test_", cleanup: bool = True) -> None:
        super().__init__()
        self._prefix = prefix
        self._cleanup = cleanup
        self._temp_dir: Path | None = None

    async def _setup(self) -> Path:
        self._temp_dir = Path(tempfile.mkdtemp(prefix=self._prefix))
        return self._temp_dir

    async def _teardown(self) -> None:
        if self._cleanup and self._temp_dir and self._temp_dir.exists():
            shutil.rmtree(self._temp_dir, ignore_errors=True)


class TempFileFixture(AsyncFixture[Path]):
    """Temporary file fixture."""

    def __init__(
        self,
        content: str | bytes = "",
        suffix: str = "",
        prefix: str = "async_test_",
        mode: str = "w",
        cleanup: bool = True,
    ) -> None:
        super().__init__()
        self._content = content
        self._suffix = suffix
        self._prefix = prefix
        self._mode = mode
        self._cleanup = cleanup
        self._temp_file: Path | None = None

    async def _setup(self) -> Path:
        if isinstance(self._content, str):
            mode = self._mode
        else:
            mode = self._mode.replace("w", "wb").replace("r", "rb")

        fd, path = tempfile.mkstemp(suffix=self._suffix, prefix=self._prefix)
        self._temp_file = Path(path)

        try:
            with os.fdopen(fd, mode) as f:
                f.write(self._content)
        except Exception:
            os.close(fd)
            raise

        return self._temp_file

    async def _teardown(self) -> None:
        if self._cleanup and self._temp_file and self._temp_file.exists():
            with suppress(Exception):
                self._temp_file.unlink()


class MockServerFixture(AsyncFixture[Any]):
    """Mock HTTP server fixture."""

    def __init__(
        self, host: str = "127.0.0.1", port: int = 0, routes: dict[str, Callable] | None = None
    ) -> None:
        super().__init__()
        self._host = host
        self._port = port
        self._routes = routes or {}
        self._server = None
        self._actual_port = None

    async def _setup(self) -> str:
        from aiohttp import web

        app = web.Application()

        for path, handler in self._routes.items():
            app.router.add_route("*", path, handler)

        runner = web.AppRunner(app)
        await runner.setup()

        site = web.TCPSite(runner, self._host, self._port)
        await site.start()

        self._actual_port = site._server.sockets[0].getsockname()[1]
        self._server = runner

        return f"http://{self._host}:{self._actual_port}"

    async def _teardown(self) -> None:
        if self._server:
            await self._server.cleanup()


class DatabaseFixture(AsyncFixture[Any]):
    """Database connection fixture."""

    def __init__(
        self,
        db_url: str,
        *,
        create_tables: Callable | None = None,
        drop_tables: Callable | None = None,
    ) -> None:
        super().__init__()
        self._db_url = db_url
        self._create_tables = create_tables
        self._drop_tables = drop_tables
        self._pool = None

    async def _setup(self) -> Any:
        # This would connect to actual database
        # Implementation depends on database driver
        if self._create_tables:
            await self._create_tables(self._pool)
        return self._pool

    async def _teardown(self) -> None:
        if self._drop_tables:
            await self._drop_tables(self._pool)
        if self._pool:
            await self._pool.close()


class MessageQueueFixture(AsyncFixture[Any]):
    """Message queue fixture for testing."""

    def __init__(self) -> None:
        super().__init__()
        self._queues: dict[str, asyncio.Queue] = {}
        self._messages: dict[str, list[Any]] = {}

    async def _setup(self) -> "MessageQueueFixture":
        return self

    async def _teardown(self) -> None:
        self._queues.clear()
        self._messages.clear()

    def get_queue(self, name: str) -> asyncio.Queue:
        if name not in self._queues:
            self._queues[name] = asyncio.Queue()
        return self._queues[name]

    async def publish(self, queue: str, message: Any) -> None:
        q = self.get_queue(queue)
        await q.put(message)
        if queue not in self._messages:
            self._messages[queue] = []
        self._messages[queue].append(message)

    async def consume(self, queue: str, timeout: float = 1.0) -> Any:
        q = self.get_queue(queue)
        try:
            return await asyncio.wait_for(q.get(), timeout=timeout)
        except TimeoutError:
            return None

    def get_messages(self, queue: str) -> list[Any]:
        return self._messages.get(queue, []).copy()


class AsyncClientFixture(AsyncFixture[Any]):
    """Async HTTP client fixture."""

    def __init__(self, base_url: str = "", **client_kwargs) -> None:
        super().__init__()
        self._base_url = base_url
        self._client_kwargs = client_kwargs
        self._client = None

    async def _setup(self) -> Any:
        from aiohttp import ClientSession

        self._client = ClientSession(base_url=self._base_url, **self._client_kwargs)
        return self._client

    async def _teardown(self) -> None:
        if self._client:
            await self._client.close()


class RedisFixture(AsyncFixture[Any]):
    """Redis connection fixture."""

    def __init__(self, url: str = "redis://localhost:6379", db: int = 0) -> None:
        super().__init__()
        self._url = url
        self._db = db
        self._redis = None

    async def _setup(self) -> Any:
        try:
            import aioredis

            self._redis = await aioredis.create_redis_pool(self._url, db=self._db)
            return self._redis
        except ImportError:
            raise RuntimeError("aioredis required for RedisFixture")

    async def _teardown(self) -> None:
        if self._redis:
            self._redis.close()
            await self._redis.wait_closed()


class KafkaFixture(AsyncFixture[Any]):
    """Kafka producer/consumer fixture."""

    def __init__(
        self, bootstrap_servers: str = "localhost:9092", topics: list[str] | None = None
    ) -> None:
        super().__init__()
        self._bootstrap_servers = bootstrap_servers
        self._topics = topics or []
        self._producer = None
        self._consumer = None

    async def _setup(self) -> dict[str, Any]:
        try:
            from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

            self._producer = AIOKafkaProducer(bootstrap_servers=self._bootstrap_servers)
            await self._producer.start()

            consumers = {}
            for topic in self._topics:
                consumer = AIOKafkaConsumer(
                    topic, bootstrap_servers=self._bootstrap_servers, group_id=f"test-{topic}"
                )
                await consumer.start()
                consumers[topic] = consumer

            return {"producer": self._producer, "consumers": consumers}
        except ImportError:
            raise RuntimeError("aiokafka required for KafkaFixture")

    async def _teardown(self) -> None:
        if self._producer:
            await self._producer.stop()

        if self._consumer:
            await self._consumer.stop()


# Fixture composition
class CompositeFixture(AsyncFixture[dict[str, Any]]):
    """Combine multiple fixtures."""

    def __init__(self, fixtures: dict[str, AsyncFixture]) -> None:
        super().__init__()
        self._fixtures = fixtures
        self._results: dict[str, Any] = {}

    async def _setup(self) -> dict[str, Any]:
        for name, fixture in self._fixtures.items():
            self._results[name] = await fixture.setup()
        return self._results

    async def _teardown(self) -> None:
        for fixture in self._fixtures.values():
            await fixture.teardown()


# Pytest fixtures
if PYTEST_AVAILABLE:

    @pytest.fixture
    def event_loop():
        """Event loop fixture."""
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()

    @pytest.fixture
    async def async_event_loop():
        """Async event loop fixture."""
        loop = asyncio.get_running_loop()
        yield loop

    @pytest.fixture
    def temp_dir():
        """Temporary directory fixture."""
        fixture = TempDirFixture()
        path = asyncio.run(fixture.setup())
        yield path
        asyncio.run(fixture.teardown())

    @pytest.fixture
    def temp_file():
        """Temporary file fixture."""
        fixture = TempFileFixture()
        path = asyncio.run(fixture.setup())
        yield path
        asyncio.run(fixture.teardown())

    @pytest.fixture
    def mock_server():
        """Mock HTTP server fixture."""
        fixture = MockServerFixture()
        url = asyncio.run(fixture.setup())
        yield url
        asyncio.run(fixture.teardown())

    @pytest.fixture
    def message_queue():
        """Message queue fixture."""
        fixture = MessageQueueFixture()
        yield asyncio.run(fixture.setup())
        asyncio.run(fixture.teardown())

    @pytest.fixture
    def async_client():
        """Async HTTP client fixture."""
        fixture = AsyncClientFixture()
        client = asyncio.run(fixture.setup())
        yield client
        asyncio.run(fixture.teardown())


# Async fixture decorator
def async_fixture(scope: str = "function", **kwargs) -> Callable:
    """Decorator to create async pytest fixture."""
    if not PYTEST_AVAILABLE:

        def decorator(func):
            return func

        return decorator

    def decorator(func: Callable) -> Callable:
        @pytest.fixture(scope=scope, **kwargs)
        @functools.wraps(func)
        def wrapper(*args, **kw):
            return asyncio.run(func(*args, **kw))

        return wrapper

    return decorator


def async_fixture_factory(
    setup_func: Callable[[], Awaitable[T]],
    teardown_func: Callable[[T], Awaitable[None]] | None = None,
    **fixture_kwargs,
) -> Callable:
    """Create fixture from setup/teardown functions."""

    class FunctionFixture(AsyncFixture[T]):
        async def _setup(self) -> T:
            return await setup_func()

        async def _teardown(self) -> None:
            if teardown_func and hasattr(self, "_value"):
                await teardown_func(self._value)

        async def setup(self) -> T:
            self._value = await super().setup()
            return self._value

    fixture = FunctionFixture()

    @async_fixture(**fixture_kwargs)
    async def _fixture():
        async with fixture.use() as value:
            yield value

    return _fixture


# Built-in fixtures
event_loop_fixture = async_fixture_factory(
    lambda: asyncio.get_event_loop_policy().new_event_loop(),
    lambda loop: loop.close() if not loop.is_closed() else None,
)

temp_dir_fixture = async_fixture_factory(
    lambda: Path(tempfile.mkdtemp(prefix="test_")),
    lambda p: shutil.rmtree(p, ignore_errors=True) if p.exists() else None,
)


# Fixture utilities
async def create_test_environment(fixtures: dict[str, AsyncFixture]) -> dict[str, Any]:
    """Set up multiple fixtures at once."""
    results = {}
    for name, fixture in fixtures.items():
        results[name] = await fixture.setup()
    return results


async def cleanup_test_environment(fixtures: dict[str, AsyncFixture]) -> None:
    """Clean up multiple fixtures."""
    for fixture in fixtures.values():
        with suppress(Exception):
            await fixture.teardown()


class FixtureManager:
    """Manage multiple fixtures in tests."""

    def __init__(self) -> None:
        self._fixtures: dict[str, AsyncFixture] = {}

    def register(self, name: str, fixture: AsyncFixture) -> None:
        self._fixtures[name] = fixture

    async def setup_all(self) -> dict[str, Any]:
        results = {}
        for name, fixture in self._fixtures.items():
            results[name] = await fixture.setup()
        return results

    async def teardown_all(self) -> None:
        for fixture in self._fixtures.values():
            with suppress(Exception):
                await fixture.teardown()

    def get(self, name: str) -> AsyncFixture | None:
        return self._fixtures.get(name)

    @asynccontextmanager
    async def use_all(self) -> AsyncGenerator[dict[str, Any], None]:
        results = await self.setup_all()
        try:
            yield results
        finally:
            await self.teardown_all()


def fixture_manager() -> FixtureManager:
    """Create fixture manager."""
    return FixtureManager()
