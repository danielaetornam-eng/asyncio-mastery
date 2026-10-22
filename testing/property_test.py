"""
Property-Based Testing for Async Code

Utilities for property-based testing of async systems.
"""

import asyncio
import functools
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, TypeVar

try:
    import hypothesis
    from hypothesis import HealthCheck, Verbosity, given, settings
    from hypothesis import strategies as st
    from hypothesis.strategies import SearchStrategy

    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    hypothesis = None
    st = None
    given = None
    settings = None
    Verbosity = None
    HealthCheck = None
    SearchStrategy = None


T = TypeVar("T")
S = TypeVar("S")


class AsyncPropertyTest:
    """Base class for async property tests."""

    def __init__(self) -> None:
        self._setup_funcs: list[Callable[[], Awaitable[None]]] = []
        self._teardown_funcs: list[Callable[[], Awaitable[None]]] = []
        self._invariants: list[Callable[[], Awaitable[bool]]] = []

    async def setup(self) -> None:
        """Run setup functions."""
        for func in self._setup_funcs:
            await func()

    async def teardown(self) -> None:
        """Run teardown functions."""
        for func in self._teardown_funcs:
            await func()

    def add_setup(self, func: Callable[[], Awaitable[None]]) -> None:
        self._setup_funcs.append(func)

    def add_teardown(self, func: Callable[[], Awaitable[None]]) -> None:
        self._teardown_funcs.append(func)

    def add_invariant(self, func: Callable[[], Awaitable[bool]]) -> None:
        self._invariants.append(func)

    async def check_invariants(self) -> bool:
        """Check all invariants."""
        for invariant in self._invariants:
            if not await invariant():
                return False
        return True

    @asynccontextmanager
    async def test_context(self) -> AsyncGenerator[None, None]:
        """Context manager for test lifecycle."""
        await self.setup()
        try:
            yield
        finally:
            await self.teardown()


def async_property(*strategies, settings_kwargs: dict | None = None, **kwargs) -> Callable:
    """Decorator for async property tests with Hypothesis."""
    if not HYPOTHESIS_AVAILABLE:

        def decorator(func):
            return func

        return decorator

    test_settings = settings(**(settings_kwargs or {}))

    def decorator(func: Callable) -> Callable:
        @given(*strategies, **kwargs)
        @settings(test_settings)
        @functools.wraps(func)
        def wrapper(*args, **kw):
            async def run():
                return await func(*args, **kw)

            return asyncio.run(run())

        return wrapper

    return decorator


def async_stateful_test(
    state_machine_class: type, *, settings_kwargs: dict | None = None
) -> Callable:
    """Decorator for stateful async tests."""
    if not HYPOTHESIS_AVAILABLE:

        def decorator(func):
            return func

        return decorator

    test_settings = settings(**(settings_kwargs or {}))

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Hypothesis stateful testing
            from hypothesis.stateful import RuleBasedStateMachine, run_state_machine_as_test

            class AsyncStateMachine(state_machine_class, RuleBasedStateMachine):
                pass

            @test_settings
            class TestRunner:
                def test_run(self) -> None:
                    run_state_machine_as_test(AsyncStateMachine)

            return TestRunner().test_run()

        return wrapper

    return decorator


# Async strategies for Hypothesis
if HYPOTHESIS_AVAILABLE:

    def async_just(value: T) -> SearchStrategy[T]:
        return st.just(value)

    def async_none() -> SearchStrategy[None]:
        return st.none()

    def async_booleans() -> SearchStrategy[bool]:
        return st.booleans()

    def async_integers(
        min_value: int | None = None, max_value: int | None = None
    ) -> SearchStrategy[int]:
        return st.integers(min_value=min_value, max_value=max_value)

    def async_floats(
        min_value: float | None = None,
        max_value: float | None = None,
        allow_nan: bool = False,
        allow_infinity: bool = False,
    ) -> SearchStrategy[float]:
        return st.floats(
            min_value=min_value,
            max_value=max_value,
            allow_nan=allow_nan,
            allow_infinity=allow_infinity,
        )

    def async_text(
        alphabet: str | None = None, min_size: int = 0, max_size: int | None = None
    ) -> SearchStrategy[str]:
        return st.text(alphabet=alphabet, min_size=min_size, max_size=max_size)

    def async_binary(min_size: int = 0, max_size: int | None = None) -> SearchStrategy[bytes]:
        return st.binary(min_size=min_size, max_size=max_size)

    def async_lists(
        elements: SearchStrategy[T],
        min_size: int = 0,
        max_size: int | None = None,
        unique: bool = False,
    ) -> SearchStrategy[list[T]]:
        return st.lists(elements, min_size=min_size, max_size=max_size, unique=unique)

    def async_sets(
        elements: SearchStrategy[T], min_size: int = 0, max_size: int | None = None
    ) -> SearchStrategy[set[T]]:
        return st.sets(elements, min_size=min_size, max_size=max_size)

    def async_dicts(
        keys: SearchStrategy,
        values: SearchStrategy[T],
        min_size: int = 0,
        max_size: int | None = None,
    ) -> SearchStrategy[dict]:
        return st.dictionaries(keys, values, min_size=min_size, max_size=max_size)

    def async_tuples(*elements: SearchStrategy) -> SearchStrategy[tuple]:
        return st.tuples(*elements)

    def async_one_of(*strategies: SearchStrategy) -> SearchStrategy:
        return st.one_of(*strategies)

    def async_sampled_from(elements: list[T]) -> SearchStrategy[T]:
        return st.sampled_from(elements)

    def async_builds(
        target: Callable[..., T], *args: SearchStrategy, **kwargs: SearchStrategy
    ) -> SearchStrategy[T]:
        return st.builds(target, *args, **kwargs)

    def async_composite(
        func: Callable[..., T], *args: SearchStrategy, **kwargs: SearchStrategy
    ) -> SearchStrategy[T]:
        return st.composite(func)(*args, **kwargs)

    def async_recursive(
        base: SearchStrategy[T],
        extend: Callable[[SearchStrategy[T]], SearchStrategy[T]],
        max_leaves: int = 50,
    ) -> SearchStrategy[T]:
        return st.recursive(base, extend, max_leaves=max_leaves)

    # Async-specific strategies

    def async_coroutines(
        return_strategy: SearchStrategy[T],
    ) -> SearchStrategy[Callable[..., Awaitable[T]]]:
        """Generate async functions that return values from strategy."""

        @st.composite
        def _coro(draw):
            value = draw(return_strategy)

            async def coro(*args, **kwargs):
                return value

            return coro

        return _coro()

    def async_delays(min_delay: float = 0.0, max_delay: float = 1.0) -> SearchStrategy[float]:
        """Generate random delays for testing."""
        return st.floats(min_value=min_delay, max_value=max_delay)

    def async_timeouts(
        min_timeout: float = 0.1, max_timeout: float = 10.0
    ) -> SearchStrategy[float]:
        """Generate timeout values."""
        return st.floats(min_value=min_timeout, max_value=max_timeout)

    def async_task_results(
        success_strategy: SearchStrategy[T],
        error_strategy: SearchStrategy[Exception] = st.exceptions(),
        success_probability: float = 0.8,
    ) -> SearchStrategy[T | Exception]:
        """Generate either success results or exceptions."""

        @st.composite
        def _result(draw):
            if draw(st.floats(0, 1)) < success_probability:
                return draw(success_strategy)
            return draw(error_strategy)

        return _result()

    def async_sequences(
        element_strategy: SearchStrategy[T], min_length: int = 0, max_length: int = 10
    ) -> SearchStrategy[list[T]]:
        """Generate sequences of async operations."""
        return st.lists(element_strategy, min_size=min_length, max_size=max_length)

    def async_concurrent_operations(
        operation_strategy: SearchStrategy[Callable[..., Awaitable[T]]],
        min_count: int = 1,
        max_count: int = 5,
    ) -> SearchStrategy[list[Callable[..., Awaitable[T]]]]:
        """Generate sets of concurrent operations."""
        return st.lists(operation_strategy, min_size=min_count, max_size=max_count)


# Stateful testing for async systems
if HYPOTHESIS_AVAILABLE:
    from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant

    class AsyncStateMachine(RuleBasedStateMachine):
        """Base class for async state machine tests."""

        def __init__(self) -> None:
            super().__init__()
            self._async_setup_done = False

        @initialize()
        def async_init(self) -> None:
            self._async_setup_done = True

        async def run_async(self, coro: Awaitable[T]) -> T:
            """Run coroutine in test loop."""
            return asyncio.run(coro)

        def run_sync(self, coro: Awaitable[T]) -> T:
            """Run coroutine synchronously."""
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)

    class AsyncRuleBasedStateMachine(AsyncStateMachine):
        """State machine with async rules."""

        def __init__(self) -> None:
            super().__init__()
            self._rule_results: list[Any] = []

        async def execute_rule(self, rule_name: str, coro: Awaitable[T]) -> T:
            """Execute a rule and record result."""
            result = await coro
            self._rule_results.append((rule_name, result))
            return result

        @invariant()
        async def check_invariants(self) -> bool:
            """Check all invariants - override in subclass."""
            return True


# Async model-based testing
class AsyncModel:
    """Model for async system behavior."""

    def __init__(self) -> None:
        self._state = {}
        self._history: list[dict] = []

    async def apply(self, operation: str, *args, **kwargs) -> Any:
        """Apply operation to model."""
        raise NotImplementedError

    async def check_invariant(self) -> bool:
        """Check model invariant."""
        return True

    def record(self, operation: str, args: tuple, kwargs: dict, result: Any) -> None:
        self._history.append(
            {"operation": operation, "args": args, "kwargs": kwargs, "result": result}
        )


class AsyncSystemUnderTest:
    """Wrapper for system under test."""

    def __init__(self, sut: Any) -> None:
        self._sut = sut

    async def apply(self, operation: str, *args, **kwargs) -> Any:
        method = getattr(self._sut, operation)
        return await method(*args, **kwargs)


class AsyncModelTest:
    """Model-based test for async systems."""

    def __init__(self, model: AsyncModel, sut: AsyncSystemUnderTest) -> None:
        self._model = model
        self._sut = sut

    async def check_operation(self, operation: str, *args, **kwargs) -> bool:
        """Check that SUT matches model for operation."""
        # Apply to model
        model_result = await self._model.apply(operation, *args, **kwargs)

        # Apply to SUT
        sut_result = await self._sut.apply(operation, *args, **kwargs)

        # Compare results (customize as needed)
        return model_result == sut_result

    async def check_invariants(self) -> bool:
        """Check all invariants."""
        return await self._model.check_invariant()


# Concurrency testing strategies
def async_interleavings(
    operations: list[Callable[..., Awaitable[T]]], max_concurrent: int = 3
) -> SearchStrategy[list[Callable[..., Awaitable[T]]]]:
    """Generate interleavings of concurrent operations."""

    @st.composite
    def _interleavings(draw):
        selected = draw(st.lists(st.sampled_from(operations), min_size=1, max_size=max_concurrent))
        return selected

    return _interleavings()


def async_race_conditions(
    operations: list[Callable[..., Awaitable[T]]], shared_state: Any
) -> SearchStrategy[list[Callable[..., Awaitable[T]]]]:
    """Generate operations that might race."""
    return st.lists(st.sampled_from(operations), min_size=2, max_size=5)


# Property test helpers
def assert_eventually_true(
    condition: Callable[[], bool], timeout: float = 5.0, interval: float = 0.1
) -> None:
    """Assert condition becomes true eventually."""
    start = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start < timeout:
        if condition():
            return
        asyncio.sleep(interval)
    raise AssertionError("Condition never became true")


async def assert_async_eventually_true(
    condition: Callable[[], Awaitable[bool]], timeout: float = 5.0, interval: float = 0.1
) -> None:
    """Assert async condition becomes true eventually."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if await condition():
            return
        await asyncio.sleep(interval)
    raise AssertionError("Condition never became true")


def run_property_test(
    test_func: Callable, *args, max_examples: int = 100, timeout: float = 60.0, **kwargs
) -> dict[str, Any]:
    """Run a property test with custom settings."""
    if not HYPOTHESIS_AVAILABLE:
        raise RuntimeError("hypothesis is required. Install with: pip install hypothesis")

    # Run with custom settings
    test = hypothesis.given(*args, **kwargs)(test_func)
    test = hypothesis.settings(
        max_examples=max_examples,
        deadline=timeout * 1000,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )(test)

    try:
        test()
        return {"status": "passed"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


import time
