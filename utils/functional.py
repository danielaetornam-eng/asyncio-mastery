"""
Functional Programming Utilities

Functional programming helpers for async code.
"""

import asyncio
import functools
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable
from typing import Any, Generic, TypeVar

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


def compose(*funcs: Callable) -> Callable:
    """Compose functions right to left."""

    def composed(*args, **kwargs):
        result = funcs[-1](*args, **kwargs)
        for func in reversed(funcs[:-1]):
            result = func(result)
        return result

    return composed


def pipe(*funcs: Callable) -> Callable:
    """Pipe functions left to right."""

    def piped(*args, **kwargs):
        result = funcs[0](*args, **kwargs)
        for func in funcs[1:]:
            result = func(result)
        return result

    return piped


def curry(func: Callable) -> Callable:
    """Curry a function."""

    def curried(*args, **kwargs):
        if len(args) + len(kwargs) >= func.__code__.co_argcount:
            return func(*args, **kwargs)

        def partial_func(*more_args, **more_kwargs):
            return func(*(args + more_args), **{**kwargs, **more_kwargs})

        return partial_func

    return curried


def partial(func: Callable, *args, **kwargs) -> Callable:
    """Create partial function."""
    return functools.partial(func, *args, **kwargs)


def identity(x: T) -> T:
    """Identity function."""
    return x


def constant(value: T) -> Callable[..., T]:
    """Constant function."""
    return lambda *args, **kwargs: value


def flip(func: Callable) -> Callable:
    """Flip first two arguments of a function."""

    def flipped(*args, **kwargs):
        if len(args) >= 2:
            return func(args[1], args[0], *args[2:])
        return func(*args, **kwargs)

    return func


# Async functional helpers


async def map_async(
    func: Callable[[T], Awaitable[U]], iterable: Iterable[T], max_concurrent: int | None = None
) -> list[U]:
    """Apply async function to each item in iterable."""
    if max_concurrent:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def limited(item):
            async with semaphore:
                return await func(item)

        return await asyncio.gather(*[limited(item) for item in iterable])

    return await asyncio.gather(*[func(item) for item in iterable])


async def filter_async(predicate: Callable[[T], Awaitable[bool]], iterable: list[T]) -> list[T]:
    """Filter iterable with async predicate."""
    results = []
    for item in iterable:
        if await predicate(item):
            results.append(item)
    return results


async def reduce_async(func: Callable[[U, T], Awaitable[U]], iterable: list[T], initial: U) -> U:
    """Reduce iterable with async function."""
    accumulator = initial
    for item in iterable:
        accumulator = await func(accumulator, item)
    return accumulator


async def flat_map_async(func: Callable[[T], Awaitable[list[U]]], iterable: Iterable[T]) -> list[U]:
    """Map and flatten with async function."""
    results = []
    for item in iterable:
        mapped = await func(item)
        results.extend(mapped)
    return results


async def chunked_async(iterable: list[T], chunk_size: int) -> AsyncGenerator[list[T], None]:
    """Yield chunks of iterable."""
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i : i + chunk_size]


async def batch_process_async(
    items: list[T], processor: Callable[[list[T]], Awaitable[list[U]]], batch_size: int = 100
) -> list[U]:
    """Process items in batches."""
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        batch_results = await processor(batch)
        results.extend(batch_results)
    return results


def async_partial(func: Callable, *args, **kwargs) -> Callable:
    """Create partial async function."""

    async def wrapper(*more_args, **more_kwargs):
        return await func(*args, *more_args, **{**kwargs, **more_kwargs})

    return wrapper


def async_partial_method(method: Callable, *args, **kwargs) -> Callable:
    """Create partial method for async function."""

    async def wrapper(self, *more_args, **more_kwargs):
        return await method(self, *args, *more_args, **{**kwargs, **more_kwargs})

    return wrapper


class AsyncFunction(Generic[T, U]):
    """Wrapper for async functions with functional methods."""

    def __init__(self, func: Callable[..., Awaitable[U]]) -> None:
        self._func = func

    async def __call__(self, *args, **kwargs) -> Any:
        return await self._func(*args, **kwargs)

    def then(self, next_func: Callable[[Any], Awaitable[V]]) -> "AsyncFunction[T, V]":
        """Chain with another function."""

        async def chained(*args, **kwargs):
            result = await self._func(*args, **kwargs)
            return await next_func(result)

        return AsyncFunction(chained)

    def map(self, transform: Callable[[Any], Awaitable[V]]) -> "AsyncFunction[T, V]":
        """Transform result."""

        async def mapped(*args, **kwargs):
            result = await self._func(*args, **kwargs)
            return await transform(result)

        return AsyncFunction(mapped)

    def filter(self, predicate: Callable[[Any], Awaitable[bool]]) -> "AsyncFunction[T, U]":
        """Filter result."""

        async def filtered(*args, **kwargs):
            result = await self._func(*args, **kwargs)
            if await predicate(result):
                return result
            raise ValueError("Filter condition not met")

        return AsyncFunction(filtered)

    def catch(self, handler: Callable[[Exception], Awaitable[V]]) -> "AsyncFunction[T, U | V]":
        """Handle errors."""

        async def caught(*args, **kwargs):
            try:
                return await self._func(*args, **kwargs)
            except Exception as e:
                return await handler(e)

        return AsyncFunction(caught)


def async_fn(func: Callable[..., Awaitable[U]]) -> AsyncFunction[Any, U]:
    """Create AsyncFunction from async function."""
    return AsyncFunction(func)


# Higher-order async functions


def async_memoize(maxsize: int = 128) -> Callable:
    """Memoize async function."""

    def decorator(func):
        lock = asyncio.Lock()

        async def wrapper(*args, **kwargs):
            key = (args, frozenset(kwargs.items()))

            async with lock:
                if key in wrapper.cache:
                    return wrapper.cache[key]

            result = await func(*args, **kwargs)

            async with lock:
                if len(wrapper.cache) >= 128:
                    # Remove oldest
                    oldest = next(iter(wrapper.cache))
                    del wrapper.cache[oldest]
                wrapper.cache[key] = result

            return result

        wrapper.cache = {}
        wrapper.cache_clear = wrapper.cache.clear
        return wrapper


def async_lru_cache(maxsize: int = 128) -> Callable:
    """LRU cache for async functions."""

    def decorator(func):
        order = []
        lock = asyncio.Lock()

        async def wrapper(*args, **kwargs):
            key = (args, frozenset(kwargs.items()))

            async with lock:
                if key in wrapper.cache:
                    # Move to end
                    order.remove(key)
                    order.append(key)
                    return wrapper.cache[key]

            result = await func(*args, **kwargs)

            async with lock:
                if len(wrapper.cache) >= 128:
                    oldest = order.pop(0)
                    del wrapper.cache[oldest]
                wrapper.cache[key] = result
                order.append(key)

            return result

        wrapper.cache = {}
        wrapper.order = []
        wrapper.cache_clear = lambda: (wrapper.cache.clear(), order.clear())
        return wrapper

    return decorator
