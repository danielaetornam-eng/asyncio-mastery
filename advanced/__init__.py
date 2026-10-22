"""
Advanced Patterns

Circuit breakers, retry policies, bulkheads, and resilience patterns.
"""

from .bulkhead import (
    Bulkhead,
    BulkheadConfig,
    BulkheadRejected,
    SemaphoreBulkhead,
    ThreadPoolBulkhead,
)
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerOpen, CircuitState
from .fallback import Fallback, FallbackConfig, fallback
from .hedging import HedgingPolicy, hedge
from .retry import ExponentialBackoff, FibonacciBackoff, FixedDelay, RetryPolicy, retry, retry_async
from .timeout import Timeout, TimeoutConfig, TimeoutError, with_timeout

__all__ = [
    "Bulkhead",
    "BulkheadConfig",
    "BulkheadRejected",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerOpen",
    "CircuitState",
    "ExponentialBackoff",
    "Fallback",
    "FallbackConfig",
    "FibonacciBackoff",
    "FixedDelay",
    "HedgingPolicy",
    "RetryPolicy",
    "SemaphoreBulkhead",
    "ThreadPoolBulkhead",
    "Timeout",
    "TimeoutConfig",
    "TimeoutError",
    "fallback",
    "hedge",
    "retry",
    "retry_async",
    "with_timeout",
]
