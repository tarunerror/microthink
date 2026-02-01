"""
Metrics and telemetry for MicroThink.

Provides built-in observability for request counts,
latencies, cache statistics, and error rates.
"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Metrics:
    """
    Telemetry metrics for MicroThink operations.

    Example:
        >>> client = MicroThinkClient()
        >>> client.generate("Hello")
        >>> print(client.metrics.total_requests)
        1
        >>> print(client.metrics.avg_latency_ms)
        150.5
    """

    # Request counters
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_retries: int = 0

    # Cache statistics
    cache_hits: int = 0
    cache_misses: int = 0

    # Latency tracking
    total_latency_ms: float = 0.0
    _latency_count: int = field(default=0, repr=False)

    @property
    def avg_latency_ms(self) -> float:
        """Average request latency in milliseconds."""
        if self._latency_count == 0:
            return 0.0
        return self.total_latency_ms / self._latency_count

    @property
    def cache_hit_rate(self) -> float:
        """Cache hit rate as a ratio (0.0 to 1.0)."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total

    @property
    def success_rate(self) -> float:
        """Success rate as a ratio (0.0 to 1.0)."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests

    @property
    def retry_rate(self) -> float:
        """Average retries per request."""
        if self.total_requests == 0:
            return 0.0
        return self.total_retries / self.total_requests

    def record_request(self, success: bool, latency_ms: float) -> None:
        """Record a completed request."""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        self.total_latency_ms += latency_ms
        self._latency_count += 1

    def record_retry(self) -> None:
        """Record a retry attempt."""
        self.total_retries += 1

    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        self.cache_hits += 1

    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        self.cache_misses += 1

    def reset(self) -> None:
        """Reset all metrics to zero."""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_retries = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_latency_ms = 0.0
        self._latency_count = 0

    def to_dict(self) -> Dict[str, Any]:
        """Export metrics as a dictionary."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_retries": self.total_retries,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "avg_latency_ms": self.avg_latency_ms,
            "cache_hit_rate": self.cache_hit_rate,
            "success_rate": self.success_rate,
            "retry_rate": self.retry_rate,
        }
