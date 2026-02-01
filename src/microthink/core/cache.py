"""
Response caching for MicroThink.

Provides an LRU cache with TTL to avoid redundant model calls
for identical prompts.
"""

import hashlib
import time
from collections import OrderedDict
from threading import Lock
from typing import Any, Dict, Optional


def make_cache_key(
    model: str,
    behavior: str,
    prompt: str,
    expect_json: bool,
    web_search: bool,
) -> str:
    """
    Generate a deterministic cache key from request parameters.

    Args:
        model: The model name.
        behavior: The persona behavior.
        prompt: The user prompt.
        expect_json: Whether JSON output is expected.
        web_search: Whether web search is enabled.

    Returns:
        A hex digest string suitable as a cache key.
    """
    key_parts = f"{model}|{behavior}|{prompt}|{expect_json}|{web_search}"
    return hashlib.sha256(key_parts.encode()).hexdigest()


class ResponseCache:
    """
    Thread-safe LRU cache with TTL for caching model responses.

    Attributes:
        max_size: Maximum number of entries to store.
        ttl: Time-to-live in seconds (0 = disabled).

    Example:
        >>> cache = ResponseCache(max_size=1000, ttl=3600)
        >>> cache.set("key", "value")
        >>> cache.get("key")
        'value'
    """

    def __init__(self, max_size: int = 1000, ttl: float = 3600) -> None:
        """
        Initialize the cache.

        Args:
            max_size: Maximum entries to store (default: 1000).
            ttl: Time-to-live in seconds (default: 3600). Set to 0 to disable.
        """
        self.max_size = max_size
        self.ttl = ttl
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value from the cache.

        Args:
            key: The cache key.

        Returns:
            The cached value, or None if not found or expired.
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]

            # Check TTL
            if self.ttl > 0 and time.time() - entry["timestamp"] > self.ttl:
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return entry["value"]

    def set(self, key: str, value: Any) -> None:
        """
        Store a value in the cache.

        Args:
            key: The cache key.
            value: The value to store.
        """
        # If TTL is 0, caching is disabled
        if self.ttl <= 0:
            return

        with self._lock:
            # Remove if exists (to update timestamp)
            if key in self._cache:
                del self._cache[key]

            # Evict oldest if at capacity
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)

            self._cache[key] = {
                "value": value,
                "timestamp": time.time(),
            }

    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with hits, misses, size, and hit_rate.
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._cache),
                "hit_rate": hit_rate,
            }
