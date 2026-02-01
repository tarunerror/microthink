"""Tests for response caching."""

import time

import pytest
from microthink.core.cache import ResponseCache, make_cache_key


class TestResponseCache:
    """Tests for the ResponseCache class."""

    def test_cache_stores_and_retrieves(self):
        """Cache stores and retrieves values."""
        cache = ResponseCache(max_size=100, ttl=60)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_cache_returns_none_for_missing(self):
        """Cache returns None for missing keys."""
        cache = ResponseCache(max_size=100, ttl=60)
        assert cache.get("nonexistent") is None

    def test_cache_respects_ttl(self):
        """Cache entries expire after TTL."""
        cache = ResponseCache(max_size=100, ttl=0.1)  # 100ms TTL
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        time.sleep(0.15)  # Wait for expiry
        assert cache.get("key1") is None

    def test_cache_respects_max_size(self):
        """Cache evicts oldest entries when full."""
        cache = ResponseCache(max_size=2, ttl=60)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # Should evict key1
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

    def test_cache_clear(self):
        """Cache can be cleared."""
        cache = ResponseCache(max_size=100, ttl=60)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_cache_stats(self):
        """Cache tracks hit/miss statistics."""
        cache = ResponseCache(max_size=100, ttl=60)
        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("missing")  # Miss
        stats = cache.stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(0.667, rel=0.01)

    def test_cache_disabled_when_ttl_zero(self):
        """Cache with ttl=0 acts as disabled."""
        cache = ResponseCache(max_size=100, ttl=0)
        cache.set("key1", "value1")
        assert cache.get("key1") is None  # Immediately expired


class TestCacheKeyGeneration:
    """Tests for cache key generation."""

    def test_make_cache_key(self):
        """Cache key is deterministic."""
        key1 = make_cache_key(
            model="llama3.2:3b",
            behavior="coder",
            prompt="Hello",
            expect_json=True,
            web_search=False,
        )
        key2 = make_cache_key(
            model="llama3.2:3b",
            behavior="coder",
            prompt="Hello",
            expect_json=True,
            web_search=False,
        )
        assert key1 == key2

    def test_different_prompts_different_keys(self):
        """Different prompts produce different keys."""
        key1 = make_cache_key("model", "general", "Hello", False, False)
        key2 = make_cache_key("model", "general", "World", False, False)
        assert key1 != key2

    def test_different_behaviors_different_keys(self):
        """Different behaviors produce different keys."""
        key1 = make_cache_key("model", "general", "Hello", False, False)
        key2 = make_cache_key("model", "coder", "Hello", False, False)
        assert key1 != key2


class TestClientCacheIntegration:
    """Tests for caching in MicroThinkClient."""

    def test_client_accepts_cache_params(self):
        """Client accepts cache configuration."""
        from microthink import MicroThinkClient

        client = MicroThinkClient(
            cache=True,
            cache_ttl=3600,
            cache_max_size=500,
        )
        assert client._cache is not None
        assert client._cache.ttl == 3600
        assert client._cache.max_size == 500

    def test_client_cache_disabled_by_default(self):
        """Caching is disabled by default."""
        from microthink import MicroThinkClient

        client = MicroThinkClient()
        assert client._cache is None

    def test_client_cache_stats(self):
        """Client exposes cache statistics."""
        from microthink import MicroThinkClient

        client = MicroThinkClient(cache=True)
        stats = client.cache_stats()
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats

    def test_client_cache_clear(self):
        """Client can clear the cache."""
        from microthink import MicroThinkClient

        client = MicroThinkClient(cache=True)
        client.clear_cache()  # Should not raise
