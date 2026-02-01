"""Tests for metrics/telemetry."""

from unittest.mock import MagicMock, patch

import pytest


class TestMetrics:
    """Tests for Metrics class."""

    def test_metrics_importable(self):
        """Metrics class is importable."""
        from microthink.metrics import Metrics

        assert Metrics is not None

    def test_metrics_has_counters(self):
        """Metrics has request counters."""
        from microthink.metrics import Metrics

        m = Metrics()
        assert hasattr(m, "total_requests")
        assert hasattr(m, "successful_requests")
        assert hasattr(m, "failed_requests")
        assert hasattr(m, "total_retries")

    def test_metrics_has_cache_stats(self):
        """Metrics has cache statistics."""
        from microthink.metrics import Metrics

        m = Metrics()
        assert hasattr(m, "cache_hits")
        assert hasattr(m, "cache_misses")

    def test_metrics_has_latency(self):
        """Metrics tracks latency."""
        from microthink.metrics import Metrics

        m = Metrics()
        assert hasattr(m, "avg_latency_ms")
        assert hasattr(m, "total_latency_ms")

    def test_metrics_record_request(self):
        """record_request increments counters."""
        from microthink.metrics import Metrics

        m = Metrics()
        m.record_request(success=True, latency_ms=100)

        assert m.total_requests == 1
        assert m.successful_requests == 1
        assert m.total_latency_ms == 100

    def test_metrics_cache_hit_rate(self):
        """cache_hit_rate calculated correctly."""
        from microthink.metrics import Metrics

        m = Metrics()
        m.record_cache_hit()
        m.record_cache_miss()
        m.record_cache_hit()

        assert m.cache_hit_rate == pytest.approx(2 / 3)

    def test_metrics_reset(self):
        """reset() clears all metrics."""
        from microthink.metrics import Metrics

        m = Metrics()
        m.record_request(success=True, latency_ms=100)
        m.reset()

        assert m.total_requests == 0


class TestClientMetrics:
    """Tests for metrics integration with client."""

    def test_client_has_metrics(self):
        """Client has metrics attribute."""
        from microthink import MicroThinkClient

        client = MicroThinkClient()
        assert hasattr(client, "metrics")

    def test_metrics_updated_on_request(self):
        """Metrics are updated after requests."""
        from microthink import MicroThinkClient

        with patch("microthink.client.ollama.Client") as mock_ollama:
            mock_client = MagicMock()
            mock_ollama.return_value = mock_client
            mock_client.chat.return_value = {
                "message": {"content": "<answer>Response</answer>"}
            }

            client = MicroThinkClient()
            client.generate("Test prompt")

            assert client.metrics.total_requests == 1
            assert client.metrics.successful_requests == 1

    def test_metrics_track_cache(self):
        """Metrics track cache hits/misses."""
        from microthink import MicroThinkClient

        with patch("microthink.client.ollama.Client") as mock_ollama:
            mock_client = MagicMock()
            mock_ollama.return_value = mock_client
            mock_client.chat.return_value = {
                "message": {"content": "<answer>Response</answer>"}
            }

            client = MicroThinkClient(cache=True)
            client.generate("Test prompt")  # Miss
            client.generate("Test prompt")  # Hit

            assert client.metrics.cache_misses == 1
            assert client.metrics.cache_hits == 1
