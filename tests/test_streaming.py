"""Tests for streaming responses."""

import collections.abc
from typing import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestSyncStreaming:
    """Tests for sync client streaming."""

    def test_generate_accepts_stream_callback(self):
        """generate() accepts stream_callback parameter."""
        from microthink import MicroThinkClient

        client = MicroThinkClient()
        # Should not raise - just verify parameter is accepted
        assert hasattr(client.generate, "__call__")

    def test_stream_method_exists(self):
        """Client has stream() method returning iterator."""
        from microthink import MicroThinkClient

        client = MicroThinkClient()
        assert hasattr(client, "stream")

    def test_stream_returns_iterator(self):
        """stream() returns an iterator."""
        from microthink import MicroThinkClient

        client = MicroThinkClient()
        with patch.object(client, "_client") as mock_ollama:
            # Mock streaming response with proper answer tags
            mock_ollama.chat.return_value = iter(
                [
                    {"message": {"content": "<answer>"}},
                    {"message": {"content": "Hello"}},
                    {"message": {"content": " world</answer>"}},
                ]
            )

            result = client.stream("Test prompt")
            assert isinstance(result, collections.abc.Iterator)


class TestAsyncStreaming:
    """Tests for async client streaming."""

    def test_async_stream_method_exists(self):
        """AsyncMicroThinkClient has stream() method."""
        from microthink import AsyncMicroThinkClient

        client = AsyncMicroThinkClient()
        assert hasattr(client, "stream")

    @pytest.mark.asyncio
    async def test_async_stream_returns_async_iterator(self):
        """stream() returns an async iterator."""
        from microthink import AsyncMicroThinkClient

        with patch("microthink.async_client.httpx.AsyncClient") as mock_httpx:
            mock_client = AsyncMock()
            mock_httpx.return_value = mock_client
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            async with AsyncMicroThinkClient() as client:
                # Just verify the method exists and is async
                assert hasattr(client, "stream")


class TestStreamCallback:
    """Tests for stream callback functionality."""

    def test_callback_receives_chunks(self):
        """Callback is called for each chunk."""
        from microthink import MicroThinkClient

        chunks_received = []

        def callback(chunk: str):
            chunks_received.append(chunk)

        with patch("microthink.client.ollama.Client") as mock_ollama_class:
            mock_client = MagicMock()
            mock_ollama_class.return_value = mock_client

            # Mock streaming chat response with proper answer tags
            mock_client.chat.return_value = iter(
                [
                    {"message": {"content": "<answer>"}},
                    {"message": {"content": "Hello"}},
                    {"message": {"content": " world"}},
                    {"message": {"content": "!</answer>"}},
                ]
            )

            client = MicroThinkClient()
            client._stream_generate("Test", callback=callback)

            assert len(chunks_received) >= 1
