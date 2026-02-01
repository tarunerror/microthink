"""Tests for async client."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestAsyncMicroThinkClient:
    """Tests for AsyncMicroThinkClient."""

    def test_async_client_importable(self):
        """AsyncMicroThinkClient is importable."""
        from microthink import AsyncMicroThinkClient

        assert AsyncMicroThinkClient is not None

    def test_async_client_has_generate(self):
        """AsyncMicroThinkClient has async generate method."""
        from microthink import AsyncMicroThinkClient

        client = AsyncMicroThinkClient()
        assert hasattr(client, "generate")
        assert asyncio.iscoroutinefunction(client.generate)

    def test_async_client_has_generate_with_schema(self):
        """AsyncMicroThinkClient has async generate_with_schema method."""
        from microthink import AsyncMicroThinkClient

        client = AsyncMicroThinkClient()
        assert hasattr(client, "generate_with_schema")
        assert asyncio.iscoroutinefunction(client.generate_with_schema)

    def test_async_client_accepts_same_params(self):
        """AsyncMicroThinkClient accepts same init params as sync."""
        from microthink import AsyncMicroThinkClient

        client = AsyncMicroThinkClient(
            model="llama3.2:3b",
            timeout=60.0,
            cache=True,
            cache_ttl=1800,
        )
        assert client.model == "llama3.2:3b"
        assert client.timeout == 60.0

    def test_async_client_has_context_manager(self):
        """AsyncMicroThinkClient supports async context manager."""
        from microthink import AsyncMicroThinkClient

        assert hasattr(AsyncMicroThinkClient, "__aenter__")
        assert hasattr(AsyncMicroThinkClient, "__aexit__")

    @pytest.mark.asyncio
    async def test_async_context_manager_usage(self):
        """Can use AsyncMicroThinkClient as async context manager."""
        from microthink import AsyncMicroThinkClient

        async with AsyncMicroThinkClient() as client:
            assert client is not None

    def test_async_client_has_cache_methods(self):
        """AsyncMicroThinkClient has cache methods."""
        from microthink import AsyncMicroThinkClient

        client = AsyncMicroThinkClient(cache=True)
        assert hasattr(client, "cache_stats")
        assert hasattr(client, "clear_cache")

    def test_async_client_has_register_persona(self):
        """AsyncMicroThinkClient has register_persona method."""
        from microthink import AsyncMicroThinkClient

        client = AsyncMicroThinkClient()
        assert hasattr(client, "register_persona")


@pytest.mark.asyncio
class TestAsyncClientMocked:
    """Tests for async client with mocked HTTP."""

    async def test_generate_returns_string(self):
        """Async generate returns string for non-JSON."""
        from microthink.async_client import AsyncMicroThinkClient

        with patch("microthink.async_client.httpx.AsyncClient") as mock_httpx:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "message": {"content": "<thinking>Step 1</thinking><answer>42</answer>"}
            }
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_httpx.return_value = mock_client

            async with AsyncMicroThinkClient() as client:
                result = await client.generate("What is 2+2?")

            assert result == "42"

    async def test_generate_returns_json(self):
        """Async generate returns parsed JSON when expect_json=True."""
        from microthink.async_client import AsyncMicroThinkClient

        with patch("microthink.async_client.httpx.AsyncClient") as mock_httpx:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "message": {
                    "content": '<thinking>Creating JSON</thinking><answer>{"name": "Alice"}</answer>'
                }
            }
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_httpx.return_value = mock_client

            async with AsyncMicroThinkClient() as client:
                result = await client.generate("Create user", expect_json=True)

            assert result == {"name": "Alice"}
