"""Tests for batch processing."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestBatchProcessor:
    """Tests for BatchProcessor class."""

    def test_batch_processor_importable(self):
        """BatchProcessor is importable."""
        from microthink.batch import BatchProcessor

        assert BatchProcessor is not None

    def test_batch_processor_accepts_client(self):
        """BatchProcessor accepts async client."""
        from microthink.batch import BatchProcessor

        from microthink import AsyncMicroThinkClient

        client = AsyncMicroThinkClient()
        processor = BatchProcessor(client)
        assert processor.client is client

    def test_batch_processor_has_process_method(self):
        """BatchProcessor has async process method."""
        import asyncio

        from microthink.batch import BatchProcessor

        from microthink import AsyncMicroThinkClient

        client = AsyncMicroThinkClient()
        processor = BatchProcessor(client)
        assert hasattr(processor, "process")
        assert asyncio.iscoroutinefunction(processor.process)

    def test_batch_processor_accepts_concurrency(self):
        """BatchProcessor accepts max_concurrent parameter."""
        from microthink.batch import BatchProcessor

        from microthink import AsyncMicroThinkClient

        client = AsyncMicroThinkClient()
        processor = BatchProcessor(client, max_concurrent=5)
        assert processor.max_concurrent == 5


@pytest.mark.asyncio
class TestBatchProcessing:
    """Tests for batch processing with mocked client."""

    async def test_process_multiple_prompts(self):
        """Processes multiple prompts."""
        from microthink.batch import BatchProcessor

        from microthink import AsyncMicroThinkClient

        with patch("microthink.async_client.httpx.AsyncClient") as mock_httpx:
            mock_client = AsyncMock()
            mock_httpx.return_value = mock_client
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            # Mock responses
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "message": {"content": "<answer>Result</answer>"}
            }
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response

            async with AsyncMicroThinkClient() as client:
                processor = BatchProcessor(client, max_concurrent=2)
                prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
                results = await processor.process(prompts)

            assert len(results) == 3

    async def test_process_with_progress_callback(self):
        """Calls progress callback during processing."""
        from microthink.batch import BatchProcessor

        from microthink import AsyncMicroThinkClient

        progress_calls = []

        def on_progress(completed: int, total: int):
            progress_calls.append((completed, total))

        with patch("microthink.async_client.httpx.AsyncClient") as mock_httpx:
            mock_client = AsyncMock()
            mock_httpx.return_value = mock_client
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            mock_response = MagicMock()
            mock_response.json.return_value = {
                "message": {"content": "<answer>Result</answer>"}
            }
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response

            async with AsyncMicroThinkClient() as client:
                processor = BatchProcessor(client)
                prompts = ["P1", "P2"]
                await processor.process(prompts, on_progress=on_progress)

            assert len(progress_calls) >= 2

    async def test_process_handles_errors(self):
        """Handles individual prompt errors gracefully."""
        from microthink.batch import BatchProcessor, BatchResult

        from microthink import AsyncMicroThinkClient

        with patch("microthink.async_client.httpx.AsyncClient") as mock_httpx:
            mock_client = AsyncMock()
            mock_httpx.return_value = mock_client
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            # First call succeeds, second fails
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "message": {"content": "<answer>Result</answer>"}
            }
            mock_response.raise_for_status = MagicMock()

            call_count = [0]

            async def mock_post(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 2:
                    raise Exception("API Error")
                return mock_response

            mock_client.post = mock_post

            async with AsyncMicroThinkClient() as client:
                processor = BatchProcessor(client)
                prompts = ["P1", "P2", "P3"]
                results = await processor.process(prompts, continue_on_error=True)

            # Should have results for all prompts
            assert len(results) == 3
            # At least one should have an error
            assert any(r.error is not None for r in results)
