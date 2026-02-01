"""
Batch processing for MicroThink.

Provides concurrent processing of multiple prompts with
configurable parallelism and progress tracking.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from microthink.async_client import AsyncMicroThinkClient


@dataclass
class BatchResult:
    """
    Result of a batch processing operation.

    Attributes:
        prompt: The original prompt.
        result: The generated result (None if error).
        error: Error message if failed (None if success).
        index: Original index in the batch.
    """

    prompt: str
    result: Optional[Union[str, Dict[str, Any], List[Any]]]
    error: Optional[str]
    index: int

    @property
    def success(self) -> bool:
        """Whether this result was successful."""
        return self.error is None


class BatchProcessor:
    """
    Process multiple prompts concurrently.

    Example:
        >>> async with AsyncMicroThinkClient() as client:
        ...     processor = BatchProcessor(client, max_concurrent=5)
        ...     results = await processor.process([
        ...         "What is Python?",
        ...         "What is JavaScript?",
        ...         "What is Rust?",
        ...     ])
    """

    def __init__(
        self,
        client: AsyncMicroThinkClient,
        max_concurrent: int = 5,
    ) -> None:
        """
        Initialize the batch processor.

        Args:
            client: The async client to use for requests.
            max_concurrent: Maximum concurrent requests (default: 5).
        """
        self.client = client
        self.max_concurrent = max_concurrent

    async def process(
        self,
        prompts: List[str],
        behavior: str = "general",
        expect_json: bool = False,
        on_progress: Optional[Callable[[int, int], None]] = None,
        continue_on_error: bool = False,
    ) -> List[BatchResult]:
        """
        Process multiple prompts concurrently.

        Args:
            prompts: List of prompts to process.
            behavior: The persona to use for all prompts.
            expect_json: If True, parse JSON output.
            on_progress: Callback called with (completed, total) after each prompt.
            continue_on_error: If True, continue processing on individual errors.

        Returns:
            List of BatchResult objects in original order.
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)
        results: List[BatchResult] = [None] * len(prompts)  # type: ignore
        completed = [0]

        async def process_one(index: int, prompt: str) -> None:
            async with semaphore:
                try:
                    result = await self.client.generate(
                        prompt=prompt,
                        behavior=behavior,
                        expect_json=expect_json,
                    )
                    results[index] = BatchResult(
                        prompt=prompt,
                        result=result,
                        error=None,
                        index=index,
                    )
                except Exception as e:
                    if not continue_on_error:
                        raise
                    results[index] = BatchResult(
                        prompt=prompt,
                        result=None,
                        error=str(e),
                        index=index,
                    )
                finally:
                    completed[0] += 1
                    if on_progress:
                        on_progress(completed[0], len(prompts))

        # Create tasks for all prompts
        tasks = [
            asyncio.create_task(process_one(i, prompt))
            for i, prompt in enumerate(prompts)
        ]

        # Wait for all to complete
        await asyncio.gather(*tasks, return_exceptions=continue_on_error)

        return results

    async def process_with_schema(
        self,
        prompts: List[str],
        schema: Union[Dict[str, Any], List[Any]],
        behavior: str = "general",
        on_progress: Optional[Callable[[int, int], None]] = None,
        continue_on_error: bool = False,
    ) -> List[BatchResult]:
        """
        Process multiple prompts with schema validation.

        Args:
            prompts: List of prompts to process.
            schema: The schema to validate against.
            behavior: The persona to use.
            on_progress: Progress callback.
            continue_on_error: Continue on individual errors.

        Returns:
            List of BatchResult objects.
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)
        results: List[BatchResult] = [None] * len(prompts)  # type: ignore
        completed = [0]

        async def process_one(index: int, prompt: str) -> None:
            async with semaphore:
                try:
                    result = await self.client.generate_with_schema(
                        prompt=prompt,
                        schema=schema,
                        behavior=behavior,
                    )
                    results[index] = BatchResult(
                        prompt=prompt,
                        result=result,
                        error=None,
                        index=index,
                    )
                except Exception as e:
                    if not continue_on_error:
                        raise
                    results[index] = BatchResult(
                        prompt=prompt,
                        result=None,
                        error=str(e),
                        index=index,
                    )
                finally:
                    completed[0] += 1
                    if on_progress:
                        on_progress(completed[0], len(prompts))

        tasks = [
            asyncio.create_task(process_one(i, prompt))
            for i, prompt in enumerate(prompts)
        ]
        await asyncio.gather(*tasks, return_exceptions=continue_on_error)
        return results
