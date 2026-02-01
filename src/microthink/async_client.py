"""
Async MicroThink Client.

Provides an async/await interface to MicroThink for concurrent requests.
Uses httpx for async HTTP communication with Ollama.
"""

import json
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import httpx

from microthink.client import MicroThinkError
from microthink.core.cache import ResponseCache, make_cache_key
from microthink.core.parser import clean_json_text, parse_response
from microthink.core.prompts import (
    SYSTEM_PERSONAS,
    build_system_prompt,
)
from microthink.core.prompts import (
    register_persona as _register_persona,
)
from microthink.core.schema import SchemaValidationError, validate_schema


class AsyncMicroThinkClient:
    """
    Async wrapper around Ollama that enhances small LLM performance.

    Example:
        >>> async with AsyncMicroThinkClient() as client:
        ...     result = await client.generate("What is 2+2?")
        ...     print(result)
        '4'
    """

    DEFAULT_MODEL = "llama3.2:3b"
    MAX_RETRIES = 3
    RETRY_TEMPERATURE = 0.2
    DEFAULT_HOST = "http://localhost:11434"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: Optional[str] = None,
        timeout: float = 120.0,
        cache: bool = False,
        cache_ttl: float = 3600.0,
        cache_max_size: int = 1000,
    ) -> None:
        if not model:
            raise ValueError("Model name cannot be empty")

        self.model = model
        self.host = host or self.DEFAULT_HOST
        self.timeout = timeout
        self._http_client: Optional[httpx.AsyncClient] = None

        self._cache: Optional[ResponseCache] = None
        if cache:
            self._cache = ResponseCache(max_size=cache_max_size, ttl=cache_ttl)

    async def __aenter__(self) -> "AsyncMicroThinkClient":
        self._http_client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    @property
    def available_behaviors(self) -> List[str]:
        return list(SYSTEM_PERSONAS.keys())

    def register_persona(
        self, name: str, prompt: str, allow_override: bool = False
    ) -> None:
        _register_persona(name, prompt, allow_override)

    def cache_stats(self) -> Dict[str, Any]:
        if self._cache is None:
            return {
                "hits": 0,
                "misses": 0,
                "size": 0,
                "hit_rate": 0.0,
                "enabled": False,
            }
        stats = self._cache.stats()
        stats["enabled"] = True
        return stats

    def clear_cache(self) -> None:
        if self._cache is not None:
            self._cache.clear()

    async def _chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        if self._http_client is None:
            raise RuntimeError(
                "Client not initialized. Use 'async with AsyncMicroThinkClient():'"
            )

        payload = {"model": self.model, "messages": messages, "stream": False}
        if temperature is not None:
            payload["options"] = {"temperature": temperature}

        response = await self._http_client.post(f"{self.host}/api/chat", json=payload)
        response.raise_for_status()
        return response.json()

    async def generate(
        self,
        prompt: str,
        behavior: str = "general",
        expect_json: bool = False,
        debug: bool = False,
        brief: bool = False,
    ) -> Union[str, Dict[str, Any], List[Any]]:
        if behavior not in SYSTEM_PERSONAS:
            raise ValueError(
                f"Invalid behavior '{behavior}'. Available: {self.available_behaviors}"
            )

        cache_key = None
        if self._cache is not None:
            cache_key = make_cache_key(self.model, behavior, prompt, expect_json, False)
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        system_prompt = build_system_prompt(behavior, expect_json, brief)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        response = await self._chat(messages)
        raw_content = response["message"]["content"]
        parsed = parse_response(raw_content)
        answer_content = parsed["answer"]

        if not expect_json:
            if self._cache is not None and cache_key:
                self._cache.set(cache_key, answer_content)
            return answer_content

        messages.append({"role": "assistant", "content": raw_content})
        retries = 0
        last_error: Optional[str] = None

        while retries < self.MAX_RETRIES:
            try:
                cleaned = clean_json_text(answer_content)
                result = json.loads(cleaned)
                if self._cache is not None and cache_key:
                    self._cache.set(cache_key, result)
                return result
            except json.JSONDecodeError as e:
                retries += 1
                last_error = str(e)
                if retries >= self.MAX_RETRIES:
                    raise MicroThinkError(
                        f"JSON parsing failed after {self.MAX_RETRIES} retries",
                        last_output=answer_content,
                        attempts=retries,
                        json_error=last_error,
                    )
                correction_msg = f"Your previous response was invalid JSON.\nError: {last_error}\nOutput ONLY valid JSON inside <answer> tags."
                messages.append({"role": "user", "content": correction_msg})
                response = await self._chat(
                    messages, temperature=self.RETRY_TEMPERATURE
                )
                raw_content = response["message"]["content"]
                parsed = parse_response(raw_content)
                answer_content = parsed["answer"]
                messages.append({"role": "assistant", "content": raw_content})

    async def generate_with_schema(
        self,
        prompt: str,
        schema: Union[Dict[str, Any], List[Any]],
        behavior: str = "general",
        debug: bool = False,
        brief: bool = False,
        validate: bool = True,
    ) -> Union[Dict[str, Any], List[Any]]:
        schema_str = json.dumps(schema, indent=2)
        enhanced_prompt = (
            f"{prompt}\n\nReturn JSON matching this structure:\n{schema_str}"
        )
        result = await self.generate(enhanced_prompt, behavior, True, debug, brief)
        if validate:
            validate_schema(result, schema)
        if isinstance(schema, dict) and not isinstance(result, dict):
            raise MicroThinkError(
                f"Expected dict but got {type(result).__name__}",
                last_output=str(result),
            )
        if isinstance(schema, list) and not isinstance(result, list):
            raise MicroThinkError(
                f"Expected list but got {type(result).__name__}",
                last_output=str(result),
            )
        return result

    async def stream(
        self,
        prompt: str,
        behavior: str = "general",
        brief: bool = False,
    ) -> AsyncIterator[str]:
        """
        Stream a response token by token asynchronously.

        Args:
            prompt: The user's input prompt.
            behavior: The persona to use.
            brief: If True, output just the result.

        Yields:
            String chunks as they are generated.

        Example:
            >>> async for chunk in client.stream("Write a story"):
            ...     print(chunk, end="", flush=True)
        """
        if self._http_client is None:
            raise RuntimeError(
                "Client not initialized. Use 'async with AsyncMicroThinkClient():'"
            )

        if behavior not in SYSTEM_PERSONAS:
            raise ValueError(
                f"Invalid behavior '{behavior}'. Available: {self.available_behaviors}"
            )

        system_prompt = build_system_prompt(behavior, expect_json=False, brief=brief)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }

        async with self._http_client.stream(
            "POST",
            f"{self.host}/api/chat",
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            yield data["message"]["content"]
                    except json.JSONDecodeError:
                        continue
