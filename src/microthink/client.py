"""
MicroThink Client - The core interface for enhanced LLM interactions.

This module provides the MicroThinkClient class, which wraps the Ollama
library to automatically inject Chain-of-Thought reasoning, enforce
structured outputs, and handle self-correction for small language models.
"""

import json
import re
from typing import Any, Dict, List, Optional, Union

import ollama

from microthink.core.cache import ResponseCache, make_cache_key
from microthink.core.parser import (
    clean_json_text,
    extract_answer_safely,
    parse_response,
)
from microthink.core.prompts import (
    SYSTEM_PERSONAS,
    build_system_prompt,
)
from microthink.core.prompts import (
    register_persona as _register_persona,
)
from microthink.core.schema import SchemaValidationError, validate_schema
from microthink.tools.search import extract_facts_from_results, search_web
from microthink.utils.logger import (
    log_answer,
    log_error,
    log_info,
    log_retry,
    log_thinking,
)


class MicroThinkError(Exception):
    """
    Custom exception for MicroThink failures with diagnostic context.

    Provides rich error messages with:
    - The last model output before failure
    - Number of retry attempts made
    - JSON error details with position
    - Actionable suggestions for fixing the issue

    Attributes:
        message: Human-readable error description.
        last_output: The last output from the model before failure.
        attempts: Number of attempts made before failure.
        json_error: Specific JSON parsing error message.
        suggestion: Actionable suggestion for fixing the issue.
    """

    # Common error patterns and their suggestions
    _SUGGESTIONS = {
        "trailing comma": (
            "Model added a trailing comma. Try:\n"
            "  - Use behavior='coder' for cleaner JSON\n"
            "  - Add 'no trailing commas' to your prompt"
        ),
        "expecting property name": (
            "JSON structure is malformed. Try:\n"
            "  - Simplify the requested schema\n"
            "  - Use behavior='coder' for better formatting"
        ),
        "expecting value": (
            "JSON value is missing or malformed. Try:\n"
            "  - Ensure your prompt clearly specifies the expected structure\n"
            "  - Use generate_with_schema() for complex objects"
        ),
        "unterminated string": (
            "Model produced incomplete string. Try:\n"
            "  - Increase timeout if model is slow\n"
            "  - Use a smaller/faster model"
        ),
    }

    def __init__(
        self,
        message: str,
        last_output: Optional[str] = None,
        attempts: int = 0,
        json_error: Optional[str] = None,
    ) -> None:
        """
        Initialize MicroThinkError.

        Args:
            message: The error message.
            last_output: The last raw output from the model.
            attempts: How many attempts were made.
            json_error: Specific JSON parsing error.
        """
        super().__init__(message)
        self.message = message
        self.last_output = last_output
        self.attempts = attempts
        self.json_error = json_error
        self.suggestion = self._generate_suggestion()

    def _generate_suggestion(self) -> str:
        """Generate actionable suggestion based on error type."""
        if not self.json_error:
            return "Run with debug=True to see the model's thinking process."

        error_lower = self.json_error.lower()
        for pattern, suggestion in self._SUGGESTIONS.items():
            if pattern in error_lower:
                return suggestion

        return (
            "Run with debug=True to see the model's thinking process.\n"
            "Consider using behavior='coder' for more reliable JSON output."
        )

    def _format_output_with_pointer(self) -> str:
        """Format output with error position pointer if available."""
        if not self.last_output:
            return ""

        # Truncate long output
        max_len = 200
        output = self.last_output
        truncated = False
        if len(output) > max_len:
            output = output[:max_len]
            truncated = True

        # Try to extract position from json_error
        pointer_line = ""
        if self.json_error:
            # Match patterns like "column 18" or "position 17"
            col_match = re.search(r"column (\d+)", self.json_error, re.IGNORECASE)
            pos_match = re.search(r"position (\d+)", self.json_error, re.IGNORECASE)

            pos = None
            if col_match:
                pos = int(col_match.group(1)) - 1  # 0-indexed
            elif pos_match:
                pos = int(pos_match.group(1))

            if pos is not None and pos < len(output):
                pointer_line = "\n    " + " " * pos + "^"

        if truncated:
            output += "..."

        return output + pointer_line

    def __str__(self) -> str:
        """Return a formatted, actionable error string."""
        parts = [f"MicroThinkError: {self.message}"]

        if self.attempts > 0:
            parts.append(f"  Attempts: {self.attempts}")

        if self.last_output:
            parts.append(
                f"\n  Last model output:\n    {self._format_output_with_pointer()}"
            )

        if self.json_error:
            parts.append(f"\n  JSON Error: {self.json_error}")

        if self.suggestion:
            parts.append(f"\n  Suggestion: {self.suggestion}")

        return "\n".join(parts)

    def __repr__(self) -> str:
        """Return a detailed repr for debugging."""
        return (
            f"MicroThinkError(message={self.message!r}, "
            f"attempts={self.attempts}, "
            f"has_output={self.last_output is not None})"
        )


class MicroThinkClient:
    """
    A smart wrapper around Ollama that enhances small LLM performance.

    Features:
        - Automatic Chain-of-Thought injection with <thinking>/<answer> tags
        - JSON output validation with self-correction retry loop
        - Configurable personas optimized for different tasks
        - Debug mode for inspecting the reasoning process

    Example:
        >>> client = MicroThinkClient(model="llama3.2:3b")
        >>> result = client.generate("How many r's in strawberry?", debug=True)
        >>> print(result)
        'There are 3 r's in strawberry.'

        >>> data = client.generate(
        ...     "Return a JSON list of 3 Python keywords",
        ...     behavior="coder",
        ...     expect_json=True
        ... )
        >>> print(data)
        ['def', 'class', 'return']
    """

    DEFAULT_MODEL = "llama3.2:3b"
    MAX_RETRIES = 3
    RETRY_TEMPERATURE = 0.2
    BAD_OUTPUT_TRUNCATE_LENGTH = 500

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: Optional[str] = None,
        timeout: float = 120.0,
        cache: bool = False,
        cache_ttl: float = 3600.0,
        cache_max_size: int = 1000,
    ) -> None:
        """
        Initialize the MicroThink client.

        Args:
            model: The Ollama model to use (default: "llama3.2:3b").
            host: Optional Ollama host URL (default: uses Ollama default).
            timeout: Request timeout in seconds (default: 120.0).
            cache: Enable response caching (default: False).
            cache_ttl: Cache time-to-live in seconds (default: 3600.0).
            cache_max_size: Maximum cache entries (default: 1000).

        Raises:
            ValueError: If the model name is empty.
        """
        if not model:
            raise ValueError("Model name cannot be empty")

        self.model = model
        self.host = host
        self.timeout = timeout

        # Initialize Ollama client
        if host:
            self._client = ollama.Client(host=host, timeout=timeout)
        else:
            self._client = ollama.Client(timeout=timeout)

        # Initialize cache
        self._cache: Optional[ResponseCache] = None
        if cache:
            self._cache = ResponseCache(max_size=cache_max_size, ttl=cache_ttl)

    @property
    def available_behaviors(self) -> List[str]:
        """Return list of available behavior personas."""
        return list(SYSTEM_PERSONAS.keys())

    def register_persona(
        self,
        name: str,
        prompt: str,
        allow_override: bool = False,
    ) -> None:
        """Register a custom persona for use with generate()."""
        _register_persona(name, prompt, allow_override)

    def cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
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
        """Clear the response cache."""
        if self._cache is not None:
            self._cache.clear()

    def generate(
        self,
        prompt: str,
        behavior: str = "general",
        expect_json: bool = False,
        debug: bool = False,
        brief: bool = False,
        web_search: bool = False,
    ) -> Union[str, Dict[str, Any], List[Any]]:
        """
        Generate a response with automatic CoT reasoning and optional JSON validation.

        This is the main entry point for interacting with the model. It:
        1. Constructs an optimized prompt with persona and CoT instructions
        2. Optionally searches the web and injects results into context
        3. Calls the model and parses the structured response
        4. If expect_json=True, validates and retries up to 3 times on failure
        5. Returns only the final answer (thinking is stripped unless debug=True)

        Args:
            prompt: The user's input prompt.
            behavior: The persona to use ('general', 'coder', 'analyst', 'reasoner').
            expect_json: If True, parse and validate JSON output with retries.
            debug: If True, log the thinking process and answers to console.
            brief: If True, answer contains only the result without explanation.
            web_search: If True, search the web and include results in context.

        Returns:
            If expect_json=False: The answer as a string.
            If expect_json=True: The parsed JSON as a dict or list.

        Raises:
            MicroThinkError: If JSON parsing fails after all retries.
            ValueError: If an invalid behavior is specified.

        Example:
            >>> client = MicroThinkClient()
            >>> # Simple text response
            >>> answer = client.generate("What is 2+2?")
            >>> print(answer)
            '4'

            >>> # JSON response with validation
            >>> data = client.generate(
            ...     "List 3 colors as JSON array",
            ...     expect_json=True
            ... )
            >>> print(data)
            ['red', 'blue', 'green']

            >>> # Web search enabled
            >>> answer = client.generate(
            ...     "What is the current price of Bitcoin?",
            ...     web_search=True
            ... )
        """
        # Validate behavior
        if behavior not in SYSTEM_PERSONAS:
            raise ValueError(
                f"Invalid behavior '{behavior}'. Available: {self.available_behaviors}"
            )

        # Check cache first
        cache_key = None
        if self._cache is not None and not web_search:
            cache_key = make_cache_key(
                model=self.model,
                behavior=behavior,
                prompt=prompt,
                expect_json=expect_json,
                web_search=web_search,
            )
            cached = self._cache.get(cache_key)
            if cached is not None:
                if debug:
                    log_info("Cache hit - returning cached response")
                return cached

        # Build the system prompt with dynamic injection
        system_prompt = build_system_prompt(behavior, expect_json, brief)

        # Web search: fetch results, extract facts, inject into prompt
        enhanced_prompt = prompt
        if web_search:
            if debug:
                log_info(f"Searching web for: {prompt[:50]}...")

            # Step 1: Get raw search results
            search_results = search_web(prompt, max_results=5)

            if search_results:
                if debug:
                    log_info(f"Found {len(search_results)} web results")

                # Step 2: Extract key facts from results (pre-processing)
                extracted_facts = extract_facts_from_results(search_results, prompt)

                if extracted_facts:
                    if debug:
                        log_info("Extracted facts from web results")
                        log_info(f"Facts:\n{extracted_facts}")

                    # Two-step approach: Present facts as knowledge, not "search results"
                    # Small models respond better to direct facts than "use this context"
                    enhanced_prompt = (
                        f"Current information:\n{extracted_facts}\n\n"
                        f"Based on the information above, answer: {prompt}"
                    )
            elif debug:
                log_info("No web results found, proceeding without")

        # Initialize message history (stateful for retries)
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": enhanced_prompt},
        ]

        # Initial model call
        if debug:
            log_info(f"Calling {self.model} with behavior='{behavior}'")

        response = self._client.chat(model=self.model, messages=messages)
        raw_content = response["message"]["content"]

        # Parse the structured response
        parsed = parse_response(raw_content)
        thinking = parsed["thinking"]
        answer_content = parsed["answer"]

        # Debug logging
        if debug:
            if thinking:
                log_thinking(thinking)
            log_answer(answer_content, is_json=expect_json)

        # If not expecting JSON, return the answer directly
        if not expect_json:
            if self._cache is not None and cache_key is not None:
                self._cache.set(cache_key, answer_content)
            return answer_content

        # JSON Reflexion Loop
        # Add the first response to history for stateful retries
        messages.append({"role": "assistant", "content": raw_content})

        retries = 0
        last_error: Optional[str] = None

        while retries < self.MAX_RETRIES:
            try:
                # Clean and parse JSON
                cleaned = clean_json_text(answer_content)
                result = json.loads(cleaned)

                if debug and retries > 0:
                    log_info(f"JSON parsed successfully after {retries} retries")

                if self._cache is not None and cache_key is not None:
                    self._cache.set(cache_key, result)
                return result

            except json.JSONDecodeError as e:
                retries += 1
                last_error = str(e)

                if debug:
                    log_retry(retries, self.MAX_RETRIES, last_error)

                if retries >= self.MAX_RETRIES:
                    raise MicroThinkError(
                        f"JSON parsing failed after {self.MAX_RETRIES} retries",
                        last_output=answer_content,
                        attempts=retries,
                        json_error=last_error,
                    )

                # Build correction prompt
                # Truncate bad output to avoid context window bloat
                truncated_output = answer_content[: self.BAD_OUTPUT_TRUNCATE_LENGTH]
                if len(answer_content) > self.BAD_OUTPUT_TRUNCATE_LENGTH:
                    truncated_output += "..."

                correction_msg = (
                    f"Your previous response was invalid JSON.\n"
                    f"Error: {last_error}\n"
                    f"Your output: {truncated_output}\n\n"
                    f"INSTRUCTIONS:\n"
                    f"1. Fix the syntax error.\n"
                    f"2. Output ONLY valid JSON inside <answer> tags.\n"
                    f"3. Do not add any explanatory text inside <answer>.\n"
                    f"4. Do not use markdown code blocks."
                )

                # Append correction to history
                messages.append({"role": "user", "content": correction_msg})

                # Retry with lower temperature for stability
                response = self._client.chat(
                    model=self.model,
                    messages=messages,
                    options={"temperature": self.RETRY_TEMPERATURE},
                )

                raw_content = response["message"]["content"]
                parsed = parse_response(raw_content)
                answer_content = parsed["answer"]

                # Add response to history for next potential retry
                messages.append({"role": "assistant", "content": raw_content})

                if debug and parsed["thinking"]:
                    log_thinking(parsed["thinking"])
                    log_answer(answer_content, is_json=True)

    def generate_with_schema(
        self,
        prompt: str,
        schema: Dict[str, Any],
        behavior: str = "general",
        debug: bool = False,
        brief: bool = False,
        web_search: bool = False,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate JSON output that conforms to a specified schema.

        This is a convenience method that includes the schema in the prompt
        to guide the model's output structure.

        Args:
            prompt: The user's input prompt.
            schema: A JSON schema or example structure to follow.
            behavior: The persona to use.
            debug: If True, log the reasoning process.
            brief: If True, output just the result without explanation.
            web_search: If True, search the web for current information.
            validate: If True (default), validate output against schema.

        Returns:
            The parsed JSON response as a dictionary.

        Raises:
            MicroThinkError: If JSON parsing fails after all retries.
            SchemaValidationError: If validate=True and output doesn't match schema.

        Example:
            >>> client = MicroThinkClient()
            >>> schema = {"name": "string", "age": "number"}
            >>> data = client.generate_with_schema(
            ...     "Create a person named Alice who is 30",
            ...     schema=schema
            ... )
            >>> print(data)
            {'name': 'Alice', 'age': 30}
        """
        schema_str = json.dumps(schema, indent=2)
        enhanced_prompt = (
            f"{prompt}\n\nReturn JSON matching this structure:\n{schema_str}"
        )

        result = self.generate(
            prompt=enhanced_prompt,
            behavior=behavior,
            expect_json=True,
            debug=debug,
            brief=brief,
            web_search=web_search,
        )

        # Ensure we return a dict (not a list)
        if isinstance(result, dict):
            if validate:
                validate_schema(result, schema)
            return result

        raise MicroThinkError(
            f"Expected dict but got {type(result).__name__}",
            last_output=str(result),
            attempts=self.MAX_RETRIES,
        )
