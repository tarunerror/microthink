"""
Smart retry strategies for MicroThink.

Provides error-aware retry strategies that adapt based on
the type of JSON parsing error encountered.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RetryStrategy:
    """
    A retry strategy with specific parameters.

    Attributes:
        temperature: The temperature to use for retry.
        correction_prompt: The prompt to send for correction.
        max_tokens: Optional max tokens limit.
    """

    temperature: float
    correction_prompt: str
    max_tokens: Optional[int] = None


def analyze_json_error(error_message: str) -> dict:
    """
    Analyze a JSON parsing error to determine its type.

    Args:
        error_message: The error message from json.loads().

    Returns:
        Dict with 'error_type' and optional 'position' keys.
    """
    error_lower = error_message.lower()

    # Trailing comma detection
    if "expecting property name" in error_lower:
        return {"error_type": "trailing_comma"}

    # Unterminated string
    if "unterminated string" in error_lower:
        return {"error_type": "unterminated"}

    # Missing value
    if "expecting value" in error_lower:
        return {"error_type": "missing_value"}

    # Structure/delimiter issues
    if "expecting" in error_lower and "delimiter" in error_lower:
        return {"error_type": "structure"}

    # Extra data after JSON
    if "extra data" in error_lower:
        return {"error_type": "extra_data"}

    return {"error_type": "unknown"}


def get_retry_strategy(error_type: str, attempt: int) -> RetryStrategy:
    """
    Get the appropriate retry strategy for an error type.

    Args:
        error_type: The type of error from analyze_json_error().
        attempt: The current attempt number (1-indexed).

    Returns:
        A RetryStrategy with appropriate parameters.
    """
    # Base temperature decreases with attempts
    base_temp = max(0.1, 0.5 - (attempt * 0.15))

    strategies = {
        "trailing_comma": RetryStrategy(
            temperature=0.1,
            correction_prompt=(
                "Your JSON has a trailing comma which is invalid. "
                "Remove the comma before the closing brace/bracket. "
                "Output ONLY valid JSON inside <answer> tags."
            ),
        ),
        "unterminated": RetryStrategy(
            temperature=0.2,
            correction_prompt=(
                "Your JSON has an unterminated string. "
                "Ensure all strings are properly closed with quotes. "
                "Output ONLY valid JSON inside <answer> tags."
            ),
            max_tokens=1000,  # Limit to prevent runaway generation
        ),
        "missing_value": RetryStrategy(
            temperature=0.2,
            correction_prompt=(
                "Your JSON is missing a value. "
                "Ensure all keys have corresponding values. "
                "Output ONLY valid JSON inside <answer> tags."
            ),
        ),
        "structure": RetryStrategy(
            temperature=0.3,
            correction_prompt=(
                "Your JSON structure is malformed. "
                "Check that all braces, brackets, and commas are correct. "
                "Output ONLY valid JSON inside <answer> tags."
            ),
        ),
        "extra_data": RetryStrategy(
            temperature=0.2,
            correction_prompt=(
                "Your response contains extra text after the JSON. "
                "Output ONLY the JSON object/array, nothing else. "
                "Put it inside <answer> tags."
            ),
        ),
    }

    if error_type in strategies:
        strategy = strategies[error_type]
        # Adjust temperature based on attempt
        return RetryStrategy(
            temperature=max(0.1, strategy.temperature - (attempt - 1) * 0.05),
            correction_prompt=strategy.correction_prompt,
            max_tokens=strategy.max_tokens,
        )

    # Unknown error - generic strategy
    return RetryStrategy(
        temperature=base_temp,
        correction_prompt=(
            "Your previous response was invalid JSON. "
            f"Error: This is attempt {attempt}. "
            "Please fix the syntax error and output ONLY valid JSON inside <answer> tags. "
            "Do not add any explanatory text."
        ),
    )
