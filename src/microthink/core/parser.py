"""
Parser utilities for extracting structured content from LLM responses.

This module implements a 4-layer resilience model for handling messy outputs
from small language models:

Layer 1 (extract_tag_content): Handles broken closing tags
Layer 2 (extract_answer_safely): Handles completely missing answer tags
Layer 3 (clean_json_text): Handles conversational fluff around JSON
Layer 4 (generate loop): Handles actual syntax errors via retry
"""

import re
from typing import Dict, List, Optional


def extract_tag_content(text: str, tag: str) -> Optional[str]:
    """
    Extract content between <tag>...</tag> with resilience to malformed tags.

    Args:
        text: The raw text to parse.
        tag: The tag name to look for (without angle brackets).

    Returns:
        The extracted content, or None if the tag is not found.

    Resilience features:
        - Handles missing closing tag (captures to end of string)
        - Case insensitive matching
        - Handles attributes in opening tag (e.g., <tag attr="val">)

    Example:
        >>> extract_tag_content("<thinking>Step 1...</thinking>", "thinking")
        'Step 1...'
        >>> extract_tag_content("<THINKING>Reasoning</THINKING>", "thinking")
        'Reasoning'
        >>> extract_tag_content("<thinking>No closing tag", "thinking")
        'No closing tag'
    """
    # Pattern handles optional attributes and missing closing tag
    pattern = rf"<{tag}\b[^>]*>(.*?)(?:</{tag}>|$)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)

    if match:
        return match.group(1).strip()
    return None


def extract_answer_safely(text: str) -> str:
    """
    Retrieves the final answer with fallback strategies.

    If <answer> tags are missing, attempts to salvage the response
    by removing the <thinking> block and returning the remainder.

    Args:
        text: The raw LLM response text.

    Returns:
        The extracted answer content. Never returns None - always
        returns at least the stripped raw text as a last resort.

    Example:
        >>> extract_answer_safely("<thinking>...</thinking><answer>42</answer>")
        '42'
        >>> extract_answer_safely("<thinking>hmm</thinking>The answer is 42")
        'The answer is 42'
        >>> extract_answer_safely("Just a plain response")
        'Just a plain response'
    """
    # Try standard extraction first
    answer = extract_tag_content(text, "answer")
    if answer:
        return answer

    # FALLBACK: Model forgot <answer> tags
    # Remove <thinking> block and return the rest
    thought = extract_tag_content(text, "thinking")
    if thought:
        # Remove the thinking block entirely
        text_no_thought = re.sub(
            r"<thinking\b[^>]*>.*?(?:</thinking>|$)",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        cleaned = text_no_thought.strip()
        if cleaned:
            return cleaned

    # Last resort: return raw text
    return text.strip()


def clean_json_text(text: str) -> str:
    """
    Aggressively extracts JSON from text with conversational fluff.

    Uses a "Search & Rescue" strategy to find the first JSON object
    or array, ignoring prefixes like "Sure! Here is the JSON:" and
    markdown code blocks.

    Args:
        text: Text that should contain JSON, possibly with extra content.

    Returns:
        The extracted JSON string, or the cleaned original text if
        no JSON structure is found (letting json.loads fail naturally).

    Example:
        >>> clean_json_text('```json\\n{"a": 1}\\n```')
        '{"a": 1}'
        >>> clean_json_text('Sure! Here is the JSON: {"a": 1}')
        '{"a": 1}'
        >>> clean_json_text('{"items": [1, 2, 3]}')
        '{"items": [1, 2, 3]}'
    """
    # Step 1: Remove markdown code blocks
    text = re.sub(r"```(?:json)?", "", text)
    text = re.sub(r"```", "", text)
    text = text.strip()

    # Step 2: Find JSON boundaries using brace/bracket matching
    # Find first opening brace or bracket
    match_start = re.search(r"[\{\[]", text)
    if not match_start:
        return text  # No JSON structure found, let json.loads handle it

    start_char = text[match_start.start()]
    end_char = "}" if start_char == "{" else "]"

    # Find the matching closing brace/bracket by counting nesting
    start_index = match_start.start()
    depth = 0
    in_string = False
    escape_next = False
    end_index = len(text)

    for i in range(start_index, len(text)):
        char = text[i]

        if escape_next:
            escape_next = False
            continue

        if char == "\\" and in_string:
            escape_next = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == start_char:
            depth += 1
        elif char == end_char:
            depth -= 1
            if depth == 0:
                end_index = i + 1
                break

    return text[start_index:end_index]


def parse_response(text: str) -> Dict[str, Optional[str]]:
    """
    Parse a complete LLM response into its components.

    Args:
        text: The raw LLM response.

    Returns:
        A dictionary with 'thinking' and 'answer' keys.
        Values may be None if the respective tag was not found.

    Example:
        >>> result = parse_response("<thinking>Step 1</thinking><answer>42</answer>")
        >>> result['thinking']
        'Step 1'
        >>> result['answer']
        '42'
    """
    return {
        "thinking": extract_tag_content(text, "thinking"),
        "answer": extract_answer_safely(text),
    }
