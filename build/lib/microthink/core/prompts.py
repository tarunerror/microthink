"""
Prompt templates and system personas for MicroThink.

This module contains the building blocks for constructing effective prompts
that guide small LLMs to produce better outputs through Chain-of-Thought
reasoning and structured formatting.
"""

# The Chain-of-Thought instruction that ensures structured reasoning
COT_INSTRUCTION = """
[IMPORTANT FORMAT INSTRUCTIONS]
1. You must FIRST think step-by-step inside <thinking> tags.
2. Then, output your final result inside <answer> tags.

Example format:
<thinking>
[Your step-by-step reasoning here]
</thinking>
<answer>
[Your final answer here]
</answer>
"""

# JSON enforcement constraint - only injected when expect_json=True
JSON_ENFORCEMENT = (
    "\n[JSON CONSTRAINT]\n"
    "The content inside <answer> tags must be valid JSON only. "
    "Do not use Markdown code blocks (```json). Do not add explanatory text. "
    "Return pure, parseable JSON text inside <answer> tags."
)

# System personas - focused on identity only, no formatting rules
SYSTEM_PERSONAS = {
    "general": (
        "You are a helpful assistant. Be concise, accurate, and thorough. "
        "Think carefully before answering."
    ),

    "coder": (
        "You are an expert Python programmer with deep knowledge of software engineering. "
        "You prioritize clean, efficient, and well-documented code. "
        "You catch edge cases and write robust solutions."
    ),

    "analyst": (
        "You are a meticulous data analyst. You are precise with numbers and logic. "
        "You always verify your calculations step by step. "
        "You identify patterns and provide actionable insights."
    ),

    "reasoner": (
        "You are a logical reasoning expert. You break down complex problems "
        "into smaller parts and solve them systematically. "
        "You consider multiple angles before reaching conclusions."
    ),
}


def build_system_prompt(behavior: str = "general", expect_json: bool = False) -> str:
    """
    Constructs a complete system prompt with dynamic injection.

    Args:
        behavior: The persona to use ('general', 'coder', 'analyst', 'reasoner').
        expect_json: Whether to inject JSON formatting constraints.

    Returns:
        A complete system prompt string.

    Example:
        >>> prompt = build_system_prompt("coder", expect_json=True)
        >>> "expert Python programmer" in prompt
        True
        >>> "valid JSON" in prompt
        True
    """
    # Start with the persona identity
    system_text = SYSTEM_PERSONAS.get(behavior, SYSTEM_PERSONAS["general"])

    # Always add CoT instruction
    system_text += f"\n{COT_INSTRUCTION}"

    # Conditionally add JSON enforcement
    if expect_json:
        system_text += JSON_ENFORCEMENT

    return system_text
