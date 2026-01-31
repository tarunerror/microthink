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
3. The <answer> MUST show the complete solution process, not just the final result.
   - For math: Show the full calculation (e.g., "25% of 200 = 0.25 × 200 = 50")
   - For reasoning: Show the key steps that led to the conclusion
   - For code: Include the complete working code

Example format:
<thinking>
[Your detailed step-by-step reasoning here]
</thinking>
<answer>
[Complete solution with steps + final result]
Example for math: "25% of 552.65 = 0.25 × 552.65 = 138.16"
Example for reasoning: "The word 'strawberry' contains: s-t-r-a-w-b-e-r-r-y. Counting 'r's: positions 3, 8, 9. Total: 3 r's"
</answer>
"""

# Brief mode - just the answer without explanation
COT_INSTRUCTION_BRIEF = """
[IMPORTANT FORMAT INSTRUCTIONS]
1. You must FIRST think step-by-step inside <thinking> tags.
2. Then, output ONLY the final result inside <answer> tags.
   - No explanation, no steps, just the answer.
   - For math: Just the number (e.g., "138.16")
   - For questions: Just the direct answer (e.g., "3")

Example format:
<thinking>
[Your step-by-step reasoning here]
</thinking>
<answer>
[Just the final answer, nothing else]
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
        "You are a helpful assistant. "
        "When information is provided, use it directly in your answer. "
        "State specific data (temperatures, prices, names, dates) from provided information."
    ),
    "coder": (
        "You are an expert Python programmer with deep knowledge of software engineering. "
        "You prioritize clean, efficient, and well-documented code. "
        "You catch edge cases and write robust solutions."
    ),
    "analyst": (
        "You are a meticulous data analyst. You are precise with numbers and logic. "
        "You always verify your calculations step by step. "
        "You identify patterns and provide actionable insights. "
        "When data is provided, extract and report the specific numbers."
    ),
    "reasoner": (
        "You are a logical reasoning expert. You break down complex problems "
        "into smaller parts and solve them systematically. "
        "You consider multiple angles before reaching conclusions."
    ),
}


def build_system_prompt(
    behavior: str = "general", expect_json: bool = False, brief: bool = False
) -> str:
    """
    Constructs a complete system prompt with dynamic injection.

    Args:
        behavior: The persona to use ('general', 'coder', 'analyst', 'reasoner').
        expect_json: Whether to inject JSON formatting constraints.
        brief: If True, instruct model to give just the answer without explanation.

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
    system_text = SYSTEM_PERSONAS[behavior]

    # Add CoT instruction (brief or detailed)
    if brief:
        system_text += f"\n{COT_INSTRUCTION_BRIEF}"
    else:
        system_text += f"\n{COT_INSTRUCTION}"

    # Conditionally add JSON enforcement
    if expect_json:
        system_text += JSON_ENFORCEMENT

    return system_text
