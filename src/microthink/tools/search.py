"""
Web search functionality using DuckDuckGo.

Provides web search capabilities for MicroThink to answer
questions about current events, real-time data, and topics
beyond the model's training data.
"""

from typing import Dict, List, Optional

try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        raise ImportError(
            "Web search requires either 'ddgs' or 'duckduckgo-search' package. "
            "Install with: pip install ddgs"
        )


def search_web(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """
    Search the web using DuckDuckGo.

    Args:
        query: The search query.
        max_results: Maximum number of results to return (default: 3).

    Returns:
        List of search results, each containing:
        - title: The page title
        - url: The page URL
        - snippet: A text snippet from the page

    Example:
        >>> results = search_web("current Bitcoin price")
        >>> for r in results:
        ...     print(r['title'], r['snippet'])
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                }
                for r in results
            ]
    except KeyboardInterrupt:
        raise  # Don't swallow Ctrl+C
    except Exception as e:
        # Log the error for debugging while gracefully degrading
        import logging

        logging.warning(f"Web search failed: {e}")
        return []


def format_search_results(results: List[Dict[str, str]]) -> str:
    """
    Format search results into a string for injection into the prompt.

    Args:
        results: List of search results from search_web().

    Returns:
        Formatted string with search results.

    Example:
        >>> results = search_web("Python programming")
        >>> context = format_search_results(results)
        >>> print(context)
    """
    if not results:
        return "No web results found."

    formatted = ""
    for i, r in enumerate(results, 1):
        formatted += f"Source {i}: {r['title']}\n"
        formatted += f"Content: {r['snippet']}\n\n"

    return formatted.strip()


def search_and_format(query: str, max_results: int = 3) -> str:
    """
    Search the web and return formatted results ready for prompt injection.

    Args:
        query: The search query.
        max_results: Maximum number of results.

    Returns:
        Formatted search results string.
    """
    results = search_web(query, max_results)
    return format_search_results(results)


def extract_facts_from_results(results: List[Dict[str, str]], query: str) -> str:
    """
    Extract key facts from search results as simple bullet points.

    This pre-processes search results into a format that small models
    can reliably use. Instead of asking the model to "use this context",
    we give it extracted facts that look like knowledge.

    Args:
        results: Search results from search_web().
        query: Original query for context.

    Returns:
        Bullet-pointed facts extracted from results.
    """
    if not results:
        return ""

    # Extract numbers, temperatures, prices, dates, names from snippets
    import re

    facts = []

    for r in results:
        snippet = r.get("snippet", "")
        title = r.get("title", "")

        # Extract temperature patterns (e.g., "18°C", "65°F", "high of 68")
        # Pattern 1: Explicit unit (18°C, 65°F)
        temp_with_unit = re.findall(
            r"(\d+(?:\.\d+)?)\s*°\s*([CFcf])",
            snippet,
        )
        for temp, unit in temp_with_unit:
            try:
                t = float(temp)
                unit_upper = unit.upper()
                # Validate based on unit
                if unit_upper == "C" and -50 <= t <= 60:
                    facts.append(f"Temperature: {int(t)}°C")
                elif unit_upper == "F" and -58 <= t <= 140:
                    facts.append(f"Temperature: {int(t)}°F")
            except ValueError:
                pass

        # Pattern 2: No explicit unit (high of 68) - assume Fahrenheit for weather
        temp_no_unit = re.findall(
            r"(?:high|low|temperature|temp)\s*(?:of|:)?\s*(\d+)",
            snippet,
            re.IGNORECASE,
        )
        for temp in temp_no_unit:
            try:
                t = int(temp)
                if -58 <= t <= 140:  # Fahrenheit range
                    facts.append(f"Temperature: {t}°F")
            except ValueError:
                pass

        # Extract percentages
        pct_patterns = re.findall(r"(\d+(?:\.\d+)?)\s*%", snippet)
        for pct in pct_patterns:
            facts.append(f"Percentage: {pct}%")

        # Extract currency/prices (e.g., "$45,000", "€100")
        price_patterns = re.findall(
            r"[\$€£]\s*[\d,]+(?:\.\d+)?|[\d,]+(?:\.\d+)?\s*(?:USD|EUR|GBP)", snippet
        )
        for price in price_patterns:
            facts.append(f"Price: {price}")

        # Extract years and dates
        date_patterns = re.findall(
            r"\b(20\d{2}|19\d{2})\b|"
            r"\b(\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*(?:\s+20\d{2})?)\b",
            snippet,
            re.IGNORECASE,
        )
        for match in date_patterns:
            date = match[0] or match[1]
            if date:
                facts.append(f"Date: {date}")

        # Include key sentences that might contain the answer
        # Look for sentences with numbers or key terms
        sentences = re.split(r"[.!?]", snippet)
        for sent in sentences[:2]:  # First 2 sentences often have the answer
            sent = sent.strip()
            if len(sent) > 20 and len(sent) < 150:
                # Check if sentence has useful content
                if re.search(r"\d+|is|are|was|were|will be", sent, re.IGNORECASE):
                    facts.append(f"Info: {sent}")

    # Deduplicate and limit
    unique_facts = list(dict.fromkeys(facts))[:10]

    if not unique_facts:
        # Fallback: include first snippet directly
        if results:
            return f"According to web search: {results[0].get('snippet', '')}"

    return "\n".join(f"• {fact}" for fact in unique_facts)
