"""Tests for smarter retry strategies."""

import pytest
from microthink.core.retry import (
    RetryStrategy,
    analyze_json_error,
    get_retry_strategy,
)


class TestAnalyzeJsonError:
    """Tests for error analysis."""

    def test_detects_trailing_comma(self):
        """Detects trailing comma errors."""
        result = analyze_json_error("Expecting property name: line 1 column 20")
        assert result["error_type"] == "trailing_comma"

    def test_detects_missing_quotes(self):
        """Detects missing quote errors."""
        result = analyze_json_error("Expecting ',' delimiter: line 1 column 10")
        assert result["error_type"] in ["structure", "syntax"]

    def test_detects_unterminated_string(self):
        """Detects unterminated string errors."""
        result = analyze_json_error("Unterminated string starting at")
        assert result["error_type"] == "unterminated"

    def test_unknown_error(self):
        """Returns unknown for unrecognized errors."""
        result = analyze_json_error("Some random error")
        assert result["error_type"] == "unknown"


class TestGetRetryStrategy:
    """Tests for getting appropriate retry strategy."""

    def test_trailing_comma_strategy(self):
        """Trailing comma gets specific fix instruction."""
        strategy = get_retry_strategy("trailing_comma", attempt=1)
        assert strategy.temperature == 0.1
        assert "trailing comma" in strategy.correction_prompt.lower()

    def test_structure_error_strategy(self):
        """Structure errors get schema re-explanation."""
        strategy = get_retry_strategy("structure", attempt=1)
        assert strategy.temperature <= 0.3
        assert (
            "structure" in strategy.correction_prompt.lower()
            or "schema" in strategy.correction_prompt.lower()
        )

    def test_temperature_decreases_with_attempts(self):
        """Temperature decreases as attempts increase."""
        strategy1 = get_retry_strategy("unknown", attempt=1)
        strategy2 = get_retry_strategy("unknown", attempt=2)
        strategy3 = get_retry_strategy("unknown", attempt=3)
        assert strategy1.temperature >= strategy2.temperature >= strategy3.temperature

    def test_unknown_gets_generic_strategy(self):
        """Unknown errors get generic retry strategy."""
        strategy = get_retry_strategy("unknown", attempt=1)
        assert strategy.temperature <= 0.5
        assert len(strategy.correction_prompt) > 0


class TestRetryStrategy:
    """Tests for RetryStrategy dataclass."""

    def test_strategy_has_required_fields(self):
        """RetryStrategy has temperature and correction_prompt."""
        strategy = RetryStrategy(temperature=0.2, correction_prompt="Fix it")
        assert strategy.temperature == 0.2
        assert strategy.correction_prompt == "Fix it"

    def test_strategy_has_optional_max_tokens(self):
        """RetryStrategy can have max_tokens."""
        strategy = RetryStrategy(
            temperature=0.2,
            correction_prompt="Fix it",
            max_tokens=500,
        )
        assert strategy.max_tokens == 500
