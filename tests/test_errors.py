"""Tests for enhanced error messages."""

import pytest

from microthink.client import MicroThinkError


class TestMicroThinkErrorEnhancements:
    """Tests for enhanced MicroThinkError."""

    def test_error_includes_last_output(self):
        """Error includes the last model output."""
        error = MicroThinkError(
            message="JSON parsing failed",
            last_output='{"name": "Alice",}',
            attempts=3,
        )
        assert '{"name": "Alice",}' in str(error)

    def test_error_includes_attempt_count(self):
        """Error includes number of attempts."""
        error = MicroThinkError(
            message="JSON parsing failed",
            last_output="bad output",
            attempts=3,
        )
        assert "3" in str(error)

    def test_error_includes_json_error_position(self):
        """Error pinpoints JSON syntax error position."""
        error = MicroThinkError(
            message="JSON parsing failed",
            last_output='{"name": "Alice",}',
            attempts=3,
            json_error="Expecting property name: line 1 column 18",
        )
        error_str = str(error)
        assert "column 18" in error_str or "position" in error_str.lower()

    def test_error_includes_suggestion(self):
        """Error includes actionable suggestion."""
        error = MicroThinkError(
            message="JSON parsing failed",
            last_output='{"name": "Alice",}',
            attempts=3,
            json_error="Trailing comma",
        )
        error_str = str(error)
        assert "suggestion" in error_str.lower() or "try" in error_str.lower()

    def test_error_suggestion_for_trailing_comma(self):
        """Trailing comma error gets specific suggestion."""
        error = MicroThinkError(
            message="JSON parsing failed",
            last_output='{"a": 1,}',
            attempts=3,
            json_error="Trailing comma",
        )
        assert "trailing comma" in str(error).lower()

    def test_error_preserves_exception_behavior(self):
        """Enhanced error still works as exception."""
        error = MicroThinkError("Test error")
        with pytest.raises(MicroThinkError):
            raise error

    def test_error_repr(self):
        """Error has useful repr."""
        error = MicroThinkError(
            message="Test",
            last_output="output",
            attempts=2,
        )
        repr_str = repr(error)
        assert "MicroThinkError" in repr_str

    def test_error_without_json_error(self):
        """Error without json_error still provides suggestion."""
        error = MicroThinkError(
            message="Some error",
            last_output="output",
            attempts=1,
        )
        error_str = str(error)
        assert "debug=True" in error_str.lower() or "suggestion" in error_str.lower()

    def test_error_truncates_long_output(self):
        """Long output is truncated in error message."""
        long_output = "x" * 500
        error = MicroThinkError(
            message="Error",
            last_output=long_output,
            attempts=1,
        )
        error_str = str(error)
        # Should be truncated to ~200 chars + "..."
        assert len(error_str) < len(long_output)
        assert "..." in error_str

    def test_error_shows_pointer_at_position(self):
        """Error shows ^ pointer at error position."""
        error = MicroThinkError(
            message="JSON parsing failed",
            last_output='{"name": "Alice",}',
            attempts=3,
            json_error="Expecting property name: line 1 column 18",
        )
        error_str = str(error)
        assert "^" in error_str
