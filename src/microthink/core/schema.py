"""
Schema validation for MicroThink.

Validates JSON results against example schemas to ensure
the model output matches the expected structure.
"""

from typing import Any


class SchemaValidationError(Exception):
    """Raised when JSON output doesn't match the expected schema."""

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f"Schema validation failed at '{path}': {message}")


TYPE_PLACEHOLDERS = {"string", "number", "integer", "boolean", "array", "object"}


def infer_type(value: Any) -> str:
    """Infer the expected type from an example value."""
    if isinstance(value, str) and value in TYPE_PLACEHOLDERS:
        return value
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"


def _check_type(value: Any, expected_type: str) -> bool:
    """Check if a value matches the expected type."""
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "object":
        return isinstance(value, dict)
    return True


def validate_schema(data: Any, schema: Any, path: str = "") -> None:
    """
    Validate data against an example schema.

    Args:
        data: The JSON data to validate.
        schema: The example schema to validate against.
        path: Current path for error messages.

    Raises:
        SchemaValidationError: If the data doesn't match the schema.
    """
    # Handle list schema
    if isinstance(schema, list):
        if not isinstance(data, list):
            raise SchemaValidationError(
                path or "root", f"Expected array, got {type(data).__name__}"
            )
        if len(schema) > 0:
            element_schema = schema[0]
            for i, item in enumerate(data):
                item_path = f"{path}[{i}]" if path else f"[{i}]"
                _validate_element(item, element_schema, item_path)
        return

    # Handle dict schema
    if isinstance(schema, dict):
        if not isinstance(data, dict):
            raise SchemaValidationError(
                path or "root", f"Expected object, got {type(data).__name__}"
            )
        for key, expected in schema.items():
            field_path = f"{path}.{key}" if path else key
            if key not in data:
                raise SchemaValidationError(field_path, "Missing required field")
            _validate_element(data[key], expected, field_path)
        return

    _validate_element(data, schema, path or "root")


def _validate_element(value: Any, expected: Any, path: str) -> None:
    """Validate a single element against its expected type or nested schema."""
    if isinstance(expected, dict):
        validate_schema(value, expected, path)
        return
    if isinstance(expected, list):
        validate_schema(value, expected, path)
        return

    expected_type = infer_type(expected)
    if not _check_type(value, expected_type):
        actual_type = infer_type(value)
        raise SchemaValidationError(
            path, f"Expected {expected_type}, got {actual_type}"
        )
