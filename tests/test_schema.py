"""Tests for schema validation."""

import pytest

from microthink.core.schema import (
    SchemaValidationError,
    infer_type,
    validate_schema,
)


class TestInferType:
    """Tests for type inference from example values."""

    def test_infer_string(self):
        assert infer_type("string") == "string"

    def test_infer_number(self):
        assert infer_type("number") == "number"

    def test_infer_integer(self):
        assert infer_type("integer") == "integer"

    def test_infer_boolean(self):
        assert infer_type("boolean") == "boolean"

    def test_infer_from_actual_string(self):
        assert infer_type("hello") == "string"

    def test_infer_from_actual_int(self):
        assert infer_type(42) == "integer"

    def test_infer_from_actual_float(self):
        assert infer_type(3.14) == "number"

    def test_infer_from_actual_bool(self):
        assert infer_type(True) == "boolean"

    def test_infer_array(self):
        assert infer_type([1, 2, 3]) == "array"

    def test_infer_object(self):
        assert infer_type({"a": 1}) == "object"


class TestValidateSchema:
    """Tests for schema validation."""

    def test_valid_simple_object(self):
        schema = {"name": "string", "age": "number"}
        data = {"name": "Alice", "age": 30}
        validate_schema(data, schema)  # Should not raise

    def test_missing_required_field(self):
        schema = {"name": "string", "age": "number"}
        data = {"name": "Alice"}
        with pytest.raises(SchemaValidationError, match="Missing required field"):
            validate_schema(data, schema)

    def test_wrong_type_string_vs_number(self):
        schema = {"name": "string", "age": "number"}
        data = {"name": "Alice", "age": "thirty"}
        with pytest.raises(SchemaValidationError, match="Expected number"):
            validate_schema(data, schema)

    def test_extra_fields_allowed(self):
        schema = {"name": "string"}
        data = {"name": "Alice", "extra": "field"}
        validate_schema(data, schema)  # Should not raise

    def test_nested_object(self):
        schema = {"user": {"name": "string", "age": "number"}, "active": "boolean"}
        data = {"user": {"name": "Bob", "age": 25}, "active": True}
        validate_schema(data, schema)  # Should not raise

    def test_nested_object_invalid(self):
        schema = {"user": {"name": "string", "age": "number"}}
        data = {"user": {"name": "Bob", "age": "young"}}
        with pytest.raises(SchemaValidationError, match="user.age"):
            validate_schema(data, schema)

    def test_array_of_strings(self):
        schema = {"tags": ["string"]}
        data = {"tags": ["a", "b", "c"]}
        validate_schema(data, schema)  # Should not raise

    def test_array_with_wrong_element_type(self):
        schema = {"tags": ["string"]}
        data = {"tags": ["a", 123, "c"]}
        with pytest.raises(SchemaValidationError, match=r"tags\[1\]"):
            validate_schema(data, schema)

    def test_empty_array_valid(self):
        schema = {"items": ["number"]}
        data = {"items": []}
        validate_schema(data, schema)  # Should not raise

    def test_type_coercion_int_as_number(self):
        schema = {"value": "number"}
        data = {"value": 42}
        validate_schema(data, schema)  # Should not raise

    def test_validates_list_result(self):
        schema = ["string"]
        data = ["a", "b", "c"]
        validate_schema(data, schema)  # Should not raise

    def test_list_schema_wrong_type(self):
        schema = ["number"]
        data = ["a", "b"]
        with pytest.raises(SchemaValidationError, match=r"\[0\]"):
            validate_schema(data, schema)

    def test_bool_not_accepted_as_integer(self):
        """Boolean is not accepted where integer is expected."""
        schema = {"count": "integer"}
        data = {"count": True}
        with pytest.raises(SchemaValidationError, match="Expected integer"):
            validate_schema(data, schema)

    def test_bool_not_accepted_as_number(self):
        """Boolean is not accepted where number is expected."""
        schema = {"value": "number"}
        data = {"value": False}
        with pytest.raises(SchemaValidationError, match="Expected number"):
            validate_schema(data, schema)


class TestSchemaValidationError:
    """Tests for SchemaValidationError."""

    def test_error_message(self):
        error = SchemaValidationError("user.name", "Expected string, got int")
        assert "user.name" in str(error)
        assert "Expected string" in str(error)

    def test_error_is_exception(self):
        with pytest.raises(SchemaValidationError):
            raise SchemaValidationError("field", "message")
