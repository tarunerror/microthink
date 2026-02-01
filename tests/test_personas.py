"""Tests for custom persona registration."""

import pytest

from microthink.core.prompts import (
    SYSTEM_PERSONAS,
    PersonaError,
    get_persona,
    register_persona,
    unregister_persona,
)


class TestRegisterPersona:
    """Tests for registering custom personas."""

    def test_register_new_persona(self):
        """Can register a new custom persona."""
        register_persona(
            name="sql_expert",
            prompt="You are an expert SQL developer.",
        )
        assert "sql_expert" in SYSTEM_PERSONAS
        assert SYSTEM_PERSONAS["sql_expert"] == "You are an expert SQL developer."
        unregister_persona("sql_expert")

    def test_register_persona_rejects_empty_name(self):
        """Cannot register persona with empty name."""
        with pytest.raises(PersonaError, match="name cannot be empty"):
            register_persona(name="", prompt="Some prompt")

    def test_register_persona_rejects_empty_prompt(self):
        """Cannot register persona with empty prompt."""
        with pytest.raises(PersonaError, match="prompt cannot be empty"):
            register_persona(name="test", prompt="")

    def test_register_persona_rejects_builtin_override(self):
        """Cannot override built-in personas by default."""
        with pytest.raises(PersonaError, match="Cannot override built-in persona"):
            register_persona(name="general", prompt="Override attempt")

    def test_register_persona_allows_override_with_flag(self):
        """Can override built-in personas with allow_override=True."""
        original = SYSTEM_PERSONAS["general"]
        register_persona(
            name="general",
            prompt="Custom general",
            allow_override=True,
        )
        assert SYSTEM_PERSONAS["general"] == "Custom general"
        SYSTEM_PERSONAS["general"] = original


class TestGetPersona:
    """Tests for retrieving personas."""

    def test_get_builtin_persona(self):
        """Can retrieve built-in personas."""
        persona = get_persona("coder")
        assert "Python programmer" in persona

    def test_get_custom_persona(self):
        """Can retrieve custom personas."""
        register_persona(name="test_persona", prompt="Test prompt")
        assert get_persona("test_persona") == "Test prompt"
        unregister_persona("test_persona")

    def test_get_unknown_persona_raises(self):
        """Raises PersonaError for unknown persona."""
        with pytest.raises(PersonaError, match="Unknown persona"):
            get_persona("nonexistent")


class TestUnregisterPersona:
    """Tests for unregistering personas."""

    def test_unregister_custom_persona(self):
        """Can unregister custom personas."""
        register_persona(name="temp", prompt="Temporary")
        unregister_persona("temp")
        assert "temp" not in SYSTEM_PERSONAS

    def test_unregister_builtin_raises(self):
        """Cannot unregister built-in personas."""
        with pytest.raises(PersonaError, match="Cannot unregister built-in"):
            unregister_persona("general")

    def test_unregister_unknown_raises(self):
        """Raises PersonaError for unknown persona."""
        with pytest.raises(PersonaError, match="Unknown persona"):
            unregister_persona("nonexistent")


class TestClientPersonaIntegration:
    """Tests for using custom personas with MicroThinkClient."""

    def test_client_register_persona_method(self):
        """Client exposes register_persona method."""
        from microthink import MicroThinkClient

        client = MicroThinkClient()
        client.register_persona(
            name="friendly_bot",
            prompt="You are a friendly and helpful assistant.",
        )
        assert "friendly_bot" in client.available_behaviors
        unregister_persona("friendly_bot")

    def test_client_available_behaviors_includes_custom(self):
        """available_behaviors property includes custom personas."""
        from microthink import MicroThinkClient

        register_persona(name="custom1", prompt="Custom persona 1")
        client = MicroThinkClient()
        assert "custom1" in client.available_behaviors
        unregister_persona("custom1")
