"""Tests for custom persona registration."""

import pytest

from microthink.core.prompts import (
    SYSTEM_PERSONAS,
    PersonaError,
    get_persona,
    register_persona,
    unregister_persona,
)


@pytest.fixture
def cleanup_personas():
    """Fixture to track and cleanup any personas registered during tests."""
    registered = []
    yield registered
    # Cleanup: remove any personas that were registered
    for name in registered:
        if name in SYSTEM_PERSONAS and name not in (
            "general",
            "coder",
            "analyst",
            "reasoner",
        ):
            del SYSTEM_PERSONAS[name]


@pytest.fixture
def restore_general():
    """Fixture to restore the general persona after override tests."""
    original = SYSTEM_PERSONAS["general"]
    yield
    SYSTEM_PERSONAS["general"] = original


class TestRegisterPersona:
    """Tests for registering custom personas."""

    def test_register_new_persona(self, cleanup_personas):
        """Can register a new custom persona."""
        register_persona(
            name="sql_expert",
            prompt="You are an expert SQL developer.",
        )
        cleanup_personas.append("sql_expert")
        assert "sql_expert" in SYSTEM_PERSONAS
        assert SYSTEM_PERSONAS["sql_expert"] == "You are an expert SQL developer."

    def test_register_persona_rejects_empty_name(self):
        """Cannot register persona with empty name."""
        with pytest.raises(PersonaError, match="name cannot be empty"):
            register_persona(name="", prompt="Some prompt")

    def test_register_persona_rejects_none_name(self):
        """Cannot register persona with None name."""
        with pytest.raises(PersonaError, match="name cannot be empty"):
            register_persona(name=None, prompt="Some prompt")

    def test_register_persona_rejects_empty_prompt(self):
        """Cannot register persona with empty prompt."""
        with pytest.raises(PersonaError, match="prompt cannot be empty"):
            register_persona(name="test", prompt="")

    def test_register_persona_rejects_none_prompt(self, cleanup_personas):
        """Cannot register persona with None prompt."""
        with pytest.raises(PersonaError, match="prompt cannot be empty"):
            register_persona(name="test", prompt=None)

    def test_register_persona_rejects_builtin_override(self):
        """Cannot override built-in personas by default."""
        with pytest.raises(PersonaError, match="Cannot override built-in persona"):
            register_persona(name="general", prompt="Override attempt")

    def test_register_persona_allows_override_with_flag(self, restore_general):
        """Can override built-in personas with allow_override=True."""
        register_persona(
            name="general",
            prompt="Custom general",
            allow_override=True,
        )
        assert SYSTEM_PERSONAS["general"] == "Custom general"

    def test_register_duplicate_overwrites(self, cleanup_personas):
        """Registering the same name twice overwrites the previous persona."""
        register_persona(name="my_persona", prompt="First version")
        cleanup_personas.append("my_persona")
        assert SYSTEM_PERSONAS["my_persona"] == "First version"

        register_persona(name="my_persona", prompt="Second version")
        assert SYSTEM_PERSONAS["my_persona"] == "Second version"


class TestGetPersona:
    """Tests for retrieving personas."""

    def test_get_builtin_persona(self):
        """Can retrieve built-in personas."""
        persona = get_persona("coder")
        assert "Python programmer" in persona

    def test_get_custom_persona(self, cleanup_personas):
        """Can retrieve custom personas."""
        register_persona(name="test_persona", prompt="Test prompt")
        cleanup_personas.append("test_persona")
        assert get_persona("test_persona") == "Test prompt"

    def test_get_unknown_persona_raises(self):
        """Raises PersonaError for unknown persona."""
        with pytest.raises(PersonaError, match="Unknown persona"):
            get_persona("nonexistent")


class TestUnregisterPersona:
    """Tests for unregistering personas."""

    def test_unregister_custom_persona(self, cleanup_personas):
        """Can unregister custom personas."""
        register_persona(name="temp", prompt="Temporary")
        cleanup_personas.append("temp")
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

    def test_client_register_persona_method(self, cleanup_personas):
        """Client exposes register_persona method."""
        from microthink import MicroThinkClient

        client = MicroThinkClient()
        client.register_persona(
            name="friendly_bot",
            prompt="You are a friendly and helpful assistant.",
        )
        cleanup_personas.append("friendly_bot")
        assert "friendly_bot" in client.available_behaviors

    def test_client_available_behaviors_includes_custom(self, cleanup_personas):
        """available_behaviors property includes custom personas."""
        from microthink import MicroThinkClient

        register_persona(name="custom1", prompt="Custom persona 1")
        cleanup_personas.append("custom1")
        client = MicroThinkClient()
        assert "custom1" in client.available_behaviors
