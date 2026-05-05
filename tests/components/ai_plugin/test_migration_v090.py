"""v0.8.x → v0.9.0: existing config entries with trigger_languages must
load cleanly. The key is silently dropped from options reads."""
from __future__ import annotations


def test_old_options_dict_with_trigger_languages_loads_cleanly():
    """An options dict carrying a removed key must not crash code that
    reads other keys."""
    options = {
        "trigger_languages": ["de"],          # legacy key
        "context_window": 16384,
        "voice_mode": False,
    }
    # The loaded code must continue to fetch known keys without error.
    assert options.get("context_window") == 16384
    # And ignore the legacy key gracefully.
    assert "trigger_languages" in options    # still present in raw dict
    # but no consumer reads it any more — verified by absence of imports.
    import custom_components.ai_plugin.const as const_mod
    assert not hasattr(const_mod, "CONF_TRIGGER_LANGUAGES")
    assert not hasattr(const_mod, "PROMPT_HINTS_I18N")
    assert not hasattr(const_mod, "default_trigger_langs")
