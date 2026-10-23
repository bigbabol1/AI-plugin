"""Voluptuous schema for i18n YAML files."""
from __future__ import annotations

import re

import voluptuous as vol

# Placeholders allowed in templates. Engine substitutes these via str.format.
ALLOWED_PLACEHOLDERS = {"label", "val", "unit", "time", "area"}


class LocalizationError(Exception):
    """Raised when an i18n YAML file is invalid."""


def _non_empty_str(value):
    if not isinstance(value, str) or not value.strip():
        raise vol.Invalid("must be a non-empty string")
    return value


def _lowercase_phrase(value):
    if not isinstance(value, str) or not value.strip():
        raise vol.Invalid("must be a non-empty string")
    return value.lower().strip()


def _valid_template(value):
    if not isinstance(value, str) or not value.strip():
        raise vol.Invalid("template must be a non-empty string")
    # Find placeholders {name} (ignoring {{ }} which str.format escapes).
    placeholders = set(re.findall(r"(?<!\{)\{([a-z_]+)\}(?!\})", value))
    unknown = placeholders - ALLOWED_PLACEHOLDERS
    if unknown:
        raise vol.Invalid(
            f"template uses unknown placeholders: {sorted(unknown)} "
            f"(allowed: {sorted(ALLOWED_PLACEHOLDERS)})"
        )
    return value


def _valid_regex(value):
    if not isinstance(value, str) or not value.strip():
        raise vol.Invalid("regex pattern must be a non-empty string")
    try:
        re.compile(value)
    except re.error as exc:
        raise vol.Invalid(f"invalid regex {value!r}: {exc}") from None
    return value


META_SCHEMA = vol.Schema({
    vol.Required("code"): vol.All(str, vol.Length(min=2, max=5)),
    vol.Required("name"): _non_empty_str,
    vol.Optional("contributors", default=list): [str],
})


SCHEMA = vol.Schema({
    vol.Required("meta"): META_SCHEMA,
    vol.Required("labels"): {str: _non_empty_str},
    vol.Required("templates"): {str: _valid_template},
    vol.Required("keywords"): {str: vol.All([_lowercase_phrase], vol.Length(min=1))},
    vol.Required("patterns"): {str: [_valid_regex]},
}, extra=vol.PREVENT_EXTRA)
