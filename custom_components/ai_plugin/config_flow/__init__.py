"""Config flow and options flow for AI Plugin.

    config_steps   first-time setup, creates the entry
    options        post-install Configure
    mcp_steps      the MCP server screens of the options flow
    schemas        the voluptuous schemas both flows show

Home Assistant imports this package by name; both flow classes register
themselves on definition, so importing them here is what wires them up.
"""

from __future__ import annotations

from .config_steps import AIPluginConfigFlow
from .options import AIPluginOptionsFlow

__all__ = ["AIPluginConfigFlow", "AIPluginOptionsFlow"]
