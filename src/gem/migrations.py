"""Config migrations — graceful upgrades when config format changes.

Each migration has a version number and a function that transforms the config dict.
Migrations run in order on load if config version is behind.
"""
from __future__ import annotations

from pathlib import Path

CURRENT_VERSION = 2


def _v1_to_v2(data: dict) -> dict:
    """v1 → v2: rename thinking_mode 'summary' default to 'full'."""
    ui = data.get("ui", {})
    if ui.get("thinking_mode") == "summary":
        ui["thinking_mode"] = "full"
    data["ui"] = ui
    # Add version field
    data["_version"] = 2
    return data


MIGRATIONS = {
    1: _v1_to_v2,
}


def migrate_config(data: dict) -> dict:
    """Run all pending migrations on a config dict."""
    version = data.get("_version", 1)
    while version < CURRENT_VERSION:
        migration = MIGRATIONS.get(version)
        if migration:
            data = migration(data)
        version += 1
    data["_version"] = CURRENT_VERSION
    return data


def needs_migration(data: dict) -> bool:
    return data.get("_version", 1) < CURRENT_VERSION
