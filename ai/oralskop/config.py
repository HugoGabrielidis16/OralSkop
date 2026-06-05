"""Tiny YAML-config helpers shared by the entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text()) or {}


def _coerce(value: str) -> Any:
    """Best-effort scalar coercion for CLI overrides (int/float/bool/str)."""
    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``key=value`` strings onto a config dict (in place)."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override {item!r} must be key=value.")
        key, _, raw = item.partition("=")
        cfg[key.strip()] = _coerce(raw.strip())
    return cfg
