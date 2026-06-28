"""Configuration loader — reads config.yaml into a typed accessor.

Centralizes all paths, seeds, and hyperparameters so no module hardcodes
magic numbers. Paths are resolved relative to the project root.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and cache the project config from YAML."""
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def resolve(rel_path: str) -> Path:
    """Resolve a config-relative path against the project root."""
    return ROOT / rel_path


# Convenience top-level constants (loaded once at import).
CONFIG = load_config()
SEED: int = CONFIG["seed"]
