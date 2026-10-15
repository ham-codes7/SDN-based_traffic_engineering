"""Loads configs/default.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml

from sdnte.topology import REPO_ROOT

DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: Path | None = None) -> dict:
    return yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text())
