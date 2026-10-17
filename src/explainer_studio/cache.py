"""Cache: full pipeline results keyed by content hash, so re-running the
same demo input during rehearsal is instant and free -- directly fixes
the "every test run pays full latency" problem diagnosed in the adintel
project (see PROJECT_DESIGN.md "Lessons carried from adintel").
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import get_settings
from .models import ExplainerResult


def _cache_path(cache_key: str) -> Path:
    settings = get_settings()
    cache_dir = Path(settings.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{cache_key}.json"


def read(cache_key: str) -> ExplainerResult | None:
    path = _cache_path(cache_key)
    if not path.exists():
        return None
    return ExplainerResult.model_validate_json(path.read_text())


def write(result: ExplainerResult) -> None:
    path = _cache_path(result.cache_key)
    path.write_text(result.model_dump_json(indent=2))
