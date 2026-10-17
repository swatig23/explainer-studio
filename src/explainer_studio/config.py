"""Runtime configuration, loaded from .env.

Kept deliberately small. Every field here is something we actually read
somewhere else in the codebase -- no speculative config (YAGNI).
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Google AI Studio key -- text + image gen (Nano Banana) both go through this.
    google_api_key: str = ""
    gemini_text_model: str = "gemini-2.5-flash"
    gemini_image_model: str = "gemini-2.5-flash-image"

    # Document length guardrail (see PROJECT_DESIGN.md "Known limitations").
    max_source_chars: int = 40_000  # roughly 8-10 pages of plain text

    # Pipeline knobs
    max_verify_retries: int = 2
    scene_count_target: int = 4

    # Feature flags -- let the demo degrade gracefully instead of hard-failing.
    image_generation_enabled: bool = True
    tts_enabled: bool = True

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # Where generated results get cached, keyed by content hash.
    cache_dir: str = "data/cache"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
