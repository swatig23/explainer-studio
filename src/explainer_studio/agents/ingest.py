"""Ingest: turn raw pasted text into clean text + a stable cache key.

Deliberately minimal for the hackathon build -- no PDF/URL fetching yet
(see PROJECT_DESIGN.md open items). Paste-text is the supported path;
PDF/URL ingestion is a thin layer that can be added on top without
touching anything downstream, since everything after this just sees
clean text.
"""

from __future__ import annotations

import hashlib

from ..config import get_settings


class DocumentTooLongError(ValueError):
    """Raised when source text exceeds the length guardrail."""


def clean_and_hash(raw_text: str, *, audience: str, language: str) -> tuple[str, str]:
    """Return (clean_text, cache_key). Raises DocumentTooLongError if over
    the configured limit -- fail loud and early, never silently truncate
    and pretend nothing was lost.
    """
    settings = get_settings()
    clean = " ".join(raw_text.split())  # collapse whitespace/newlines

    if len(clean) > settings.max_source_chars:
        raise DocumentTooLongError(
            f"Source is {len(clean):,} chars, limit is {settings.max_source_chars:,}. "
            "Trim the document or split it -- chunking is on the roadmap, not in this build."
        )

    key_material = f"{clean}|{audience}|{language}".encode("utf-8")
    cache_key = hashlib.sha256(key_material).hexdigest()[:16]
    return clean, cache_key
