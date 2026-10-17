"""Core data shapes for the pipeline. Pydantic models -- validated at every
boundary (Gemini output, cache read/write, API response) so a malformed
LLM response fails loudly here instead of silently corrupting the demo.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SourceSpan(BaseModel):
    """A verbatim snippet of the original document a claim is grounded in."""

    text: str
    start_hint: str = ""  # short prefix to help the UI locate it, not exact offsets


class Scene(BaseModel):
    """One beat of the explainer: a narration line, its visual, and its proof."""

    order: int
    narration: str
    visual_prompt: str
    source_span: SourceSpan
    image_url: str | None = None
    audio_duration_ms: int | None = None


class VerifyVerdict(BaseModel):
    """Output of the critic agent for a single scene."""

    scene_order: int
    entailed: bool
    reason: str
    unsupported_terms: list[str] = Field(default_factory=list)


class QuizQuestion(BaseModel):
    question: str
    options: list[str]
    correct_index: int


class ExplainerRequest(BaseModel):
    source_text: str
    audience: str  # e.g. "farmer", "ca_student", "class8"
    language: str = "English"


class ExplainerResult(BaseModel):
    request: ExplainerRequest
    scenes: list[Scene]
    quiz: list[QuizQuestion]
    dropped_scene_count: int = 0
    needs_review: bool = False
    source_char_count: int = 0
    cache_key: str = ""
