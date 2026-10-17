"""Quiz step -- generates the comprehension check from the finished script.

Non-critical by design: if this fails, the pipeline still returns a valid
explainer with an empty quiz rather than failing the whole request
(see PROJECT_DESIGN.md stage table).
"""

from __future__ import annotations

from pydantic import BaseModel

from ..clients.gemini import generate_json
from .script_agent import SceneDraft


class _QuizQuestionDraft(BaseModel):
    question: str
    options: list[str]
    correct_index: int


class _QuizDraft(BaseModel):
    questions: list[_QuizQuestionDraft]


async def write_quiz(scenes: list[SceneDraft], *, language: str, n_questions: int = 3) -> list[_QuizQuestionDraft]:
    narration_block = "\n".join(f"- {s.narration}" for s in scenes)
    prompt = f"""Based only on these facts from an explainer video:

{narration_block}

Write exactly {n_questions} multiple-choice comprehension questions in
{language}, each with exactly 4 options and one correct_index (0-based).
Questions must be answerable purely from the facts above -- no outside
knowledge required."""

    draft = await generate_json(prompt, _QuizDraft)
    return draft.questions
