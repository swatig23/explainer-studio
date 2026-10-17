"""Orchestrator: wires ingest -> script (writer) -> [verify, quiz] in
parallel -> assembled result. The real fork-join lesson from PROJECT_DESIGN.md
applied where it actually matters: verify and quiz both only need the
script draft, not each other, so they run concurrently instead of adding
their latencies together.
"""

from __future__ import annotations

from loguru import logger

from . import cache
from .agents.ingest import clean_and_hash
from .agents.quiz_agent import write_quiz
from .agents.script_agent import write_script
from .agents.verify_agent import verify_scenes
from .models import ExplainerRequest, ExplainerResult, QuizQuestion, Scene, SourceSpan

import asyncio

_DROP_THRESHOLD = 0.3  # if more than 30% of scenes are dropped, flag for review


async def run_pipeline(request: ExplainerRequest) -> ExplainerResult:
    clean_text, cache_key = clean_and_hash(
        request.source_text, audience=request.audience, language=request.language
    )

    cached = cache.read(cache_key)
    if cached is not None:
        logger.info(f"[pipeline] cache hit for {cache_key}")
        return cached

    logger.info(f"[pipeline] cache miss for {cache_key}, running fresh")
    draft = await write_script(clean_text, audience=request.audience, language=request.language)

    # Fork-join: verify and quiz both only depend on the draft, not each other.
    verdicts, quiz_draft = await asyncio.gather(
        verify_scenes(draft.scenes),
        write_quiz(draft.scenes, language=request.language),
    )

    scenes: list[Scene] = []
    dropped = 0
    for s in draft.scenes:
        verdict = verdicts.get(s.order, {"entailed": False, "reason": "no verdict"})
        if not verdict["entailed"]:
            logger.warning(f"[pipeline] dropping scene {s.order}: {verdict['reason']}")
            dropped += 1
            continue
        scenes.append(
            Scene(
                order=s.order,
                narration=s.narration,
                visual_prompt=s.visual_prompt,
                source_span=SourceSpan(text=s.source_span_text, start_hint=s.source_span_text[:40]),
            )
        )

    total = len(draft.scenes) or 1
    needs_review = (dropped / total) > _DROP_THRESHOLD

    result = ExplainerResult(
        request=request,
        scenes=scenes,
        quiz=[
            QuizQuestion(question=q.question, options=q.options, correct_index=q.correct_index)
            for q in quiz_draft
        ],
        dropped_scene_count=dropped,
        needs_review=needs_review,
        source_char_count=len(clean_text),
        cache_key=cache_key,
    )
    cache.write(result)
    return result
