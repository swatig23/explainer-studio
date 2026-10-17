"""Verify step -- the "critic agent" half of the writer/critic pair.

Two independent checks, deliberately not trusting the LLM alone for the
thing that matters most (per PROJECT_DESIGN.md -- this is the same
"Code measures, Gemini decides" pattern the Patchamomma winners used):

1. An LLM entailment judge: does the cited source_span actually support
   the narration?
2. A deterministic regex check: does every number in the narration
   literally appear somewhere in the source span? Numbers are the
   highest-harm hallucination for policy/financial content, so this
   gets a non-LLM safety net, not just a judge's opinion.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from ..clients.gemini import generate_json
from .script_agent import SceneDraft


class _Verdict(BaseModel):
    scene_order: int
    entailed: bool
    reason: str


class _VerdictBatch(BaseModel):
    verdicts: list[_Verdict]


_NUMBER_RE = re.compile(r"\d[\d,.]*%?")


def _unsupported_numbers(narration: str, source_span: str) -> list[str]:
    narration_numbers = set(_NUMBER_RE.findall(narration))
    source_numbers = set(_NUMBER_RE.findall(source_span))
    return sorted(n for n in narration_numbers if n not in source_numbers)


async def verify_scenes(scenes: list[SceneDraft]) -> dict[int, dict]:
    """Returns {scene_order: {"entailed": bool, "reason": str, "unsupported_numbers": [...]}}

    One batched judge call for all scenes (cheaper and faster than one
    call per scene), plus the deterministic number check applied locally.
    """
    if not scenes:
        return {}

    scenes_block = "\n\n".join(
        f"SCENE {s.order}:\nnarration: {s.narration}\nsource_span: {s.source_span_text}"
        for s in scenes
    )
    prompt = f"""You are a strict fact-checker. For each scene below, decide
whether the narration is actually supported by (entailed by) its cited
source_span. Flag anything that adds claims, numbers, or specifics not
present in the span, even if plausible-sounding.

{scenes_block}

Return a verdict for every scene_order listed above."""

    batch = await generate_json(prompt, _VerdictBatch)
    verdict_map = {v.scene_order: v for v in batch.verdicts}

    results: dict[int, dict] = {}
    for scene in scenes:
        verdict = verdict_map.get(scene.order)
        unsupported_numbers = _unsupported_numbers(scene.narration, scene.source_span_text)
        entailed = bool(verdict and verdict.entailed) and not unsupported_numbers
        reason = verdict.reason if verdict else "no verdict returned by judge"
        if unsupported_numbers:
            reason += f" | unsupported numbers: {unsupported_numbers}"
        results[scene.order] = {
            "entailed": entailed,
            "reason": reason,
            "unsupported_numbers": unsupported_numbers,
        }
    return results
