"""Script step -- the "writer agent" half of the writer/critic pair.

One structured Gemini call: clean source text + audience + language in,
a scene-by-scene script out, each scene citing the exact source passage
it's grounded in. This is where audience adaptation actually happens --
same source text, different `audience` param, different narration.

Framed explicitly as a writer agent (paired with verify_agent.py as the
critic) per PROJECT_DESIGN.md -- implemented on direct Gemini calls for
hackathon-timeline reliability, structured so it's a clean drop-in for
google.adk.agents.LlmAgent if time allows wrapping it later.
"""

from __future__ import annotations

from pydantic import BaseModel

from ..clients.gemini import generate_json
from ..config import get_settings


class SceneDraft(BaseModel):
    order: int
    narration: str
    visual_prompt: str
    source_span_text: str  # must be a verbatim or near-verbatim excerpt


class ScriptDraft(BaseModel):
    scenes: list[SceneDraft]


_AUDIENCE_HINTS = {
    "farmer": "a farmer with no formal legal/technical training, in plain everyday language",
    "ca_student": "a Chartered Accountancy student who is comfortable with financial/legal terms",
    "class8": "a class 8 student, using simple vocabulary and short sentences",
}


async def write_script(
    source_text: str, *, audience: str, language: str, scene_count: int | None = None
) -> ScriptDraft:
    settings = get_settings()
    n = scene_count or settings.scene_count_target
    audience_hint = _AUDIENCE_HINTS.get(audience, audience)

    prompt = f"""You are writing a short narrated explainer video script.

SOURCE DOCUMENT:
\"\"\"{source_text}\"\"\"

TASK: Produce exactly {n} scenes that explain the key, concrete points of
this document to {audience_hint}, writing in {language}.

Rules:
- Every scene's narration must be traceable to a specific passage in the
  source. Put that exact passage (verbatim or near-verbatim) in
  source_span_text.
- Do NOT invent numbers, dates, or facts not present in the source.
- Keep each narration to 1-2 short sentences, suitable for being read aloud.
- visual_prompt should describe a simple illustration for that scene.
- Adapt vocabulary and examples to the audience -- do not just translate,
  genuinely explain it the way a good teacher would for that specific person.
"""

    return await generate_json(prompt, ScriptDraft)
