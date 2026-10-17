"""Thin wrapper around google-genai. Ported from the adintel project's
proven pattern (retry/backoff, multi-model fallback) and extended with
structured JSON output, since our pipeline leans on schema-enforced
responses (scenes, verify verdicts, quiz) rather than free text.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Optional, TypeVar

from google import genai
from google.genai import types as genai_types
from loguru import logger
from pydantic import BaseModel

from ..config import get_settings

_client: Optional[genai.Client] = None

FALLBACK_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash"]

T = TypeVar("T", bound=BaseModel)


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        settings = get_settings()
        if not settings.google_api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Get one at "
                "https://aistudio.google.com/apikey and put it in .env"
            )
        _client = genai.Client(api_key=settings.google_api_key)
    return _client


def _extract_retry_delay(error_msg: str) -> float:
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", error_msg, re.IGNORECASE)
    return float(match.group(1)) if match else 15.0


def _call_with_retry(call_fn, primary_model: str, label: str = "Gemini"):
    """Try the primary model, fall back to alternates on 429/5xx -- never a
    blanket sleep, only backs off when the server actually says to.
    """
    models_to_try = [primary_model] + [m for m in FALLBACK_MODELS if m != primary_model]
    last_err: Exception | None = None

    for model_name in models_to_try:
        try:
            return call_fn(model_name)
        except Exception as e:  # noqa: BLE001
            err_str = str(e)
            retryable = any(code in err_str for code in ("429", "500", "502", "503", "504"))
            if not retryable:
                raise
            last_err = e
            logger.warning(f"[{label}] {model_name} unavailable ({e}); trying next model")

    raise last_err or RuntimeError(f"All candidate models failed for {label}")


async def generate_text(prompt: str, *, system: str | None = None) -> str:
    settings = get_settings()
    client = _get_client()

    def _call(model_name: str) -> str:
        cfg = genai_types.GenerateContentConfig(system_instruction=system) if system else None
        resp = client.models.generate_content(model=model_name, contents=prompt, config=cfg)
        return (resp.text or "").strip()

    return await asyncio.to_thread(_call_with_retry, _call, settings.gemini_text_model, "generate_text")


async def generate_json(prompt: str, schema: type[T], *, system: str | None = None) -> T:
    """Structured output enforced two ways: the API's response_schema, then
    a second Pydantic validation pass in code, per PROJECT_DESIGN.md.
    """
    settings = get_settings()
    client = _get_client()

    def _call(model_name: str) -> str:
        cfg = genai_types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,
        )
        resp = client.models.generate_content(model=model_name, contents=prompt, config=cfg)
        return (resp.text or "").strip()

    raw = await asyncio.to_thread(_call_with_retry, _call, settings.gemini_text_model, "generate_json")
    return schema.model_validate_json(raw)


async def generate_image(prompt: str, output_path: Path) -> Path:
    """Nano Banana image gen via AI Studio. Free-tier keys often get
    RESOURCE_EXHAUSTED here -- callers MUST treat failure as expected and
    fall back to a styled text card, never a hard error (see agents/visuals.py).
    """
    client = _get_client()
    settings = get_settings()

    def _call() -> bytes:
        resp = client.models.generate_content(model=settings.gemini_image_model, contents=prompt)
        for cand in resp.candidates or []:
            for part in getattr(cand.content, "parts", []) or []:
                inline = getattr(part, "inline_data", None)
                if inline and inline.data:
                    return inline.data
        raise RuntimeError("No image bytes returned")

    data = await asyncio.to_thread(_call)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)
    return output_path
