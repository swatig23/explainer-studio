"""Standalone connectivity smoke test -- run this FIRST, before anything
else, so a network/VPN/key problem is diagnosed in 10 seconds instead of
discovered halfway through a pipeline run. Ported from the adintel
project's proven version of this exact script.

Usage:
    uv run python scripts/test_gemini_connection.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from explainer_studio.clients.gemini import generate_text  # noqa: E402
from explainer_studio.config import get_settings  # noqa: E402


async def main() -> int:
    settings = get_settings()
    print("=" * 60)
    print("Vernacular Explainer Studio -- Gemini connectivity smoke test")
    print("=" * 60)
    print(f"GOOGLE_API_KEY set: {bool(settings.google_api_key)}")
    print(f"Text model: {settings.gemini_text_model}")

    if not settings.google_api_key:
        print("\nFAILED: GOOGLE_API_KEY is not set in .env")
        print("Get one at https://aistudio.google.com/apikey")
        return 1

    try:
        result = await generate_text("Say hello in exactly 5 words.")
        print(f"\nGemini says: {result!r}")
        if not result.strip():
            print("FAILED: empty response")
            return 1
        print("\nOK -- text generation works. Ready to build.")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"\nFAILED: {type(e).__name__}: {e}")
        print("\nCommon causes:")
        print("  - Invalid/expired key -- regenerate at https://aistudio.google.com/apikey")
        print("  - Corporate VPN blocking generativelanguage.googleapis.com")
        print("    (adintel hit this before -- try disconnecting VPN or switching networks)")
        print("  - Free tier rate limit -- wait a minute and retry")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
