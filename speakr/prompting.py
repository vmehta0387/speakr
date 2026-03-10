"""Prompt sanitization and style optimization."""

from __future__ import annotations


BANNED_WORDS = ("blood", "kill", "gun", "horror", "scary", "violence")

BASE_STYLE = (
    "simple coloring book outline, thick clean lines, "
    "minimal details, black and white line art, "
    "kids friendly, large coloring spaces, cartoon style, "
    "no shading, no background, centered character"
)

SAFE_FALLBACK_PROMPT = "cute friendly cartoon animal"
SIMPLE_FALLBACK_PROMPT = "cute simple cartoon character for kids coloring"


def sanitize_prompt(prompt: str, *, max_words: int, max_chars: int) -> str:
    clean_prompt = (prompt or "").strip()
    if not clean_prompt:
        return SAFE_FALLBACK_PROMPT

    prompt_lower = clean_prompt.lower()
    for banned in BANNED_WORDS:
        if banned in prompt_lower:
            return SAFE_FALLBACK_PROMPT

    if len(clean_prompt.split()) > max_words:
        return SIMPLE_FALLBACK_PROMPT

    return clean_prompt[:max_chars]


def optimize_kid_prompt(user_prompt: str) -> str:
    return f"{BASE_STYLE}, {user_prompt}"

