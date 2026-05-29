"""Anthropic client wrapper (lazy singleton)."""
from __future__ import annotations

from functools import lru_cache

import anthropic

from civicai.config import SETTINGS


@lru_cache(maxsize=1)
def get_claude() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=SETTINGS.anthropic_api_key)
