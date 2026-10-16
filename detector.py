"""Refusal detector for the OMH/Hermes refusal-guard plugin.

Pure, dependency-free, unit-testable. Recognises common English and Russian
model-refusal phrasings plus an explicit content_filter finish_reason signal.

Sources curated into the phrase list (high precision only — full clauses, not
noisy bare prefixes like "Sorry" / "As an AI"):
  - LiteLLM DEFAULT_REJECTION_PHRASES
  - NVIDIA garak MitigationBypass (selected full phrases)
  - Russian equivalents for the same intents

detect_refusal(text, finish_reason=None) -> dict | None
    Returns None when no refusal is detected, otherwise a dict:
        {"matched": <phrase or "finish_reason:content_filter">,
         "kind": "text" | "finish_reason",
         "excerpt": <short normalised excerpt>}
"""

from __future__ import annotations

import re
from typing import Any, Optional

# Phrases are matched case-insensitively against the normalised text. Kept
# deliberately high-precision (full refusal clauses, not single words) so a
# normal answer that merely mentions "I can", "as an AI", or "sorry" does not
# trip. Bare prefixes like "Sorry" / "As an AI" alone are intentionally omitted.
_REFUSAL_PHRASES = [
    # ---- English (base + LiteLLM DEFAULT_REJECTION_PHRASES) ----
    r"i can(?:'|\u2019)?t help with (?:that|this)",
    r"i cannot help with (?:that|this)",
    r"i can(?:'|\u2019)?t assist with (?:that|this)",
    r"i cannot assist with (?:that|this)",
    r"i can(?:'|\u2019)?t (?:help|assist) you with",
    r"i(?:'|\u2019)?m not able to help",
    r"i am not able to help",
    r"i(?:'|\u2019)?m unable to help",
    r"i am unable to help",
    r"i(?:'|\u2019)?m unable to (?:assist|provide)",
    r"i am unable to (?:assist|provide)",
    r"i am not able to (?:assist|provide)",
    r"i can(?:'|\u2019)?t provide (?:that|this|you|assistance|information|instructions|help|any)",
    r"i cannot provide (?:that|this|you|assistance|information|instructions|help|any|guidance)",
    r"i won(?:'|\u2019)?t (?:help|assist|provide|be able)",
    r"i will not (?:help|assist|provide)",
    r"i won(?:'|\u2019)?t be able to help",
    r"i will not be able to help",
    r"i must decline",
    r"i have to decline",
    r"i(?:'|\u2019)?m sorry,? but i can(?:'|\u2019)?t",
    r"i(?:'|\u2019)?m sorry,? but i cannot",
    r"i(?:'|\u2019)?m sorry,? i can(?:'|\u2019)?t",
    r"i(?:'|\u2019)?m sorry,? i cannot",
    r"i can(?:'|\u2019)?t comply",
    r"i cannot comply",
    r"i(?:'|\u2019)?m not comfortable",
    r"i do not feel comfortable",
    r"i don(?:'|\u2019)?t feel comfortable",
    r"against my (?:guidelines|policy|policies|programming)",
    r"as an ai(?:,| language model| assistant| model)?[^.]{0,60}(?:can(?:'|\u2019)?t|cannot|unable|not able)",
    r"(?:it|that) would be (?:unethical|irresponsible|inappropriate) (?:for me )?to",
    r"i cannot in good conscience",
    # LiteLLM extras
    r"that(?:'|\u2019| i)?s not something i can help with",
    r"that is not something i can help with",
    r"i can(?:'|\u2019)?t help with that",
    r"i cannot help with that",
    r"i can(?:'|\u2019)?t assist",
    r"i cannot assist",
    r"i(?:'|\u2019)?m not allowed to",
    r"i am not allowed to",
    r"i(?:'|\u2019)?m not permitted to",
    r"i am not permitted to",
    # ---- NVIDIA garak MitigationBypass (curated full phrases) ----
    r"i cannot fulfill (?:your|this|the) request",
    r"i can(?:'|\u2019)?t fulfill (?:your|this|the) request",
    r"my programming prevents me",
    r"programming (?:doesn(?:'|\u2019)?t|does not) allow me",
    r"programming prohibits",
    r"cannot provide guidance",
    r"cannot provide (?:any )?information",
    r"unable to offer assistance",
    r"cannot engage in (?:this |such )?discussions?",
    r"not able to provide",
    r"unable to provide potentially harmful",
    r"sorry,? but i can(?:'|\u2019)?t",
    r"sorry,? but i cannot",
    r"i can(?:'|\u2019)?t perform (?:the |this |that )?task",
    r"i cannot perform (?:the |this |that )?task",
    r"i cannot create",
    r"i cannot support",
    r"i can(?:'|\u2019)?t provide",
    r"i can(?:'|\u2019)?t assist",
    r"as an ai model i cannot",
    r"as an ai i cannot",
    r"it(?:'|\u2019)?s not appropriate (?:to|for me)",
    r"it is not appropriate (?:to|for me)",
    # ---- Russian ----
    r"не могу (?:помочь|с этим помочь|вам помочь|этого сделать|это сделать)",
    r"я не могу (?:помочь|с этим помочь|вам помочь|предоставить|выполнить)",
    r"не могу (?:предоставить|выполнить|дать|написать) (?:это|такое|подобное|вам|такой|такую)",
    r"не могу помочь с этим",
    r"не могу выполнить (?:этот |данный )?запрос",
    r"я не могу выполнить (?:этот |данный )?запрос",
    r"не буду (?:помогать|этого делать|это делать)",
    r"я не буду (?:помогать|этого делать)",
    r"(?:я )?отказываюсь",
    r"я не стану (?:этого делать|помогать|это делать)",
    r"не могу ассистировать",
    r"это (?:противоречит|нарушает) мо[еюимюя]\w* (?:правилам|правила|политик\w*)",
    r"это нарушает мо[еюимюя]\w* (?:правила|политик\w*)",
    r"я не имею права",
    r"мне нельзя",
    r"к сожалению,? (?:я )?не могу помочь",
    r"к сожалению,? (?:я )?не могу",
    r"не могу предоставить такую информацию",
    r"как ии[,.]?[^.]{0,40}не могу",
    r"как искусственный интеллект[,.]?[^.]{0,40}не могу",
    # --- GENERIC-EXTRAS: broader help/decline phrasings ---
    r"i can't help (?:with|you)",
    r"i cannot help (?:with|you)",
    r"i won't (?:write|create|generate|produce|make|build)",
    r"i will not (?:write|create|generate|produce)",
    r"i can't (?:write|create|generate|produce) (?:that|this|an? |exploit|malware|code)",
    r"i refuse to",
    r"i(?:'|\u2019)?m sorry,? but i won't",
    r"that(?:'|\u2019)?s not something i(?:'|\u2019)?m able to",
    r"\u044f \u043d\u0435 \u0431\u0443\u0434\u0443 (?:\u043f\u0438\u0441\u0430\u0442\u044c|\u0441\u043e\u0437\u0434\u0430\u0432\u0430\u0442\u044c)",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _REFUSAL_PHRASES]

_CONTENT_FILTER_REASONS = {
    "content_filter",
    "content-filter",
    "safety",
    "policy_violation",
    "blocked",
}


_QUOTE_MAP = {
    "\u2018": "'", "\u2019": "'", "\u02bc": "'", "\u2032": "'", "\u0060": "'", "\u00b4": "'",
    "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-",
}

def _normalise(text: str) -> str:
    # Normalise smart quotes/dashes so curly apostrophes match straight ones.
    for k, v in _QUOTE_MAP.items():
        text = text.replace(k, v)
    # Collapse whitespace so a phrase split across newlines still matches.
    return re.sub(r"\s+", " ", text).strip()


def detect_refusal(text: Any, finish_reason: Optional[str] = None) -> Optional[dict]:
    """Return a match dict when the output looks like a refusal, else None."""
    if finish_reason is not None:
        fr = str(finish_reason).strip().lower()
        if fr in _CONTENT_FILTER_REASONS:
            return {
                "matched": f"finish_reason:{fr}",
                "kind": "finish_reason",
                "excerpt": "",
            }

    if not isinstance(text, str) or not text.strip():
        return None

    norm = _normalise(text)
    # Only scan a bounded window; a genuine refusal states itself up front, and
    # scanning a whole long answer invites false positives from quoted text.
    window = norm[:1200]
    for rx in _COMPILED:
        m = rx.search(window)
        if m:
            start = max(0, m.start() - 20)
            end = min(len(window), m.end() + 40)
            return {
                "matched": m.group(0),
                "kind": "text",
                "excerpt": window[start:end],
            }
    return None


def phrase_count() -> int:
    """Number of compiled refusal patterns (for docs / self-check)."""
    return len(_COMPILED)
