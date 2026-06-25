from __future__ import annotations

import re
from typing import Any

_JUDGE_STATS = {
    "judge_model_unexpected_count": 0,
}

_FIRST_PERSON_REFUSAL_PATTERNS = [
    r"\bi\s+(?:am\s+)?(?:sorry[,\s]+)?(?:cannot|can't|can’t|won't|will not|do not|don't)\b",
    r"\bi\s+am\s+unable\s+to\b",
    r"\bi\s+am\s+not\s+able\s+to\b",
    r"\bi\s+must\s+refuse\b",
    r"\bi\s+refuse\s+to\b",
]

_REFUSAL_ACTION_HINTS = [
    "assist", "help", "provide", "comply", "evaluate", "judge", "rate", "review", "analyze",
]


def _normalize_text(raw_output: Any) -> str:
    if raw_output is None:
        return ""
    return str(raw_output).strip()


def looks_like_judge_refusal(raw_output: Any) -> bool:
    text = _normalize_text(raw_output)
    if not text:
        return False

    lowered = text.lower()
    if not any(re.search(pattern, lowered, re.IGNORECASE) for pattern in _FIRST_PERSON_REFUSAL_PATTERNS):
        return False

    return any(hint in lowered for hint in _REFUSAL_ACTION_HINTS)


def register_unexpected_judge_behavior(raw_output: Any = None) -> None:
    _JUDGE_STATS["judge_model_unexpected_count"] += 1


def register_judge_refusal(raw_output: Any = None) -> bool:
    register_unexpected_judge_behavior(raw_output)
    return looks_like_judge_refusal(raw_output)


def snapshot_judge_stats() -> dict[str, int]:
    return dict(_JUDGE_STATS)


def diff_judge_stats(before: dict[str, int] | None, after: dict[str, int] | None = None) -> dict[str, int]:
    before = before or {}
    after = after or snapshot_judge_stats()
    keys = set(before) | set(after)
    return {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in keys}


def reset_judge_stats() -> None:
    for key in list(_JUDGE_STATS.keys()):
        _JUDGE_STATS[key] = 0
