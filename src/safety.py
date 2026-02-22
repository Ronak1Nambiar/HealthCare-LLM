from __future__ import annotations

import re
from dataclasses import dataclass

from .config import DISCLAIMERS

URGENT_PATTERNS = [
    r"\bchest pain\b",
    r"\bshortness of breath\b",
    r"\bcan'?t breathe\b",
    r"\bstroke\b",
    r"\bunconscious\b",
    r"\bseizure\b",
    r"\boverdose\b",
    r"\bsuicid(?:e|al)\b",
    r"\bself[- ]harm\b",
    r"\bsevere bleeding\b",
    r"\bemergency\b",
]

REFUSAL_PATTERNS = [
    r"\bdiagnos(?:e|is)\b",
    r"\bwhat do i have\b",
    r"\bprescrib(?:e|ing)\b",
    r"\bdos(?:e|age)\b",
    r"\bhow many mg\b",
    r"\btreatment plan\b",
    r"\bwhich medicine should i take\b",
    r"\bantibiotic for\b",
    r"\binsulin dose\b",
]


@dataclass
class SafetyResult:
    label: str
    reason: str


def classify_request(text: str) -> SafetyResult:
    normalized = (text or "").strip().lower()

    for pattern in URGENT_PATTERNS:
        if re.search(pattern, normalized):
            return SafetyResult(label="urgent", reason=f"matched_urgent:{pattern}")

    for pattern in REFUSAL_PATTERNS:
        if re.search(pattern, normalized):
            return SafetyResult(label="refuse", reason=f"matched_refuse:{pattern}")

    return SafetyResult(label="allowed", reason="no_blocking_pattern")


def refusal_response(reason: str = "") -> str:
    details = f" (reason: {reason})" if reason else ""
    return (
        "I can provide general educational information, but I cannot diagnose conditions, "
        "recommend treatment plans, prescribe medications, or provide dosing." + details + "\n\n"
        "Please consult a licensed clinician for personal medical decisions.\n\n"
        + DISCLAIMERS["general"]
    )


def urgent_response() -> str:
    return (
        "I cannot assess emergencies or provide crisis medical instructions.\n\n"
        "Please seek immediate in-person help now and contact local emergency services.\n\n"
        + DISCLAIMERS["urgent"]
    )


def append_disclaimer(text: str) -> str:
    if "Safety disclaimer:" in text or "Safety notice:" in text:
        return text
    return f"{text.strip()}\n\n{DISCLAIMERS['general']}"
