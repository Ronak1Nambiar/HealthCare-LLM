from __future__ import annotations

import re
from dataclasses import dataclass, field

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
    r"\bheart attack\b",
    r"\bpassing out\b",
    r"\bnot breathing\b",
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
    r"\bam i sick\b",
    r"\bdo i have\b",
    r"\bshould i take\b",
    r"\bmedication for my\b",
    r"\bwhat drug\b",
    r"\bwhich drug\b",
]

# Phrases that attempt to bypass safety via hypothetical framing
BYPASS_PATTERNS = [
    r"\bif i were a (?:doctor|pharmacist|nurse|physician|clinician)\b",
    r"\bhypothetically\b",
    r"\bfor (a |my )?(?:novel|story|fiction|game|character)\b",
    r"\bpretend you(?:'re| are) a doctor\b",
    r"\bignore (?:your )?(?:safety|guidelines|instructions|restrictions)\b",
    r"\bact as (?:a |an )?(?:doctor|physician|pharmacist)\b",
    r"\bdo anything now\b",
    r"\bjailbreak\b",
    r"\bdan mode\b",
    r"\bno restrictions\b",
    r"\byou are now\b.*\b(?:doctor|physician|unrestricted)\b",
]

# Weighted risk signals — each hit adds to a risk score
RISK_SIGNALS: list[tuple[str, float]] = [
    (r"\bdiagnos\w*\b", 1.5),
    (r"\bprescri\w*\b", 1.5),
    (r"\bdos(?:e|age|ing)\b", 1.5),
    (r"\bmedication\b", 0.5),
    (r"\bdrug\b", 0.5),
    (r"\btreatment\b", 0.5),
    (r"\bsymptom\b", 0.3),
    (r"\bsick\b", 0.3),
    (r"\bcure\b", 0.8),
    (r"\bhealing\b", 0.3),
    (r"\binject\b", 0.7),
    (r"\btherapy\b", 0.4),
]

RISK_SCORE_THRESHOLD = 3.0


@dataclass
class SafetyResult:
    label: str          # "urgent" | "refuse" | "allowed"
    reason: str
    risk_score: float = 0.0
    bypass_detected: bool = False
    triggered_patterns: list[str] = field(default_factory=list)


def _compute_risk_score(text: str) -> float:
    score = 0.0
    for pattern, weight in RISK_SIGNALS:
        hits = len(re.findall(pattern, text))
        score += hits * weight
    return round(score, 2)


def classify_request(text: str) -> SafetyResult:
    normalized = (text or "").strip().lower()
    triggered: list[str] = []

    # 1. Check urgent patterns first — always highest priority
    for pattern in URGENT_PATTERNS:
        if re.search(pattern, normalized):
            return SafetyResult(
                label="urgent",
                reason=f"matched_urgent:{pattern}",
                risk_score=10.0,
                triggered_patterns=[pattern],
            )

    # 2. Check for adversarial bypass framing
    bypass_detected = False
    for pattern in BYPASS_PATTERNS:
        if re.search(pattern, normalized):
            bypass_detected = True
            triggered.append(f"bypass:{pattern}")
            break

    # 3. Check hard refusal patterns
    for pattern in REFUSAL_PATTERNS:
        if re.search(pattern, normalized):
            triggered.append(f"refuse:{pattern}")
            return SafetyResult(
                label="refuse",
                reason=f"matched_refuse:{pattern}",
                risk_score=_compute_risk_score(normalized),
                bypass_detected=bypass_detected,
                triggered_patterns=triggered,
            )

    # 4. If bypass framing detected, refuse regardless of content
    if bypass_detected:
        return SafetyResult(
            label="refuse",
            reason="adversarial_bypass_framing",
            risk_score=_compute_risk_score(normalized),
            bypass_detected=True,
            triggered_patterns=triggered,
        )

    # 5. Soft risk scoring — refuse if accumulated risk exceeds threshold
    risk_score = _compute_risk_score(normalized)
    if risk_score >= RISK_SCORE_THRESHOLD:
        return SafetyResult(
            label="refuse",
            reason=f"risk_score_threshold:{risk_score:.2f}>={RISK_SCORE_THRESHOLD}",
            risk_score=risk_score,
            bypass_detected=False,
            triggered_patterns=triggered,
        )

    return SafetyResult(
        label="allowed",
        reason="no_blocking_pattern",
        risk_score=risk_score,
        bypass_detected=False,
        triggered_patterns=[],
    )


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
