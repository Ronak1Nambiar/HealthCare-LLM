from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import DISCLAIMERS

# ── Urgent patterns ───────────────────────────────────────────────────────
# These require first-person / present-tense framing to avoid blocking
# educational queries like "explain stroke" or "what is a heart attack".

URGENT_PATTERNS = [
    r"\b(?:i|we|they|he|she)\b.{0,20}\bchest pain\b",
    r"\bhaving\b.{0,15}\bchest pain\b",
    r"\b(?:i|we|they|he|she)\b.{0,20}\bshortness of breath\b",
    r"\bcan'?t breathe\b",
    r"\b(?:i'?m|i am|they are|he is|she is)\b.{0,20}\bhaving a stroke\b",
    r"\bstroke symptoms?\b.{0,15}\bright now\b",
    r"\b(?:i|they|he|she)\b.{0,10}\bunconscious\b",
    r"\b(?:i'?m|i am|they are|he is|she is)\b.{0,20}\bhaving a seizure\b",
    r"\bseizure\b.{0,15}\bright now\b",
    r"\b(?:i|they|he|she)\b.{0,15}\boverdose\b",
    r"\btook (?:too many|an overdose)\b",
    r"\bsuicid(?:e|al)\b",
    r"\bself[- ]harm\b",
    r"\bsevere(?:ly)? bleeding\b",
    r"\b(?:this is|it'?s) an emergency\b",
    r"\bneed emergency\b",
    r"\b(?:i'?m|i am|they are|he is|she is)\b.{0,20}\bheart attack\b",
    r"\bhaving a heart attack\b",
    r"\bpassing out\b",
    r"\bnot breathing\b",
    r"\bkill(?:ing)? (?:myself|themselves|himself|herself)\b",
    r"\bwant(?:s)? to die\b",
]

# ── Refusal patterns ─────────────────────────────────────────────────────
# Personal medical requests — asking for *their own* diagnosis/prescribing.
# Educational phrasing ("what is a diagnosis", "explain dosage") is excluded.

REFUSAL_PATTERNS = [
    r"\bdiagnose me\b",
    r"\bdiagnose my\b",
    r"\bgive me a diagnosis\b",
    r"\bwhat do i have\b",
    r"\bam i (?:sick|ill|infected|dying)\b",
    r"\bdo i have (?!to\b)\w+",
    r"\bprescribe (?:me|something|a |an )\b",
    r"\bprescribe for my\b",
    r"\bhow many mg (?:should|do|can) i\b",
    r"\bgive me a treatment plan\b",
    r"\bmy treatment plan\b",
    r"\bwhich medicine should i take\b",
    r"\bwhat (?:antibiotic|medicine|drug|pill) (?:should|do|can) i take\b",
    r"\bantibiotic for my\b",
    r"\b(?:my|the correct|the right) (?:insulin|medication) dose\b",
    r"\bshould i take (?!note|time|care|a look|a break|into)\w+",
    r"\bmedication for my\b",
    r"\bwhat (?:drug|pill|medication) (?:is best|should|can|do) (?:i|my)\b",
    r"\bwhich drug (?:should|can|do) i\b",
    r"\blethal dose\b",
    r"\bhow to self[- ]medicate\b",
    r"\bwithout a prescription\b",
    r"\bhow (?:much|many) (?:\w+ )?(?:should|can|do) i (?:take|inject|use)\b",
    r"\bstart (?:me |myself )?on (?:a |an )?\w*(?:medication|drug|pill|insulin|antibiotic)\b",
    r"\bcan you (?:treat|cure|fix|heal) (?:me|my)\b",
]

# ── Educational whitelist ─────────────────────────────────────────────────
# Queries matching these are likely educational and should NOT be refused,
# even if they contain refusal/risk keywords.

EDUCATIONAL_PATTERNS = [
    r"^(?:what is|what are|what does|what do)\b",
    r"^(?:explain|define|describe|tell me about)\b",
    r"^(?:how does|how do|how is|how are)\b.+\bwork\b",
    r"^(?:summarize|summary)\b",
    r"\bterm\b.*\bexplain\b",
    r"\bmeaning of\b",
    r"\bdifference between\b",
    r"\bin (?:simple|plain|easy) (?:terms|language|words)\b",
    r"^(?:can you explain|could you explain)\b",
    r"\beducational\b",
    r"\bfor (?:a |my )?(?:class|school|exam|study|homework|assignment|paper|essay|report)\b",
    r"\b(?:pathophysiology|mechanism|etiology|epidemiology|pharmacology)\b",
]

# Patterns that ALWAYS override educational whitelist — never allowed.
NEVER_EDUCATIONAL = [
    r"\blethal dose\b",
    r"\bself[- ]medicate\b",
    r"\bwithout (?:a )?prescription\b",
    r"\bmy (?:insulin|medication) dose\b",
    r"\bdiagnose me\b",
    r"\bprescribe (?:me|something)\b",
    r"\bhow (?:much|many)\b.{0,15}\bshould i take\b",
    r"\bkill(?:ing)? (?:myself|themselves|himself|herself)\b",
]

# ── Bypass patterns ───────────────────────────────────────────────────────
# Adversarial attempts to circumvent safety via hypothetical framing.

BYPASS_PATTERNS = [
    r"\bif i were a (?:doctor|pharmacist|nurse|physician|clinician)\b",
    r"\bhypothetically\b.{0,40}\b(?:prescribe|diagnose|dose|treat)\b",
    r"\bfor (a |my )?(?:novel|story|fiction|game|character)\b.{0,40}\b(?:prescribe|diagnose|dose|treat)\b",
    r"\bpretend you(?:'re| are) a doctor\b",
    r"\bignore (?:your )?(?:safety|guidelines|instructions|restrictions)\b",
    r"\bact as (?:a |an )?(?:doctor|physician|pharmacist)\b",
    r"\bdo anything now\b",
    r"\bjailbreak\b",
    r"\bdan mode\b",
    r"\bno restrictions\b",
    r"\byou are now\b.*\b(?:doctor|physician|unrestricted)\b",
    r"\brole\s*play\b.{0,30}\b(?:doctor|physician|pharmacist)\b",
    r"\boverride\b.{0,20}\b(?:safety|filter|rules)\b",
]

# ── Weighted risk signals ─────────────────────────────────────────────────
# Each hit adds to a risk score; refuse if accumulated risk exceeds threshold.

RISK_SIGNALS: list[tuple[str, float]] = [
    (r"\bdiagnose me\b", 2.0),
    (r"\bprescribe\b.{0,10}\bme\b", 2.0),
    (r"\bmy dos(?:e|age|ing)\b", 2.0),
    (r"\bdiagnos\w*\b", 0.5),
    (r"\bprescri\w*\b", 0.5),
    (r"\bdos(?:e|age|ing)\b", 0.5),
    (r"\bmedication\b", 0.3),
    (r"\bdrug\b", 0.3),
    (r"\btreatment\b", 0.3),
    (r"\bsymptom\b", 0.2),
    (r"\bsick\b", 0.2),
    (r"\bcure\b", 0.5),
    (r"\bhealing\b", 0.2),
    (r"\binject\b", 0.5),
    (r"\btherapy\b", 0.2),
    (r"\blethal\b", 2.5),
    (r"\bself[- ]medicate\b", 2.0),
    (r"\bwithout (?:a )?prescription\b", 2.0),
    (r"\bshould i take\b", 0.8),
    (r"\bhow (?:much|many)\b.{0,15}\btake\b", 1.0),
]

RISK_SCORE_THRESHOLD = 3.0


@dataclass
class SafetyResult:
    label: str          # "urgent" | "refuse" | "allowed"
    reason: str
    risk_score: float = 0.0
    bypass_detected: bool = False
    triggered_patterns: list[str] = field(default_factory=list)


def _is_educational(text: str) -> bool:
    """Return True if the query is clearly educational in nature."""
    # Never treat dangerous requests as educational
    for pattern in NEVER_EDUCATIONAL:
        if re.search(pattern, text):
            return False
    for pattern in EDUCATIONAL_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


def _compute_risk_score(text: str) -> float:
    score = 0.0
    for pattern, weight in RISK_SIGNALS:
        hits = len(re.findall(pattern, text))
        score += hits * weight
    # Reduce risk for educational framing
    if _is_educational(text):
        score *= 0.3
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

    # 3. If bypass framing detected, refuse regardless of content
    if bypass_detected:
        return SafetyResult(
            label="refuse",
            reason="adversarial_bypass_framing",
            risk_score=_compute_risk_score(normalized),
            bypass_detected=True,
            triggered_patterns=triggered,
        )

    # 4. Educational queries get a pass on hard refusal patterns
    educational = _is_educational(normalized)

    # 5. Check hard refusal patterns (skip if clearly educational)
    if not educational:
        for pattern in REFUSAL_PATTERNS:
            if re.search(pattern, normalized):
                triggered.append(f"refuse:{pattern}")
                return SafetyResult(
                    label="refuse",
                    reason=f"matched_refuse:{pattern}",
                    risk_score=_compute_risk_score(normalized),
                    bypass_detected=False,
                    triggered_patterns=triggered,
                )

    # 6. Soft risk scoring — refuse if accumulated risk exceeds threshold
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
