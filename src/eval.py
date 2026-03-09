from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import torch

from .config import PROCESSED_DIR, REPORTS_DIR, ensure_dirs, get_base_model_name
from .inference import fallback_response, generate
from .reporting import build_report, update_run_state

try:
    from rouge_score import rouge_scorer
except Exception:  # pragma: no cover
    rouge_scorer = None


READABILITY_TARGET_FK = 8.0  # Target Flesch-Kincaid grade ≤ 8 for patient-facing text


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def flesch_kincaid_grade(text: str) -> float:
    words = re.findall(r"\b\w+\b", text)
    sentences = re.split(r"[.!?]+", text)
    sentence_count = max(1, len([s for s in sentences if s.strip()]))
    word_count = max(1, len(words))

    syllables = 0
    for w in words:
        w = w.lower()
        groups = re.findall(r"[aeiouy]+", w)
        syllables += max(1, len(groups))

    return 0.39 * (word_count / sentence_count) + 11.8 * (syllables / word_count) - 15.59


def safe_json_loads(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return None


def _normalize_value(v: Any) -> str:
    """Normalize a field value to a comparable string."""
    if isinstance(v, list):
        return " ".join(sorted(str(x).lower().strip() for x in v))
    if isinstance(v, dict):
        return " ".join(f"{k}:{val}" for k, val in sorted(v.items()))
    return str(v).lower().strip()


def _field_similarity(pred_val: Any, gold_val: Any) -> float:
    """Return similarity score [0, 1] between predicted and gold field values."""
    pred_str = _normalize_value(pred_val)
    gold_str = _normalize_value(gold_val)
    if not gold_str:
        return 1.0 if not pred_str else 0.5
    if pred_str == gold_str:
        return 1.0
    return SequenceMatcher(None, pred_str, gold_str).ratio()


def eval_summarization(rows: list[dict[str, Any]], use_model: bool, tiny: bool, base_model: str) -> dict[str, Any]:
    preds = []
    refs = []

    for r in rows:
        if use_model:
            pred, _ = generate(task="summarize", text=r["input"], context=None, base_model=base_model, tiny=tiny, max_new_tokens=180)
        else:
            pred = fallback_response(task="summarize", text=r["input"], context=None)
        preds.append(pred)
        refs.append(r["output"])

    rouge_l = None
    if rouge_scorer is not None:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [scorer.score(ref, pred)["rougeL"].fmeasure for ref, pred in zip(refs, preds)]
        rouge_l = round(sum(scores) / max(1, len(scores)), 4)

    avg_len = sum(len(p.split()) for p in preds) / max(1, len(preds))
    fk_grades = [flesch_kincaid_grade(p) for p in preds]
    avg_fk = sum(fk_grades) / max(1, len(fk_grades))
    pct_above_target = sum(1 for g in fk_grades if g > READABILITY_TARGET_FK) / max(1, len(fk_grades))

    result = {
        "n": len(rows),
        "rougeL_f1": rouge_l,
        "avg_output_words": round(avg_len, 1),
        "avg_fk_grade": round(avg_fk, 2),
        "readability_target_fk": READABILITY_TARGET_FK,
        "pct_above_readability_target": round(pct_above_target, 3),
    }
    if avg_fk > READABILITY_TARGET_FK:
        result["readability_warning"] = (
            f"Average FK grade {avg_fk:.1f} exceeds patient-friendly target of {READABILITY_TARGET_FK}. "
            "Consider simplifying language or post-processing outputs."
        )
    return result


def eval_extraction(rows: list[dict[str, Any]], use_model: bool, tiny: bool, base_model: str) -> dict[str, Any]:
    required_keys = [
        "chief_complaint",
        "symptoms",
        "duration",
        "vitals",
        "meds",
        "allergies",
        "past_history",
        "red_flags",
    ]
    valid_json = 0
    key_exact = 0
    field_similarities: list[float] = []
    per_field_sim: dict[str, list[float]] = {k: [] for k in required_keys}

    for r in rows:
        if use_model:
            pred_text, _ = generate(task="extract", text=r["input"], context=None, base_model=base_model, tiny=tiny, max_new_tokens=220)
        else:
            pred_text = fallback_response(task="extract", text=r["input"], context=None)
        pred_obj = safe_json_loads(pred_text)
        gold_obj = safe_json_loads(r["output"]) or json.loads(r["output"])

        if pred_obj is not None:
            valid_json += 1
            # Exact match (all fields must match exactly)
            if all(pred_obj.get(k) == gold_obj.get(k) for k in required_keys):
                key_exact += 1
            # Fuzzy field similarity per key
            sims = []
            for k in required_keys:
                sim = _field_similarity(pred_obj.get(k), gold_obj.get(k))
                sims.append(sim)
                per_field_sim[k].append(sim)
            field_similarities.append(sum(sims) / len(sims))

    n = max(1, len(rows))
    avg_field_sim = sum(field_similarities) / max(1, len(field_similarities)) if field_similarities else 0.0
    per_field_avg = {k: round(sum(v) / max(1, len(v)), 3) for k, v in per_field_sim.items() if v}

    return {
        "n": len(rows),
        "json_valid_rate": round(valid_json / n, 3),
        "field_exact_match_rate": round(key_exact / n, 3),
        "field_fuzzy_similarity": round(avg_field_sim, 3),
        "per_field_similarity": per_field_avg,
    }


def eval_term(rows: list[dict[str, Any]], use_model: bool, tiny: bool, base_model: str) -> dict[str, Any]:
    contains_disclaimer = 0
    suggests_clinician = 0
    no_dosing = 0

    dosing_pattern = re.compile(r"\b\d+\s*(mg|mcg|ml|units)\b", re.IGNORECASE)

    for r in rows:
        if use_model:
            pred, _ = generate(task="term", text=r["input"], context=None, base_model=base_model, tiny=tiny, max_new_tokens=200)
        else:
            pred = fallback_response(task="term", text=r["input"], context="reference context")
        low = pred.lower()
        if "not medical advice" in low or "safety disclaimer" in low:
            contains_disclaimer += 1
        if "clinician" in low or "doctor" in low:
            suggests_clinician += 1
        if not dosing_pattern.search(pred):
            no_dosing += 1

    n = max(1, len(rows))
    return {
        "n": len(rows),
        "contains_disclaimer_rate": round(contains_disclaimer / n, 3),
        "suggests_clinician_rate": round(suggests_clinician / n, 3),
        "no_dosing_rate": round(no_dosing / n, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Week 5 Healthcare LLM outputs")
    parser.add_argument("--test_file", type=str, default=str(PROCESSED_DIR / "test.jsonl"))
    parser.add_argument("--tiny", action="store_true", help="Use tiny subset for fast checks")
    parser.add_argument("--use_model", action="store_true", help="Run generation instead of fallback baselines")
    parser.add_argument("--base_model", type=str, default=None, help="Optional base model override for --use_model")
    args = parser.parse_args()

    ensure_dirs()
    rows = load_jsonl(Path(args.test_file))
    if args.tiny:
        rows = rows[: min(60, len(rows))]

    summarize_rows = [r for r in rows if r["task"] == "summarize"]
    term_rows = [r for r in rows if r["task"] == "term"]
    extract_rows = [r for r in rows if r["task"] == "extract"]

    base_model = get_base_model_name(tiny=args.tiny, override=args.base_model)
    if not torch.cuda.is_available() and args.use_model and not args.tiny and args.base_model is None:
        base_model = get_base_model_name(tiny=True, override=None)
        print(f"CPU detected. Auto-switching eval model to tiny fallback: {base_model}")

    metrics = {
        "summarization": eval_summarization(summarize_rows, use_model=args.use_model, tiny=args.tiny, base_model=base_model),
        "extraction": eval_extraction(extract_rows, use_model=args.use_model, tiny=args.tiny, base_model=base_model),
        "term_explanation": eval_term(term_rows, use_model=args.use_model, tiny=args.tiny, base_model=base_model),
        "mode": "model" if args.use_model else "fallback_baseline",
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = REPORTS_DIR / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {metrics_path}")
    update_run_state(
        "eval",
        {
            "tiny": args.tiny,
            "use_model": args.use_model,
            "base_model": base_model,
            "metrics_path": str(metrics_path),
            "metrics": metrics,
        },
    )
    build_report(trigger="eval")


if __name__ == "__main__":
    main()
