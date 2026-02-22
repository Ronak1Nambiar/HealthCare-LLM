from __future__ import annotations

import argparse
import json
import re
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


def eval_summarization(rows: list[dict[str, Any]], use_model: bool, tiny: bool, base_model: str) -> dict[str, Any]:
    preds = []
    refs = []

    for r in rows:
        if use_model:
            pred = generate(task="summarize", text=r["input"], context=None, base_model=base_model, tiny=tiny, max_new_tokens=180)
        else:
            pred = fallback_response(task="summarize", text=r["input"], context=None)
        preds.append(pred)
        refs.append(r["output"])

    rouge_l = None
    if rouge_scorer is not None:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [scorer.score(ref, pred)["rougeL"].fmeasure for ref, pred in zip(refs, preds)]
        rouge_l = sum(scores) / max(1, len(scores))

    avg_len = sum(len(p.split()) for p in preds) / max(1, len(preds))
    avg_fk = sum(flesch_kincaid_grade(p) for p in preds) / max(1, len(preds))

    return {
        "n": len(rows),
        "rougeL_f1": rouge_l,
        "avg_output_words": avg_len,
        "avg_fk_grade": avg_fk,
    }


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

    for r in rows:
        pred_text = (
            generate(task="extract", text=r["input"], context=None, base_model=base_model, tiny=tiny, max_new_tokens=220)
            if use_model
            else fallback_response(task="extract", text=r["input"], context=None)
        )
        pred_obj = safe_json_loads(pred_text)
        gold_obj = json.loads(r["output"])

        if pred_obj is not None:
            valid_json += 1
            if all(pred_obj.get(k) == gold_obj.get(k) for k in required_keys):
                key_exact += 1

    n = max(1, len(rows))
    return {
        "n": len(rows),
        "json_valid_rate": valid_json / n,
        "field_exact_match_rate": key_exact / n,
    }


def eval_term(rows: list[dict[str, Any]], use_model: bool, tiny: bool, base_model: str) -> dict[str, Any]:
    contains_disclaimer = 0
    suggests_clinician = 0
    no_dosing = 0

    dosing_pattern = re.compile(r"\b\d+\s*(mg|mcg|ml|units)\b", re.IGNORECASE)

    for r in rows:
        pred = (
            generate(task="term", text=r["input"], context=None, base_model=base_model, tiny=tiny, max_new_tokens=200)
            if use_model
            else fallback_response(task="term", text=r["input"], context="reference context")
        )
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
        "contains_disclaimer_rate": contains_disclaimer / n,
        "suggests_clinician_rate": suggests_clinician / n,
        "no_dosing_rate": no_dosing / n,
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
