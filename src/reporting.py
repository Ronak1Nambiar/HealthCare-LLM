from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import LORA_DIR, PROCESSED_DIR, REPORTS_DIR, ensure_dirs

RUN_STATE_PATH = REPORTS_DIR / "run_state.json"
QUESTION_LOG_PATH = REPORTS_DIR / "question_log.jsonl"
HISTORY_DIR = REPORTS_DIR / "history"
LATEST_REPORT_PATH = REPORTS_DIR / "latest_run_report.md"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def update_run_state(stage: str, payload: dict[str, Any]) -> None:
    ensure_dirs()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    state = load_json(RUN_STATE_PATH)
    state.setdefault("history", [])
    record = {"timestamp_utc": utc_now_iso(), "stage": stage, "payload": payload}
    state[stage] = record
    state["last_stage"] = stage
    state["history"].append(record)
    state["history"] = state["history"][-80:]
    save_json(RUN_STATE_PATH, state)


def log_question(task: str, prompt_text: str, safety_label: str, response_preview: str, used_model: bool) -> None:
    ensure_dirs()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    append_jsonl(
        QUESTION_LOG_PATH,
        {
            "timestamp_utc": utc_now_iso(),
            "task": task,
            "input_preview": prompt_text[:240],
            "safety_label": safety_label,
            "response_preview": response_preview[:320],
            "used_model": used_model,
        },
    )


def summarize_data() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "files_present": False,
        "train_rows": 0,
        "val_rows": 0,
        "test_rows": 0,
        "task_counts_all": {},
    }
    paths = {
        "train": PROCESSED_DIR / "train.jsonl",
        "val": PROCESSED_DIR / "val.jsonl",
        "test": PROCESSED_DIR / "test.jsonl",
        "all": PROCESSED_DIR / "all.jsonl",
    }
    if not all(p.exists() for p in paths.values()):
        return summary

    train_rows = load_jsonl(paths["train"])
    val_rows = load_jsonl(paths["val"])
    test_rows = load_jsonl(paths["test"])
    all_rows = load_jsonl(paths["all"])
    task_counts = Counter([r.get("task", "unknown") for r in all_rows])

    summary.update(
        {
            "files_present": True,
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "test_rows": len(test_rows),
            "task_counts_all": dict(task_counts),
        }
    )
    return summary


def summarize_training() -> dict[str, Any]:
    meta_path = LORA_DIR / "train_meta.json"
    if not meta_path.exists():
        return {"available": False}
    meta = load_json(meta_path)
    return {"available": True, **meta}


def summarize_eval() -> dict[str, Any]:
    metrics_path = REPORTS_DIR / "metrics.json"
    if not metrics_path.exists():
        return {"available": False}
    metrics = load_json(metrics_path)
    metrics["available"] = True
    return metrics


def summarize_questions(limit: int = 8) -> dict[str, Any]:
    rows = load_jsonl(QUESTION_LOG_PATH)
    if not rows:
        return {"available": False, "count": 0, "safety_counts": {}, "recent": []}
    recent = rows[-limit:]
    safety_counts = Counter([r.get("safety_label", "unknown") for r in rows])
    return {
        "available": True,
        "count": len(rows),
        "safety_counts": dict(safety_counts),
        "recent": recent,
    }


def accomplishment_lines(data: dict[str, Any], train: dict[str, Any], eval_summary: dict[str, Any], questions: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if data.get("files_present"):
        lines.append(
            f"Prepared instruction data with {data['train_rows']} train / {data['val_rows']} val / {data['test_rows']} test rows."
        )
    if train.get("available"):
        lines.append(
            "Completed LoRA training run "
            f"(base model: {train.get('base_model', 'n/a')}, max steps: {train.get('max_steps', 'n/a')})."
        )
    if eval_summary.get("available"):
        term = eval_summary.get("term_explanation", {})
        extract = eval_summary.get("extraction", {})
        lines.append(
            "Validated safety-oriented term behavior "
            f"(disclaimer rate: {term.get('contains_disclaimer_rate', 'n/a')}, no-dosing rate: {term.get('no_dosing_rate', 'n/a')})."
        )
        lines.append(
            "Validated extraction formatting "
            f"(JSON validity: {extract.get('json_valid_rate', 'n/a')})."
        )
    if questions.get("available"):
        lines.append(
            f"Processed {questions.get('count', 0)} user questions with safety routing counts: {questions.get('safety_counts', {})}."
        )
    if not lines:
        lines.append("Run has started, but no completed artifacts were found yet.")
    return lines


def build_report(trigger: str = "manual") -> Path:
    ensure_dirs()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    state = load_json(RUN_STATE_PATH)
    data = summarize_data()
    train = summarize_training()
    eval_summary = summarize_eval()
    questions = summarize_questions()
    now = utc_now_iso()

    accomplishments = accomplishment_lines(data, train, eval_summary, questions)

    lines: list[str] = []
    lines.append("# Automated Run Report")
    lines.append("")
    lines.append(f"- Generated at (UTC): {now}")
    lines.append(f"- Trigger: {trigger}")
    lines.append(f"- Last pipeline stage: {state.get('last_stage', 'unknown')}")
    lines.append("")
    lines.append("## What The LLM Has Accomplished")
    for item in accomplishments:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Training Data And Fine-tuning Snapshot")
    if data.get("files_present"):
        lines.append(
            f"- Split sizes: train={data['train_rows']}, val={data['val_rows']}, test={data['test_rows']}"
        )
        lines.append(f"- Task distribution: {data.get('task_counts_all', {})}")
    else:
        lines.append("- Processed data files not found yet.")

    if train.get("available"):
        lines.append(
            f"- Training config: base_model={train.get('base_model')}, "
            f"max_steps={train.get('max_steps')}, max_seq_len={train.get('max_seq_len')}, "
            f"lora_r={train.get('lora_r')}, lr={train.get('learning_rate')}"
        )
    else:
        lines.append("- Training metadata not found yet.")
    lines.append("")

    lines.append("## Evaluation Snapshot")
    if eval_summary.get("available"):
        sum_m = eval_summary.get("summarization", {})
        ext_m = eval_summary.get("extraction", {})
        term_m = eval_summary.get("term_explanation", {})
        lines.append(
            f"- Summarization: n={sum_m.get('n')}, rougeL_f1={sum_m.get('rougeL_f1')}, "
            f"avg_fk_grade={sum_m.get('avg_fk_grade')}"
        )
        lines.append(
            f"- Extraction: n={ext_m.get('n')}, json_valid_rate={ext_m.get('json_valid_rate')}, "
            f"field_exact_match_rate={ext_m.get('field_exact_match_rate')}"
        )
        lines.append(
            f"- Term explanation: n={term_m.get('n')}, contains_disclaimer_rate={term_m.get('contains_disclaimer_rate')}, "
            f"suggests_clinician_rate={term_m.get('suggests_clinician_rate')}, no_dosing_rate={term_m.get('no_dosing_rate')}"
        )
    else:
        lines.append("- Metrics not available yet.")
    lines.append("")

    lines.append("## Adaptive Question Handling")
    if questions.get("available"):
        lines.append(f"- Total logged questions: {questions.get('count', 0)}")
        lines.append(f"- Safety label counts: {questions.get('safety_counts', {})}")
        lines.append("- Recent question outcomes:")
        for q in questions.get("recent", []):
            lines.append(
                f"  - [{q.get('timestamp_utc')}] task={q.get('task')} safety={q.get('safety_label')} used_model={q.get('used_model')}"
            )
            lines.append(f"    input: {q.get('input_preview')}")
            lines.append(f"    output: {q.get('response_preview')}")
    else:
        lines.append("- No question logs yet. Run inference to populate adaptive behavior traces.")
    lines.append("")

    lines.append("## Safety Statement")
    lines.append(
        "- This system is education-only and is designed to refuse diagnosis, prescribing, dosing, and emergency triage decisions."
    )
    lines.append(
        "- For urgent symptoms, users are redirected to immediate in-person emergency care."
    )

    content = "\n".join(lines).strip() + "\n"
    LATEST_REPORT_PATH.write_text(content, encoding="utf-8")
    stamped = HISTORY_DIR / f"run_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.md"
    stamped.write_text(content, encoding="utf-8")
    return LATEST_REPORT_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate automated Week 5 run report")
    parser.add_argument("--trigger", type=str, default="manual")
    args = parser.parse_args()
    path = build_report(trigger=args.trigger)
    print(f"Wrote report: {path}")


if __name__ == "__main__":
    main()
