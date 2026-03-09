from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset

from .config import PROCESSED_DIR, RAW_DIR, ensure_dirs
from .reporting import build_report, update_run_state

RNG = random.Random(42)  # default seed; overridden in main() via --seed


@dataclass
class Record:
    id: str
    task: str
    instruction: str
    input: str
    output: str


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def bulletize(text: str, n: int = 5) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", clean_text(text))
    kept = [s for s in sentences if len(s.split()) > 5][:n]
    if not kept:
        kept = ["This text discusses general health information."]
    while len(kept) < n:
        kept.append("Ask your clinician how this applies to your personal history.")
    return kept[:n]


def parse_pubmedqa_record(ex: dict[str, Any], idx: int) -> tuple[Record, Record] | None:
    question = clean_text(ex.get("question", ""))

    context_candidates = []
    context_obj = ex.get("context")
    if isinstance(context_obj, dict):
        context_candidates.extend(context_obj.get("contexts", []) or [])
        context_candidates.extend(context_obj.get("labels", []) or [])
        context_candidates.extend(context_obj.get("meshes", []) or [])
    elif isinstance(context_obj, list):
        context_candidates.extend(context_obj)

    long_answer = clean_text(ex.get("long_answer", ""))
    answer = clean_text(ex.get("final_decision", ""))

    joined_context = clean_text(" ".join([str(x) for x in context_candidates[:3] if x]))
    if not joined_context and not long_answer:
        return None

    summary_source = long_answer if long_answer else joined_context
    bullets = bulletize(summary_source, n=5)
    summary_output = "\n".join([f"- {b}" for b in bullets])
    summary_output += (
        "\n\nSafety disclaimer: This is educational content only, not medical advice. "
        "I cannot diagnose or prescribe. Please consult a clinician."
    )

    summarize = Record(
        id=f"sum_{idx}",
        task="summarize",
        instruction=(
            "Summarize the provided medical education text for a patient in exactly 5 simple bullet points. "
            "Do not diagnose. End with a safety disclaimer."
        ),
        input=joined_context if joined_context else summary_source,
        output=summary_output,
    )

    term = question
    if question.lower().startswith("what is "):
        term = question[8:].rstrip(" ?.")

    term_context = joined_context if joined_context else summary_source
    term_output = (
        f"Definition: {clean_text(summary_source)[:280]}\n\n"
        "What to ask your clinician:\n"
        "- How does this apply to my health history?\n"
        "- What warning signs should I watch for?\n"
        "- When should I follow up?\n\n"
        f"Citation:\n[context-1] \"{clean_text(term_context)[:220]}\"\n\n"
        "Safety disclaimer: This is educational information, not medical advice. "
        "I cannot diagnose, prescribe, or provide dosing."
    )

    explain = Record(
        id=f"term_{idx}",
        task="term",
        instruction=(
            "Explain the medical term in simple language using provided context, cite context as [context-1], "
            "include what to ask a clinician, and include a safety disclaimer."
        ),
        input=f"Term: {term}\nContext: {term_context}\nAnswer label: {answer}",
        output=term_output,
    )

    return summarize, explain


def make_synthetic_note() -> tuple[str, dict[str, Any]]:
    complaints = ["headache", "cough", "fatigue", "sore throat", "nausea", "back pain"]
    symptoms_pool = ["dizziness", "fever", "chills", "body aches", "runny nose", "poor sleep", "vomiting"]
    durations = ["2 days", "1 week", "3 weeks", "since yesterday", "for 4 days"]
    meds_pool = ["acetaminophen", "ibuprofen", "lisinopril", "metformin", "none"]
    allergy_pool = ["penicillin", "none known", "peanuts", "sulfa"]
    history_pool = ["asthma", "hypertension", "type 2 diabetes", "seasonal allergies", "none"]
    red_flags_pool = ["chest pain", "fainting", "shortness of breath", "none"]

    chief = RNG.choice(complaints)
    symptom_count = RNG.randint(1, 3)
    symptoms = RNG.sample(symptoms_pool, symptom_count)
    duration = RNG.choice(durations)

    vitals = {
        "temp_f": round(RNG.uniform(97.8, 102.2), 1),
        "heart_rate": RNG.randint(62, 118),
        "bp": f"{RNG.randint(102, 156)}/{RNG.randint(62, 98)}",
        "spo2": RNG.randint(93, 100),
    }

    meds = [m for m in RNG.sample(meds_pool, RNG.randint(1, 2)) if m != "none"]
    allergies = [RNG.choice(allergy_pool)]
    past_history = [h for h in RNG.sample(history_pool, RNG.randint(1, 2)) if h != "none"]
    red_flags = [RNG.choice(red_flags_pool)]

    note = (
        f"Chief complaint: {chief}. Symptoms include {', '.join(symptoms)} for {duration}. "
        f"Vitals today: temp {vitals['temp_f']} F, HR {vitals['heart_rate']}, BP {vitals['bp']}, SpO2 {vitals['spo2']}%. "
        f"Current meds: {', '.join(meds) if meds else 'none'}. "
        f"Allergies: {', '.join(allergies)}. Past history: {', '.join(past_history) if past_history else 'none'}. "
        f"Red flags discussed: {', '.join(red_flags)}."
    )

    target = {
        "chief_complaint": chief,
        "symptoms": symptoms,
        "duration": duration,
        "vitals": vitals,
        "meds": meds,
        "allergies": allergies,
        "past_history": past_history,
        "red_flags": [] if red_flags == ["none"] else red_flags,
    }
    return note, target


def build_extraction_records(n: int) -> list[Record]:
    records: list[Record] = []
    for i in range(n):
        note, target = make_synthetic_note()
        output = json.dumps(target, ensure_ascii=True)
        records.append(
            Record(
                id=f"extract_{i}",
                task="extract",
                instruction=(
                    "Extract structured fields from this de-identified note as strict JSON with keys: "
                    "chief_complaint, symptoms, duration, vitals, meds, allergies, past_history, red_flags. "
                    "Do not infer diagnosis."
                ),
                input=note,
                output=output,
            )
        )
    return records


def split_records(records: list[Record]) -> dict[str, list[Record]]:
    RNG.shuffle(records)
    n = len(records)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    return {
        "train": records[:train_end],
        "val": records[train_end:val_end],
        "test": records[val_end:],
    }


def write_jsonl(path: Path, rows: list[Record]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=True) + "\n")


def load_pubmedqa(max_items: int) -> list[Record]:
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    out: list[Record] = []
    for i, ex in enumerate(dataset):
        if len(out) >= max_items * 2:
            break
        pair = parse_pubmedqa_record(ex, i)
        if not pair:
            continue
        out.extend(pair)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Week 5 healthcare instruction dataset")
    parser.add_argument("--tiny", action="store_true", help="Build a tiny sample dataset for quick tests")
    parser.add_argument("--max_pubmed", type=int, default=800, help="Max PubMedQA source rows to read")
    parser.add_argument("--synthetic_notes", type=int, default=600, help="Number of synthetic extraction samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Apply seed globally so synthetic generation and splits are reproducible
    global RNG
    RNG = random.Random(args.seed)

    ensure_dirs()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    max_pubmed = 80 if args.tiny else args.max_pubmed
    synthetic_n = 80 if args.tiny else args.synthetic_notes

    records: list[Record] = []
    try:
        pubmed_records = load_pubmedqa(max_pubmed)
        records.extend(pubmed_records)
        print(f"Loaded PubMedQA-derived records: {len(pubmed_records)}")
    except Exception as exc:
        print(f"Warning: failed to load PubMedQA ({exc}). Falling back to synthetic-only text tasks.")
        fallback = [
            Record(
                id="sum_fallback_1",
                task="summarize",
                instruction="Summarize this health education text in 5 bullets and include disclaimer.",
                input="High blood pressure can raise the risk of stroke and heart disease over time. Lifestyle changes may help.",
                output=(
                    "- High blood pressure can increase long-term health risks.\n"
                    "- Risk can rise for heart and brain conditions.\n"
                    "- Healthy eating can support blood pressure control.\n"
                    "- Physical activity can also help.\n"
                    "- Follow-up visits help track progress.\n\n"
                    "Safety disclaimer: This is educational content only, not medical advice. I cannot diagnose or prescribe."
                ),
            ),
            Record(
                id="term_fallback_1",
                task="term",
                instruction="Explain the term simply with citation and disclaimer.",
                input="Term: Hypertension\nContext: Hypertension means persistently elevated blood pressure.",
                output=(
                    "Definition: Hypertension means blood pressure stays high over time.\n\n"
                    "What to ask your clinician:\n- What is my target blood pressure?\n"
                    "- How often should I check it?\n- What lifestyle changes help me?\n\n"
                    "Citation:\n[context-1] \"Hypertension means persistently elevated blood pressure.\"\n\n"
                    "Safety disclaimer: This is educational information, not medical advice. I cannot diagnose, prescribe, or provide dosing."
                ),
            ),
        ]
        records.extend(fallback)

    records.extend(build_extraction_records(synthetic_n))

    splits = split_records(records)
    for split_name, split_rows in splits.items():
        out_path = PROCESSED_DIR / f"{split_name}.jsonl"
        write_jsonl(out_path, split_rows)
        print(f"Wrote {split_name}: {len(split_rows)} -> {out_path}")

    all_path = PROCESSED_DIR / "all.jsonl"
    write_jsonl(all_path, records)
    print(f"Wrote all records: {len(records)} -> {all_path}")

    split_sizes = {name: len(rows) for name, rows in splits.items()}
    task_counts: dict[str, int] = {}
    for row in records:
        task = row.task
        task_counts[task] = task_counts.get(task, 0) + 1
    update_run_state(
        "data_prep",
        {
            "tiny": args.tiny,
            "total_records": len(records),
            "split_sizes": split_sizes,
            "task_counts": task_counts,
        },
    )
    build_report(trigger="data_prep")


if __name__ == "__main__":
    main()
