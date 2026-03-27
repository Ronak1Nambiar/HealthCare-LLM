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
    # Expanded pools based on common clinical encounter categories
    complaints = [
        # General / constitutional
        "headache", "fatigue", "dizziness", "fever", "weight loss",
        # Respiratory
        "cough", "sore throat", "nasal congestion", "wheezing", "shortness of breath on exertion",
        # GI
        "nausea", "abdominal pain", "diarrhea", "constipation", "heartburn",
        # MSK
        "back pain", "knee pain", "joint stiffness", "shoulder pain", "neck pain",
        # Neuro
        "numbness in hands", "migraine", "blurred vision", "tingling in feet",
        # Cardiovascular
        "palpitations", "swelling in legs", "lightheadedness on standing",
        # Dermatological
        "skin rash", "itching", "wound that won't heal",
        # Mental health (non-urgent)
        "insomnia", "anxiety", "low mood", "difficulty concentrating",
        # Urinary
        "frequent urination", "burning with urination", "blood in urine",
        # ENT
        "ear pain", "hearing loss", "sinus pressure",
    ]

    symptoms_pool = [
        "dizziness", "fever", "chills", "body aches", "runny nose", "poor sleep",
        "vomiting", "sweating", "loss of appetite", "dry mouth", "muscle cramps",
        "joint swelling", "bruising easily", "night sweats", "unintentional weight loss",
        "difficulty swallowing", "hoarse voice", "chronic cough", "blood in stool",
        "excessive thirst", "frequent headaches", "cold intolerance", "heat intolerance",
        "hair loss", "brittle nails", "puffy face", "swollen lymph nodes",
        "trouble sleeping", "racing heart", "shortness of breath with activity",
        "leg cramps at night", "urinary urgency", "flank pain",
    ]

    durations = [
        "2 days", "3 days", "5 days", "1 week", "10 days", "2 weeks",
        "3 weeks", "1 month", "6 weeks", "2 months", "3 months",
        "since yesterday", "for 4 days", "for about a week",
        "on and off for 2 months", "worsening over 3 weeks",
    ]

    meds_pool = [
        "acetaminophen", "ibuprofen", "lisinopril", "metformin", "atorvastatin",
        "amlodipine", "omeprazole", "metoprolol", "levothyroxine", "albuterol inhaler",
        "sertraline", "gabapentin", "prednisone", "aspirin 81mg", "hydrochlorothiazide",
        "losartan", "fluticasone nasal spray", "cetirizine", "montelukast",
        "insulin glargine", "sitagliptin", "empagliflozin", "warfarin",
        "none",
    ]

    allergy_pool = [
        "penicillin", "none known", "peanuts", "sulfa drugs", "latex",
        "iodine contrast", "shellfish", "amoxicillin", "aspirin",
        "codeine", "NSAIDs", "bee stings", "eggs",
        "no known drug allergies (NKDA)",
    ]

    history_pool = [
        "asthma", "hypertension", "type 2 diabetes", "seasonal allergies",
        "GERD", "hypothyroidism", "hyperlipidemia", "obesity (BMI > 30)",
        "coronary artery disease", "atrial fibrillation", "COPD",
        "chronic kidney disease stage 3", "osteoarthritis", "osteoporosis",
        "migraine disorder", "anxiety disorder", "major depressive disorder",
        "sleep apnea (on CPAP)", "gout", "benign prostatic hyperplasia",
        "iron-deficiency anemia", "vitamin D deficiency",
        "none",
    ]

    red_flags_pool = [
        "chest pain", "fainting", "shortness of breath at rest",
        "sudden severe headache", "unexplained weight loss > 10 lbs",
        "blood in stool or vomit", "new onset confusion",
        "unilateral leg swelling", "fever > 103 F not improving",
        "none",
    ]

    # Vary note structure/format for diversity
    note_templates = [
        # Template 1: standard clinical note
        (
            "Chief complaint: {chief}. Symptoms include {symptoms} for {duration}. "
            "Vitals today: temp {temp} F, HR {hr}, BP {bp}, SpO2 {spo2}%. "
            "Current meds: {meds}. "
            "Allergies: {allergies}. Past history: {history}. "
            "Red flags discussed: {red_flags}."
        ),
        # Template 2: SOAP-style
        (
            "S: Patient presents with {chief} x {duration}. Associated symptoms: {symptoms}. "
            "O: Vitals — T {temp} F, HR {hr}, BP {bp}, SpO2 {spo2}%. "
            "Medications: {meds}. Allergies: {allergies}. "
            "PMH: {history}. "
            "Red flags screened: {red_flags}."
        ),
        # Template 3: narrative
        (
            "A patient presents today reporting {chief} for {duration}. "
            "They also describe {symptoms}. "
            "Vital signs are as follows: temperature {temp} F, heart rate {hr} bpm, "
            "blood pressure {bp} mmHg, oxygen saturation {spo2}%. "
            "The patient currently takes {meds} and reports allergies to {allergies}. "
            "Relevant past medical history includes {history}. "
            "Red flags reviewed: {red_flags}."
        ),
        # Template 4: triage note
        (
            "Triage note — CC: {chief} ({duration}). "
            "Additional: {symptoms}. "
            "VS: T={temp}, HR={hr}, BP={bp}, O2={spo2}%. "
            "Meds: {meds}. Allergy: {allergies}. Hx: {history}. "
            "Red flags: {red_flags}."
        ),
    ]

    chief = RNG.choice(complaints)
    symptom_count = RNG.randint(1, 4)
    symptoms = RNG.sample(symptoms_pool, min(symptom_count, len(symptoms_pool)))
    duration = RNG.choice(durations)

    vitals = {
        "temp_f": round(RNG.uniform(97.0, 103.0), 1),
        "heart_rate": RNG.randint(55, 125),
        "bp": f"{RNG.randint(95, 170)}/{RNG.randint(55, 105)}",
        "spo2": RNG.randint(90, 100),
    }

    med_count = RNG.randint(0, 4)
    meds = [m for m in RNG.sample(meds_pool, min(med_count + 1, len(meds_pool))) if m != "none"]
    allergy_count = RNG.randint(1, 2)
    allergies = RNG.sample(allergy_pool, min(allergy_count, len(allergy_pool)))
    hist_count = RNG.randint(0, 3)
    past_history = [h for h in RNG.sample(history_pool, min(hist_count + 1, len(history_pool))) if h != "none"]
    red_flag_count = RNG.randint(1, 2)
    red_flags = RNG.sample(red_flags_pool, min(red_flag_count, len(red_flags_pool)))

    template = RNG.choice(note_templates)
    note = template.format(
        chief=chief,
        symptoms=", ".join(symptoms),
        duration=duration,
        temp=vitals["temp_f"],
        hr=vitals["heart_rate"],
        bp=vitals["bp"],
        spo2=vitals["spo2"],
        meds=", ".join(meds) if meds else "none",
        allergies=", ".join(allergies),
        history=", ".join(past_history) if past_history else "none",
        red_flags=", ".join(red_flags),
    )

    target = {
        "chief_complaint": chief,
        "symptoms": symptoms,
        "duration": duration,
        "vitals": vitals,
        "meds": meds,
        "allergies": allergies,
        "past_history": past_history,
        "red_flags": [rf for rf in red_flags if rf != "none"],
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
