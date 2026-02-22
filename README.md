# Healthcare based LLM

Week 5 experiment for a **small, safety-constrained healthcare domain adaptation** using QLoRA.

This repo builds a model workflow that can:
1. Summarize patient-facing medical education text.
2. Extract structured, non-diagnostic fields from synthetic de-identified clinical notes.
3. Explain medical terms using provided context with citations to that context.

All outputs include a safety disclaimer and the system refuses diagnosis, treatment planning, prescribing, and dosing requests.

## Safety Scope

- No real PHI is used.
- Primary dataset is public (`PubMedQA`).
- Clinical note extraction data is synthetic and template-generated.
- Safety layer blocks/redirects diagnosis, prescribing, dosage, and emergency requests.

## Model Choice

Default base model: `Qwen/Qwen2.5-1.5B-Instruct`

Why:
- Small enough for consumer hardware with 4-bit quantization.
- Good instruction-following behavior for constrained outputs.

Supported alternatives via CLI/config:
- `google/gemma-2-2b-it`
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

## Repo Structure

```text
healthcare-llm-week5/
  README.md
  requirements.txt
  data/
    raw/
    processed/
  models/
    lora/
  src/
    __init__.py
    config.py
    data_prep.py
    train_lora.py
    eval.py
    inference.py
    safety.py
    reporting.py
  prompts/
    system.txt
    refusal_templates.txt
  reports/
    week5_report.md
    metrics.json
  notebooks/
```

## Install

```bash
pip install -r requirements.txt
```

## Quick Start (Tiny Mode)

Tiny mode is designed to run end-to-end quickly on CPU.

```bash
python -m src.data_prep --tiny
python -m src.train_lora --tiny --max_steps 10
python -m src.eval --tiny
python -m src.inference --task term --input "What is hypertension?" --context "Hypertension means persistently elevated blood pressure."
```

Notes:
- On CPU, inference defaults to deterministic safety-preserving fallback outputs for speed.
- To force actual model generation on CPU, add `--force_model`.
- On CPU, `train_lora.py` auto-switches to the tiny base model unless `--base_model` is explicitly set.

## Full Run (GPU Recommended)

```bash
python -m src.data_prep
python -m src.train_lora
python -m src.eval
python -m src.inference --task summarize --input_file sample.txt
python -m src.inference --task extract --input_file note.txt
```

## CLI Commands

1. Data prep
```bash
python -m src.data_prep --tiny
```

2. Train LoRA adapters
```bash
python -m src.train_lora
```

3. Evaluate
```bash
python -m src.eval
```

4. Inference
```bash
python -m src.inference --task term --input "What is anemia?"
python -m src.inference --task summarize --input_file sample.txt
python -m src.inference --task extract --input_file note.txt
```

5. Manual report generation (optional)
```bash
python -m src.reporting --trigger manual
```

## How To Interact

1. Ask a normal term question (allowed)
```bash
python -m src.inference --task term --input "What is hypertension?" --context "Hypertension means persistently elevated blood pressure over time."
```

2. Summarize patient education text from a file
```bash
python -m src.inference --task summarize --input_file sample.txt
```

3. Extract structured fields from a de-identified note
```bash
python -m src.inference --task extract --input_file note.txt
```

4. Test refusal behavior (diagnosis/dosing request)
```bash
python -m src.inference --task term --input "Can you diagnose me and give me a dose in mg?"
```

5. Test urgent routing behavior
```bash
python -m src.inference --task term --input "I have chest pain and can't breathe, what should I do?"
```

Expected behavior:
- Allowed prompts: educational response + disclaimer.
- Refusal prompts: no diagnosis/prescribing/dosing, recommends clinician.
- Urgent prompts: immediate emergency-care guidance message.

## Output Format Expectations

- `summarize`: patient-friendly bullet summary + safety disclaimer.
- `term`: simple explanation + "what to ask your clinician" + context citation.
- `extract`: strict JSON keys only:
  - `chief_complaint`
  - `symptoms[]`
  - `duration`
  - `vitals{...}`
  - `meds[]`
  - `allergies[]`
  - `past_history[]`
  - `red_flags[]`

No diagnosis field is allowed.

## Sample Inference Output

Command:

```bash
python -m src.inference --task term --input "What is hypertension?" --context "Hypertension means persistently elevated blood pressure over time."
```

Example output:

```text
Definition:
Hypertension means blood pressure that stays higher than normal over time.

What to ask your clinician:
- What blood pressure target is appropriate for me?
- How often should I monitor blood pressure at home?
- What lifestyle changes are most helpful in my case?

Citation:
[context-1] "Hypertension means persistently elevated blood pressure over time."

Safety disclaimer: This is general educational information, not medical advice. I cannot diagnose conditions, prescribe treatment, or provide medication dosing. Please talk to a licensed clinician.
```

## Notes

- In CPU-only environments, training falls back to lightweight settings.
- If LoRA adapters are unavailable, inference can still run with base model + safety layer.
- `reports/week5_report.md` includes hypothesis, config, results, failure analysis, safety outcomes, and next steps.

## Presentation + Notes Pack

Use these files to present and track incoming results:
- `reports/presentation_brief.md`: ready-to-read presentation narrative.
- `reports/meeting_notes_template.md`: structured meeting notes template.
- `reports/data_intake_log.csv`: starter log for metrics/safety outcomes over time.
- `reports/latest_run_report.md`: auto-generated adaptive report after each `data_prep`, `train_lora`, `eval`, or `inference` run.
- `reports/question_log.jsonl`: logged user questions with safety labels and response previews.
