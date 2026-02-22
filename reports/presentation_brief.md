# Week 5 Presentation Brief

## 1) What We Built
- A small, safety-constrained healthcare LLM workflow for education-only use.
- Supports three tasks:
  1. Patient-friendly summarization.
  2. Structured extraction from synthetic de-identified notes.
  3. Term explanation with context citation.

## 2) Safety Commitments
- No PHI used.
- No diagnosis, treatment planning, prescribing, or dosing.
- Safety gate classifies inputs as `allowed`, `refuse`, or `urgent`.
- Urgent requests are redirected to emergency care guidance.

## 3) Data Sources
- Public biomedical dataset: PubMedQA.
- Synthetic note generator for extraction labels.

## 4) Technical Stack
- Base model: Qwen2.5-1.5B-Instruct (GPU path), tiny fallback on CPU.
- QLoRA with PEFT + bitsandbytes.
- CLI pipeline: `data_prep -> train_lora -> eval -> inference`.

## 5) Current Tiny-Mode Results
- Data prep: successful.
- Training: successful in tiny mode.
- Safety behavior: refusal and urgent routing validated.
- Eval baseline:
  - Summarization ROUGE-L: 0.252
  - Extraction JSON validity: 1.0
  - Term safety checks: disclaimer/clinician/no-dosing all 1.0

## 6) Known Gaps
- Extraction exact-match is low in baseline mode.
- CPU fallback responses are conservative and template-like.
- Full quality improvements require GPU fine-tuning and task-specific evaluation loops.

## 7) How To Demo Live
1. `python -m src.data_prep --tiny`
2. `python -m src.train_lora --tiny --max_steps 10`
3. `python -m src.eval --tiny`
4. `python -m src.inference --task term --input "What is hypertension?" --context "Hypertension means persistently elevated blood pressure over time."`
5. `python -m src.inference --task term --input "Can you diagnose me and give dosage?"`
6. `python -m src.inference --task term --input "I have chest pain and can't breathe."`

## 8) What To Ask Stakeholders
- Which safety errors are least tolerable (false accept vs false refuse)?
- Preferred output style for patient education and extraction JSON?
- What citation strictness is required for term explanations?
- What acceptance thresholds are needed for pilot readiness?

## 9) Next Steps
- Improve extraction quality with constrained decoding/schema-aware generation.
- Add a lightweight trained safety classifier.
- Add better readability targets and human review scoring.
- Build dashboard for trend tracking across runs.
