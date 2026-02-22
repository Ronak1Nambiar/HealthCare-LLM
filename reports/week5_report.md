# Week 5 Report - Healthcare LLM Fine-tune

## Hypothesis
A small instruction model with QLoRA domain adaptation on public biomedical text plus synthetic de-identified note extraction data can improve healthcare education utility while maintaining strict non-diagnostic safety behavior.

## Dataset(s) + Why Chosen
- Primary: PubMedQA (`qiaojin/PubMedQA`, labeled subset) for public biomedical QA/context.
  - Rationale: open, biomedical domain language, no PHI.
- Optional synthetic: template-based de-identified note generator for structured extraction task.
  - Rationale: avoids restricted data and controls output schema.

## Training Config
- Base model: `Qwen/Qwen2.5-1.5B-Instruct` (tiny mode fallback: `sshleifer/tiny-gpt2`)
- Quantization: 4-bit NF4 (`bitsandbytes`)
- Adapter: LoRA (r=16, alpha=32, dropout=0.05)
- Typical LR: 2e-4
- Sequence length: 768 (tiny: 384)
- Batching: per-device batch 1, grad accumulation 8
- Steps: tiny 10-20, full run user-configurable

## Results Table
| Task | Metric | Value |
|---|---|---|
| Summarization | ROUGE-L F1 | see `reports/metrics.json` |
| Summarization | Avg FK grade | see `reports/metrics.json` |
| Extraction | JSON validity | see `reports/metrics.json` |
| Extraction | Field exact match | see `reports/metrics.json` |
| Term explanation | Disclaimer present | see `reports/metrics.json` |
| Term explanation | Clinician suggestion | see `reports/metrics.json` |

## Failure Analysis (3 Examples)
1. Summarization occasionally repeats sentence fragments when source context is fragmented.
2. Term explanation can over-quote context and produce shallow explanations for ambiguous terms.
3. Extraction may miss multi-token medication names with punctuation variants.

## Safety Outcomes
- Safety layer classes: `allowed`, `refuse`, `urgent`.
- Refusal trigger categories: diagnose/prescribe/dosage/treatment-plan requests.
- Urgent trigger categories: chest pain, severe breathing problems, stroke/self-harm markers.
- Final counts and false positives should be measured by running `python -m src.eval --use_model` with adversarial prompts.

## Next Steps (Week 6 Ideas)
1. Add a compact safety classifier trained on synthetic refusal/allowed examples.
2. Add citation span checking to ensure term answers quote context minimally and accurately.
3. Improve extraction robustness with constrained decoding or JSON schema decoding.
4. Add calibration set for refusal false-positive/false-negative tuning.
