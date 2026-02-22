from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import LORA_DIR, PROMPTS_DIR, ensure_dirs, get_base_model_name
from .reporting import build_report, log_question, update_run_state
from .safety import append_disclaimer, classify_request, refusal_response, urgent_response


def read_text(input_text: str | None, input_file: str | None) -> str:
    if input_file:
        return Path(input_file).read_text(encoding="utf-8").strip()
    return (input_text or "").strip()


def load_prompt_file(path: Path, fallback: str) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return fallback


def format_user_prompt(task: str, text: str, context: str | None) -> str:
    ctx_block = f"\nContext:\n{context.strip()}\n" if context else ""
    if task == "summarize":
        return (
            "Task: Summarize the following medical education content for a patient in exactly 5 bullets. "
            "Use simple language and do not provide diagnosis or treatment instructions."
            f"{ctx_block}\nContent:\n{text}"
        )
    if task == "extract":
        return (
            "Task: Extract non-diagnostic structured fields into strict JSON with keys "
            "chief_complaint, symptoms, duration, vitals, meds, allergies, past_history, red_flags. "
            "No diagnosis field.\n"
            f"Clinical note:\n{text}"
        )
    return (
        "Task: Explain the term in plain language using the provided context. "
        "Include a short list of what to ask a clinician and cite context as [context-1].\n"
        f"Question/Term:\n{text}{ctx_block}"
    )


def load_model_and_tokenizer(base_model: str, tiny: bool = False):
    use_cuda = torch.cuda.is_available()
    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if use_cuda:
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        kwargs["device_map"] = "auto"

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)

    adapter_config = LORA_DIR / "adapter_config.json"
    if adapter_config.exists() and not tiny:
        model = PeftModel.from_pretrained(model, str(LORA_DIR))

    return model, tokenizer


def fallback_response(task: str, text: str, context: str | None) -> str:
    if task == "extract":
        skeleton = {
            "chief_complaint": "",
            "symptoms": [],
            "duration": "",
            "vitals": {"temp_f": None, "heart_rate": None, "bp": "", "spo2": None},
            "meds": [],
            "allergies": [],
            "past_history": [],
            "red_flags": [],
        }
        m = re.search(r"Chief complaint:\s*([^\.]+)", text, re.IGNORECASE)
        if m:
            skeleton["chief_complaint"] = m.group(1).strip()
        return append_disclaimer(json.dumps(skeleton, ensure_ascii=True))

    if task == "summarize":
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        bullets = sentences[:5] if sentences else ["This content provides general health education."]
        while len(bullets) < 5:
            bullets.append("Discuss personal questions with a licensed clinician.")
        return append_disclaimer("\n".join([f"- {b}" for b in bullets[:5]]))

    context_line = context.strip() if context else "No context provided."
    term = text.strip()
    m = re.search(r"Term:\s*(.+)", text, re.IGNORECASE)
    if m:
        term = m.group(1).splitlines()[0].strip()
    out = (
        f"Definition:\n{term} is a medical term. It is best understood in the context of your personal health history.\n\n"
        "What to ask your clinician:\n"
        "- How does this relate to my health history?\n"
        "- What warning signs should I watch for?\n"
        "- When should I follow up?\n\n"
        f"Citation:\n[context-1] \"{context_line[:240]}\""
    )
    return append_disclaimer(out)


def generate(
    task: str,
    text: str,
    context: str | None,
    base_model: str,
    tiny: bool,
    max_new_tokens: int,
    attempt_model: bool = True,
) -> str:
    system_prompt = load_prompt_file(
        PROMPTS_DIR / "system.txt",
        "You are an education-only healthcare assistant. No diagnosis, no treatment, no dosing.",
    )

    guard = classify_request(f"{task}\n{text}\n{context or ''}")
    if guard.label == "urgent":
        return urgent_response()
    if guard.label == "refuse":
        return refusal_response(guard.reason)
    if not attempt_model:
        return fallback_response(task=task, text=text, context=context)

    user_prompt = format_user_prompt(task=task, text=text, context=context)
    full_prompt = f"{system_prompt}\n\n{user_prompt}\n\nAnswer:"

    try:
        model, tokenizer = load_model_and_tokenizer(base_model=base_model, tiny=tiny)
        device = model.device
        inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = decoded.split("Answer:")[-1].strip() if "Answer:" in decoded else decoded.strip()
        final = append_disclaimer(answer)
        if task == "term":
            low = final.lower()
            if "what to ask your clinician" not in low or "citation" not in low:
                return fallback_response(task=task, text=text, context=context)
        if task == "extract" and "{" not in final:
            return fallback_response(task=task, text=text, context=context)
        if task == "summarize" and final.count("- ") < 3:
            return fallback_response(task=task, text=text, context=context)
        return final
    except Exception as exc:
        print(f"Warning: generation fallback activated ({exc})")
        return fallback_response(task=task, text=text, context=context)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference for Healthcare LLM Week 5")
    parser.add_argument("--task", choices=["term", "summarize", "extract"], required=True)
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--context", type=str, default=None)
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--force_model", action="store_true", help="Attempt model generation on CPU")
    parser.add_argument("--max_new_tokens", type=int, default=220)
    args = parser.parse_args()

    ensure_dirs()
    text = read_text(args.input, args.input_file)
    if not text:
        raise ValueError("Provide --input or --input_file with non-empty content")

    has_cuda = torch.cuda.is_available()
    model_name = get_base_model_name(tiny=args.tiny, override=args.base_model)
    if not has_cuda and not args.tiny and args.base_model is None:
        model_name = get_base_model_name(tiny=True, override=None)
        print(f"CPU detected. Auto-switching model to tiny fallback: {model_name}")
    attempt_model = has_cuda or args.force_model
    output = generate(
        task=args.task,
        text=text,
        context=args.context,
        base_model=model_name,
        tiny=args.tiny,
        max_new_tokens=args.max_new_tokens,
        attempt_model=attempt_model,
    )
    guard = classify_request(f"{args.task}\n{text}\n{args.context or ''}")
    log_question(
        task=args.task,
        prompt_text=text,
        safety_label=guard.label,
        response_preview=output,
        used_model=(attempt_model and guard.label == "allowed"),
    )
    update_run_state(
        "inference",
        {
            "task": args.task,
            "tiny": args.tiny,
            "base_model": model_name,
            "safety_label": guard.label,
            "attempt_model": attempt_model,
        },
    )
    build_report(trigger="inference")
    print(output)


if __name__ == "__main__":
    main()
