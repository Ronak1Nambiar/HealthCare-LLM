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

# ── Constants ──────────────────────────────────────────────────────────────

# LLM mode identifiers (used throughout inference + GUI)
MODE_BASE = "base"
MODE_FINETUNED = "finetuned"
MODE_RAG = "rag"
ALL_MODES = (MODE_BASE, MODE_FINETUNED, MODE_RAG)

# Prompt-injection tokens to strip from user input
_INJECTION_TOKENS = re.compile(
    r"(###\s*(Instruction|Input|Response|System)|"
    r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>|"
    r"\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>|"
    r"<s>|</s>)",
    re.IGNORECASE,
)


# ── Input helpers ──────────────────────────────────────────────────────────

def sanitize_input(text: str) -> str:
    """Strip prompt-injection tokens from user-supplied text."""
    return _INJECTION_TOKENS.sub("", text).strip()


def read_text(input_text: str | None, input_file: str | None) -> str:
    if input_file:
        raw = Path(input_file).read_text(encoding="utf-8").strip()
    else:
        raw = (input_text or "").strip()
    return sanitize_input(raw)


def load_prompt_file(path: Path, fallback: str) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return fallback


def format_user_prompt(task: str, text: str, context: str | None) -> str:
    safe_context = sanitize_input(context) if context else None
    ctx_block = f"\nContext:\n{safe_context.strip()}\n" if safe_context else ""
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


# ── Model loading ──────────────────────────────────────────────────────────

def load_model_and_tokenizer(
    base_model: str,
    tiny: bool = False,
    use_lora: bool = True,
):
    """
    Load tokenizer + model.
    use_lora=False → pure base model, no adapter (Base Model mode).
    use_lora=True  → loads LoRA adapter if models/lora/adapter_config.json exists.
    """
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
    if use_lora and adapter_config.exists() and not tiny:
        model = PeftModel.from_pretrained(model, str(LORA_DIR))

    return model, tokenizer


# ── Fallback (no-model) response ───────────────────────────────────────────

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


# ── Core generation ────────────────────────────────────────────────────────

def generate(
    task: str,
    text: str,
    context: str | None,
    base_model: str,
    tiny: bool,
    max_new_tokens: int,
    attempt_model: bool = True,
    mode: str = MODE_FINETUNED,
) -> tuple[str, bool]:
    """
    Generate a response.

    mode:
        "base"      — raw base model, no LoRA adapter, no retrieval
        "finetuned" — base model + LoRA adapter (if adapter exists)
        "rag"       — base model + LoRA adapter + TF-IDF retrieved context

    Returns (response_text, used_fallback).
    used_fallback=True means the response came from the deterministic fallback.
    """
    system_prompt = load_prompt_file(
        PROMPTS_DIR / "system.txt",
        "You are an education-only healthcare assistant. No diagnosis, no treatment, no dosing.",
    )

    guard = classify_request(f"{task}\n{text}\n{context or ''}")
    if guard.label == "urgent":
        return urgent_response(), True
    if guard.label == "refuse":
        return refusal_response(guard.reason), True
    if not attempt_model:
        return fallback_response(task=task, text=text, context=context), True

    # ── RAG: inject retrieved context ─────────────────────────────────────
    effective_context = context
    if mode == MODE_RAG:
        try:
            from .rag import get_retriever
            retriever = get_retriever()
            if retriever.is_built:
                rag_ctx = retriever.format_context(f"{task} {text}", top_k=3)
                if rag_ctx:
                    effective_context = (
                        f"{rag_ctx}\n\nUser-provided context:\n{context}"
                        if context else rag_ctx
                    )
            else:
                print("[RAG] Index not built — falling back to no-retrieval mode.")
        except Exception as exc:
            print(f"[RAG] Retrieval error: {exc}")

    # ── Load model ────────────────────────────────────────────────────────
    use_lora = mode in (MODE_FINETUNED, MODE_RAG)
    user_prompt = format_user_prompt(task=task, text=text, context=effective_context)
    full_prompt = f"{system_prompt}\n\n{user_prompt}\n\nAnswer:"

    try:
        model, tokenizer = load_model_and_tokenizer(
            base_model=base_model, tiny=tiny, use_lora=use_lora
        )
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

        # Validate output quality; fall back to deterministic if needed
        if task == "term":
            low = final.lower()
            if "what to ask your clinician" not in low or "citation" not in low:
                return fallback_response(task=task, text=text, context=context), True
        if task == "extract" and "{" not in final:
            return fallback_response(task=task, text=text, context=context), True
        if task == "summarize" and final.count("- ") < 3:
            return fallback_response(task=task, text=text, context=context), True

        return final, False

    except Exception as exc:
        print(f"Warning: generation fallback activated ({exc})")
        return fallback_response(task=task, text=text, context=context), True


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Inference for Healthcare LLM")
    parser.add_argument("--task", choices=["term", "summarize", "extract"], required=True)
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--context", type=str, default=None)
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--force_model", action="store_true", help="Attempt model generation on CPU")
    parser.add_argument("--max_new_tokens", type=int, default=220)
    parser.add_argument(
        "--mode",
        choices=list(ALL_MODES),
        default=MODE_FINETUNED,
        help="LLM mode: base | finetuned | rag",
    )
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

    output, used_fallback = generate(
        task=args.task,
        text=text,
        context=args.context,
        base_model=model_name,
        tiny=args.tiny,
        max_new_tokens=args.max_new_tokens,
        attempt_model=attempt_model,
        mode=args.mode,
    )

    if used_fallback and attempt_model:
        print("[Notice] Response generated using deterministic fallback (model output failed validation).")

    guard = classify_request(f"{args.task}\n{text}\n{args.context or ''}")
    log_question(
        task=args.task,
        prompt_text=text,
        safety_label=guard.label,
        response_preview=output,
        used_model=(attempt_model and guard.label == "allowed" and not used_fallback),
    )
    update_run_state(
        "inference",
        {
            "task": args.task,
            "tiny": args.tiny,
            "base_model": model_name,
            "safety_label": guard.label,
            "attempt_model": attempt_model,
            "used_fallback": used_fallback,
            "mode": args.mode,
        },
    )
    build_report(trigger="inference")
    print(output)


if __name__ == "__main__":
    main()
