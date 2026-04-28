#!/usr/bin/env python3
"""
Evaluate three models on healthcare eval tasks → reports/model_comparison.json.

Run from repo root:
    python generate_model_comparison.py

Hardware notes:
- Uses 4-bit NF4 quantization if CUDA is available, float32 on CPU.
- On CPU, evaluation is reduced to N_SAMPLES_PER_TASK=5 and max_new_tokens
  is capped at MAX_NEW_TOKENS_CPU=80 to keep wall-clock time manageable.
- Gemma-2-2b-it is gated and requires a HF_TOKEN environment variable.
- The LoRA adapter in models/lora/ was trained on sshleifer/tiny-gpt2; loading
  it onto Qwen or Gemma will fail. All comparisons therefore use base-model-only
  inference (use_lora=False at the loader level) via the monkey-patched loader.
"""
from __future__ import annotations

import gc
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

# ── ensure repo root is on sys.path ───────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import psutil
import torch

from src.config import PROCESSED_DIR, REPORTS_DIR, ensure_dirs
from src.eval import eval_extraction, eval_summarization, eval_term, load_jsonl

# ── tunables ──────────────────────────────────────────────────────────────────
N_SAMPLES_PER_TASK = 5          # drop from 20 on CPU (noted in report)
MAX_NEW_TOKENS_CPU = 80         # cap tokens to limit wall-clock on CPU

MODELS = [
    ("qwen2.5-1.5b-instruct", "Qwen/Qwen2.5-1.5B-Instruct"),
    ("gemma-2-2b-it",          "google/gemma-2-2b-it"),
    ("tiny-gpt2",              "sshleifer/tiny-gpt2"),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def hardware_info() -> dict:
    info: dict = {
        "cpu": platform.processor() or platform.machine(),
        "gpu": None,
        "gpu_vram_total_gb": None,
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 2),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        _, total = torch.cuda.mem_get_info(0)
        info["gpu_vram_total_gb"] = round(total / 1e9, 2)
    return info


def sample_rows(rows: list[dict], task: str, n: int) -> list[dict]:
    task_rows = [r for r in rows if r["task"] == task]
    return task_rows[:n]


# ── per-model evaluation ──────────────────────────────────────────────────────

def eval_one_model(
    model_key: str,
    model_name: str,
    rows_by_task: dict[str, list[dict]],
) -> dict:
    import src.eval as eval_mod
    import src.inference as inf_mod

    use_cuda = torch.cuda.is_available()
    max_tokens = None if use_cuda else MAX_NEW_TOKENS_CPU

    # save originals before any patching
    orig_loader = inf_mod.load_model_and_tokenizer
    orig_gen_in_eval = eval_mod.generate

    model = None
    result: dict = {
        "model_name": model_name,
        "load_status": "pending",
        "error": None,
        "device_used": "cuda" if use_cuda else "cpu",
        "n_samples_per_task": N_SAMPLES_PER_TASK,
        "max_new_tokens_used": max_tokens or "model-default",
    }

    try:
        print(f"\n{'='*60}")
        print(f"[{model_key}] Loading: {model_name}")

        t_load = time.perf_counter()
        model, tokenizer = orig_loader(
            base_model=model_name, tiny=False, use_lora=False
        )
        load_s = round(time.perf_counter() - t_load, 2)
        result["load_status"] = "success"
        result["load_time_s"] = load_s
        print(f"[{model_key}] Loaded in {load_s}s  "
              f"params={sum(p.numel() for p in model.parameters()):,}")

        # ── patch 1: cache the model so eval functions don't reload it ────────
        _m, _tok = model, tokenizer

        def _cached_loader(*args, **kwargs):
            return _m, _tok

        inf_mod.load_model_and_tokenizer = _cached_loader

        # ── patch 2: cap max_new_tokens for CPU runs ──────────────────────────
        if max_tokens is not None:
            _orig_gen = inf_mod.generate

            def _fast_gen(
                task, text, context, base_model, tiny,
                max_new_tokens, **kwargs
            ):
                return _orig_gen(
                    task, text, context, base_model, tiny,
                    min(max_new_tokens, max_tokens), **kwargs,
                )

            eval_mod.generate = _fast_gen

        # ── summarization ─────────────────────────────────────────────────────
        print(f"[{model_key}] Running summarization ({N_SAMPLES_PER_TASK} samples)...")
        t0 = time.perf_counter()
        try:
            summ = eval_summarization(
                rows_by_task["summarize"],
                use_model=True, tiny=False, base_model=model_name,
            )
            summ["eval_wall_s"] = round(time.perf_counter() - t0, 2)
            result["summarization"] = summ
            print(f"[{model_key}] summarization: {summ}")
        except Exception:
            result["summarization"] = {"error": traceback.format_exc().strip()}
            print(f"[{model_key}] summarization FAILED:\n{traceback.format_exc()}")

        # ── extraction ────────────────────────────────────────────────────────
        print(f"[{model_key}] Running extraction ({N_SAMPLES_PER_TASK} samples)...")
        t0 = time.perf_counter()
        try:
            extr = eval_extraction(
                rows_by_task["extract"],
                use_model=True, tiny=False, base_model=model_name,
            )
            extr["eval_wall_s"] = round(time.perf_counter() - t0, 2)
            result["extraction"] = extr
            print(f"[{model_key}] extraction: {extr}")
        except Exception:
            result["extraction"] = {"error": traceback.format_exc().strip()}
            print(f"[{model_key}] extraction FAILED:\n{traceback.format_exc()}")

        # ── safety / term ─────────────────────────────────────────────────────
        print(f"[{model_key}] Running safety/term ({N_SAMPLES_PER_TASK} samples)...")
        t0 = time.perf_counter()
        try:
            saf = eval_term(
                rows_by_task["term"],
                use_model=True, tiny=False, base_model=model_name,
            )
            saf["eval_wall_s"] = round(time.perf_counter() - t0, 2)
            result["safety"] = saf
            print(f"[{model_key}] safety: {saf}")
        except Exception:
            result["safety"] = {"error": traceback.format_exc().strip()}
            print(f"[{model_key}] safety FAILED:\n{traceback.format_exc()}")

    except Exception:
        tb = traceback.format_exc().strip()
        result["load_status"] = "failed"
        result["error"] = tb
        print(f"[{model_key}] FAILED to load:\n{tb}")

    finally:
        # always restore originals
        inf_mod.load_model_and_tokenizer = orig_loader
        eval_mod.generate = orig_gen_in_eval
        if model is not None:
            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ensure_dirs()

    print("=== CAPABILITY REPORT ===")
    hw = hardware_info()
    for k, v in hw.items():
        print(f"  {k}: {v}")
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    print(f"  HF_TOKEN set: {bool(hf_token)}")
    print(f"  Gemma gated (requires token): yes")
    print(f"  n_samples_per_task: {N_SAMPLES_PER_TASK} (CPU mode)")
    print(f"  max_new_tokens cap: {MAX_NEW_TOKENS_CPU}")
    print("=========================\n")

    # ── load test split once ─────────────────────────────────────────────────
    test_path = PROCESSED_DIR / "test.jsonl"
    if not test_path.exists():
        print(f"ERROR: {test_path} not found. Run src/data_prep.py first.")
        sys.exit(1)

    all_rows = load_jsonl(test_path)
    rows_by_task = {
        "summarize": sample_rows(all_rows, "summarize", N_SAMPLES_PER_TASK),
        "extract":   sample_rows(all_rows, "extract",   N_SAMPLES_PER_TASK),
        "term":      sample_rows(all_rows, "term",       N_SAMPLES_PER_TASK),
    }
    for task, rows in rows_by_task.items():
        print(f"  {task}: {len(rows)} samples")

    # ── run each model ────────────────────────────────────────────────────────
    results: dict = {"hardware": hw}
    t_total = time.perf_counter()

    for model_key, model_name in MODELS:
        result = eval_one_model(model_key, model_name, rows_by_task)
        results[model_key] = result

    results["total_wall_s"] = round(time.perf_counter() - t_total, 2)

    # ── write output ──────────────────────────────────────────────────────────
    out_path = REPORTS_DIR / "model_comparison.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Wrote {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
