#!/usr/bin/env python3
"""
Measure lightweight/IoT-applicability deployment metrics → reports/deployment_metrics.json.

Run from repo root (after generate_model_comparison.py has cached models):
    python generate_deployment_metrics.py

Measures:
  - Disk footprint (base model HF cache, LoRA adapter, RAG index)
  - Peak RAM during inference (no CUDA on this machine)
  - Cold-start latency (model load + first inference)
  - Per-query latency mean / p95 across 10 queries
  - Tokens-per-second during generation
  - Per-component overhead: sanitize, safety-classify, RAG-retrieve

Notes:
  - The LoRA adapter in models/lora/ was trained on sshleifer/tiny-gpt2
    (target_modules=c_attn,c_proj). Loading it onto Qwen2.5-1.5B will fail
    because those module names don't exist in the Qwen architecture.
    Deployment metrics therefore use Qwen base model (mode="base") without
    the LoRA adapter.  This is recorded in load_status.
  - All measurements are CPU-only (no CUDA); values are not representative
    of a GPU production deployment.
"""
from __future__ import annotations

import gc
import json
import os
import platform
import subprocess
import sys
import time
import timeit
import traceback
from pathlib import Path
from statistics import mean, quantiles
from threading import Thread

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import psutil
import torch

from src.config import (
    LORA_DIR, PROCESSED_DIR, RAG_DIR, REPORTS_DIR, ensure_dirs
)
from src.eval import load_jsonl
from src.inference import (
    fallback_response, load_model_and_tokenizer, sanitize_input
)
from src.safety import classify_request

PRODUCTION_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
N_LATENCY_QUERIES = 10
N_COMPONENT_CALLS = 100
MAX_NEW_TOKENS = 80   # cap for CPU speed; matches model_comparison run


# ── helpers ───────────────────────────────────────────────────────────────────

def dir_bytes(path: Path) -> int | None:
    """Return total bytes used by all files under path, or None if missing."""
    if not path.exists():
        return None
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def hf_cache_dir_for_model(model_id: str) -> Path | None:
    """Return the HuggingFace cache directory for a given model id."""
    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    # HF hub layout: hub/models--<org>--<name>/snapshots/<hash>
    slug = "models--" + model_id.replace("/", "--")
    candidate = cache_root / "hub" / slug
    if candidate.exists():
        return candidate
    return None


def peak_rss_mb_during(fn) -> tuple[float, float]:
    """
    Run fn() in the main thread while a background thread polls RSS every 100 ms.
    Returns (result_value, peak_rss_mb).  fn must return a single value.
    """
    proc = psutil.Process()
    peak = [proc.memory_info().rss]
    done = [False]

    def _poll():
        while not done[0]:
            try:
                peak[0] = max(peak[0], proc.memory_info().rss)
            except Exception:
                pass
            time.sleep(0.1)

    t = Thread(target=_poll, daemon=True)
    t.start()
    result = fn()
    done[0] = True
    t.join()
    return result, peak[0] / 1e6


def hardware_info() -> dict:
    info = {
        "cpu": platform.processor() or platform.machine(),
        "gpu": None,
        "gpu_vram_total_gb": None,
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 2),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "python_version": platform.python_version(),
        "os": platform.platform(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        _, total = torch.cuda.mem_get_info(0)
        info["gpu_vram_total_gb"] = round(total / 1e9, 2)
    return info


def try_load_lora_status() -> dict:
    """
    Attempt to load Qwen + LoRA; capture success or failure verbatim.
    This is purely diagnostic — we revert immediately so the real measurements
    use the already-loaded base model.
    """
    status = {"attempted": True}
    try:
        _m, _t = load_model_and_tokenizer(
            base_model=PRODUCTION_MODEL, tiny=False, use_lora=True
        )
        status["load_status"] = "success"
        del _m, _t
        gc.collect()
    except Exception:
        status["load_status"] = "failed"
        status["error"] = traceback.format_exc().strip()
    return status


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ensure_dirs()

    print("=== DEPLOYMENT METRICS ===")
    hw = hardware_info()
    for k, v in hw.items():
        print(f"  {k}: {v}")

    metrics: dict = {
        "hardware": hw,
        "production_model": PRODUCTION_MODEL,
        "notes": (
            "All measurements are CPU-only (no CUDA). "
            "The LoRA adapter was trained on sshleifer/tiny-gpt2 "
            "(target_modules=[c_attn,c_proj]) and is incompatible with "
            "Qwen2.5-1.5B-Instruct. Deployment metrics use base model only."
        ),
    }

    # ── 1. Disk sizes ─────────────────────────────────────────────────────────
    print("\n[1/5] Measuring disk footprint...")
    lora_bytes = dir_bytes(LORA_DIR)
    rag_bytes   = dir_bytes(RAG_DIR)
    hf_dir      = hf_cache_dir_for_model(PRODUCTION_MODEL)
    base_bytes  = dir_bytes(hf_dir) if hf_dir else None

    total_bytes: int | None = None
    if base_bytes is not None and lora_bytes is not None and rag_bytes is not None:
        total_bytes = base_bytes + lora_bytes + rag_bytes

    metrics["disk"] = {
        "base_model_hf_cache_mb": round(base_bytes / 1e6, 1) if base_bytes is not None else None,
        "base_model_hf_cache_path": str(hf_dir) if hf_dir else "not_cached",
        "lora_adapter_mb": round(lora_bytes / 1e6, 1) if lora_bytes is not None else None,
        "lora_adapter_path": str(LORA_DIR),
        "rag_index_mb": round(rag_bytes / 1e6, 1) if rag_bytes is not None else None,
        "rag_index_path": str(RAG_DIR),
        "total_deployment_mb": round(total_bytes / 1e6, 1) if total_bytes is not None else None,
        "note": (
            "base_model_hf_cache_mb is null until Qwen is downloaded. "
            "Run generate_model_comparison.py first to cache the model."
            if base_bytes is None else ""
        ),
    }
    print(f"  LoRA adapter:  {metrics['disk']['lora_adapter_mb']} MB")
    print(f"  RAG index:     {metrics['disk']['rag_index_mb']} MB")
    print(f"  Base model HF: {metrics['disk']['base_model_hf_cache_mb']} MB")

    # ── 2. LoRA compatibility check ───────────────────────────────────────────
    print("\n[2/5] Checking LoRA adapter compatibility with Qwen...")
    # Only probe if the model is already cached (avoids a multi-GB download here)
    if hf_dir is not None:
        lora_status = try_load_lora_status()
    else:
        lora_status = {
            "attempted": False,
            "load_status": "skipped",
            "reason": "Qwen not yet in HF cache; run generate_model_comparison.py first",
        }
    metrics["lora_compatibility"] = lora_status
    print(f"  LoRA load status: {lora_status.get('load_status')}")
    if lora_status.get("error"):
        print(f"  Error (first line): {lora_status['error'].splitlines()[0]}")

    # ── 3. Load base model for latency / memory measurements ──────────────────
    print(f"\n[3/5] Loading {PRODUCTION_MODEL} (base, no LoRA)...")
    if hf_dir is None:
        print("  Model not in HF cache — downloading now (~3 GB)...")

    proc = psutil.Process()
    rss_before_load = proc.memory_info().rss
    t_cold_start = time.perf_counter()

    model = tokenizer = None
    try:
        model, tokenizer = load_model_and_tokenizer(
            base_model=PRODUCTION_MODEL, tiny=False, use_lora=False
        )
        t_model_loaded = time.perf_counter()
        rss_after_load = proc.memory_info().rss
        load_ram_mb = round((rss_after_load - rss_before_load) / 1e6, 1)
        load_time_s = round(t_model_loaded - t_cold_start, 2)
        params = sum(p.numel() for p in model.parameters())
        print(f"  Loaded in {load_time_s}s, RAM delta={load_ram_mb} MB, "
              f"params={params:,}")
    except Exception:
        tb = traceback.format_exc().strip()
        metrics["model_load"] = {"status": "failed", "error": tb}
        print(f"  Model load FAILED:\n{tb}")
        _write_and_exit(metrics)
        return

    # ── 4. Latency measurements ───────────────────────────────────────────────
    print(f"\n[4/5] Measuring inference latency ({N_LATENCY_QUERIES} queries)...")

    # Pull sample queries from test set
    test_rows = load_jsonl(PROCESSED_DIR / "test.jsonl")
    # Use a mix of tasks (mostly "term" / "summarize" which are faster)
    query_rows = [r for r in test_rows if r["task"] == "term"][:N_LATENCY_QUERIES]
    if len(query_rows) < N_LATENCY_QUERIES:
        query_rows += [r for r in test_rows if r["task"] == "summarize"][
            : N_LATENCY_QUERIES - len(query_rows)
        ]

    # Patch loader to use cached model
    import src.inference as inf_mod
    _m, _t = model, tokenizer
    orig_loader = inf_mod.load_model_and_tokenizer

    def _cached_loader(*args, **kwargs):
        return _m, _t

    inf_mod.load_model_and_tokenizer = _cached_loader

    from src.inference import generate

    latencies_s: list[float] = []
    tokens_counts: list[int] = []
    peak_rss_mb_inference = 0.0

    for i, row in enumerate(query_rows):
        print(f"  query {i+1}/{N_LATENCY_QUERIES}: task={row['task']}, "
              f"input={row['input'][:60]!r}...")

        rss_before_q = proc.memory_info().rss
        t0 = time.perf_counter()

        try:
            output, _used_fallback = generate(
                task=row["task"],
                text=row["input"],
                context=None,
                base_model=PRODUCTION_MODEL,
                tiny=False,
                max_new_tokens=MAX_NEW_TOKENS,
                attempt_model=True,
                mode="base",
            )
        except Exception as exc:
            print(f"  [warn] query {i+1} failed: {exc}")
            continue

        elapsed = time.perf_counter() - t0
        rss_after_q = proc.memory_info().rss
        peak_rss_mb_inference = max(
            peak_rss_mb_inference,
            (rss_after_q - rss_before_load) / 1e6,
        )

        # approximate output tokens
        ntok = len(tokenizer.encode(output)) if output else 0
        latencies_s.append(elapsed)
        tokens_counts.append(ntok)
        print(f"    → {elapsed:.2f}s, ~{ntok} tokens, fallback={_used_fallback}")

        if i == 0:
            # Record cold-start (load + first query)
            cold_start_s = round(time.perf_counter() - t_cold_start, 2)

    inf_mod.load_model_and_tokenizer = orig_loader  # restore

    if latencies_s:
        sorted_lat = sorted(latencies_s)
        mean_lat = mean(latencies_s)
        p95_lat = sorted_lat[max(0, int(0.95 * len(sorted_lat)) - 1)]
        mean_tokens = mean(tokens_counts) if tokens_counts else 0
        tps = mean_tokens / mean_lat if mean_lat > 0 else None

        metrics["memory"] = {
            "load_time_s": load_time_s,
            "ram_delta_on_load_mb": load_ram_mb,
            "peak_rss_during_inference_mb": round(peak_rss_mb_inference, 1),
            "device": "cpu",
            "gpu_vram_peak_mb": None,
            "note": "No CUDA — VRAM measurements not applicable.",
        }
        metrics["latency"] = {
            "cold_start_s": cold_start_s,
            "n_queries": len(latencies_s),
            "mean_query_s": round(mean_lat, 3),
            "p95_query_s": round(p95_lat, 3),
            "min_query_s": round(min(latencies_s), 3),
            "max_query_s": round(max(latencies_s), 3),
            "mean_output_tokens": round(mean_tokens, 1),
            "tokens_per_second": round(tps, 2) if tps else None,
            "device": "cpu",
            "max_new_tokens_cap": MAX_NEW_TOKENS,
            "note": (
                "CPU-only measurements. GPU latency expected to be 10-50x faster. "
                "cold_start_s includes model loading from HF cache."
            ),
        }
    else:
        metrics["memory"] = {"error": "No successful queries"}
        metrics["latency"] = {"error": "No successful queries"}

    # ── 5. Pipeline component overhead ────────────────────────────────────────
    print(f"\n[5/5] Measuring pipeline component overhead "
          f"({N_COMPONENT_CALLS} calls each)...")

    sample_query = "What is hypertension and how is it treated?"
    sample_urgent = "I have severe chest pain and cannot breathe"
    sample_allowed = "Explain the term 'myocardial infarction' for a patient."
    sample_refused = "Diagnose my symptoms and prescribe medication."
    sample_outputs = [
        "- Hypertension is high blood pressure...\n- It is treated with medication.",
        "- Risk factors include smoking and obesity.\nSafety disclaimer: Not medical advice.",
        "{\"chief_complaint\": \"headache\", \"symptoms\": [\"nausea\"]}",
    ]

    component_metrics: dict = {}

    # a) input sanitization
    try:
        san_time = (
            timeit.timeit(lambda: sanitize_input(sample_query), number=N_COMPONENT_CALLS)
            / N_COMPONENT_CALLS
        )
        component_metrics["input_sanitization_mean_ms"] = round(san_time * 1000, 4)
        print(f"  sanitize:  {san_time*1000:.4f} ms/call")
    except Exception as exc:
        component_metrics["input_sanitization_mean_ms"] = None
        component_metrics["input_sanitization_error"] = str(exc)

    # b) safety classification (mix of urgent/refused/allowed)
    try:
        inputs_mix = [sample_urgent, sample_allowed, sample_refused] * (
            N_COMPONENT_CALLS // 3 + 1
        )
        inputs_mix = inputs_mix[:N_COMPONENT_CALLS]
        idx = [0]

        def _classify():
            classify_request(inputs_mix[idx[0] % len(inputs_mix)])
            idx[0] += 1

        saf_time = timeit.timeit(_classify, number=N_COMPONENT_CALLS) / N_COMPONENT_CALLS
        component_metrics["safety_classification_mean_ms"] = round(saf_time * 1000, 4)
        print(f"  safety:    {saf_time*1000:.4f} ms/call")
    except Exception as exc:
        component_metrics["safety_classification_mean_ms"] = None
        component_metrics["safety_classification_error"] = str(exc)

    # c) TF-IDF retrieval
    try:
        from src.rag import get_retriever

        retriever = get_retriever()
        if retriever.is_built:
            query_pool = [
                "hypertension treatment options",
                "diabetes management patient education",
                "myocardial infarction symptoms",
                "asthma inhaler technique",
                "antibiotic resistance mechanisms",
            ]
            q_idx = [0]

            def _retrieve():
                retriever.retrieve(query_pool[q_idx[0] % len(query_pool)])
                q_idx[0] += 1

            ret_time = timeit.timeit(_retrieve, number=N_COMPONENT_CALLS) / N_COMPONENT_CALLS
            component_metrics["rag_retrieval_mean_ms"] = round(ret_time * 1000, 4)
            print(f"  RAG:       {ret_time*1000:.4f} ms/call")
        else:
            component_metrics["rag_retrieval_mean_ms"] = None
            component_metrics["rag_retrieval_note"] = "RAG index not built"
    except Exception as exc:
        component_metrics["rag_retrieval_mean_ms"] = None
        component_metrics["rag_retrieval_error"] = str(exc)

    # d) output validation (inline string checks from inference.py)
    try:
        out_idx = [0]

        def _validate():
            txt = sample_outputs[out_idx[0] % len(sample_outputs)]
            low = txt.lower()
            _ = "what to ask your clinician" in low and "citation" in low
            _ = txt.count("- ") >= 3
            _ = "{" in txt
            out_idx[0] += 1

        val_time = timeit.timeit(_validate, number=N_COMPONENT_CALLS) / N_COMPONENT_CALLS
        component_metrics["output_validation_mean_ms"] = round(val_time * 1000, 4)
        print(f"  validation:{val_time*1000:.4f} ms/call")
    except Exception as exc:
        component_metrics["output_validation_mean_ms"] = None
        component_metrics["output_validation_error"] = str(exc)

    metrics["pipeline_component_overhead"] = component_metrics

    # ── cleanup & write ───────────────────────────────────────────────────────
    if model is not None:
        del model, tokenizer
        gc.collect()

    _write_and_exit(metrics)


def _write_and_exit(metrics: dict) -> None:
    ensure_dirs()
    out_path = REPORTS_DIR / "deployment_metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Wrote {out_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
