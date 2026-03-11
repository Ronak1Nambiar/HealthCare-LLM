"""
HealthCare LLM — Local Gradio GUI
Run with:  python app.py  (or  python app.py --share  for a public link)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from pathlib import Path

# Make sure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr

from src.config import LORA_DIR, RAG_DIR, REPORTS_DIR, ensure_dirs, get_base_model_name, login_huggingface
from src.inference import (
    MODE_BASE,
    MODE_FINETUNED,
    MODE_RAG,
    fallback_response,
    generate,
    sanitize_input,
)
from src.reporting import log_question, update_run_state
from src.safety import classify_request

# ── GUI mode labels (maps display name → internal mode key) ───────────────
LLM_MODES = {
    "🧠 Base Model": MODE_BASE,
    "🎯 Fine-tuned": MODE_FINETUNED,
    "🔍 RAG": MODE_RAG,
}
LLM_MODE_LABELS = list(LLM_MODES.keys())
DEFAULT_LLM_MODE = "🎯 Fine-tuned"

# ── Helpers ────────────────────────────────────────────────────────────────

def _safety_badge(label: str) -> str:
    color = {"allowed": "green", "refuse": "red", "urgent": "orange"}.get(label, "gray")
    return (
        f'<span style="background:{color};color:white;padding:3px 10px;'
        f'border-radius:4px;font-weight:bold">{label.upper()}</span>'
    )


def _mode_badge(mode_label: str) -> str:
    colors = {
        "🧠 Base Model": "#607d8b",
        "🎯 Fine-tuned": "#1976d2",
        "🔍 RAG": "#7b1fa2",
    }
    color = colors.get(mode_label, "#555")
    return (
        f'<span style="background:{color};color:white;padding:3px 10px;'
        f'border-radius:4px;font-weight:bold">{mode_label}</span>'
    )


def _load_metrics() -> str:
    path = REPORTS_DIR / "metrics.json"
    if not path.exists():
        return "No metrics file found yet. Run `python -m src.eval` first."
    metrics = json.loads(path.read_text(encoding="utf-8"))
    lines = ["### Evaluation Metrics", ""]
    mode = metrics.get("mode", "unknown")
    lines.append(f"**Mode:** `{mode}`\n")
    for section, data in metrics.items():
        if section == "mode" or not isinstance(data, dict):
            continue
        lines.append(f"#### {section.replace('_', ' ').title()}")
        for k, v in data.items():
            if isinstance(v, dict):
                lines.append(f"- **{k}:**")
                for sk, sv in v.items():
                    lines.append(f"  - {sk}: `{sv}`")
            elif isinstance(v, str) and len(v) > 80:
                lines.append(f"- **{k}:** ⚠️ {v}")
            else:
                lines.append(f"- **{k}:** `{v}`")
        lines.append("")
    return "\n".join(lines)


def _load_question_log(limit: int = 20) -> str:
    path = REPORTS_DIR / "question_log.jsonl"
    if not path.exists():
        return "No questions logged yet."
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    recent = rows[-limit:]
    if not recent:
        return "No questions logged yet."
    lines = [f"### Last {len(recent)} Questions\n"]
    for q in reversed(recent):
        ts = q.get("timestamp_utc", "?")
        task = q.get("task", "?")
        label = q.get("safety_label", "?")
        llm_mode = q.get("llm_mode", "?")
        used = "model" if q.get("used_model") else "fallback"
        inp = q.get("input_preview", "")[:120]
        lines.append(
            f"**[{ts}]** task=`{task}` mode=`{llm_mode}` safety=`{label}` source=`{used}`"
        )
        lines.append(f"> {inp}")
        lines.append("")
    return "\n".join(lines)


def _adapter_status() -> str:
    meta_path = LORA_DIR / "train_meta.json"
    best_path = LORA_DIR / "best_checkpoint.txt"
    adapter_path = LORA_DIR / "adapter_config.json"

    if not meta_path.exists():
        return "⚠️ No LoRA adapter found. Run **Prepare Data** then **Train Fine-tuned** below."

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    lines = ["**LoRA Adapter**"]
    lines.append(f"- Base model: `{meta.get('base_model', '?')}`")
    compatible = meta.get("adapter_compatible_with_production", True)
    lines.append(f"- Production compatible: {'✅' if compatible else '⚠️ trained on CPU fallback'}")
    lines.append(f"- LoRA r={meta.get('lora_r')} alpha={meta.get('lora_alpha')} lr={meta.get('learning_rate')}")
    if best_path.exists():
        best = json.loads(best_path.read_text(encoding="utf-8"))
        lines.append(f"- Train loss: `{best.get('train_loss', 'n/a')}`")
    loaded = "✅ Loaded" if adapter_path.exists() else "❌ Not saved"
    lines.append(f"- Adapter weights: {loaded}")
    return "\n".join(lines)


def _rag_status() -> str:
    index_path = RAG_DIR / "rag_index.pkl"
    docs_path = RAG_DIR / "rag_docs.json"
    if not index_path.exists() or not docs_path.exists():
        return "⚠️ RAG index not built. Run **Build RAG Index** below."
    try:
        docs = json.loads(docs_path.read_text(encoding="utf-8"))
        return f"✅ RAG index ready — **{len(docs):,} documents** indexed at `{RAG_DIR}`"
    except Exception:
        return "⚠️ RAG index files found but could not be read."


# ── Core inference wrapper ─────────────────────────────────────────────────

def run_inference(
    task: str,
    user_input: str,
    context: str,
    llm_mode_label: str,
    base_model_override: str,
    use_tiny: bool,
    max_new_tokens: int,
    force_fallback: bool,
) -> tuple[str, str, str]:
    """Returns (response_text, combined_badges_html, status_text)."""
    ensure_dirs()

    text = sanitize_input(user_input.strip())
    ctx = sanitize_input(context.strip()) if context.strip() else None

    if not text:
        return "Please enter some text first.", "", "No input provided."

    task_key = {"Summarize": "summarize", "Extract Fields": "extract", "Explain Term": "term"}[task]
    mode = LLM_MODES.get(llm_mode_label, MODE_FINETUNED)

    guard = classify_request(f"{task_key}\n{text}\n{ctx or ''}")
    badges = _safety_badge(guard.label) + "&nbsp;&nbsp;" + _mode_badge(llm_mode_label)

    model_name = get_base_model_name(
        tiny=use_tiny,
        override=base_model_override.strip() if base_model_override.strip() else None,
    )

    import torch
    has_cuda = torch.cuda.is_available()
    attempt_model = (has_cuda or use_tiny) and not force_fallback

    response, used_fallback = generate(
        task=task_key,
        text=text,
        context=ctx,
        base_model=model_name,
        tiny=use_tiny,
        max_new_tokens=int(max_new_tokens),
        attempt_model=attempt_model,
        mode=mode,
    )

    source = "deterministic fallback" if used_fallback else f"model ({model_name})"
    if guard.label in ("refuse", "urgent"):
        source = f"safety filter ({guard.label})"

    status = (
        f"Mode: {llm_mode_label} | Source: {source} | "
        f"Safety: {guard.label} | Risk: {guard.risk_score:.2f}"
    )
    if guard.bypass_detected:
        status += " | ⚠️ Bypass attempt detected"

    log_question(
        task=task_key,
        prompt_text=text,
        safety_label=guard.label,
        response_preview=response,
        used_model=attempt_model and guard.label == "allowed" and not used_fallback,
    )
    update_run_state("inference", {
        "task": task_key,
        "llm_mode": mode,
        "tiny": use_tiny,
        "base_model": model_name,
        "safety_label": guard.label,
        "attempt_model": attempt_model,
        "used_fallback": used_fallback,
    })

    return response, badges, status


# ── Background training helpers ────────────────────────────────────────────

_training_log: list[str] = []
_training_lock = threading.Lock()


def _stream_subprocess(cmd: list[str]) -> None:
    """Run *cmd* in a subprocess; capture stdout+stderr into _training_log."""
    global _training_log
    with _training_lock:
        _training_log.clear()
        _training_log.append(f"$ {' '.join(cmd)}\n")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in proc.stdout:  # type: ignore[union-attr]
            with _training_lock:
                _training_log.append(line)
        proc.wait()
        with _training_lock:
            _training_log.append(f"\n[Done — exit code {proc.returncode}]")
    except Exception as exc:
        with _training_lock:
            _training_log.append(f"\n[Error: {exc}]")


def _get_log() -> str:
    with _training_lock:
        return "".join(_training_log[-300:])


def _run_data_prep(tiny: bool) -> str:
    cmd = [sys.executable, "-m", "src.data_prep"] + (["--tiny"] if tiny else [])
    t = threading.Thread(target=_stream_subprocess, args=(cmd,), daemon=True)
    t.start()
    return "Data preparation started — see log below (refresh to update)."


def _run_train_lora(tiny: bool) -> str:
    cmd = [sys.executable, "-m", "src.train_lora"] + (["--tiny"] if tiny else [])
    t = threading.Thread(target=_stream_subprocess, args=(cmd,), daemon=True)
    t.start()
    return "Fine-tuning started — see log below (refresh to update). This may take a while on CPU."


def _run_rag_train() -> str:
    cmd = [sys.executable, "-m", "src.rag_train"]
    t = threading.Thread(target=_stream_subprocess, args=(cmd,), daemon=True)
    t.start()
    return "RAG indexing started — see log below (refresh to update)."


# ── CSS ────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
    .response-box textarea { font-size: 14px; line-height: 1.6; }
    .status-bar { font-family: monospace; font-size: 12px; color: #555; }
    #header { text-align: center; margin-bottom: 8px; }
    .mode-selector .wrap { gap: 6px; }
    .log-box textarea { font-family: monospace; font-size: 12px; background: #1e1e1e; color: #d4d4d4; }
"""


# ── Build UI ───────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="HealthCare LLM") as demo:

        gr.Markdown(
            """
# 🏥 HealthCare LLM — Local Interface
**Education-only** · No diagnosis · No prescribing · No dosing
_All outputs are general educational information. Always consult a licensed clinician._
            """,
            elem_id="header",
        )

        with gr.Tabs():

            # ── Tab 1: Chat ──────────────────────────────────────────────
            with gr.TabItem("Chat"):
                with gr.Row():
                    with gr.Column(scale=2):
                        llm_mode_1 = gr.Radio(
                            choices=LLM_MODE_LABELS,
                            value=DEFAULT_LLM_MODE,
                            label="LLM Mode",
                            info=(
                                "Base Model: raw Qwen · "
                                "Fine-tuned: QLoRA adapter · "
                                "RAG: retrieval-augmented"
                            ),
                            elem_classes=["mode-selector"],
                        )
                        task_radio = gr.Radio(
                            choices=["Summarize", "Extract Fields", "Explain Term"],
                            value="Explain Term",
                            label="Task",
                        )
                        user_input = gr.Textbox(
                            label="Your Input",
                            placeholder=(
                                "Summarize: paste medical education text here\n"
                                "Extract: paste a de-identified clinical note\n"
                                "Explain Term: type a medical term or question"
                            ),
                            lines=6,
                        )
                        context_box = gr.Textbox(
                            label="Context (optional)",
                            placeholder="Paste reference text / article excerpt...",
                            lines=3,
                        )
                        with gr.Row():
                            submit_btn = gr.Button("Submit", variant="primary", scale=3)
                            clear_btn = gr.Button("Clear", scale=1)

                    with gr.Column(scale=3):
                        badges_html = gr.HTML(label="Safety & Mode")
                        status_txt = gr.Textbox(
                            label="Status",
                            interactive=False,
                            elem_classes=["status-bar"],
                        )
                        response_box = gr.Textbox(
                            label="Response",
                            lines=18,
                            interactive=False,
                            elem_classes=["response-box"],
                        )

                def do_submit(mode, task, inp, ctx):
                    return run_inference(task, inp, ctx, mode, "", False, 220, False)

                def do_clear():
                    return "", "", "", ""

                submit_btn.click(
                    fn=do_submit,
                    inputs=[llm_mode_1, task_radio, user_input, context_box],
                    outputs=[response_box, badges_html, status_txt],
                )
                clear_btn.click(
                    fn=do_clear,
                    outputs=[user_input, context_box, response_box, status_txt],
                )

            # ── Tab 2: Chat + Settings ───────────────────────────────────
            with gr.TabItem("Chat + Settings"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### LLM Mode")
                        llm_mode_2 = gr.Radio(
                            choices=LLM_MODE_LABELS,
                            value=DEFAULT_LLM_MODE,
                            label="Select LLM",
                            info=(
                                "**Base Model**: raw Qwen2.5-1.5B, no fine-tuning, no retrieval\n"
                                "**Fine-tuned**: QLoRA-trained on PubMedQA + synthetic notes\n"
                                "**RAG**: retrieves relevant documents → injects as context"
                            ),
                            elem_classes=["mode-selector"],
                        )
                        gr.Markdown("### Advanced Settings")
                        model_override = gr.Textbox(
                            label="Base Model Override",
                            placeholder="Leave blank to use default (Qwen/Qwen2.5-1.5B-Instruct)",
                        )
                        use_tiny = gr.Checkbox(label="Use tiny model (CPU-safe)", value=False)
                        force_fallback = gr.Checkbox(
                            label="Force deterministic fallback (no LLM)",
                            value=False,
                            info="Useful for testing safety layer in isolation.",
                        )
                        max_new_tokens = gr.Slider(
                            minimum=50, maximum=512, value=220, step=10,
                            label="Max New Tokens",
                        )

                    with gr.Column(scale=2):
                        task_radio2 = gr.Radio(
                            choices=["Summarize", "Extract Fields", "Explain Term"],
                            value="Explain Term",
                            label="Task",
                        )
                        user_input2 = gr.Textbox(label="Your Input", lines=6)
                        context_box2 = gr.Textbox(label="Context (optional)", lines=3)
                        submit_btn2 = gr.Button("Submit", variant="primary")

                    with gr.Column(scale=2):
                        badges_html2 = gr.HTML(label="Safety & Mode")
                        status_txt2 = gr.Textbox(
                            label="Status", interactive=False, elem_classes=["status-bar"]
                        )
                        response_box2 = gr.Textbox(
                            label="Response", lines=15, interactive=False,
                            elem_classes=["response-box"],
                        )

                submit_btn2.click(
                    fn=run_inference,
                    inputs=[
                        task_radio2, user_input2, context_box2,
                        llm_mode_2, model_override, use_tiny, max_new_tokens, force_fallback,
                    ],
                    outputs=[response_box2, badges_html2, status_txt2],
                )

            # ── Tab 3: Training & Setup ──────────────────────────────────
            with gr.TabItem("Training & Setup"):
                gr.Markdown("""
### Build & Train All Three LLM Modes

Run these steps in order (or individually if you already have the data):
1. **Prepare Data** — downloads PubMedQA + generates synthetic notes → `data/processed/`
2. **Train Fine-tuned** — runs QLoRA training → saves adapter to `models/lora/`
3. **Build RAG Index** — indexes processed data → saves TF-IDF index to `models/rag/`
                """)

                with gr.Row():
                    # ── Step 1: Data Prep ────────────────────────────────
                    with gr.Column():
                        gr.Markdown("#### Step 1 — Prepare Data")
                        data_tiny_cb = gr.Checkbox(
                            label="Tiny mode (fast, small dataset for testing)", value=False
                        )
                        prep_btn = gr.Button("Prepare Data", variant="secondary")
                        prep_status = gr.Textbox(label="Status", interactive=False, lines=1)
                        prep_btn.click(
                            fn=_run_data_prep,
                            inputs=[data_tiny_cb],
                            outputs=[prep_status],
                        )

                    # ── Step 2: Fine-tune ────────────────────────────────
                    with gr.Column():
                        gr.Markdown("#### Step 2 — Fine-tune (QLoRA)")
                        train_tiny_cb = gr.Checkbox(
                            label="Tiny mode (20 steps, CPU-safe, quick test)", value=False
                        )
                        train_btn = gr.Button("Train Fine-tuned Model", variant="primary")
                        train_status = gr.Textbox(label="Status", interactive=False, lines=1)
                        adapter_md = gr.Markdown(_adapter_status())
                        refresh_adapter_btn = gr.Button("Refresh Adapter Status", size="sm")
                        train_btn.click(
                            fn=_run_train_lora,
                            inputs=[train_tiny_cb],
                            outputs=[train_status],
                        )
                        refresh_adapter_btn.click(fn=_adapter_status, outputs=[adapter_md])

                    # ── Step 3: RAG ──────────────────────────────────────
                    with gr.Column():
                        gr.Markdown("#### Step 3 — Build RAG Index")
                        rag_status_md = gr.Markdown(_rag_status())
                        rag_btn = gr.Button("Build RAG Index", variant="primary")
                        rag_status_txt = gr.Textbox(label="Status", interactive=False, lines=1)
                        refresh_rag_btn = gr.Button("Refresh RAG Status", size="sm")
                        rag_btn.click(
                            fn=_run_rag_train,
                            outputs=[rag_status_txt],
                        )
                        refresh_rag_btn.click(fn=_rag_status, outputs=[rag_status_md])

                gr.Markdown("---")
                gr.Markdown("#### Live Training Log")
                log_box = gr.Textbox(
                    label="Output",
                    lines=20,
                    interactive=False,
                    placeholder="Logs appear here while training is running...",
                    elem_classes=["log-box"],
                )
                refresh_log_btn = gr.Button("Refresh Log")
                refresh_log_btn.click(fn=_get_log, outputs=[log_box])

            # ── Tab 4: Metrics ───────────────────────────────────────────
            with gr.TabItem("Metrics"):
                metrics_md = gr.Markdown(_load_metrics())
                refresh_metrics = gr.Button("Refresh Metrics")
                refresh_metrics.click(fn=_load_metrics, outputs=metrics_md)

            # ── Tab 5: Question Log ──────────────────────────────────────
            with gr.TabItem("Question Log"):
                log_limit = gr.Slider(minimum=5, maximum=100, value=20, step=5, label="Show last N")
                log_md = gr.Markdown(_load_question_log())
                refresh_log_tab = gr.Button("Refresh Log")
                refresh_log_tab.click(fn=_load_question_log, inputs=[log_limit], outputs=log_md)

            # ── Tab 6: Examples ──────────────────────────────────────────
            with gr.TabItem("Examples"):
                gr.Markdown("""
### Quick-start Examples

**Explain Term:** Input: `Hypertension`
**Summarize:** Input: `High blood pressure can raise the risk of stroke...`
**Extract Fields:** Input: `Chief complaint: headache. Symptoms include dizziness...`
**Safety-blocked (Refuse):** Input: `Diagnose me with diabetes and give me my insulin dose`
**Safety-blocked (Urgent):** Input: `I have severe chest pain and shortness of breath`
""")
                with gr.Row():
                    example_mode = gr.Radio(
                        choices=LLM_MODE_LABELS,
                        value=DEFAULT_LLM_MODE,
                        label="LLM Mode",
                        elem_classes=["mode-selector"],
                    )
                    example_task = gr.Radio(
                        choices=["Summarize", "Extract Fields", "Explain Term"],
                        value="Explain Term",
                        label="Task",
                    )
                with gr.Row():
                    example_input = gr.Textbox(label="Example Input", lines=4)
                    example_ctx = gr.Textbox(label="Example Context", lines=3)

                run_example_btn = gr.Button("Run Example", variant="secondary")
                example_badges = gr.HTML(label="Safety & Mode")
                example_status = gr.Textbox(label="Status", interactive=False)
                example_response = gr.Textbox(label="Response", lines=10, interactive=False)

                run_example_btn.click(
                    fn=lambda mode, task, inp, ctx: run_inference(
                        task, inp, ctx, mode, "", False, 220, False
                    ),
                    inputs=[example_mode, example_task, example_input, example_ctx],
                    outputs=[example_response, example_badges, example_status],
                )

            # ── Tab 7: System Info ───────────────────────────────────────
            with gr.TabItem("System"):
                import torch as _torch
                cuda_info = f"CUDA available: {_torch.cuda.is_available()}"
                if _torch.cuda.is_available():
                    cuda_info += f"\nGPU: {_torch.cuda.get_device_name(0)}"
                    cuda_info += f"\nVRAM: {_torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"

                gr.Markdown(f"""
### System Information
```
{cuda_info}
```

### Three LLM Modes
| Mode | Description | Adapter | Retrieval |
|------|-------------|---------|-----------|
| 🧠 Base Model | Raw Qwen2.5-1.5B-Instruct | ❌ No | ❌ No |
| 🎯 Fine-tuned | QLoRA-trained on PubMedQA + synthetic notes | ✅ Yes | ❌ No |
| 🔍 RAG | TF-IDF retrieval from knowledge base | ✅ Yes | ✅ Yes |

### Pipeline Commands
```bash
# 1. Prepare data
python -m src.data_prep --seed 42

# 2. Train fine-tuned model (GPU recommended)
python -m src.train_lora

# 3. Build RAG index
python -m src.rag_train

# 4. CLI inference
python -m src.inference --task term --mode rag --input "What is hypertension?"

# 5. Evaluate
python -m src.eval
```

### Safety Architecture
1. **Urgent patterns** — emergency guidance routing
2. **Hard refusal patterns** — blocks diagnosis/prescribing/dosing
3. **Bypass detection** — catches adversarial framing
4. **Risk score threshold** — soft refusal above 3.0
""")

    return demo


# ── Entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="HealthCare LLM Gradio GUI")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    parser.add_argument("--port", type=int, default=8080, help="Port to run on (default: 8080)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    args = parser.parse_args()

    ensure_dirs()
    logged_in = login_huggingface()
    if logged_in:
        print("Logged in to Hugging Face.")

    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
