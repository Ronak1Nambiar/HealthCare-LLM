"""
HealthCare LLM — Local Gradio GUI
Run with:  python app.py  (or  python app.py --share  for a public link)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make sure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr

from src.config import LORA_DIR, REPORTS_DIR, ensure_dirs, get_base_model_name, login_huggingface
from src.inference import fallback_response, generate, sanitize_input
from src.reporting import log_question, update_run_state
from src.safety import classify_request

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _safety_badge(label: str) -> str:
    color = {"allowed": "green", "refuse": "red", "urgent": "orange"}.get(label, "gray")
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-weight:bold">{label.upper()}</span>'


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
        used = "model" if q.get("used_model") else "fallback"
        inp = q.get("input_preview", "")[:120]
        lines.append(f"**[{ts}]** task=`{task}` safety=`{label}` source=`{used}`")
        lines.append(f"> {inp}")
        lines.append("")
    return "\n".join(lines)


def _adapter_status() -> str:
    meta_path = LORA_DIR / "train_meta.json"
    best_path = LORA_DIR / "best_checkpoint.txt"
    if not meta_path.exists():
        return "No LoRA adapter found. Run `python -m src.train_lora` to train one."
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    lines = ["**LoRA Adapter Status**"]
    lines.append(f"- Base model: `{meta.get('base_model', '?')}`")
    lines.append(f"- Production model: `{meta.get('intended_production_model', '?')}`")
    compatible = meta.get("adapter_compatible_with_production", True)
    lines.append(f"- Compatible with production: {'✅' if compatible else '⚠️ NO — trained on CPU fallback model'}")
    lines.append(f"- LoRA r={meta.get('lora_r')} alpha={meta.get('lora_alpha')} lr={meta.get('learning_rate')}")
    if best_path.exists():
        best = json.loads(best_path.read_text(encoding="utf-8"))
        lines.append(f"- Train loss: `{best.get('train_loss', 'n/a')}`")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Core inference wrapper
# ──────────────────────────────────────────────────────────────

def run_inference(
    task: str,
    user_input: str,
    context: str,
    base_model_override: str,
    use_tiny: bool,
    max_new_tokens: int,
    force_fallback: bool,
) -> tuple[str, str, str]:
    """
    Returns (response_text, safety_html, status_text)
    """
    ensure_dirs()

    text = sanitize_input(user_input.strip())
    ctx = sanitize_input(context.strip()) if context.strip() else None

    if not text:
        return "Please enter some text first.", "", "No input provided."

    task_key = {"Summarize": "summarize", "Extract Fields": "extract", "Explain Term": "term"}[task]

    guard = classify_request(f"{task_key}\n{text}\n{ctx or ''}")
    safety_html = _safety_badge(guard.label)

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
    )

    source = "deterministic fallback" if used_fallback else f"model ({model_name})"
    if guard.label in ("refuse", "urgent"):
        source = f"safety filter ({guard.label})"

    status = f"Source: {source} | Safety: {guard.label} | Risk score: {guard.risk_score:.2f}"
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
        "tiny": use_tiny,
        "base_model": model_name,
        "safety_label": guard.label,
        "attempt_model": attempt_model,
        "used_fallback": used_fallback,
    })

    return response, safety_html, status


# ──────────────────────────────────────────────────────────────
# Build UI
# ──────────────────────────────────────────────────────────────

CUSTOM_CSS = """
    .response-box textarea { font-size: 14px; line-height: 1.6; }
    .status-bar { font-family: monospace; font-size: 13px; color: #555; }
    #header { text-align: center; margin-bottom: 8px; }
    .badge-row { display: flex; align-items: center; gap: 8px; }
"""

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

            # ── Tab 1: Chat / Inference ──────────────────────────────
            with gr.TabItem("Chat"):
                with gr.Row():
                    with gr.Column(scale=2):
                        task_radio = gr.Radio(
                            choices=["Summarize", "Extract Fields", "Explain Term"],
                            value="Explain Term",
                            label="Task",
                            info="Choose what you want the model to do with your input.",
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
                            placeholder="Paste reference text / article excerpt for citation...",
                            lines=3,
                        )

                        with gr.Row():
                            submit_btn = gr.Button("Submit", variant="primary", scale=3)
                            clear_btn = gr.Button("Clear", scale=1)

                    with gr.Column(scale=3):
                        safety_html = gr.HTML(label="Safety Label")
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

                def do_submit(task, inp, ctx):
                    resp, badge, status = run_inference(task, inp, ctx, "", False, 220, False)
                    return resp, badge, status

                def do_clear():
                    return "", "", "", ""

                submit_btn.click(
                    fn=do_submit,
                    inputs=[task_radio, user_input, context_box],
                    outputs=[response_box, safety_html, status_txt],
                )
                clear_btn.click(
                    fn=do_clear,
                    inputs=[],
                    outputs=[user_input, context_box, response_box, status_txt],
                )

            # ── Tab 2: Full Chat with Settings ─────────────────────────
            with gr.TabItem("Chat + Settings"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Model Settings")
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
                        adapter_md = gr.Markdown(_adapter_status())
                        refresh_adapter = gr.Button("Refresh Adapter Status")
                        refresh_adapter.click(fn=_adapter_status, outputs=adapter_md)

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
                        safety_html2 = gr.HTML(label="Safety Label")
                        status_txt2 = gr.Textbox(label="Status", interactive=False, elem_classes=["status-bar"])
                        response_box2 = gr.Textbox(label="Response", lines=15, interactive=False)

                submit_btn2.click(
                    fn=run_inference,
                    inputs=[task_radio2, user_input2, context_box2, model_override, use_tiny, max_new_tokens, force_fallback],
                    outputs=[response_box2, safety_html2, status_txt2],
                )

            # ── Tab 3: Metrics ───────────────────────────────────────
            with gr.TabItem("Metrics"):
                metrics_md = gr.Markdown(_load_metrics())
                refresh_metrics = gr.Button("Refresh Metrics")
                refresh_metrics.click(fn=_load_metrics, outputs=metrics_md)

            # ── Tab 4: Question Log ──────────────────────────────────
            with gr.TabItem("Question Log"):
                log_limit = gr.Slider(minimum=5, maximum=100, value=20, step=5, label="Show last N questions")
                log_md = gr.Markdown(_load_question_log())
                refresh_log = gr.Button("Refresh Log")
                refresh_log.click(fn=_load_question_log, inputs=[log_limit], outputs=log_md)

            # ── Tab 5: Examples ──────────────────────────────────────
            with gr.TabItem("Examples"):
                gr.Markdown("""
### Quick-start Examples

**Explain Term:**
- Input: `Hypertension`
- Context: `Hypertension is a condition where blood pressure remains persistently elevated above 140/90 mmHg.`

**Summarize:**
- Input: `High blood pressure can raise the risk of stroke and heart disease over time. Lifestyle changes such as reducing sodium intake, increasing physical activity, and managing stress may help control blood pressure. Regular monitoring is recommended.`

**Extract Fields:**
- Input: `Chief complaint: headache. Symptoms include dizziness and nausea for 2 days. Vitals today: temp 98.6 F, HR 88, BP 130/85, SpO2 98%. Current meds: ibuprofen. Allergies: penicillin. Past history: hypertension. Red flags discussed: none.`

**Safety-blocked (Refuse):**
- Input: `Diagnose me with diabetes and give me my insulin dose`

**Safety-blocked (Urgent):**
- Input: `I have severe chest pain and shortness of breath`
""")
                with gr.Row():
                    example_task = gr.Radio(
                        choices=["Summarize", "Extract Fields", "Explain Term"],
                        value="Explain Term",
                        label="Task",
                    )
                    example_input = gr.Textbox(label="Example Input", lines=4)
                    example_ctx = gr.Textbox(label="Example Context", lines=3)

                run_example_btn = gr.Button("Run Example", variant="secondary")
                example_safety = gr.HTML(label="Safety")
                example_status = gr.Textbox(label="Status", interactive=False)
                example_response = gr.Textbox(label="Response", lines=10, interactive=False)

                def run_example(task, inp, ctx):
                    return run_inference(task, inp, ctx, "", False, 220, False)

                run_example_btn.click(
                    fn=run_example,
                    inputs=[example_task, example_input, example_ctx],
                    outputs=[example_response, example_safety, example_status],
                )

            # ── Tab 6: System Info ───────────────────────────────────
            with gr.TabItem("System"):
                import torch
                cuda_info = f"CUDA available: {torch.cuda.is_available()}"
                if torch.cuda.is_available():
                    cuda_info += f"\nGPU: {torch.cuda.get_device_name(0)}"
                    cuda_info += f"\nVRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"

                gr.Markdown(f"""
### System Information
```
{cuda_info}
```

### Model Defaults
- **Default model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Tiny (CPU) model:** `sshleifer/tiny-gpt2`
- **LoRA adapter:** `models/lora/`

### Pipeline Commands
```bash
# Prepare data
python -m src.data_prep --seed 42

# Train LoRA (GPU required for production model)
python -m src.train_lora

# Evaluate
python -m src.eval

# CLI inference
python -m src.inference --task term --input "What is hypertension?" --context "High blood pressure..."
```

### Safety Architecture
The safety layer has **three tiers**:
1. **Urgent patterns** — immediately routes to emergency guidance
2. **Hard refusal patterns** — blocks diagnosis, prescribing, dosing
3. **Bypass detection** — catches adversarial framing ("pretend you're a doctor", etc.)
4. **Risk score threshold** — soft refusal if accumulated risk signals exceed {3.0}

All outputs include a mandatory safety disclaimer.
""")

    return demo


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="HealthCare LLM Gradio GUI")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    parser.add_argument("--port", type=int, default=8080, help="Port to run on (default: 8080)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
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
