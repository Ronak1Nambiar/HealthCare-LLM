from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Load .env if present (won't fail if python-dotenv is not installed)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
LORA_DIR = MODELS_DIR / "lora"
RAG_DIR = MODELS_DIR / "rag"
PROMPTS_DIR = ROOT / "prompts"
REPORTS_DIR = ROOT / "reports"


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ALTERNATE_MODELS = [
    "google/gemma-2-2b-it",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]
TINY_BASE_MODEL = "sshleifer/tiny-gpt2"


DISCLAIMERS = {
    "general": (
        "Safety disclaimer: This is general educational information, not medical advice. "
        "I cannot diagnose conditions, prescribe treatment, or provide medication dosing. "
        "Please talk to a licensed clinician."
    ),
    "urgent": (
        "Safety notice: This could be urgent. I cannot provide emergency medical advice. "
        "If there is severe pain, trouble breathing, chest pain, stroke symptoms, "
        "or risk of self-harm, seek emergency care now (call local emergency services)."
    ),
}


@dataclass
class TrainConfig:
    base_model: str = DEFAULT_BASE_MODEL
    output_dir: str = str(LORA_DIR)
    max_seq_len: int = 768
    learning_rate: float = 2e-4
    batch_size: int = 1
    grad_accum_steps: int = 8
    num_epochs: int = 1
    max_steps: int = -1
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    val_size: int = 200


def ensure_dirs() -> None:
    for directory in [DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR, LORA_DIR, RAG_DIR, PROMPTS_DIR, REPORTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def login_huggingface() -> bool:
    """Log in to Hugging Face using HF_TOKEN from environment/.env if available."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        return False
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        return True
    except Exception:
        return False


def get_base_model_name(tiny: bool = False, override: str | None = None) -> str:
    if override:
        return override
    return TINY_BASE_MODEL if tiny else DEFAULT_BASE_MODEL
