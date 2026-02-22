from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from .config import LORA_DIR, PROCESSED_DIR, TrainConfig, ensure_dirs, get_base_model_name
from .reporting import build_report, update_run_state


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def format_example(row: dict[str, Any]) -> str:
    return (
        "### Instruction:\n"
        f"{row['instruction']}\n\n"
        "### Input:\n"
        f"{row['input']}\n\n"
        "### Response:\n"
        f"{row['output']}"
    )


def detect_target_modules(model: torch.nn.Module) -> list[str]:
    candidates = set()
    for name, module in model.named_modules():
        lname = name.lower()
        if any(k in lname for k in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
            leaf = name.split(".")[-1]
            candidates.add(leaf)
    if not candidates:
        return ["c_attn", "c_proj"]
    return sorted(candidates)


def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer, max_len: int) -> Dataset:
    def _tok(batch: dict[str, list[str]]) -> dict[str, Any]:
        encoded = tokenizer(batch["text"], truncation=True, max_length=max_len, padding=False)
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    return dataset.map(_tok, batched=True, remove_columns=dataset.column_names)


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA training for Healthcare LLM Week 5")
    parser.add_argument("--tiny", action="store_true", help="Use tiny model/settings for fast CPU checks")
    parser.add_argument("--base_model", type=str, default=None, help="Override base model name")
    parser.add_argument("--train_file", type=str, default=str(PROCESSED_DIR / "train.jsonl"))
    parser.add_argument("--val_file", type=str, default=str(PROCESSED_DIR / "val.jsonl"))
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    ensure_dirs()
    cfg = TrainConfig()
    cfg.base_model = get_base_model_name(tiny=args.tiny, override=args.base_model)
    use_cuda = torch.cuda.is_available()
    if not use_cuda and not args.tiny and args.base_model is None:
        cfg.base_model = get_base_model_name(tiny=True, override=None)
        print(f"CPU detected. Auto-switching base model to tiny fallback: {cfg.base_model}")

    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    elif args.tiny:
        cfg.max_steps = 20

    if args.max_seq_len is not None:
        cfg.max_seq_len = args.max_seq_len
    elif args.tiny:
        cfg.max_seq_len = 384

    if args.lr is not None:
        cfg.learning_rate = args.lr

    if args.tiny:
        cfg.grad_accum_steps = 2

    train_rows = load_jsonl(Path(args.train_file))
    val_rows = load_jsonl(Path(args.val_file))

    if args.tiny:
        train_rows = train_rows[: min(128, len(train_rows))]
        val_rows = val_rows[: min(32, len(val_rows))]

    train_texts = [format_example(r) for r in train_rows]
    val_texts = [format_example(r) for r in val_rows]

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = None
    model_kwargs: dict[str, Any] = {"trust_remote_code": True}

    if use_cuda:
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model_kwargs.update({"quantization_config": bnb_config, "device_map": "auto"})
    else:
        model_kwargs.update({"torch_dtype": torch.float32})

    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **model_kwargs)
    if use_cuda:
        model = prepare_model_for_kbit_training(model)

    target_modules = detect_target_modules(model)
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    train_ds = Dataset.from_dict({"text": train_texts})
    val_ds = Dataset.from_dict({"text": val_texts})

    train_tok = tokenize_dataset(train_ds, tokenizer, cfg.max_seq_len)
    val_tok = tokenize_dataset(val_ds, tokenizer, cfg.max_seq_len)

    args_out = LORA_DIR
    args_out.mkdir(parents=True, exist_ok=True)

    fp16 = use_cuda and not torch.cuda.is_bf16_supported()
    bf16 = use_cuda and torch.cuda.is_bf16_supported()

    arg_names = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    eval_key = "evaluation_strategy" if "evaluation_strategy" in arg_names else "eval_strategy"

    ta_kwargs: dict[str, Any] = {
        "output_dir": str(args_out),
        "per_device_train_batch_size": cfg.batch_size,
        "per_device_eval_batch_size": cfg.batch_size,
        "gradient_accumulation_steps": cfg.grad_accum_steps,
        "learning_rate": cfg.learning_rate,
        "num_train_epochs": cfg.num_epochs,
        "max_steps": cfg.max_steps,
        "logging_steps": 5,
        "eval_steps": 20,
        eval_key: "steps",
        "save_strategy": "steps",
        "save_steps": 20,
        "save_total_limit": 2,
        "report_to": "none",
        "fp16": fp16,
        "bf16": bf16,
        "dataloader_num_workers": 0,
        "remove_unused_columns": False,
    }

    train_args = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    print(f"Training with model={cfg.base_model}, cuda={use_cuda}, max_steps={cfg.max_steps}, seq_len={cfg.max_seq_len}")
    result = trainer.train()
    trainer.save_model(str(args_out))
    tokenizer.save_pretrained(str(args_out))

    with (args_out / "train_meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "base_model": cfg.base_model,
                "max_steps": cfg.max_steps,
                "max_seq_len": cfg.max_seq_len,
                "lora_r": cfg.lora_r,
                "lora_alpha": cfg.lora_alpha,
                "learning_rate": cfg.learning_rate,
            },
            f,
            indent=2,
        )
    print(f"Saved LoRA adapter and metadata to {args_out}")
    update_run_state(
        "train_lora",
        {
            "tiny": args.tiny,
            "base_model": cfg.base_model,
            "max_steps": cfg.max_steps,
            "max_seq_len": cfg.max_seq_len,
            "learning_rate": cfg.learning_rate,
            "train_samples": len(train_rows),
            "val_samples": len(val_rows),
            "train_runtime": getattr(result, "metrics", {}).get("train_runtime"),
            "train_loss": getattr(result, "metrics", {}).get("train_loss"),
        },
    )
    build_report(trigger="train_lora")


if __name__ == "__main__":
    main()
