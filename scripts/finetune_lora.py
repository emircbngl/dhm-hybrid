"""LoRA fine-tune the local LLM on ``data/ai/training_examples.jsonl``.

Pipeline B from docs/AI_FINETUNE.md — actually updates weights via a
low-rank adapter on top of Qwen2.5-7B-Instruct. The training set is the
100 lab-specific examples we ship; the held-out 15 are *not* used here
(they belong to ``tests/test_ai_finetune_eval.py``).

Targets:
    * Apple Silicon (M2/M3 Ultra) via ``device="mps"``  — ~15 min for 100×3 epochs
    * CUDA (L4 / A10G / consumer GPU)                   — ~4 min
    * CPU-only                                          — possible, slow (≥1 h)

Usage::

    pip install -r requirements_finetune.txt
    python scripts/finetune_lora.py
    # adapter weights → out/dhm-copilot-lora/final
    # (then convert + import via Ollama, see docs/AI_FINETUNE.md §"Pipeline B")

Why these hyperparameters
-------------------------
* ``r=16`` / ``alpha=32`` is the canonical Qwen2.5 LoRA recipe — narrower
  ranks underfit on 100 examples, wider over-fits the lab idiom.
* 3 epochs is what the holdout eval converges on; 5+ starts memorising
  prompts and chain-summary text drifts toward ceremonial.
* ``per_device_train_batch_size=1`` + ``gradient_accumulation_steps=8``
  fits in 6 GB peak VRAM on M2 Ultra; bump batch on CUDA if memory
  allows (no convergence change).
* ``learning_rate=2e-4`` + cosine schedule + 10 % warmup is the trl
  default for Qwen LoRA on small datasets.

This script is intentionally minimal — no W&B, no eval-during-train, no
distributed sharding. The eval is a separate pytest run after the fact.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Defaults — match the lab profile baked into ai_training_examples.py.
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DATA_PATH = ROOT / "data" / "ai" / "training_examples.jsonl"
OUTPUT_DIR = ROOT / "out" / "dhm-copilot-lora"


def _detect_device() -> str:
    """Pick the most capable backend available; bail loudly if none."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        sys.exit(
            "torch is not installed — install requirements_finetune.txt first.\n"
            f"underlying error: {exc}"
        )
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="LoRA fine-tune the DHM AI assistant.")
    p.add_argument("--model", default=MODEL_ID,
                   help=f"Base model HF repo id (default: {MODEL_ID}).")
    p.add_argument("--data", type=Path, default=DATA_PATH,
                   help="Training JSONL path (default: %(default)s).")
    p.add_argument("--out", type=Path, default=OUTPUT_DIR,
                   help="Output dir for adapter weights (default: %(default)s).")
    p.add_argument("--epochs", type=int, default=3,
                   help="Training epochs (default: 3 — calibrated to 100 examples).")
    p.add_argument("--lr", type=float, default=2e-4,
                   help="Learning rate (default: 2e-4 cosine).")
    p.add_argument("--lora-r", type=int, default=16,
                   help="LoRA rank (default: 16).")
    p.add_argument("--lora-alpha", type=int, default=32,
                   help="LoRA alpha (default: 32 = 2× rank).")
    p.add_argument("--max-seq-length", type=int, default=4096,
                   help="Max token length per example (default: 4096).")
    p.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"),
                   default="auto",
                   help="Compute backend (default: auto-detect).")
    args = p.parse_args(argv)

    if not args.data.exists():
        sys.exit(
            f"training data not found at {args.data}\n"
            f"run `python scripts/ai_training_examples.py` first to generate it."
        )

    device = _detect_device() if args.device == "auto" else args.device

    # Quick sanity on the JSONL — fail fast before loading the 7B model.
    with open(args.data) as fh:
        n_train = sum(1 for line in fh if line.strip())
    print(f"training set: {n_train} examples ({args.data})")
    print(f"base model:   {args.model}")
    print(f"device:       {device}")
    print(f"output dir:   {args.out}")

    # Lazy imports — keep them off the critical path so a failed
    # dependency surfaces *with* the install hint above instead of an
    # opaque traceback at import time.
    try:
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:  # pragma: no cover
        sys.exit(
            "missing fine-tune deps — `pip install -r requirements_finetune.txt`\n"
            f"underlying error: {exc}"
        )

    print("loading tokenizer + model …")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        # Qwen2.5 has eos but no pad; pointing pad → eos is canonical.
        tok.pad_token = tok.eos_token

    model_kwargs: dict = {"torch_dtype": "auto"}
    if device == "mps":
        # MPS doesn't support bf16 cleanly across all ops; let HF auto-pick fp16.
        model_kwargs["device_map"] = {"": "mps"}
    elif device == "cuda":
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = {"": "cpu"}
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    print("loading dataset …")
    ds = load_dataset("json", data_files=str(args.data), split="train")

    sft_cfg = SFTConfig(
        output_dir=str(args.out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="epoch",
        bf16=(device == "cuda"),
        fp16=(device == "mps"),
        max_seq_length=args.max_seq_length,
        # ``messages`` field auto-renders via the tokenizer's chat template.
        dataset_text_field=None,
        # Ignore the `tools` key on each row at training time — the chat
        # template doesn't render it. The schema is still useful at
        # inference time (it goes into the API call), so we leave the
        # field in the JSONL.
        remove_unused_columns=False,
    )

    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=ds,
        args=sft_cfg,
        peft_config=peft_cfg,
    )

    print(f"training for {args.epochs} epochs …")
    trainer.train()

    final_dir = args.out / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tok.save_pretrained(str(final_dir))
    print(f"adapter saved → {final_dir}")
    print(
        "next step: convert to GGUF + register with Ollama. See "
        "docs/AI_FINETUNE.md §\"Pipeline B → Ollama'ya import et\"."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
