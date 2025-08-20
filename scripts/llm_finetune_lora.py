#!/usr/bin/env python3
"""
LoRA fine-tuning for ~1.3B causal LMs with Hugging Face Trainer.

Example:
  .venv/bin/python scripts/llm_finetune_lora.py \
    --base EleutherAI/gpt-neo-1.3B \
    --data out/llm_dataset \
    --out out/lora-neo-1.3b \
    --epochs 3 --lr 2e-4 --warmup 0.0618 \
    --rank 16 --dropout 0.05 \
    --per-device-batch 1 --grad-accum 8 --bf16

Depends on: transformers, datasets, peft (and bitsandbytes for QLoRA if used).
"""
import argparse
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="EleutherAI/gpt-neo-1.3B", help="Pretrained base model")
    ap.add_argument("--data", default="out/llm_dataset", help="Tokenized dataset dir (load_from_disk)")
    ap.add_argument("--out", default="out/lora-out", help="Output directory")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=float, default=0.0618, help="Warmup ratio")
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--per-device-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.base)

    # LoRA config; adjust target modules per architecture if needed
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "fc_in", "fc_out", "dense", "proj"
        ],
    )
    model = get_peft_model(model, lora_cfg)

    ds = load_from_disk(args.data)
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    train_args = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.per_device_batch,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=args.weight_decay,
        logging_steps=25,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=["none"],
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation"),
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(f"{args.out}/final")
    print(f"Saved LoRA adapter to {args.out}/final")


if __name__ == "__main__":
    main()
