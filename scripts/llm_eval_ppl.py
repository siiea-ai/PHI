#!/usr/bin/env python3
"""
Evaluate perplexity on a validation split for a LoRA-adapted model.

Example:
  .venv/bin/python scripts/llm_eval_ppl.py \
    --base EleutherAI/gpt-neo-1.3B \
    --adapter out/lora-neo-1.3b/final \
    --data out/llm_dataset
"""
import argparse
import math
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import PeftModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="EleutherAI/gpt-neo-1.3B", help="Base pretrained model")
    ap.add_argument("--adapter", required=True, help="LoRA adapter directory (from Trainer.save_model)")
    ap.add_argument("--data", default="out/llm_dataset", help="Tokenized dataset dir (load_from_disk)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(args.base)
    model = PeftModel.from_pretrained(base, args.adapter)

    ds = load_from_disk(args.data)
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    targs = TrainingArguments(output_dir="out/eval_tmp", per_device_eval_batch_size=1)
    trainer = Trainer(model=model, args=targs, eval_dataset=ds["validation"], data_collator=collator)

    metrics = trainer.evaluate()
    ppl = math.exp(metrics["eval_loss"]) if "eval_loss" in metrics else float("nan")
    print({"perplexity": ppl, **metrics})


if __name__ == "__main__":
    main()
