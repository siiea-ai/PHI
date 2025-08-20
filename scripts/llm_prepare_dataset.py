#!/usr/bin/env python3
"""
Prepare a tokenized dataset for causal language modeling (CLM) fine-tuning.

Usage:
  .venv/bin/python scripts/llm_prepare_dataset.py \
    --train data/train.jsonl --val data/val.jsonl \
    --model EleutherAI/gpt-neo-1.3B \
    --max-length 512 \
    --out out/llm_dataset

Input data format options per example:
  - {"text": "..."}
  - {"instruction": "...", "input": "...", "output": "..."}

This script converts each example to a single text string for CLM and tokenizes it.
"""
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer


def build_text(example):
    if "text" in example and example["text"]:
        return {"text": example["text"]}
    instruction = example.get("instruction", "").strip()
    inp = example.get("input", "").strip()
    out = example.get("output", "").strip()
    text = f"Instruction: {instruction}\nInput: {inp}\nOutput: {out}\n"
    return {"text": text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Path to train.jsonl")
    ap.add_argument("--val", required=True, help="Path to val.jsonl")
    ap.add_argument("--model", default="EleutherAI/gpt-neo-1.3B", help="Tokenizer model name")
    ap.add_argument("--max-length", type=int, default=1024, help="Max sequence length")
    ap.add_argument("--out", default="out/llm_dataset", help="Output dataset dir")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    raw = load_dataset(
        "json",
        data_files={"train": args.train, "validation": args.val},
    )
    raw = raw.map(build_text)

    def tok_map(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_length)

    cols = raw["train"].column_names
    tokd = raw.map(tok_map, batched=True, remove_columns=cols)
    tokd.save_to_disk(args.out)
    print(f"Saved tokenized dataset to {args.out}")


if __name__ == "__main__":
    main()
