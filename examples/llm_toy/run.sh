#!/usr/bin/env bash
set -euo pipefail

# Toy end-to-end LLM fine-tune demo using small model for quick run.
# - Prepares tokenized dataset from examples/llm_toy/*.jsonl
# - Fine-tunes with LoRA for 1 short epoch
# - Evaluates perplexity on the validation split

ROOT_DIR="$(cd "$(dirname "$0")"/.. && pwd)"
PY_BIN="${PY_BIN:-python3}"

BASE="EleutherAI/gpt-neo-125M"
DATA_IN_TRAIN="$ROOT_DIR/examples/llm_toy/train.jsonl"
DATA_IN_VAL="$ROOT_DIR/examples/llm_toy/valid.jsonl"
DATA_OUT="$ROOT_DIR/out/llm_toy_dataset"
LORA_OUT="$ROOT_DIR/out/llm_toy_lora"

echo "[1/3] Preparing tokenized dataset -> $DATA_OUT"
"$PY_BIN" "$ROOT_DIR/scripts/llm_prepare_dataset.py" \
  --train "$DATA_IN_TRAIN" \
  --val   "$DATA_IN_VAL" \
  --model "$BASE" \
  --max-length 256 \
  --out "$DATA_OUT"

echo "[2/3] Fine-tuning (LoRA) -> $LORA_OUT"
"$PY_BIN" "$ROOT_DIR/scripts/llm_finetune_lora.py" \
  --base "$BASE" \
  --data "$DATA_OUT" \
  --out  "$LORA_OUT" \
  --epochs 1 \
  --lr 2e-4 \
  --warmup 0.0618 \
  --rank 8 \
  --alpha 16 \
  --dropout 0.05 \
  --per-device-batch 1 \
  --grad-accum 2 \
  --eval-steps 20 \
  --save-steps 20

echo "[3/3] Evaluating perplexity"
"$PY_BIN" "$ROOT_DIR/scripts/llm_eval_ppl.py" \
  --base "$BASE" \
  --adapter "$LORA_OUT/final" \
  --data "$DATA_OUT"

echo "Done. Adapter saved under: $LORA_OUT/final"
