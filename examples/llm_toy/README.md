# Toy LLM Fine-Tune Demo

This tiny demo validates the PHI + Transformers + PEFT pipeline on a very small dataset. It fine-tunes a small base model to keep runtime/memory modest.

## Files
- `train.jsonl`, `valid.jsonl`: tiny instruction-tuning dataset.
- `run.sh`: end-to-end script to prepare data, fine-tune with LoRA, and evaluate perplexity.

## Requirements
- A Python venv with project requirements plus LLM stack:
  ```bash
  python3 -m venv .venv
  .venv/bin/pip install -U pip setuptools wheel
  .venv/bin/pip install "transformers>=4.42" "datasets>=2.20" "accelerate>=0.30" "peft>=0.11" evaluate sentencepiece
  ```

## Run
```bash
bash examples/llm_toy/run.sh
```

Notes:
- Uses `EleutherAI/gpt-neo-125M` to run quickly.
- Adjust sequence length, epochs, and batch settings in `run.sh` if needed.
- For the full interactive wizard, try:
  ```bash
  python3 -m phi.cli llm wizard
  ```
