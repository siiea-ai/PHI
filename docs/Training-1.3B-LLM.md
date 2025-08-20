# Training a ~1.3B LLM with the PHI Project (via Transformers + PEFT)

This guide shows a realistic, reproducible path to train or fine‑tune an ~1.3B parameter language model on your own dataset using the Hugging Face ecosystem (Transformers, Datasets, Accelerate, PEFT). The PHI repository itself does not implement an LLM trainer; instead, use these standard tools and keep PHI for analysis, data transforms, and experiment orchestration.

Important: Full training from scratch at 1.3B is compute‑intensive. We strongly recommend parameter‑efficient fine‑tuning (LoRA/QLoRA) on a pretrained base such as `EleutherAI/gpt-neo-1.3B` or `EleutherAI/pythia-1.4b`.

## 0) Hardware expectations

- Fine‑tuning (LoRA/QLoRA):
  - Single 24–48 GB GPU (e.g., RTX 3090/4090, A5000, A6000) is typically sufficient.
  - Multi‑GPU or gradient accumulation helps larger sequence lengths and batch sizes.
  - Linux preferred; macOS MPS works for smaller runs but lacks bitsandbytes.
- From‑scratch pretraining: multiple high‑end GPUs and substantial time/budget.

## 1) Environment setup

Create a project‑local venv (see `docs/Quickstart.md` for PHI base setup), then add training deps:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip setuptools wheel

# Core training stack
.venv/bin/pip install \
  "transformers>=4.42" "datasets>=2.20" "accelerate>=0.30" \
  "peft>=0.11" "evaluate" "sentencepiece" "wandb"

# Optional: QLoRA (Linux): 8‑bit/4‑bit quantization
.venv/bin/pip install "bitsandbytes>=0.43"  # Linux CUDA only
```

If you plan to launch distributed runs, run `accelerate config` once:

```bash
.venv/bin/accelerate config
```

### Wizard: interactive setup

Prefer a guided flow? Use the CLI wizard:

```bash
python -m phi.cli llm wizard
```

It will:
- Ask for model size and curated base models to choose from.
- Optionally download tokenizer and base weights into `models/`.
- Ingest your dataset (JSONL). If a single file is provided, it applies a φ 61.8/38.2 train/val split into `datasets/raw/<project>/`.
- Tokenize into a ready dataset at `datasets/ready/<project>/`.
- Suggest golden-heuristic hyperparameters and optionally start a LoRA run under `runs/<project>/`.

Non-interactive mode (no prompts):

```bash
python -m phi.cli llm wizard \
  --auto \
  --size 125m \
  --base EleutherAI/gpt-neo-125M \
  --project toy_phi \
  --dataset examples/llm_toy/train.jsonl \
  --tokenize --train --eval \
  --epochs 1 --rank 8 --warmup 0.0618
```

Multi-dataset φ-mix (interleave sources by φ weights 0.618, 0.382, 0.236, …):

```bash
python -m phi.cli llm wizard \
  --auto --base EleutherAI/gpt-neo-125M --project mix_phi \
  --dataset data/domain_a.jsonl \
  --dataset data/domain_b.jsonl \
  --phi-mix --tokenize --train
```

Common flags:
- `--download-tokenizer/--no-download-tokenizer` (default: on)
- `--download-weights/--no-download-weights` (default: off)
- `--tokenize/--no-tokenize`, `--train/--no-train`, `--eval/--no-eval`
- Hyperparams: `--epochs`, `--lr`, `--warmup`, `--rank`, `--alpha`, `--dropout`, `--per-device-batch`, `--grad-accum`, `--precision`

## 2) Prepare your dataset

Two common formats:

- Plain text files (one example per line)
- Instruction‑tuning JSONL with fields like `{ "instruction", "input", "output" }`

Example loader/tokenizer (causal LM):

```python
# Provided script: scripts/llm_prepare_dataset.py
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

MODEL_NAME = "EleutherAI/gpt-neo-1.3B"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
TOKENIZER.pad_token = TOKENIZER.eos_token  # causal LM

# Option A: CSV/JSONL with a "text" or combined fields
def build_text(example):
    if "text" in example:
        return {"text": example["text"]}
    # For instruction tuning
    instruction = example.get("instruction", "")
    inp = example.get("input", "")
    out = example.get("output", "")
    text = f"Instruction: {instruction}\nInput: {inp}\nOutput: {out}\n"
    return {"text": text}

# Replace with your path or hf dataset name
raw = load_dataset("json", data_files={"train": "data/train.jsonl", "validation": "data/val.jsonl"})
raw = raw.map(build_text)

def tok(batch):
    return TOKENIZER(batch["text"], truncation=True, max_length=1024)

tokd = raw.map(tok, batched=True, remove_columns=raw["train"].column_names)

tokd.save_to_disk("out/llm_dataset")
print("Saved tokenized dataset to out/llm_dataset")
```

Run:

```bash
.venv/bin/python scripts/llm_prepare_dataset.py \
  --train data/train.jsonl --val data/val.jsonl \
  --model EleutherAI/gpt-neo-1.3B --max-length 1024 \
  --out out/llm_dataset
```

## 3) Choose a pretrained base

Recommended 1.3–1.4B bases on Hugging Face Hub:

- `EleutherAI/gpt-neo-1.3B`
- `EleutherAI/pythia-1.4b`

Using pretrained models dramatically reduces compute vs training from scratch.

### How to choose

- Compatibility: tokenizer/vocab that matches your language/domain, and context length sufficient for your prompts.
- Licensing & usage: verify model license fits your project.
- Training data & capabilities: prefer bases closer to your domain; broader pretraining is safer if unsure.
- Hardware fit: 1.3–1.4B is feasible on a single 24–48 GB GPU; QLoRA helps further.
- Ecosystem maturity: look for community reports, evals, and good tokenizer behavior.

If you’re targeting instruction-like tasks, it’s fine to start from a base LM and instruction-tune on your data. If a well-maintained instruction‑tuned 1.3–1.4B exists for your needs, you can start there and still add LoRA adapters.

### Task specialization tips

- Instruction/chat style: format examples consistently (e.g., fields `instruction`, `input`, `output`). See the example formatter in §2.
- Domain adaptation: keep your output style stable; mix multiple sources with `--phi-mix` to interleave domains proportionally.
- Classification/regression: consider sequence‑classification heads instead of causal LM fine‑tuning, or use prompting with a generative LM; choose based on deployment needs.
- Summarization/QA: ensure prompts include clear task delimiters; monitor length truncation at tokenization.

## 4) Parameter‑efficient fine‑tuning (LoRA/QLoRA)

LoRA injects small trainable adapters and freezes the base weights.

```python
# Reference implementation (similar to scripts/llm_finetune_lora.py)
from datasets import load_from_disk
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME = "EleutherAI/gpt-neo-1.3B"
DATA_PATH = "out/llm_dataset"

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,            # rank (try 8/16/32)
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "fc_in", "fc_out"],  # adapt to arch
)

model = get_peft_model(model, lora_cfg)

# Data
ds = load_from_disk(DATA_PATH)
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

args = TrainingArguments(
    output_dir="out/lora-neo-1.3b",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,   # increase effective batch
    num_train_epochs=3,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.0618,             # optional golden warmup (see § Golden heuristics)
    weight_decay=0.1,
    logging_steps=25,
    evaluation_strategy="steps",
    eval_steps=200,
    save_steps=200,
    save_total_limit=3,
    bf16=True,                       # if supported; else use fp16=True
    report_to=["none"],              # or ["wandb"]
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds.get("validation"),
    data_collator=collator,
)

trainer.train()
trainer.save_model("out/lora-neo-1.3b/final")
```

Run:

```bash
.venv/bin/python scripts/llm_finetune_lora.py
```

Notes:
- On Linux with CUDA + `bitsandbytes`, you can load the base in 8‑bit and fine‑tune adapters (QLoRA) to fit larger batches.
- Adjust `target_modules` to match your architecture (e.g., `gpt_neox` uses different module names like `attention.query_key_value`).

## 5) Evaluation

Compute perplexity on validation set:

```python
# Provided script: scripts/llm_eval_ppl.py
import math
from datasets import load_from_disk
from transformers import (AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)
from peft import PeftModel

MODEL_BASE = "EleutherAI/gpt-neo-1.3B"
ADAPTER_DIR = "out/lora-neo-1.3b/final"
DATA_PATH = "out/llm_dataset"

tok = AutoTokenizer.from_pretrained(MODEL_BASE, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

base = AutoModelForCausalLM.from_pretrained(MODEL_BASE)
model = PeftModel.from_pretrained(base, ADAPTER_DIR)

collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
ds = load_from_disk(DATA_PATH)

args = TrainingArguments(output_dir="out/eval_tmp", per_device_eval_batch_size=1)
trainer = Trainer(model=model, args=args, eval_dataset=ds["validation"], data_collator=collator)

metrics = trainer.evaluate()
ppl = math.exp(metrics["eval_loss"]) if "eval_loss" in metrics else float("nan")
print({"perplexity": ppl, **metrics})
```

Run:

```bash
.venv/bin/python scripts/llm_eval_ppl.py \
  --base EleutherAI/gpt-neo-1.3B \
  --adapter out/lora-neo-1.3b/final \
  --data out/llm_dataset
```

## 5.1) Monitoring and hyperparameter tuning

- Enable experiment tracking (e.g., Weights & Biases): set `report_to=["wandb"]` in `TrainingArguments`, `pip install wandb`, and `wandb login`. Track loss, eval loss, learning rate, and gradients.
- Adjust over time:
  - Learning rate: start at 2e-4; scan down if unstable, up if underfitting.
  - Warmup: `warmup_ratio=0.0618` (φ^{-3}) is a solid default; try 0.0382–0.1 ranges.
  - LoRA: scan rank in {8, 16, 32}; keep dropout near 0.05 unless overfitting.
  - Effective batch: increase via gradient accumulation; keep GPU mem in check.
  - Early stopping: stop when eval loss plateaus; save best checkpoints.

## 6) Inference

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
model = PeftModel.from_pretrained(base, "out/lora-neo-1.3b/final")
tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B", use_fast=True)

prompt = "Write a short poem about golden ratios."
ids = tok(prompt, return_tensors="pt").input_ids
out = model.generate(ids, max_new_tokens=128, do_sample=True, temperature=0.9)
print(tok.decode(out[0], skip_special_tokens=True))
```

## 7) (Optional) Golden ratio heuristics

PHI explores φ‑inspired heuristics; apply carefully and validate empirically:

- Warmup ratio near φ^{-3} ≈ 0.0618 (`TrainingArguments.warmup_ratio=0.0618`).
- Cosine schedule with restart periods scaled by φ (for long runs with restarts).
- LoRA rank search around φ‑multiples (e.g., r ∈ {8, 13, 21, 34}).
- Keep head dimension near 64; for custom architectures, depth:width decisions can be scanned around φ‑proportions.

These are heuristics, not guarantees—measure with validation loss and task metrics.

Example (Trainer) using φ warmup and cosine with hard restarts:

```python
args = TrainingArguments(
    ...,
    lr_scheduler_type="cosine_with_restarts",  # script defaults to "cosine"
    warmup_ratio=0.0618,
)
# Note: if you need fine‑grained control over restart periods/cycles,
# consider a custom scheduler or phase your training into multiple runs.
```

## 8) Reproducibility

- Set `seed` in `TrainingArguments` and control data shuffling.
- Log `transformers`, `datasets`, `accelerate`, CUDA/cudnn versions.
- Save tokenizer and adapters alongside checkpoints.

## 9) Common pitfalls

- Out‑of‑memory: lower sequence length, increase gradient accumulation, use 8‑bit loading, enable gradient checkpointing.
- Mismatch `target_modules`: verify module names with `model.named_modules()`.
- Mac without CUDA: bitsandbytes/QLoRA not available; standard LoRA with small batches recommended.

## 10) How PHI fits in

- Use PHI’s data transforms (e.g., CSV preprocessing via `phi.cli transform`) before tokenization.
- Track experiments and metrics alongside PHI’s other simulations for unified analysis.
- Integrate generated text or metrics back into PHI pipelines for downstream experiments.

---

If you want a minimal, runnable demo (tiny dataset, short sequence length) for quick validation, let us know and we’ll add a self‑contained example under `examples/`.
