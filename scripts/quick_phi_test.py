#!/usr/bin/env python3
"""
Quick PHI training test using our prepared dataset.
"""

import os
import json
import time
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from datasets import Dataset
import sys
sys.path.append('.')

from phi.training import PHITrainingConfig, PHIMath
from phi.trainer import create_phi_trainer

def load_jsonl_dataset(file_path: str, max_examples: int = 100):
    """Load dataset from JSONL file."""
    examples = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_examples:
                break
            examples.append(json.loads(line.strip()))
    return examples

def tokenize_dataset(examples, tokenizer, max_length=256):
    """Tokenize dataset for causal LM."""
    texts = []
    for ex in examples:
        if 'text' in ex:
            texts.append(ex['text'])
        else:
            # Fallback formatting
            texts.append(str(ex))
    
    # Tokenize
    tokenized = tokenizer(
        texts, 
        truncation=True, 
        padding=False, 
        max_length=max_length,
        return_tensors=None
    )
    
    # Create dataset
    dataset = Dataset.from_dict({
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask']
    })
    
    return dataset

def run_quick_experiment():
    """Run a quick PHI vs baseline comparison."""
    
    print("🚀 Starting Quick PHI Experiment")
    
    # Setup
    model_name = "gpt2"
    train_file = "Datasets/phi_test_subset/train.jsonl"
    eval_file = "Datasets/phi_test_subset/eval.jsonl"
    
    # Load tokenizer and model
    print(f"📥 Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    print("📊 Loading datasets...")
    train_examples = load_jsonl_dataset(train_file, max_examples=50)  # Small for quick test
    eval_examples = load_jsonl_dataset(eval_file, max_examples=20)
    
    print(f"   Train examples: {len(train_examples)}")
    print(f"   Eval examples: {len(eval_examples)}")
    
    # Tokenize
    train_dataset = tokenize_dataset(train_examples, tokenizer)
    eval_dataset = tokenize_dataset(eval_examples, tokenizer)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM
    )
    
    results = {}
    
    # === BASELINE EXPERIMENT ===
    print("\n🔄 Running Baseline Experiment...")
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    baseline_args = TrainingArguments(
        output_dir="./out/quick_baseline",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-4,
        warmup_steps=10,
        logging_steps=5,
        eval_steps=10,
        evaluation_strategy="steps",
        save_steps=20,
        save_total_limit=1,
        dataloader_drop_last=False,
        remove_unused_columns=False,
    )
    
    baseline_trainer = Trainer(
        model=model,
        args=baseline_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    baseline_start = time.time()
    baseline_result = baseline_trainer.train()
    baseline_time = time.time() - baseline_start
    
    baseline_eval = baseline_trainer.evaluate()
    
    results['baseline'] = {
        'training_time': baseline_time,
        'final_loss': baseline_result.training_loss,
        'eval_loss': baseline_eval['eval_loss'],
        'steps': baseline_result.global_step
    }
    
    print(f"✅ Baseline complete: {baseline_time:.1f}s, Loss: {baseline_result.training_loss:.4f}")
    
    # === PHI EXPERIMENT ===
    print("\n🌟 Running PHI Experiment...")
    
    # Fresh model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # PHI configuration
    phi_config = PHITrainingConfig(
        base_learning_rate=2e-4,
        lr_schedule_mode="phi_decay",
        base_batch_size=4,
        phi_batch_progression=True,
        phi_training_phases=True,
        phi_dropout_schedule=True,
        phi_weight_decay_schedule=True
    )
    
    phi_args = TrainingArguments(
        output_dir="./out/quick_phi",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-4,
        warmup_steps=10,
        logging_steps=5,
        eval_steps=10,
        evaluation_strategy="steps",
        save_steps=20,
        save_total_limit=1,
        dataloader_drop_last=False,
        remove_unused_columns=False,
    )
    
    phi_trainer = create_phi_trainer(
        model=model,
        args=phi_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        phi_config=phi_config
    )
    
    phi_start = time.time()
    phi_result = phi_trainer.train()
    phi_time = time.time() - phi_start
    
    phi_eval = phi_trainer.evaluate()
    
    results['phi'] = {
        'training_time': phi_time,
        'final_loss': phi_result.training_loss,
        'eval_loss': phi_eval['eval_loss'],
        'steps': phi_result.global_step
    }
    
    print(f"✅ PHI complete: {phi_time:.1f}s, Loss: {phi_result.training_loss:.4f}")
    
    # === COMPARISON ===
    print(f"\n📊 EXPERIMENT RESULTS:")
    print(f"   Baseline - Time: {results['baseline']['training_time']:.1f}s, Loss: {results['baseline']['final_loss']:.4f}, Eval: {results['baseline']['eval_loss']:.4f}")
    print(f"   PHI      - Time: {results['phi']['training_time']:.1f}s, Loss: {results['phi']['final_loss']:.4f}, Eval: {results['phi']['eval_loss']:.4f}")
    
    # Calculate improvements
    loss_improvement = results['baseline']['final_loss'] - results['phi']['final_loss']
    eval_improvement = results['baseline']['eval_loss'] - results['phi']['eval_loss']
    time_ratio = results['phi']['training_time'] / results['baseline']['training_time']
    
    print(f"\n🎯 PHI vs Baseline:")
    print(f"   Training Loss Improvement: {loss_improvement:+.6f} ({'better' if loss_improvement > 0 else 'worse'})")
    print(f"   Eval Loss Improvement: {eval_improvement:+.6f} ({'better' if eval_improvement > 0 else 'worse'})")
    print(f"   Time Ratio (PHI/Baseline): {time_ratio:.2f}x")
    
    # Save results
    results['comparison'] = {
        'loss_improvement': loss_improvement,
        'eval_improvement': eval_improvement,
        'time_ratio': time_ratio,
        'phi_better_training': loss_improvement > 0,
        'phi_better_eval': eval_improvement > 0
    }
    
    # Save to file
    output_dir = Path("./out/quick_phi_experiment")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_dir / 'results.json'}")
    
    return results

if __name__ == "__main__":
    results = run_quick_experiment()
