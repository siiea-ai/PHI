#!/usr/bin/env python3
"""
Phase 5A: Real GPT-2 PHI Training Validation

Test PHI training principles on actual GPT-2 models with real datasets.
"""

import os
import json
import time
import argparse
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, DataCollatorForLanguageModeling
)
from datasets import load_dataset, Dataset
import numpy as np
import sys
sys.path.append('.')

from phi.training import PHITrainingConfig, PHIMath
from phi.trainer import create_phi_trainer

def prepare_dataset(dataset_path: str, tokenizer, max_length: int = 256, max_examples: int = 500):
    """Prepare dataset for causal language modeling."""
    
    print(f"📊 Loading dataset from: {dataset_path}")
    
    # Load from JSONL
    examples = []
    with open(dataset_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_examples:
                break
            examples.append(json.loads(line.strip()))
    
    # Format for causal LM
    texts = []
    for ex in examples:
        if 'text' in ex:
            texts.append(ex['text'])
        else:
            # Fallback formatting
            texts.append(str(ex))
    
    print(f"   Loaded {len(texts)} examples")
    
    # Tokenize
    print("🔤 Tokenizing dataset...")
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

def run_baseline_training(
    model_name: str,
    train_dataset,
    eval_dataset,
    tokenizer,
    output_dir: str,
    epochs: int = 2,
    batch_size: int = 4,
    learning_rate: float = 2e-4
):
    """Run baseline training experiment."""
    
    print(f"🔄 Running baseline training...")
    
    # Load fresh model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=20,
        logging_steps=10,
        eval_steps=25,
        evaluation_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        dataloader_drop_last=False,
        remove_unused_columns=False,
        report_to=None,  # Disable wandb
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create trainer
    from transformers import Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    start_time = time.time()
    train_result = trainer.train()
    training_time = time.time() - start_time
    
    # Evaluate
    eval_result = trainer.evaluate()
    
    # Save results
    results = {
        'type': 'baseline',
        'model': model_name,
        'training_time': training_time,
        'train_loss': train_result.training_loss,
        'eval_loss': eval_result['eval_loss'],
        'steps': train_result.global_step,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }
    
    # Save model and results
    trainer.save_model()
    with open(os.path.join(output_dir, 'baseline_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Baseline training complete:")
    print(f"   Training Loss: {train_result.training_loss:.6f}")
    print(f"   Eval Loss: {eval_result['eval_loss']:.6f}")
    print(f"   Training Time: {training_time:.1f}s")
    
    return results

def run_phi_training(
    model_name: str,
    train_dataset,
    eval_dataset,
    tokenizer,
    output_dir: str,
    phi_config: PHITrainingConfig,
    epochs: int = 2,
    batch_size: int = 4
):
    """Run PHI training experiment."""
    
    print(f"🌟 Running PHI training...")
    
    # Load fresh model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=phi_config.base_learning_rate,
        warmup_steps=20,
        logging_steps=10,
        eval_steps=25,
        evaluation_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        dataloader_drop_last=False,
        remove_unused_columns=False,
        report_to=None,  # Disable wandb
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create PHI trainer
    phi_trainer = create_phi_trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        phi_config=phi_config
    )
    
    # Train
    start_time = time.time()
    train_result = phi_trainer.train()
    training_time = time.time() - start_time
    
    # Evaluate
    eval_result = phi_trainer.evaluate()
    
    # Get PHI analysis
    phi_analysis = phi_trainer.get_phi_analysis()
    
    # Save results
    results = {
        'type': 'phi',
        'model': model_name,
        'training_time': training_time,
        'train_loss': train_result.training_loss,
        'eval_loss': eval_result['eval_loss'],
        'steps': train_result.global_step,
        'epochs': epochs,
        'batch_size': batch_size,
        'phi_config': phi_config.__dict__,
        'phi_analysis': phi_analysis
    }
    
    # Save model and results
    phi_trainer.save_model()
    with open(os.path.join(output_dir, 'phi_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"✅ PHI training complete:")
    print(f"   Training Loss: {train_result.training_loss:.6f}")
    print(f"   Eval Loss: {eval_result['eval_loss']:.6f}")
    print(f"   Training Time: {training_time:.1f}s")
    
    return results

def compare_results(baseline_results, phi_results):
    """Compare baseline vs PHI training results."""
    
    print(f"\n📊 PHASE 5A RESULTS COMPARISON")
    print("=" * 50)
    
    # Basic metrics
    baseline_loss = baseline_results['train_loss']
    phi_loss = phi_results['train_loss']
    loss_improvement = baseline_loss - phi_loss
    
    baseline_eval = baseline_results['eval_loss']
    phi_eval = phi_results['eval_loss']
    eval_improvement = baseline_eval - phi_eval
    
    time_ratio = phi_results['training_time'] / baseline_results['training_time']
    
    print(f"Training Loss:")
    print(f"   Baseline: {baseline_loss:.6f}")
    print(f"   PHI:      {phi_loss:.6f}")
    print(f"   Improvement: {loss_improvement:+.6f} ({'✅ Better' if loss_improvement > 0 else '❌ Worse'})")
    
    print(f"\nEvaluation Loss:")
    print(f"   Baseline: {baseline_eval:.6f}")
    print(f"   PHI:      {phi_eval:.6f}")
    print(f"   Improvement: {eval_improvement:+.6f} ({'✅ Better' if eval_improvement > 0 else '❌ Worse'})")
    
    print(f"\nTraining Efficiency:")
    print(f"   Baseline Time: {baseline_results['training_time']:.1f}s")
    print(f"   PHI Time:      {phi_results['training_time']:.1f}s")
    print(f"   Time Ratio:    {time_ratio:.2f}x")
    
    # PHI-specific analysis
    if 'phi_analysis' in phi_results:
        phi_analysis = phi_results['phi_analysis']
        print(f"\n🌟 PHI Analysis:")
        if 'lr_decay_ratio' in phi_analysis:
            print(f"   LR Decay Ratio: {phi_analysis['lr_decay_ratio']:.3f}")
        if 'batch_progression' in phi_analysis:
            print(f"   Batch Progression: {phi_analysis['batch_progression']}")
        if 'phi_alignment' in phi_analysis:
            print(f"   PHI Alignment: {phi_analysis['phi_alignment']:.3f}")
    
    # Overall assessment
    print(f"\n🎯 PHASE 5A ASSESSMENT:")
    if loss_improvement > 0 and eval_improvement > 0:
        print("✅ PHI training SUCCESSFUL on real models!")
        print(f"   Training improvement: {loss_improvement:.6f}")
        print(f"   Evaluation improvement: {eval_improvement:.6f}")
    elif loss_improvement > 0 or eval_improvement > 0:
        print("⚠️ PHI training shows mixed results")
        print("   Consider parameter tuning for better performance")
    else:
        print("❌ PHI training needs optimization for this model/dataset")
        print("   Recommend adjusting PHI parameters")
    
    return {
        'loss_improvement': loss_improvement,
        'eval_improvement': eval_improvement,
        'time_ratio': time_ratio,
        'phi_better': loss_improvement > 0 and eval_improvement > 0
    }

def main():
    parser = argparse.ArgumentParser(description="Phase 5A: Real GPT-2 PHI Training")
    parser.add_argument("--model", default="gpt2", help="Model name")
    parser.add_argument("--dataset", default="Datasets/phi_test_subset/train.jsonl", help="Training dataset")
    parser.add_argument("--eval-dataset", default="Datasets/phi_test_subset/eval.jsonl", help="Eval dataset")
    parser.add_argument("--output-dir", default="./out/phase5a_real_training", help="Output directory")
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--max-examples", type=int, default=200, help="Max examples to use")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline training")
    parser.add_argument("--skip-phi", action="store_true", help="Skip PHI training")
    
    args = parser.parse_args()
    
    print("🚀 PHASE 5A: Real GPT-2 PHI Training Validation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    
    # Create output directories
    baseline_dir = Path(args.output_dir) / "baseline"
    phi_dir = Path(args.output_dir) / "phi"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    phi_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    print(f"\n📥 Loading tokenizer and datasets...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare datasets
    train_dataset = prepare_dataset(args.dataset, tokenizer, max_examples=args.max_examples)
    eval_dataset = prepare_dataset(args.eval_dataset, tokenizer, max_examples=args.max_examples//4)
    
    results = {}
    
    # Run baseline training
    if not args.skip_baseline:
        baseline_results = run_baseline_training(
            args.model, train_dataset, eval_dataset, tokenizer,
            str(baseline_dir), args.epochs, args.batch_size
        )
        results['baseline'] = baseline_results
    
    # Run PHI training
    if not args.skip_phi:
        # Create optimized PHI config for real training
        phi_config = PHITrainingConfig(
            base_learning_rate=3e-4,
            lr_schedule_mode="phi_decay",
            phi_lr_power=0.8,
            base_batch_size=args.batch_size,
            phi_batch_progression=True,
            max_batch_size=min(args.batch_size * 2, 16),
            batch_phi_phases=2,
            phi_training_phases=True,
            base_dropout=0.1,
            phi_dropout_schedule=True,
            base_weight_decay=0.01,
            phi_weight_decay_schedule=True
        )
        
        phi_results = run_phi_training(
            args.model, train_dataset, eval_dataset, tokenizer,
            str(phi_dir), phi_config, args.epochs, args.batch_size
        )
        results['phi'] = phi_results
    
    # Compare results
    if not args.skip_baseline and not args.skip_phi:
        comparison = compare_results(results['baseline'], results['phi'])
        results['comparison'] = comparison
        
        # Save comprehensive results
        with open(Path(args.output_dir) / "phase5a_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {Path(args.output_dir) / 'phase5a_results.json'}")
    
    print(f"\n🎉 Phase 5A Complete!")
    return results

if __name__ == "__main__":
    results = main()
