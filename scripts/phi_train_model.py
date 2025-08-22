#!/usr/bin/env python3
"""
PHI-based model training script.

Complete training pipeline using golden ratio principles throughout
the training process.
"""

import argparse
import os
import json
from pathlib import Path
from typing import Optional

import torch
from datasets import load_from_disk, load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling,
    TrainingArguments, set_seed
)

from phi.training import PHITrainingConfig, validate_phi_config
from phi.trainer import PHITrainer, create_phi_trainer
from phi.hf_integration import create_phi_training_args


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="PHI-based model training")
    
    # Model and data arguments
    parser.add_argument("--model", default="gpt2", help="Base model name or path")
    parser.add_argument("--data", required=True, help="Training dataset path or name")
    parser.add_argument("--eval-data", help="Evaluation dataset path (optional)")
    parser.add_argument("--tokenizer", help="Tokenizer name (defaults to model)")
    parser.add_argument("--output-dir", default="./phi_training_output", help="Output directory")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    
    # PHI configuration
    parser.add_argument("--phi-lr", type=float, default=2e-4, help="Base learning rate")
    parser.add_argument("--phi-lr-schedule", default="phi_decay", 
                       choices=["phi_decay", "phi_cosine", "phi_cyclic"],
                       help="PHI learning rate schedule")
    parser.add_argument("--phi-batch-size", type=int, default=8, help="Base batch size")
    parser.add_argument("--phi-batch-progression", action="store_true", 
                       help="Enable PHI batch size progression")
    parser.add_argument("--phi-max-batch", type=int, default=128, help="Maximum batch size")
    parser.add_argument("--phi-training-phases", action="store_true",
                       help="Enable PHI training phases")
    parser.add_argument("--phi-dropout-schedule", action="store_true",
                       help="Enable PHI dropout scheduling")
    parser.add_argument("--phi-weight-decay-schedule", action="store_true", 
                       help="Enable PHI weight decay scheduling")
    parser.add_argument("--phi-warmup-epochs", type=int, default=0,
                       help="Number of warmup epochs")
    
    # Standard training arguments
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    
    # Evaluation and logging
    parser.add_argument("--eval-steps", type=int, default=500, help="Evaluation frequency")
    parser.add_argument("--save-steps", type=int, default=500, help="Save frequency")
    parser.add_argument("--logging-steps", type=int, default=50, help="Logging frequency")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation")
    
    return parser.parse_args()


def load_and_prepare_dataset(data_path: str, tokenizer, max_length: int = 512):
    """Load and prepare dataset for training."""
    print(f"Loading dataset from: {data_path}")
    
    # Try to load as pre-tokenized dataset first
    try:
        if os.path.isdir(data_path):
            dataset = load_from_disk(data_path)
            print(f"Loaded pre-tokenized dataset: {dataset}")
            return dataset
    except Exception as e:
        print(f"Could not load as pre-tokenized dataset: {e}")
    
    # Try to load as HuggingFace dataset
    try:
        if data_path.endswith('.jsonl') or data_path.endswith('.json'):
            dataset = load_dataset('json', data_files=data_path)['train']
        else:
            dataset = load_dataset(data_path)
            if isinstance(dataset, dict):
                dataset = dataset['train']
        
        print(f"Loaded raw dataset: {dataset}")
        
        # Tokenize the dataset
        def tokenize_function(examples):
            # Assume text field exists, adjust as needed
            text_field = 'text' if 'text' in examples else list(examples.keys())[0]
            return tokenizer(
                examples[text_field],
                truncation=True,
                padding=False,
                max_length=max_length,
                return_special_tokens_mask=False,
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )
        
        print(f"Tokenized dataset: {tokenized_dataset}")
        return tokenized_dataset
        
    except Exception as e:
        raise ValueError(f"Could not load dataset from {data_path}: {e}")


def main():
    """Main training function."""
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    print("🚀 Starting PHI-based model training")
    print(f"📊 Configuration: {vars(args)}")
    
    # Create PHI training configuration
    phi_config = PHITrainingConfig(
        base_learning_rate=args.phi_lr,
        lr_schedule_mode=args.phi_lr_schedule,
        warmup_epochs=args.phi_warmup_epochs,
        base_batch_size=args.phi_batch_size,
        phi_batch_progression=args.phi_batch_progression,
        max_batch_size=args.phi_max_batch,
        phi_training_phases=args.phi_training_phases,
        base_dropout=args.dropout,
        phi_dropout_schedule=args.phi_dropout_schedule,
        base_weight_decay=args.weight_decay,
        phi_weight_decay_schedule=args.phi_weight_decay_schedule,
    )
    
    # Validate PHI configuration
    warnings = validate_phi_config(phi_config)
    if warnings:
        print("⚠️  PHI Configuration Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    
    # Load tokenizer and model
    tokenizer_name = args.tokenizer or args.model
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    
    # Load and prepare datasets
    train_dataset = load_and_prepare_dataset(args.data, tokenizer, args.max_length)
    
    eval_dataset = None
    if args.eval_data and not args.no_eval:
        eval_dataset = load_and_prepare_dataset(args.eval_data, tokenizer, args.max_length)
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Create training arguments
    training_args = create_phi_training_args(
        phi_config=phi_config,
        output_dir=args.output_dir,
        total_epochs=args.epochs,
        per_device_train_batch_size=args.phi_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=args.dataloader_num_workers,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps" if eval_dataset and not args.no_eval else "no",
        save_strategy="steps",
        load_best_model_at_end=True if eval_dataset and not args.no_eval else False,
        metric_for_best_model="eval_loss" if eval_dataset and not args.no_eval else None,
        greater_is_better=False,
        report_to=[],  # Disable wandb/tensorboard by default
    )
    
    # Create PHI trainer
    trainer = PHITrainer(
        phi_config=phi_config,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Print PHI training summary
    trainer.print_phi_summary()
    
    # Start training
    print("🎯 Starting PHI training...")
    train_result = trainer.train()
    
    # Save the final model
    print("💾 Saving final model...")
    trainer.save_model()
    
    # Print training results
    print("\n📈 Training Results:")
    print(f"  Final Loss: {train_result.training_loss:.4f}")
    print(f"  Training Steps: {train_result.global_step}")
    print(f"  Training Time: {train_result.training_time:.2f}s")
    
    # Get PHI analysis
    phi_analysis = trainer.get_phi_analysis()
    print("\n🔍 PHI Training Analysis:")
    for key, value in phi_analysis.items():
        if key != 'phase_transitions':
            print(f"  {key}: {value}")
    
    if phi_analysis.get('phase_transitions'):
        print("  Phase Transitions:")
        for transition in phi_analysis['phase_transitions']:
            print(f"    Step {transition['step']}: Phase {transition['from_phase']} → {transition['to_phase']}")
    
    # Save final analysis
    analysis_path = os.path.join(args.output_dir, "phi_analysis.json")
    with open(analysis_path, 'w') as f:
        json.dump(phi_analysis, f, indent=2)
    
    print(f"\n✅ PHI training complete! Results saved to: {args.output_dir}")
    print("📊 Generated files:")
    print(f"  - Model: {args.output_dir}/")
    print(f"  - PHI Config: {args.output_dir}/phi_training_config.json")
    print(f"  - PHI Metrics: {args.output_dir}/phi_training_metrics.json")
    print(f"  - PHI Analysis: {args.output_dir}/phi_analysis.json")


if __name__ == "__main__":
    main()
