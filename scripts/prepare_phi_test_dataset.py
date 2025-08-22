#!/usr/bin/env python3
"""
Prepare small dataset subset for PHI training experiments.

Creates a manageable subset of the constitutional_examples dataset
for rapid PHI vs baseline training comparisons.
"""

import json
import random
from pathlib import Path
import argparse
from typing import List, Dict, Any

def load_constitutional_examples(file_path: str) -> List[Dict[str, Any]]:
    """Load constitutional examples from JSONL file."""
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples

def create_phi_test_subset(
    examples: List[Dict[str, Any]], 
    train_size: int = 500,
    eval_size: int = 100,
    seed: int = 42
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Create train/eval split for PHI testing."""
    random.seed(seed)
    
    # Shuffle examples
    shuffled = examples.copy()
    random.shuffle(shuffled)
    
    # Ensure we have enough examples
    total_needed = train_size + eval_size
    if len(shuffled) < total_needed:
        print(f"Warning: Only {len(shuffled)} examples available, need {total_needed}")
        # Duplicate examples if needed
        while len(shuffled) < total_needed:
            shuffled.extend(examples[:total_needed - len(shuffled)])
    
    # Split into train/eval
    train_examples = shuffled[:train_size]
    eval_examples = shuffled[train_size:train_size + eval_size]
    
    return train_examples, eval_examples

def format_for_causal_lm(examples: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Format examples for causal language modeling."""
    formatted = []
    
    for example in examples:
        # Extract text content - adapt based on your dataset structure
        if 'prompt' in example and 'response' in example:
            # Conversation format
            text = f"Human: {example['prompt']}\n\nAssistant: {example['response']}"
        elif 'input' in example and 'output' in example:
            # Input/output format
            text = f"Input: {example['input']}\n\nOutput: {example['output']}"
        elif 'text' in example:
            # Direct text format
            text = example['text']
        else:
            # Fallback - use first available text field
            text_fields = ['content', 'message', 'data']
            text = None
            for field in text_fields:
                if field in example:
                    text = str(example[field])
                    break
            
            if text is None:
                # Last resort - convert entire example to string
                text = str(example)
        
        formatted.append({"text": text})
    
    return formatted

def save_dataset_subset(
    train_examples: List[Dict[str, str]],
    eval_examples: List[Dict[str, str]],
    output_dir: str
):
    """Save train/eval datasets to JSONL files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save training set
    train_path = output_path / "train.jsonl"
    with open(train_path, 'w', encoding='utf-8') as f:
        for example in train_examples:
            f.write(json.dumps(example) + '\n')
    
    # Save evaluation set
    eval_path = output_path / "eval.jsonl"
    with open(eval_path, 'w', encoding='utf-8') as f:
        for example in eval_examples:
            f.write(json.dumps(example) + '\n')
    
    # Save metadata
    metadata = {
        "train_size": len(train_examples),
        "eval_size": len(eval_examples),
        "total_size": len(train_examples) + len(eval_examples),
        "format": "causal_lm",
        "source": "constitutional_examples",
        "created_for": "phi_training_experiments"
    }
    
    metadata_path = output_path / "dataset_info.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    return train_path, eval_path, metadata_path

def main():
    parser = argparse.ArgumentParser(description="Prepare PHI test dataset")
    parser.add_argument("--input", required=True, help="Input JSONL file path")
    parser.add_argument("--output", default="./Datasets/phi_test_subset", help="Output directory")
    parser.add_argument("--train-size", type=int, default=500, help="Training set size")
    parser.add_argument("--eval-size", type=int, default=100, help="Evaluation set size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    print(f"🔄 Loading dataset from: {args.input}")
    
    # Load original dataset
    try:
        examples = load_constitutional_examples(args.input)
        print(f"✅ Loaded {len(examples)} examples")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return 1
    
    # Create subset
    print(f"🎯 Creating subset: {args.train_size} train + {args.eval_size} eval")
    train_examples, eval_examples = create_phi_test_subset(
        examples, args.train_size, args.eval_size, args.seed
    )
    
    # Format for causal LM
    print("📝 Formatting for causal language modeling...")
    train_formatted = format_for_causal_lm(train_examples)
    eval_formatted = format_for_causal_lm(eval_examples)
    
    # Save dataset
    print(f"💾 Saving to: {args.output}")
    train_path, eval_path, metadata_path = save_dataset_subset(
        train_formatted, eval_formatted, args.output
    )
    
    print("✅ Dataset preparation complete!")
    print(f"📁 Files created:")
    print(f"   Training: {train_path}")
    print(f"   Evaluation: {eval_path}")
    print(f"   Metadata: {metadata_path}")
    
    # Show sample
    print(f"\n📖 Sample training example:")
    if train_formatted:
        sample_text = train_formatted[0]["text"]
        # Truncate if too long
        if len(sample_text) > 200:
            sample_text = sample_text[:200] + "..."
        print(f"   {sample_text}")
    
    return 0

if __name__ == "__main__":
    exit(main())
