#!/usr/bin/env python3
"""
Run PHI training experiments comparing baseline vs PHI-based training.

This script runs controlled experiments to validate the effectiveness
of PHI training principles.
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List
import subprocess
import sys

def run_baseline_experiment(
    model_name: str,
    dataset_path: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-4,
    **kwargs
) -> Dict[str, Any]:
    """Run baseline training experiment."""
    
    print(f"🔄 Running baseline experiment...")
    print(f"   Model: {model_name}")
    print(f"   Dataset: {dataset_path}")
    print(f"   Output: {output_dir}")
    
    # Prepare baseline training command
    cmd = [
        sys.executable, "scripts/llm_finetune_lora.py",
        "--base", model_name,
        "--data", dataset_path,
        "--out", output_dir,
        "--epochs", str(epochs),
        "--lr", str(learning_rate),
        "--per-device-batch", str(batch_size),
        "--warmup", "0.1",  # Standard warmup
        "--weight-decay", "0.01",
        "--rank", "16",
        "--alpha", "32",
        "--dropout", "0.05",
        "--eval-steps", "50",
        "--save-steps", "50",
        "--bf16"  # Use bfloat16 for efficiency
    ]
    
    # Add any additional arguments
    for key, value in kwargs.items():
        if key.startswith('--'):
            cmd.extend([key, str(value)])
    
    start_time = time.time()
    
    try:
        # Run training
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        training_time = time.time() - start_time
        
        # Parse results
        experiment_result = {
            "type": "baseline",
            "model": model_name,
            "dataset": dataset_path,
            "output_dir": output_dir,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "training_time": training_time,
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "command": " ".join(cmd)
        }
        
        if result.returncode == 0:
            print(f"✅ Baseline experiment completed in {training_time:.1f}s")
        else:
            print(f"❌ Baseline experiment failed")
            print(f"Error: {result.stderr}")
        
        return experiment_result
        
    except Exception as e:
        return {
            "type": "baseline",
            "success": False,
            "error": str(e),
            "training_time": time.time() - start_time
        }

def run_phi_experiment(
    model_name: str,
    dataset_path: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-4,
    phi_schedule: str = "phi_decay",
    **kwargs
) -> Dict[str, Any]:
    """Run PHI training experiment."""
    
    print(f"🌟 Running PHI experiment...")
    print(f"   Model: {model_name}")
    print(f"   Dataset: {dataset_path}")
    print(f"   Output: {output_dir}")
    print(f"   PHI Schedule: {phi_schedule}")
    
    # Prepare PHI training command
    cmd = [
        sys.executable, "scripts/phi_train_model.py",
        "--model", model_name,
        "--data", dataset_path,
        "--output-dir", output_dir,
        "--epochs", str(epochs),
        "--phi-lr", str(learning_rate),
        "--phi-lr-schedule", phi_schedule,
        "--phi-batch-size", str(batch_size),
        "--phi-batch-progression",
        "--phi-training-phases",
        "--phi-dropout-schedule",
        "--phi-weight-decay-schedule",
        "--eval-steps", "50",
        "--save-steps", "50",
        "--bf16"
    ]
    
    # Add any additional arguments
    for key, value in kwargs.items():
        if key.startswith('--'):
            cmd.extend([key, str(value)])
    
    start_time = time.time()
    
    try:
        # Run training
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        training_time = time.time() - start_time
        
        # Parse results
        experiment_result = {
            "type": "phi",
            "model": model_name,
            "dataset": dataset_path,
            "output_dir": output_dir,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "phi_schedule": phi_schedule,
            "training_time": training_time,
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "command": " ".join(cmd)
        }
        
        if result.returncode == 0:
            print(f"✅ PHI experiment completed in {training_time:.1f}s")
            
            # Try to load PHI analysis if available
            analysis_path = os.path.join(output_dir, "phi_analysis.json")
            if os.path.exists(analysis_path):
                with open(analysis_path, 'r') as f:
                    experiment_result["phi_analysis"] = json.load(f)
        else:
            print(f"❌ PHI experiment failed")
            print(f"Error: {result.stderr}")
        
        return experiment_result
        
    except Exception as e:
        return {
            "type": "phi",
            "success": False,
            "error": str(e),
            "training_time": time.time() - start_time
        }

def extract_training_metrics(output_dir: str) -> Dict[str, Any]:
    """Extract training metrics from output directory."""
    metrics = {}
    
    # Look for trainer state
    trainer_state_path = os.path.join(output_dir, "trainer_state.json")
    if os.path.exists(trainer_state_path):
        with open(trainer_state_path, 'r') as f:
            trainer_state = json.load(f)
            
        # Extract key metrics
        if 'log_history' in trainer_state:
            log_history = trainer_state['log_history']
            
            # Get final training loss
            train_losses = [entry.get('train_loss') for entry in log_history if 'train_loss' in entry]
            if train_losses:
                metrics['final_train_loss'] = train_losses[-1]
                metrics['initial_train_loss'] = train_losses[0]
                metrics['loss_improvement'] = train_losses[0] - train_losses[-1]
            
            # Get final eval loss
            eval_losses = [entry.get('eval_loss') for entry in log_history if 'eval_loss' in entry]
            if eval_losses:
                metrics['final_eval_loss'] = eval_losses[-1]
                metrics['best_eval_loss'] = min(eval_losses)
            
            # Get learning rates
            learning_rates = [entry.get('learning_rate') for entry in log_history if 'learning_rate' in entry]
            if learning_rates:
                metrics['initial_lr'] = learning_rates[0]
                metrics['final_lr'] = learning_rates[-1]
                metrics['lr_decay_ratio'] = learning_rates[0] / learning_rates[-1] if learning_rates[-1] > 0 else float('inf')
    
    return metrics

def compare_experiments(baseline_result: Dict[str, Any], phi_result: Dict[str, Any]) -> Dict[str, Any]:
    """Compare baseline vs PHI experiment results."""
    
    comparison = {
        "baseline_success": baseline_result.get("success", False),
        "phi_success": phi_result.get("success", False),
        "baseline_time": baseline_result.get("training_time", 0),
        "phi_time": phi_result.get("training_time", 0),
    }
    
    # Extract metrics from both experiments
    if baseline_result.get("success") and phi_result.get("success"):
        baseline_metrics = extract_training_metrics(baseline_result["output_dir"])
        phi_metrics = extract_training_metrics(phi_result["output_dir"])
        
        comparison["baseline_metrics"] = baseline_metrics
        comparison["phi_metrics"] = phi_metrics
        
        # Compare key metrics
        if "final_train_loss" in baseline_metrics and "final_train_loss" in phi_metrics:
            comparison["train_loss_improvement"] = (
                baseline_metrics["final_train_loss"] - phi_metrics["final_train_loss"]
            )
        
        if "final_eval_loss" in baseline_metrics and "final_eval_loss" in phi_metrics:
            comparison["eval_loss_improvement"] = (
                baseline_metrics["final_eval_loss"] - phi_metrics["final_eval_loss"]
            )
        
        # Compare training efficiency
        comparison["time_ratio"] = phi_result["training_time"] / baseline_result["training_time"]
        
        # PHI-specific analysis
        if "phi_analysis" in phi_result:
            comparison["phi_analysis"] = phi_result["phi_analysis"]
    
    return comparison

def main():
    parser = argparse.ArgumentParser(description="Run PHI training experiments")
    parser.add_argument("--model", default="gpt2", help="Base model name")
    parser.add_argument("--dataset", required=True, help="Dataset path")
    parser.add_argument("--output-base", default="./out/phi_experiments", help="Base output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--phi-schedule", default="phi_decay", help="PHI schedule type")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline experiment")
    parser.add_argument("--skip-phi", action="store_true", help="Skip PHI experiment")
    parser.add_argument("--experiment-name", help="Custom experiment name")
    
    args = parser.parse_args()
    
    # Create experiment directory
    if args.experiment_name:
        exp_name = args.experiment_name
    else:
        exp_name = f"exp_{int(time.time())}"
    
    exp_dir = Path(args.output_base) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🧪 Starting PHI training experiments: {exp_name}")
    print(f"📁 Output directory: {exp_dir}")
    
    results = {
        "experiment_name": exp_name,
        "timestamp": time.time(),
        "config": {
            "model": args.model,
            "dataset": args.dataset,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "phi_schedule": args.phi_schedule
        }
    }
    
    # Run baseline experiment
    if not args.skip_baseline:
        baseline_dir = exp_dir / "baseline"
        baseline_result = run_baseline_experiment(
            model_name=args.model,
            dataset_path=args.dataset,
            output_dir=str(baseline_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        results["baseline"] = baseline_result
    
    # Run PHI experiment
    if not args.skip_phi:
        phi_dir = exp_dir / "phi"
        phi_result = run_phi_experiment(
            model_name=args.model,
            dataset_path=args.dataset,
            output_dir=str(phi_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            phi_schedule=args.phi_schedule
        )
        results["phi"] = phi_result
    
    # Compare results
    if not args.skip_baseline and not args.skip_phi:
        comparison = compare_experiments(results["baseline"], results["phi"])
        results["comparison"] = comparison
        
        print(f"\n📊 EXPERIMENT COMPARISON:")
        print(f"   Baseline success: {comparison['baseline_success']}")
        print(f"   PHI success: {comparison['phi_success']}")
        print(f"   Training time ratio (PHI/Baseline): {comparison.get('time_ratio', 'N/A'):.2f}")
        
        if "train_loss_improvement" in comparison:
            improvement = comparison["train_loss_improvement"]
            print(f"   Train loss improvement: {improvement:.6f} ({'better' if improvement > 0 else 'worse'})")
        
        if "eval_loss_improvement" in comparison:
            improvement = comparison["eval_loss_improvement"]
            print(f"   Eval loss improvement: {improvement:.6f} ({'better' if improvement > 0 else 'worse'})")
    
    # Save results
    results_path = exp_dir / "experiment_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Experiments complete!")
    print(f"📁 Results saved to: {results_path}")
    
    return 0

if __name__ == "__main__":
    exit(main())
