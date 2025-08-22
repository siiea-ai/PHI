"""
PHI-based Trainer implementation extending HuggingFace Trainer.

This module provides a complete PHI training system that integrates
golden ratio principles throughout the training process.
"""

from __future__ import annotations

import os
import json
import math
from typing import Dict, List, Optional, Union, Any, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from transformers import (
        Trainer, TrainingArguments, PreTrainedModel, PreTrainedTokenizer,
        DataCollator, EvalPrediction, TrainerCallback
    )
    from transformers.trainer_utils import PredictionOutput
    from transformers.training_args import OptimizerNames
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    Trainer = object
    TrainingArguments = None

from .training import PHITrainingConfig, PHIMath
from .hf_integration import PHITrainerCallback, PHILRScheduler, PHIOptimizer, setup_phi_training


class PHITrainer(Trainer if HF_AVAILABLE else object):
    """
    PHI-based trainer extending HuggingFace Trainer.
    
    Applies golden ratio principles to:
    - Learning rate scheduling
    - Batch size progression  
    - Training phase transitions
    - Regularization scheduling
    - Data sampling (when supported)
    """
    
    def __init__(
        self,
        phi_config: PHITrainingConfig,
        model: Optional[PreTrainedModel] = None,
        args: Optional[TrainingArguments] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        data_collator: Optional[DataCollator] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: tuple = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        **kwargs
    ):
        if not HF_AVAILABLE:
            raise ImportError("HuggingFace Transformers not available. Install with: pip install transformers")
        
        self.phi_config = phi_config
        self.phi_components = None
        self.phi_callback = None
        
        # Calculate total steps for PHI scheduling
        if args and train_dataset:
            total_epochs = int(args.num_train_epochs)
            steps_per_epoch = len(train_dataset) // (args.per_device_train_batch_size * args.gradient_accumulation_steps)
            if args.world_size > 1:
                steps_per_epoch //= args.world_size
            self.total_steps = steps_per_epoch * total_epochs
        else:
            self.total_steps = 1000  # Default fallback
        
        # Add PHI callback to callbacks list
        callbacks = callbacks or []
        if args:
            self.phi_callback = PHITrainerCallback(phi_config, int(args.num_train_epochs))
            callbacks.append(self.phi_callback)
        
        # Initialize parent trainer
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            **kwargs
        )
        
        # Setup PHI training components after parent initialization
        if self.optimizer is not None:
            self._setup_phi_components()
    
    def _setup_phi_components(self):
        """Set up PHI training components."""
        self.phi_components = setup_phi_training(
            model=self.model,
            optimizer=self.optimizer,
            phi_config=self.phi_config,
            total_epochs=int(self.args.num_train_epochs),
            total_steps=self.total_steps
        )
        
        # Replace optimizer and scheduler with PHI versions
        self.optimizer = self.phi_components['optimizer']
        self.lr_scheduler = self.phi_components['lr_scheduler']
    
    def create_optimizer(self):
        """Create optimizer with PHI configuration."""
        # Let parent create the base optimizer first
        super().create_optimizer()
        
        # Then wrap it with PHI components
        if self.phi_components is None:
            self._setup_phi_components()
    
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """Create PHI learning rate scheduler."""
        if optimizer is None:
            optimizer = self.optimizer
        
        # Create PHI scheduler instead of default
        self.lr_scheduler = PHILRScheduler(
            optimizer=optimizer,
            phi_config=self.phi_config,
            total_steps=num_training_steps
        )
        return self.lr_scheduler
    
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step with PHI enhancements.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # Apply PHI-based dropout if model supports it
        if hasattr(model, 'config') and hasattr(model.config, 'hidden_dropout_prob'):
            if self.phi_config.phi_dropout_schedule and self.phi_components:
                current_step = self.state.global_step
                phi_dropout = self.phi_components['optimizer'].reg_scheduler.get_dropout(current_step)
                
                # Temporarily adjust dropout for this step
                original_dropout = model.config.hidden_dropout_prob
                model.config.hidden_dropout_prob = phi_dropout
        
        # Standard training step
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        
        # Restore original dropout if modified
        if hasattr(model, 'config') and hasattr(model.config, 'hidden_dropout_prob'):
            if self.phi_config.phi_dropout_schedule and 'original_dropout' in locals():
                model.config.hidden_dropout_prob = original_dropout
        
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        
        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            loss = loss / self.args.gradient_accumulation_steps
        
        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        
        return loss.detach()
    
    def log(self, logs: Dict[str, float]) -> None:
        """
        Log training metrics with PHI-specific information.
        """
        # Add PHI-specific metrics to logs
        if self.phi_callback:
            current_epoch = self.state.epoch
            current_step = self.state.global_step
            
            # Add PHI batch size info
            if self.phi_config.phi_batch_progression:
                phi_batch_size = self.phi_callback.batch_scheduler.get_batch_size(int(current_epoch))
                logs['phi_batch_size'] = phi_batch_size
            
            # Add training phase info
            phase1_epochs = PHIMath.phi_training_phases(int(self.args.num_train_epochs))[0]
            current_phase = 1 if current_epoch < phase1_epochs else 2
            logs['phi_training_phase'] = current_phase
            
            # Add PHI ratios for analysis
            if current_step > 0:
                progress = current_step / self.total_steps
                phi_progress = PHIMath.phi_decay(1.0, current_step, self.total_steps)
                logs['phi_progress_ratio'] = phi_progress
        
        # Call parent log method
        super().log(logs)
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Save model with PHI training configuration.
        """
        # Save the model using parent method
        super().save_model(output_dir, _internal_call)
        
        # Save PHI configuration alongside model
        if output_dir is None:
            output_dir = self.args.output_dir
        
        phi_config_path = os.path.join(output_dir, "phi_training_config.json")
        with open(phi_config_path, 'w') as f:
            # Convert PHI config to dict for JSON serialization
            phi_dict = {
                'base_learning_rate': self.phi_config.base_learning_rate,
                'lr_schedule_mode': self.phi_config.lr_schedule_mode,
                'phi_lr_power': self.phi_config.phi_lr_power,
                'warmup_epochs': self.phi_config.warmup_epochs,
                'base_batch_size': self.phi_config.base_batch_size,
                'phi_batch_progression': self.phi_config.phi_batch_progression,
                'max_batch_size': self.phi_config.max_batch_size,
                'batch_phi_phases': self.phi_config.batch_phi_phases,
                'phi_training_phases': self.phi_config.phi_training_phases,
                'phase1_focus': self.phi_config.phase1_focus,
                'phase2_focus': self.phi_config.phase2_focus,
                'base_dropout': self.phi_config.base_dropout,
                'phi_dropout_schedule': self.phi_config.phi_dropout_schedule,
                'base_weight_decay': self.phi_config.base_weight_decay,
                'phi_weight_decay_schedule': self.phi_config.phi_weight_decay_schedule,
                'phi_data_weighting': self.phi_config.phi_data_weighting,
                'phi_curriculum_learning': self.phi_config.phi_curriculum_learning
            }
            json.dump(phi_dict, f, indent=2)
        
        # Save PHI training metrics if available
        if self.phi_callback:
            metrics_path = os.path.join(output_dir, "phi_training_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(self.phi_callback.get_metrics_history(), f, indent=2)
    
    def get_phi_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive PHI training analysis.
        """
        if not self.phi_callback:
            return {"error": "No PHI callback available"}
        
        from .hf_integration import analyze_phi_training_run
        return analyze_phi_training_run(self.phi_callback)
    
    def print_phi_summary(self):
        """
        Print a summary of PHI training configuration and progress.
        """
        print("\n" + "="*50)
        print("PHI TRAINING SUMMARY")
        print("="*50)
        
        print(f"📐 Golden Ratio (φ): {PHIMath.PHI:.6f}")
        print(f"🔢 Inverse φ: {PHIMath.INV_PHI:.6f}")
        print(f"🔢 Inverse φ²: {PHIMath.INV_PHI_SQUARED:.6f}")
        
        print(f"\n🎯 Training Configuration:")
        print(f"  Learning Rate: {self.phi_config.base_learning_rate}")
        print(f"  LR Schedule: {self.phi_config.lr_schedule_mode}")
        print(f"  Batch Size: {self.phi_config.base_batch_size}")
        print(f"  Batch Progression: {self.phi_config.phi_batch_progression}")
        print(f"  Training Phases: {self.phi_config.phi_training_phases}")
        
        if self.args:
            total_epochs = int(self.args.num_train_epochs)
            phase1, phase2 = PHIMath.phi_training_phases(total_epochs)
            print(f"\n📊 Training Phases:")
            print(f"  Phase 1 (Exploration): {phase1} epochs ({phase1/total_epochs*100:.1f}%)")
            print(f"  Phase 2 (Exploitation): {phase2} epochs ({phase2/total_epochs*100:.1f}%)")
            print(f"  Phase Ratio: {phase2/phase1:.3f} (φ = {PHIMath.PHI:.3f})")
        
        if self.phi_callback:
            analysis = self.get_phi_analysis()
            print(f"\n📈 Training Progress:")
            print(f"  Total Steps: {analysis.get('total_steps', 'N/A')}")
            print(f"  Phase Transitions: {len(analysis.get('phase_transitions', []))}")
            if analysis.get('lr_decay_ratio'):
                print(f"  LR Decay Ratio: {analysis['lr_decay_ratio']:.3f}")
        
        print("="*50 + "\n")


def create_phi_trainer(
    model: PreTrainedModel,
    phi_config: PHITrainingConfig,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    output_dir: str = "./phi_training_output",
    num_epochs: int = 3,
    **kwargs
) -> PHITrainer:
    """
    Convenience function to create a PHI trainer with sensible defaults.
    
    Args:
        model: PreTrained model to train
        phi_config: PHI training configuration
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset (optional)
        output_dir: Output directory
        num_epochs: Number of training epochs
        **kwargs: Additional TrainingArguments parameters
    
    Returns:
        Configured PHI trainer
    """
    if not HF_AVAILABLE:
        raise ImportError("HuggingFace Transformers not available. Install with: pip install transformers")
    
    # Create training arguments with PHI defaults
    from .hf_integration import create_phi_training_args
    
    training_args = create_phi_training_args(
        phi_config=phi_config,
        output_dir=output_dir,
        total_epochs=num_epochs,
        **kwargs
    )
    
    # Create and return PHI trainer
    trainer = PHITrainer(
        phi_config=phi_config,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    
    return trainer
