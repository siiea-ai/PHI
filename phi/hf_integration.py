"""
HuggingFace Transformers integration for PHI-based training.

This module provides PHI scheduler implementations that integrate seamlessly
with HuggingFace Transformers training pipelines.
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Union, Any
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler

try:
    from transformers import TrainingArguments, Trainer
    from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    TrainingArguments = None
    Trainer = None
    TrainerCallback = None
    TrainerControl = None
    TrainerState = None

from .training import PHITrainingConfig, PHIMath, PHILearningRateScheduler, PHIBatchScheduler, PHIRegularizationScheduler


class PHILRScheduler(_LRScheduler):
    """
    PyTorch learning rate scheduler implementing PHI-based schedules.
    
    Compatible with PyTorch optimizers and HuggingFace Transformers.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        phi_config: PHITrainingConfig,
        total_steps: int,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.phi_config = phi_config
        self.total_steps = total_steps
        self.phi_scheduler = PHILearningRateScheduler(phi_config, total_steps)
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self) -> List[float]:
        """Get learning rates for all parameter groups."""
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                         "please use `get_last_lr()`.", UserWarning)
        
        current_step = self.last_epoch + 1
        phi_lr_multiplier = self.phi_scheduler.get_lr(current_step) / self.phi_config.base_learning_rate
        
        return [base_lr * phi_lr_multiplier for base_lr in self.base_lrs]


class PHITrainerCallback(TrainerCallback):
    """
    HuggingFace Trainer callback for PHI-based training control.
    
    Handles dynamic batch size adjustment, regularization scheduling,
    and training phase transitions.
    """
    
    def __init__(self, phi_config: PHITrainingConfig, total_epochs: int):
        if not HF_AVAILABLE:
            raise ImportError("HuggingFace Transformers not available. Install with: pip install transformers")
        
        self.phi_config = phi_config
        self.total_epochs = total_epochs
        self.batch_scheduler = PHIBatchScheduler(phi_config, total_epochs)
        self.current_phase = 1
        self.phase_transition_epoch = PHIMath.phi_training_phases(total_epochs)[0]
        
        # Track metrics for analysis
        self.metrics_history = {
            'learning_rates': [],
            'batch_sizes': [],
            'dropout_rates': [],
            'weight_decay_rates': [],
            'training_phases': []
        }
    
    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, 
                      control: TrainerControl, model: torch.nn.Module = None, **kwargs):
        """Handle epoch beginning - adjust batch size and training phase."""
        current_epoch = int(state.epoch)
        
        # Check for phase transition
        if current_epoch >= self.phase_transition_epoch and self.current_phase == 1:
            self.current_phase = 2
            print(f"🔄 PHI Training Phase Transition: Phase {self.current_phase} (Exploitation)")
        
        # Get PHI batch size for this epoch
        if self.phi_config.phi_batch_progression:
            phi_batch_size = self.batch_scheduler.get_batch_size(current_epoch)
            
            # Note: Actual batch size change requires trainer restart in HF
            # This is logged for monitoring and future implementation
            self.metrics_history['batch_sizes'].append(phi_batch_size)
            
            if len(self.metrics_history['batch_sizes']) == 1:
                print(f"📊 PHI Batch Size Schedule: Starting with {phi_batch_size}")
        
        # Log current training phase
        self.metrics_history['training_phases'].append(self.current_phase)
    
    def on_step_begin(self, args: TrainingArguments, state: TrainerState,
                     control: TrainerControl, model: torch.nn.Module = None, **kwargs):
        """Handle training step beginning - log PHI metrics."""
        # This would be where we could apply dynamic regularization
        # if the model architecture supports it
        pass
    
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl,
               model: torch.nn.Module = None, optimizer: Optimizer = None, **kwargs):
        """Log PHI-specific metrics."""
        if optimizer is not None and hasattr(optimizer, 'param_groups'):
            # Log current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            self.metrics_history['learning_rates'].append(current_lr)
            
            # Log weight decay if available
            if 'weight_decay' in optimizer.param_groups[0]:
                current_wd = optimizer.param_groups[0]['weight_decay']
                self.metrics_history['weight_decay_rates'].append(current_wd)
    
    def get_metrics_history(self) -> Dict[str, List]:
        """Get complete metrics history for analysis."""
        return self.metrics_history.copy()


class PHIOptimizer:
    """
    Wrapper for optimizers to apply PHI-based regularization scheduling.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        phi_config: PHITrainingConfig,
        total_steps: int
    ):
        self.optimizer = optimizer
        self.phi_config = phi_config
        self.total_steps = total_steps
        self.reg_scheduler = PHIRegularizationScheduler(phi_config, total_steps)
        self.current_step = 0
    
    def step(self, closure=None):
        """Step the optimizer with PHI regularization updates."""
        # Update weight decay based on PHI schedule
        if self.phi_config.phi_weight_decay_schedule:
            new_weight_decay = self.reg_scheduler.get_weight_decay(self.current_step)
            for param_group in self.optimizer.param_groups:
                param_group['weight_decay'] = new_weight_decay
        
        # Step the underlying optimizer
        result = self.optimizer.step(closure)
        self.current_step += 1
        return result
    
    def zero_grad(self, set_to_none: bool = False):
        """Zero gradients."""
        return self.optimizer.zero_grad(set_to_none)
    
    def state_dict(self):
        """Get optimizer state dict."""
        state = self.optimizer.state_dict()
        state['phi_current_step'] = self.current_step
        return state
    
    def load_state_dict(self, state_dict):
        """Load optimizer state dict."""
        if 'phi_current_step' in state_dict:
            self.current_step = state_dict.pop('phi_current_step')
        return self.optimizer.load_state_dict(state_dict)
    
    def __getattr__(self, name):
        """Delegate attribute access to underlying optimizer."""
        return getattr(self.optimizer, name)


def create_phi_lr_scheduler(
    optimizer: Optimizer,
    phi_config: PHITrainingConfig,
    total_steps: int
) -> PHILRScheduler:
    """
    Create a PHI learning rate scheduler compatible with PyTorch and HuggingFace.
    
    Args:
        optimizer: PyTorch optimizer
        phi_config: PHI training configuration
        total_steps: Total number of training steps
    
    Returns:
        PHI learning rate scheduler
    """
    return PHILRScheduler(optimizer, phi_config, total_steps)


def create_phi_training_args(
    phi_config: PHITrainingConfig,
    output_dir: str,
    total_epochs: int,
    total_steps: Optional[int] = None,
    **kwargs
) -> 'TrainingArguments':
    """
    Create HuggingFace TrainingArguments with PHI-based settings.
    
    Args:
        phi_config: PHI training configuration
        output_dir: Output directory for training
        total_epochs: Total number of training epochs
        total_steps: Total training steps (calculated if not provided)
        **kwargs: Additional TrainingArguments parameters
    
    Returns:
        TrainingArguments configured for PHI training
    """
    if not HF_AVAILABLE:
        raise ImportError("HuggingFace Transformers not available. Install with: pip install transformers")
    
    # Calculate PHI training phases
    phase1_epochs, phase2_epochs = PHIMath.phi_training_phases(total_epochs)
    
    # Set up base training arguments with PHI principles
    phi_training_args = {
        'output_dir': output_dir,
        'num_train_epochs': total_epochs,
        'per_device_train_batch_size': phi_config.base_batch_size,
        'learning_rate': phi_config.base_learning_rate,
        'weight_decay': phi_config.base_weight_decay,
        'warmup_steps': phi_config.warmup_epochs if total_steps else 0,
        'lr_scheduler_type': 'constant',  # We'll override with PHI scheduler
        'save_strategy': 'epoch',
        'evaluation_strategy': 'epoch',
        'logging_steps': max(1, (total_steps or 100) // 20),  # Log ~20 times per training
        'save_steps': max(1, (total_steps or 100) // 10),     # Save ~10 times per training
        'dataloader_drop_last': True,  # Ensure consistent batch sizes
        'remove_unused_columns': False,
        'report_to': [],  # Disable wandb/tensorboard by default
    }
    
    # Override with user-provided arguments
    phi_training_args.update(kwargs)
    
    return TrainingArguments(**phi_training_args)


def setup_phi_training(
    model: torch.nn.Module,
    optimizer: Optimizer,
    phi_config: PHITrainingConfig,
    total_epochs: int,
    total_steps: int
) -> Dict[str, Any]:
    """
    Set up complete PHI training components.
    
    Args:
        model: PyTorch model to train
        optimizer: PyTorch optimizer
        phi_config: PHI training configuration
        total_epochs: Total training epochs
        total_steps: Total training steps
    
    Returns:
        Dictionary containing PHI training components
    """
    # Create PHI learning rate scheduler
    lr_scheduler = create_phi_lr_scheduler(optimizer, phi_config, total_steps)
    
    # Wrap optimizer with PHI regularization
    phi_optimizer = PHIOptimizer(optimizer, phi_config, total_steps)
    
    # Create PHI trainer callback
    callback = PHITrainerCallback(phi_config, total_epochs)
    
    return {
        'lr_scheduler': lr_scheduler,
        'optimizer': phi_optimizer,
        'callback': callback,
        'phi_config': phi_config
    }


def get_phi_schedule_lambda(phi_config: PHITrainingConfig, total_steps: int):
    """
    Create lambda function for HuggingFace get_scheduler compatibility.
    
    This allows PHI schedules to work with HuggingFace's get_scheduler function.
    """
    phi_scheduler = PHILearningRateScheduler(phi_config, total_steps)
    base_lr = phi_config.base_learning_rate
    
    def phi_lambda(current_step: int) -> float:
        return phi_scheduler.get_lr(current_step) / base_lr
    
    return phi_lambda


# Utility functions for PHI training analysis
def analyze_phi_training_run(callback: PHITrainerCallback) -> Dict[str, Any]:
    """
    Analyze a completed PHI training run.
    
    Args:
        callback: PHI trainer callback with metrics history
    
    Returns:
        Analysis results
    """
    metrics = callback.get_metrics_history()
    
    analysis = {
        'total_steps': len(metrics.get('learning_rates', [])),
        'phase_transitions': [],
        'lr_decay_ratio': None,
        'batch_size_progression': metrics.get('batch_sizes', []),
        'training_phases': metrics.get('training_phases', [])
    }
    
    # Analyze learning rate decay
    lrs = metrics.get('learning_rates', [])
    if len(lrs) >= 2:
        analysis['lr_decay_ratio'] = lrs[0] / lrs[-1] if lrs[-1] > 0 else float('inf')
    
    # Find phase transitions
    phases = metrics.get('training_phases', [])
    for i in range(1, len(phases)):
        if phases[i] != phases[i-1]:
            analysis['phase_transitions'].append({
                'step': i,
                'from_phase': phases[i-1],
                'to_phase': phases[i]
            })
    
    return analysis
