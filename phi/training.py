"""
PHI-based training utilities and mathematical framework.

This module implements golden ratio (φ ≈ 1.618) based scheduling functions
for learning rates, batch sizes, training phases, and regularization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from .constants import PHI, INV_PHI, fibonacci_sequence


# Additional PHI constants for training
INV_PHI_SQUARED = INV_PHI * INV_PHI  # ≈ 0.381966 (38.2%)
PHI_MINUS_ONE = PHI - 1.0            # ≈ 0.618034 (61.8%)


@dataclass
class PHITrainingConfig:
    """Configuration for PHI-based training parameters."""
    
    # Learning rate PHI settings
    base_learning_rate: float = 2e-4
    lr_schedule_mode: str = "phi_decay"  # phi_decay, phi_warmup, phi_cosine, phi_cyclic
    phi_lr_power: float = 1.0
    warmup_epochs: int = 0
    
    # Batch size PHI settings
    base_batch_size: int = 8
    phi_batch_progression: bool = True
    max_batch_size: Optional[int] = 128
    batch_phi_phases: int = 3
    
    # Training phase PHI settings
    phi_training_phases: bool = True
    phase1_focus: str = "exploration"  # exploration, exploitation
    phase2_focus: str = "exploitation"
    
    # Regularization PHI settings
    base_dropout: float = 0.1
    phi_dropout_schedule: bool = True
    base_weight_decay: float = 0.01
    phi_weight_decay_schedule: bool = True
    
    # Data sampling PHI settings
    phi_data_weighting: bool = False
    phi_curriculum_learning: bool = False


class PHIMath:
    """Mathematical functions for PHI-based scheduling."""
    
    # Class constants for easy access
    PHI = PHI
    INV_PHI = INV_PHI
    INV_PHI_SQUARED = INV_PHI_SQUARED
    
    @staticmethod
    def phi_decay(base_value: float, step: int, total_steps: int, power: float = 1.0) -> float:
        """
        PHI-based exponential decay.
        
        Formula: base_value * φ^(-(step/total_steps)^power)
        """
        if total_steps <= 0:
            return base_value
        progress = step / total_steps
        return base_value * (PHI ** (-(progress ** power)))
    
    @staticmethod
    def phi_warmup(base_value: float, step: int, warmup_steps: int) -> float:
        """
        PHI-based warmup schedule.
        
        Formula: base_value * (1 - φ^(-step/warmup_steps))
        """
        if warmup_steps <= 0 or step >= warmup_steps:
            return base_value
        progress = step / warmup_steps
        return base_value * (1.0 - (PHI ** (-progress)))
    
    @staticmethod
    def phi_cosine(base_value: float, step: int, total_steps: int, min_value: float = 0.0) -> float:
        """
        PHI-modulated cosine annealing.
        
        Formula: min_value + (base_value - min_value) * (1 + cos(π * step / total_steps)) / 2 * φ^(-step/total_steps)
        """
        if total_steps <= 0:
            return base_value
        progress = step / total_steps
        cosine_factor = (1.0 + math.cos(math.pi * progress)) / 2.0
        phi_factor = PHI ** (-progress)
        return min_value + (base_value - min_value) * cosine_factor * phi_factor
    
    @staticmethod
    def phi_cyclic(base_value: float, step: int, cycle_length: int, num_cycles: int = 1) -> float:
        """
        PHI-based cyclic schedule with golden ratio periods.
        
        Creates cycles where each cycle is φ times longer than the previous.
        """
        if cycle_length <= 0:
            return base_value
        
        # Calculate which cycle we're in and position within cycle
        total_cycle_length = 0
        current_cycle_length = cycle_length
        
        for cycle in range(num_cycles):
            if step < total_cycle_length + current_cycle_length:
                # We're in this cycle
                cycle_progress = (step - total_cycle_length) / current_cycle_length
                # Use PHI decay within the cycle
                return base_value * (PHI ** (-cycle_progress))
            
            total_cycle_length += current_cycle_length
            current_cycle_length = int(current_cycle_length * PHI)
        
        # If we're past all cycles, use final decay
        return base_value * (PHI ** (-1.0))
    
    @staticmethod
    def phi_batch_progression(base_batch: int, phase: int, max_batch: Optional[int] = None) -> int:
        """
        Calculate PHI-based batch size progression.
        
        Formula: base_batch * φ^phase
        """
        new_batch = int(base_batch * (PHI ** phase))
        if max_batch is not None:
            new_batch = min(new_batch, max_batch)
        return max(1, new_batch)  # Ensure at least batch size 1
    
    @staticmethod
    def phi_training_phases(total_epochs: int) -> Tuple[int, int]:
        """
        Split training into PHI-proportioned phases.
        
        Phase 1: 38.2% of epochs (1/φ²)
        Phase 2: 61.8% of epochs (1/φ)
        """
        phase1_epochs = max(1, int(total_epochs * INV_PHI_SQUARED))
        phase2_epochs = total_epochs - phase1_epochs
        return phase1_epochs, phase2_epochs
    
    @staticmethod
    def phi_fibonacci_weights(length: int) -> np.ndarray:
        """
        Generate PHI-based weights using Fibonacci sequence.
        
        Useful for data weighting and curriculum learning.
        """
        if length <= 0:
            return np.array([])
        
        # Generate Fibonacci sequence
        fibs = fibonacci_sequence(length)
        if len(fibs) < length:
            # Extend with PHI-based progression
            last_fib = fibs[-1] if fibs else 1
            while len(fibs) < length:
                next_fib = int(last_fib * PHI)
                fibs.append(next_fib)
                last_fib = next_fib
        
        # Normalize to create weights
        weights = np.array(fibs[:length], dtype=np.float32)
        return weights / weights.sum()
    
    @staticmethod
    def phi_regularization_schedule(base_value: float, step: int, total_steps: int, 
                                  schedule_type: str = "decay") -> float:
        """
        PHI-based regularization scheduling for dropout and weight decay.
        
        schedule_type: 'decay', 'increase', 'cyclic'
        """
        if total_steps <= 0:
            return base_value
        
        progress = step / total_steps
        
        if schedule_type == "decay":
            # Decrease regularization as training progresses (φ-decay)
            return base_value * (PHI ** (-progress))
        elif schedule_type == "increase":
            # Increase regularization as training progresses
            return base_value * (1.0 + progress * (PHI - 1.0))
        elif schedule_type == "cyclic":
            # Cyclic regularization with PHI period
            cycle_progress = (step % int(total_steps / PHI)) / (total_steps / PHI)
            return base_value * (1.0 + 0.5 * math.sin(2 * math.pi * cycle_progress))
        else:
            return base_value


class PHILearningRateScheduler:
    """PHI-based learning rate scheduler."""
    
    def __init__(self, config: PHITrainingConfig, total_steps: int):
        self.config = config
        self.total_steps = total_steps
        self.warmup_steps = config.warmup_epochs if config.warmup_epochs > 0 else 0
        
    def get_lr(self, step: int) -> float:
        """Get learning rate for given step."""
        base_lr = self.config.base_learning_rate
        
        # Handle warmup phase
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return PHIMath.phi_warmup(base_lr, step, self.warmup_steps)
        
        # Adjust step for post-warmup scheduling
        adjusted_step = step - self.warmup_steps
        adjusted_total = self.total_steps - self.warmup_steps
        
        if self.config.lr_schedule_mode == "phi_decay":
            return PHIMath.phi_decay(base_lr, adjusted_step, adjusted_total, self.config.phi_lr_power)
        elif self.config.lr_schedule_mode == "phi_cosine":
            return PHIMath.phi_cosine(base_lr, adjusted_step, adjusted_total)
        elif self.config.lr_schedule_mode == "phi_cyclic":
            cycle_length = adjusted_total // 4  # 4 cycles by default
            return PHIMath.phi_cyclic(base_lr, adjusted_step, cycle_length, 4)
        else:
            return base_lr


class PHIBatchScheduler:
    """PHI-based batch size scheduler."""
    
    def __init__(self, config: PHITrainingConfig, total_epochs: int):
        self.config = config
        self.total_epochs = total_epochs
        self.phase_epochs = PHIMath.phi_training_phases(total_epochs)
        
    def get_batch_size(self, epoch: int) -> int:
        """Get batch size for given epoch."""
        if not self.config.phi_batch_progression:
            return self.config.base_batch_size
        
        # Determine which phase we're in
        if epoch < self.phase_epochs[0]:
            # Phase 1: smaller batches for exploration
            phase = 0
        else:
            # Phase 2: larger batches for exploitation
            phase = min(self.config.batch_phi_phases - 1, 
                       1 + (epoch - self.phase_epochs[0]) // (self.phase_epochs[1] // max(1, self.config.batch_phi_phases - 1)))
        
        return PHIMath.phi_batch_progression(
            self.config.base_batch_size, 
            phase, 
            self.config.max_batch_size
        )


class PHIRegularizationScheduler:
    """PHI-based regularization scheduler for dropout and weight decay."""
    
    def __init__(self, config: PHITrainingConfig, total_steps: int):
        self.config = config
        self.total_steps = total_steps
        
    def get_dropout(self, step: int) -> float:
        """Get dropout rate for given step."""
        if not self.config.phi_dropout_schedule:
            return self.config.base_dropout
        
        # Use decay schedule - start high, decrease as training progresses
        return PHIMath.phi_regularization_schedule(
            self.config.base_dropout, step, self.total_steps, "decay"
        )
    
    def get_weight_decay(self, step: int) -> float:
        """Get weight decay for given step."""
        if not self.config.phi_weight_decay_schedule:
            return self.config.base_weight_decay
        
        # Use increase schedule - start low, increase as training progresses
        return PHIMath.phi_regularization_schedule(
            self.config.base_weight_decay, step, self.total_steps, "increase"
        )


class PHIDataSampler:
    """PHI-based data sampling and weighting."""
    
    def __init__(self, config: PHITrainingConfig, dataset_size: int):
        self.config = config
        self.dataset_size = dataset_size
        self._phi_weights = None
        
    def get_sample_weights(self) -> Optional[np.ndarray]:
        """Get PHI-based sample weights for the dataset."""
        if not self.config.phi_data_weighting:
            return None
        
        if self._phi_weights is None:
            self._phi_weights = PHIMath.phi_fibonacci_weights(self.dataset_size)
        
        return self._phi_weights
    
    def get_curriculum_order(self, difficulty_scores: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get PHI-based curriculum learning order.
        
        If difficulty_scores provided, orders samples by PHI-weighted difficulty.
        Otherwise, uses PHI-based random sampling.
        """
        if not self.config.phi_curriculum_learning:
            return np.arange(self.dataset_size)
        
        if difficulty_scores is not None:
            # Sort by difficulty, but weight by PHI sequence
            phi_weights = PHIMath.phi_fibonacci_weights(self.dataset_size)
            weighted_difficulty = difficulty_scores * phi_weights
            return np.argsort(weighted_difficulty)
        else:
            # PHI-based pseudo-random ordering
            phi_weights = PHIMath.phi_fibonacci_weights(self.dataset_size)
            return np.argsort(-phi_weights)  # Descending order


# Utility functions for PHI training analysis
def analyze_phi_schedule(scheduler_func, total_steps: int, **kwargs) -> Dict[str, np.ndarray]:
    """
    Analyze a PHI scheduler function over its full range.
    
    Returns arrays of steps and corresponding values for plotting/analysis.
    """
    steps = np.arange(total_steps)
    values = np.array([scheduler_func(step, **kwargs) for step in steps])
    
    return {
        "steps": steps,
        "values": values,
        "min_value": values.min(),
        "max_value": values.max(),
        "final_value": values[-1],
        "phi_ratio": values[0] / values[-1] if values[-1] != 0 else float('inf')
    }


def validate_phi_config(config: PHITrainingConfig) -> List[str]:
    """
    Validate PHI training configuration and return list of warnings/errors.
    """
    warnings = []
    
    if config.base_learning_rate <= 0:
        warnings.append("base_learning_rate must be positive")
    
    if config.base_batch_size <= 0:
        warnings.append("base_batch_size must be positive")
    
    if config.max_batch_size is not None and config.max_batch_size < config.base_batch_size:
        warnings.append("max_batch_size should be >= base_batch_size")
    
    if config.base_dropout < 0 or config.base_dropout > 1:
        warnings.append("base_dropout should be between 0 and 1")
    
    if config.base_weight_decay < 0:
        warnings.append("base_weight_decay should be non-negative")
    
    if config.lr_schedule_mode not in ["phi_decay", "phi_warmup", "phi_cosine", "phi_cyclic"]:
        warnings.append(f"Unknown lr_schedule_mode: {config.lr_schedule_mode}")
    
    return warnings
