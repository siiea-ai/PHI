"""
Unit tests for PHI HuggingFace integration components.
"""

import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW
import numpy as np
from unittest.mock import Mock, patch

from phi.training import PHITrainingConfig
from phi.hf_integration import (
    PHILRScheduler, PHITrainerCallback, PHIOptimizer,
    create_phi_lr_scheduler, setup_phi_training, get_phi_schedule_lambda
)


class DummyModel(nn.Module):
    """Simple model for testing."""
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        return self.linear2(x)


class TestPHILRScheduler:
    """Test PHI learning rate scheduler."""
    
    def test_scheduler_initialization(self):
        """Test scheduler initialization."""
        model = DummyModel()
        optimizer = AdamW(model.parameters(), lr=1e-3)
        phi_config = PHITrainingConfig(base_learning_rate=1e-3)
        total_steps = 1000
        
        scheduler = PHILRScheduler(optimizer, phi_config, total_steps)
        
        assert scheduler.phi_config == phi_config
        assert scheduler.total_steps == total_steps
        assert len(scheduler.base_lrs) == len(optimizer.param_groups)
    
    def test_learning_rate_decay(self):
        """Test that learning rate decays according to PHI schedule."""
        model = DummyModel()
        optimizer = AdamW(model.parameters(), lr=1e-3)
        phi_config = PHITrainingConfig(
            base_learning_rate=1e-3,
            lr_schedule_mode="phi_decay"
        )
        total_steps = 100
        
        scheduler = PHILRScheduler(optimizer, phi_config, total_steps)
        
        # Get initial learning rate
        initial_lrs = scheduler.get_lr()
        
        # Step forward and check decay
        for step in range(10):
            scheduler.step()
        
        current_lrs = scheduler.get_lr()
        
        # Learning rate should have decreased
        assert all(current < initial for current, initial in zip(current_lrs, initial_lrs))
    
    def test_warmup_schedule(self):
        """Test warmup functionality."""
        model = DummyModel()
        optimizer = AdamW(model.parameters(), lr=1e-3)
        phi_config = PHITrainingConfig(
            base_learning_rate=1e-3,
            lr_schedule_mode="phi_decay",
            warmup_epochs=10
        )
        total_steps = 100
        
        scheduler = PHILRScheduler(optimizer, phi_config, total_steps)
        
        # During warmup, LR should increase
        warmup_lrs = []
        for step in range(15):
            warmup_lrs.append(scheduler.get_lr()[0])
            scheduler.step()
        
        # First 10 steps should show increasing trend (warmup)
        warmup_portion = warmup_lrs[:10]
        assert warmup_portion[5] > warmup_portion[0]  # Should increase during warmup


class TestPHITrainerCallback:
    """Test PHI trainer callback."""
    
    def test_callback_initialization(self):
        """Test callback initialization."""
        phi_config = PHITrainingConfig()
        total_epochs = 10
        
        callback = PHITrainerCallback(phi_config, total_epochs)
        
        assert callback.phi_config == phi_config
        assert callback.total_epochs == total_epochs
        assert callback.current_phase == 1
        assert len(callback.metrics_history) > 0
    
    def test_phase_transition_calculation(self):
        """Test training phase transition calculation."""
        phi_config = PHITrainingConfig(phi_training_phases=True)
        total_epochs = 100
        
        callback = PHITrainerCallback(phi_config, total_epochs)
        
        # Phase transition should occur at ~38% of total epochs
        expected_transition = int(100 * 0.382)  # 38 epochs
        assert abs(callback.phase_transition_epoch - expected_transition) <= 2
    
    @patch('phi.hf_integration.TrainingArguments')
    @patch('phi.hf_integration.TrainerState')
    @patch('phi.hf_integration.TrainerControl')
    def test_epoch_begin_callback(self, mock_control, mock_state, mock_args):
        """Test epoch begin callback functionality."""
        phi_config = PHITrainingConfig(phi_batch_progression=True)
        total_epochs = 10
        callback = PHITrainerCallback(phi_config, total_epochs)
        
        # Mock state
        mock_state.epoch = 5
        
        # Call epoch begin
        callback.on_epoch_begin(mock_args, mock_state, mock_control)
        
        # Should have recorded batch size
        assert len(callback.metrics_history['batch_sizes']) > 0
        assert len(callback.metrics_history['training_phases']) > 0
    
    def test_metrics_history_tracking(self):
        """Test metrics history tracking."""
        phi_config = PHITrainingConfig()
        total_epochs = 10
        callback = PHITrainerCallback(phi_config, total_epochs)
        
        # Get initial metrics
        metrics = callback.get_metrics_history()
        
        # Should have all required keys
        required_keys = ['learning_rates', 'batch_sizes', 'dropout_rates', 
                        'weight_decay_rates', 'training_phases']
        for key in required_keys:
            assert key in metrics
            assert isinstance(metrics[key], list)


class TestPHIOptimizer:
    """Test PHI optimizer wrapper."""
    
    def test_optimizer_wrapping(self):
        """Test optimizer wrapping functionality."""
        model = DummyModel()
        base_optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        phi_config = PHITrainingConfig(phi_weight_decay_schedule=True)
        total_steps = 100
        
        phi_optimizer = PHIOptimizer(base_optimizer, phi_config, total_steps)
        
        assert phi_optimizer.optimizer == base_optimizer
        assert phi_optimizer.phi_config == phi_config
        assert phi_optimizer.total_steps == total_steps
        assert phi_optimizer.current_step == 0
    
    def test_weight_decay_scheduling(self):
        """Test weight decay scheduling during optimization."""
        model = DummyModel()
        base_optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        phi_config = PHITrainingConfig(
            phi_weight_decay_schedule=True,
            base_weight_decay=0.01
        )
        total_steps = 100
        
        phi_optimizer = PHIOptimizer(base_optimizer, phi_config, total_steps)
        
        # Get initial weight decay
        initial_wd = base_optimizer.param_groups[0]['weight_decay']
        
        # Step several times
        for _ in range(10):
            phi_optimizer.step()
        
        # Weight decay should have changed (increased for PHI schedule)
        current_wd = base_optimizer.param_groups[0]['weight_decay']
        assert current_wd != initial_wd
        assert phi_optimizer.current_step == 10
    
    def test_optimizer_delegation(self):
        """Test that optimizer methods are properly delegated."""
        model = DummyModel()
        base_optimizer = AdamW(model.parameters(), lr=1e-3)
        phi_config = PHITrainingConfig()
        total_steps = 100
        
        phi_optimizer = PHIOptimizer(base_optimizer, phi_config, total_steps)
        
        # Test delegation of common optimizer attributes
        assert hasattr(phi_optimizer, 'param_groups')
        assert phi_optimizer.param_groups == base_optimizer.param_groups
        
        # Test state dict operations
        state_dict = phi_optimizer.state_dict()
        assert 'phi_current_step' in state_dict
        
        # Test zero_grad delegation
        phi_optimizer.zero_grad()  # Should not raise error


class TestPHIIntegrationFunctions:
    """Test PHI integration utility functions."""
    
    def test_create_phi_lr_scheduler(self):
        """Test PHI LR scheduler creation function."""
        model = DummyModel()
        optimizer = AdamW(model.parameters(), lr=1e-3)
        phi_config = PHITrainingConfig()
        total_steps = 1000
        
        scheduler = create_phi_lr_scheduler(optimizer, phi_config, total_steps)
        
        assert isinstance(scheduler, PHILRScheduler)
        assert scheduler.phi_config == phi_config
        assert scheduler.total_steps == total_steps
    
    def test_setup_phi_training(self):
        """Test complete PHI training setup."""
        model = DummyModel()
        optimizer = AdamW(model.parameters(), lr=1e-3)
        phi_config = PHITrainingConfig()
        total_epochs = 5
        total_steps = 1000
        
        components = setup_phi_training(
            model, optimizer, phi_config, total_epochs, total_steps
        )
        
        # Should return all required components
        required_keys = ['lr_scheduler', 'optimizer', 'callback', 'phi_config']
        for key in required_keys:
            assert key in components
        
        # Check component types
        assert isinstance(components['lr_scheduler'], PHILRScheduler)
        assert isinstance(components['optimizer'], PHIOptimizer)
        assert isinstance(components['callback'], PHITrainerCallback)
        assert components['phi_config'] == phi_config
    
    def test_phi_schedule_lambda(self):
        """Test PHI schedule lambda function."""
        phi_config = PHITrainingConfig(base_learning_rate=1e-3)
        total_steps = 100
        
        phi_lambda = get_phi_schedule_lambda(phi_config, total_steps)
        
        # Test lambda function
        initial_multiplier = phi_lambda(0)
        mid_multiplier = phi_lambda(50)
        final_multiplier = phi_lambda(100)
        
        # Should be decreasing for decay schedule
        assert initial_multiplier >= mid_multiplier >= final_multiplier
        assert initial_multiplier == pytest.approx(1.0)  # Should start at 1.0


class TestPHIConfigValidation:
    """Test PHI configuration validation in integration context."""
    
    def test_valid_phi_config_integration(self):
        """Test that valid PHI config works in integration."""
        phi_config = PHITrainingConfig(
            base_learning_rate=2e-4,
            lr_schedule_mode="phi_decay",
            phi_batch_progression=True,
            phi_training_phases=True
        )
        
        model = DummyModel()
        optimizer = AdamW(model.parameters(), lr=phi_config.base_learning_rate)
        
        # Should not raise any errors
        components = setup_phi_training(model, optimizer, phi_config, 10, 1000)
        assert components is not None
    
    def test_scheduler_mode_validation(self):
        """Test different scheduler modes work."""
        modes = ["phi_decay", "phi_cosine", "phi_cyclic"]
        
        for mode in modes:
            phi_config = PHITrainingConfig(lr_schedule_mode=mode)
            model = DummyModel()
            optimizer = AdamW(model.parameters(), lr=1e-3)
            
            scheduler = create_phi_lr_scheduler(optimizer, phi_config, 100)
            
            # Should create scheduler without error
            assert scheduler is not None
            assert scheduler.phi_config.lr_schedule_mode == mode


if __name__ == "__main__":
    pytest.main([__file__])
