"""
Unit tests for PHI training utilities and mathematical framework.
"""

import pytest
import numpy as np
import math
from phi.training import (
    PHITrainingConfig, PHIMath, PHILearningRateScheduler, 
    PHIBatchScheduler, PHIRegularizationScheduler, PHIDataSampler,
    analyze_phi_schedule, validate_phi_config
)
from phi.constants import PHI, INV_PHI


class TestPHIMath:
    """Test PHI mathematical functions."""
    
    def test_phi_decay(self):
        """Test PHI decay function."""
        base_value = 1.0
        total_steps = 100
        
        # At step 0, should return base value
        assert PHIMath.phi_decay(base_value, 0, total_steps) == pytest.approx(base_value)
        
        # At final step, should be significantly reduced
        final_value = PHIMath.phi_decay(base_value, total_steps, total_steps)
        assert final_value < base_value
        assert final_value == pytest.approx(base_value * (PHI ** -1.0))
        
        # Should be monotonically decreasing
        values = [PHIMath.phi_decay(base_value, step, total_steps) for step in range(0, total_steps, 10)]
        assert all(values[i] >= values[i+1] for i in range(len(values)-1))
    
    def test_phi_warmup(self):
        """Test PHI warmup function."""
        base_value = 1.0
        warmup_steps = 50
        
        # At step 0, should be close to 0
        initial_value = PHIMath.phi_warmup(base_value, 0, warmup_steps)
        assert initial_value < base_value
        assert initial_value >= 0
        
        # At warmup completion, should approach base value
        final_value = PHIMath.phi_warmup(base_value, warmup_steps, warmup_steps)
        assert final_value == pytest.approx(base_value)
        
        # Should be monotonically increasing
        values = [PHIMath.phi_warmup(base_value, step, warmup_steps) for step in range(0, warmup_steps, 5)]
        assert all(values[i] <= values[i+1] for i in range(len(values)-1))
    
    def test_phi_cosine(self):
        """Test PHI-modulated cosine annealing."""
        base_value = 1.0
        total_steps = 100
        min_value = 0.1
        
        # At step 0, should be close to base value
        initial_value = PHIMath.phi_cosine(base_value, 0, total_steps, min_value)
        assert initial_value <= base_value
        assert initial_value >= min_value
        
        # At final step, should be close to min value
        final_value = PHIMath.phi_cosine(base_value, total_steps, total_steps, min_value)
        assert final_value >= min_value
        assert final_value < base_value
    
    def test_phi_batch_progression(self):
        """Test PHI batch size progression."""
        base_batch = 8
        
        # Phase 0 should return base batch
        assert PHIMath.phi_batch_progression(base_batch, 0) == base_batch
        
        # Each phase should increase by PHI
        phase1_batch = PHIMath.phi_batch_progression(base_batch, 1)
        expected = int(base_batch * PHI)
        assert phase1_batch == expected
        
        # Should respect max batch size
        max_batch = 16
        large_phase_batch = PHIMath.phi_batch_progression(base_batch, 3, max_batch)
        assert large_phase_batch <= max_batch
    
    def test_phi_training_phases(self):
        """Test PHI training phase splitting."""
        total_epochs = 100
        phase1, phase2 = PHIMath.phi_training_phases(total_epochs)
        
        # Should sum to total epochs
        assert phase1 + phase2 == total_epochs
        
        # Phase 1 should be approximately 38.2% (1/φ²)
        expected_phase1 = int(total_epochs * INV_PHI * INV_PHI)
        assert abs(phase1 - expected_phase1) <= 1
        
        # Both phases should be positive
        assert phase1 > 0
        assert phase2 > 0
    
    def test_phi_fibonacci_weights(self):
        """Test PHI Fibonacci weight generation."""
        length = 10
        weights = PHIMath.phi_fibonacci_weights(length)
        
        # Should have correct length
        assert len(weights) == length
        
        # Should sum to 1 (normalized)
        assert weights.sum() == pytest.approx(1.0)
        
        # All weights should be positive
        assert all(w > 0 for w in weights)
        
        # Should be roughly increasing (Fibonacci-based)
        # Allow some tolerance for normalization effects
        increasing_count = sum(1 for i in range(len(weights)-1) if weights[i] <= weights[i+1])
        assert increasing_count >= len(weights) * 0.7  # At least 70% increasing


class TestPHISchedulers:
    """Test PHI scheduler classes."""
    
    def test_learning_rate_scheduler(self):
        """Test PHI learning rate scheduler."""
        config = PHITrainingConfig(
            base_learning_rate=1e-3,
            lr_schedule_mode="phi_decay",
            warmup_epochs=10
        )
        total_steps = 1000
        scheduler = PHILearningRateScheduler(config, total_steps)
        
        # Test warmup phase
        warmup_lr = scheduler.get_lr(5)
        assert 0 < warmup_lr < config.base_learning_rate
        
        # Test post-warmup decay
        mid_lr = scheduler.get_lr(500)
        final_lr = scheduler.get_lr(999)
        
        assert config.base_learning_rate > mid_lr > final_lr > 0
    
    def test_batch_scheduler(self):
        """Test PHI batch size scheduler."""
        config = PHITrainingConfig(
            base_batch_size=8,
            phi_batch_progression=True,
            max_batch_size=64,
            batch_phi_phases=3
        )
        total_epochs = 100
        scheduler = PHIBatchScheduler(config, total_epochs)
        
        # Test phase 1 (early epochs)
        early_batch = scheduler.get_batch_size(10)
        assert early_batch >= config.base_batch_size
        
        # Test phase 2 (later epochs)
        late_batch = scheduler.get_batch_size(80)
        assert late_batch >= early_batch
        assert late_batch <= config.max_batch_size
    
    def test_regularization_scheduler(self):
        """Test PHI regularization scheduler."""
        config = PHITrainingConfig(
            base_dropout=0.2,
            phi_dropout_schedule=True,
            base_weight_decay=0.01,
            phi_weight_decay_schedule=True
        )
        total_steps = 1000
        scheduler = PHIRegularizationScheduler(config, total_steps)
        
        # Test dropout decay (should decrease over time)
        early_dropout = scheduler.get_dropout(100)
        late_dropout = scheduler.get_dropout(900)
        assert early_dropout > late_dropout
        assert 0 <= late_dropout <= early_dropout <= 1
        
        # Test weight decay increase (should increase over time)
        early_wd = scheduler.get_weight_decay(100)
        late_wd = scheduler.get_weight_decay(900)
        assert late_wd >= early_wd
        assert early_wd >= 0
    
    def test_data_sampler(self):
        """Test PHI data sampler."""
        config = PHITrainingConfig(
            phi_data_weighting=True,
            phi_curriculum_learning=True
        )
        dataset_size = 100
        sampler = PHIDataSampler(config, dataset_size)
        
        # Test sample weights
        weights = sampler.get_sample_weights()
        assert weights is not None
        assert len(weights) == dataset_size
        assert weights.sum() == pytest.approx(1.0)
        
        # Test curriculum order
        order = sampler.get_curriculum_order()
        assert len(order) == dataset_size
        assert set(order) == set(range(dataset_size))  # All indices present
        
        # Test with difficulty scores
        difficulty = np.random.random(dataset_size)
        order_with_difficulty = sampler.get_curriculum_order(difficulty)
        assert len(order_with_difficulty) == dataset_size


class TestPHIConfig:
    """Test PHI configuration validation."""
    
    def test_valid_config(self):
        """Test valid configuration passes validation."""
        config = PHITrainingConfig()
        warnings = validate_phi_config(config)
        assert len(warnings) == 0
    
    def test_invalid_learning_rate(self):
        """Test invalid learning rate generates warning."""
        config = PHITrainingConfig(base_learning_rate=-1.0)
        warnings = validate_phi_config(config)
        assert any("learning_rate" in w for w in warnings)
    
    def test_invalid_batch_size(self):
        """Test invalid batch size generates warning."""
        config = PHITrainingConfig(base_batch_size=0)
        warnings = validate_phi_config(config)
        assert any("batch_size" in w for w in warnings)
    
    def test_invalid_dropout(self):
        """Test invalid dropout generates warning."""
        config = PHITrainingConfig(base_dropout=1.5)
        warnings = validate_phi_config(config)
        assert any("dropout" in w for w in warnings)
    
    def test_invalid_schedule_mode(self):
        """Test invalid schedule mode generates warning."""
        config = PHITrainingConfig(lr_schedule_mode="invalid_mode")
        warnings = validate_phi_config(config)
        assert any("lr_schedule_mode" in w for w in warnings)


class TestPHIAnalysis:
    """Test PHI analysis utilities."""
    
    def test_analyze_phi_schedule(self):
        """Test PHI schedule analysis."""
        def test_scheduler(step, base_value=1.0, total_steps=100):
            return PHIMath.phi_decay(base_value, step, total_steps)
        
        analysis = analyze_phi_schedule(test_scheduler, 100, base_value=1.0, total_steps=100)
        
        # Check required keys
        required_keys = ["steps", "values", "min_value", "max_value", "final_value", "phi_ratio"]
        assert all(key in analysis for key in required_keys)
        
        # Check array lengths
        assert len(analysis["steps"]) == 100
        assert len(analysis["values"]) == 100
        
        # Check monotonic decrease for decay function
        values = analysis["values"]
        assert values[0] > values[-1]  # Should decrease
        assert analysis["max_value"] == values[0]
        assert analysis["min_value"] == values[-1]


if __name__ == "__main__":
    pytest.main([__file__])
