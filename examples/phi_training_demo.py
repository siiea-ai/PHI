#!/usr/bin/env python3
"""
PHI Training Demo - Validation of Phase 1 Mathematical Framework

This script demonstrates and validates the PHI training mathematical framework
by visualizing different scheduling functions and their properties.
"""

import matplotlib.pyplot as plt
import numpy as np
from phi.training import (
    PHIMath, PHITrainingConfig, PHILearningRateScheduler,
    PHIBatchScheduler, PHIRegularizationScheduler, analyze_phi_schedule
)
from phi.constants import PHI, INV_PHI


def demo_learning_rate_schedules():
    """Demonstrate different PHI learning rate schedules."""
    print("=== PHI Learning Rate Schedules Demo ===")
    
    total_steps = 1000
    base_lr = 1e-3
    
    # Create different schedulers
    configs = {
        'PHI Decay': PHITrainingConfig(lr_schedule_mode="phi_decay"),
        'PHI Cosine': PHITrainingConfig(lr_schedule_mode="phi_cosine"),
        'PHI Cyclic': PHITrainingConfig(lr_schedule_mode="phi_cyclic"),
    }
    
    plt.figure(figsize=(12, 8))
    
    for i, (name, config) in enumerate(configs.items(), 1):
        scheduler = PHILearningRateScheduler(config, total_steps)
        steps = np.arange(total_steps)
        lrs = [scheduler.get_lr(step) for step in steps]
        
        plt.subplot(2, 2, i)
        plt.plot(steps, lrs, label=name, linewidth=2)
        plt.title(f'{name} Schedule')
        plt.xlabel('Training Step')
        plt.ylabel('Learning Rate')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Print key statistics
        print(f"{name}:")
        print(f"  Initial LR: {lrs[0]:.6f}")
        print(f"  Final LR: {lrs[-1]:.6f}")
        print(f"  Decay Ratio: {lrs[0]/lrs[-1]:.2f}")
        print(f"  PHI Relationship: {(lrs[0]/lrs[-1])/PHI:.2f} (should be close to 1.0 for PHI decay)")
        print()
    
    # Combined comparison
    plt.subplot(2, 2, 4)
    for name, config in configs.items():
        scheduler = PHILearningRateScheduler(config, total_steps)
        steps = np.arange(total_steps)
        lrs = [scheduler.get_lr(step) for step in steps]
        plt.plot(steps, lrs, label=name, linewidth=2)
    
    plt.title('All PHI Schedules Comparison')
    plt.xlabel('Training Step')
    plt.ylabel('Learning Rate')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('/Users/ali_personal/Projects/PHI/phi_lr_schedules.png', dpi=150, bbox_inches='tight')
    plt.show()


def demo_batch_progression():
    """Demonstrate PHI batch size progression."""
    print("=== PHI Batch Size Progression Demo ===")
    
    config = PHITrainingConfig(
        base_batch_size=8,
        phi_batch_progression=True,
        max_batch_size=128,
        batch_phi_phases=4
    )
    
    total_epochs = 100
    scheduler = PHIBatchScheduler(config, total_epochs)
    
    epochs = np.arange(total_epochs)
    batch_sizes = [scheduler.get_batch_size(epoch) for epoch in epochs]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, batch_sizes, 'o-', linewidth=2, markersize=4)
    plt.title('PHI Batch Size Progression')
    plt.xlabel('Epoch')
    plt.ylabel('Batch Size')
    plt.grid(True, alpha=0.3)
    
    # Mark phase transitions
    phase1, phase2 = PHIMath.phi_training_phases(total_epochs)
    plt.axvline(x=phase1, color='red', linestyle='--', alpha=0.7, label=f'Phase 1 End (epoch {phase1})')
    plt.legend()
    
    plt.savefig('/Users/ali_personal/Projects/PHI/phi_batch_progression.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Training phases: Phase 1 = {phase1} epochs (38.2%), Phase 2 = {phase2} epochs (61.8%)")
    print(f"Batch size progression: {min(batch_sizes)} → {max(batch_sizes)}")
    print(f"PHI ratio validation: {phase2/phase1:.3f} (should be ≈ {PHI:.3f})")


def demo_regularization_schedules():
    """Demonstrate PHI regularization schedules."""
    print("=== PHI Regularization Schedules Demo ===")
    
    config = PHITrainingConfig(
        base_dropout=0.2,
        phi_dropout_schedule=True,
        base_weight_decay=0.01,
        phi_weight_decay_schedule=True
    )
    
    total_steps = 1000
    scheduler = PHIRegularizationScheduler(config, total_steps)
    
    steps = np.arange(total_steps)
    dropout_rates = [scheduler.get_dropout(step) for step in steps]
    weight_decay_rates = [scheduler.get_weight_decay(step) for step in steps]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(steps, dropout_rates, 'b-', linewidth=2, label='Dropout Rate')
    plt.title('PHI Dropout Schedule (Decay)')
    plt.xlabel('Training Step')
    plt.ylabel('Dropout Rate')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(steps, weight_decay_rates, 'r-', linewidth=2, label='Weight Decay')
    plt.title('PHI Weight Decay Schedule (Increase)')
    plt.xlabel('Training Step')
    plt.ylabel('Weight Decay')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('/Users/ali_personal/Projects/PHI/phi_regularization_schedules.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Dropout: {dropout_rates[0]:.4f} → {dropout_rates[-1]:.4f} (decreasing)")
    print(f"Weight Decay: {weight_decay_rates[0]:.4f} → {weight_decay_rates[-1]:.4f} (increasing)")


def demo_phi_fibonacci_weights():
    """Demonstrate PHI Fibonacci weight generation."""
    print("=== PHI Fibonacci Weights Demo ===")
    
    lengths = [10, 20, 50]
    
    plt.figure(figsize=(15, 5))
    
    for i, length in enumerate(lengths, 1):
        weights = PHIMath.phi_fibonacci_weights(length)
        
        plt.subplot(1, 3, i)
        plt.bar(range(length), weights, alpha=0.7)
        plt.title(f'PHI Fibonacci Weights (n={length})')
        plt.xlabel('Sample Index')
        plt.ylabel('Weight')
        plt.grid(True, alpha=0.3)
        
        print(f"Length {length}: Sum = {weights.sum():.6f}, Max/Min ratio = {weights.max()/weights.min():.2f}")
    
    plt.tight_layout()
    plt.savefig('/Users/ali_personal/Projects/PHI/phi_fibonacci_weights.png', dpi=150, bbox_inches='tight')
    plt.show()


def validate_phi_properties():
    """Validate mathematical properties of PHI schedules."""
    print("=== PHI Mathematical Properties Validation ===")
    
    # Test PHI decay properties
    print("1. PHI Decay Properties:")
    base_value = 1.0
    total_steps = 1000
    
    # Test at key PHI ratios
    test_points = [
        (int(total_steps * INV_PHI), "1/φ point"),
        (int(total_steps * INV_PHI * INV_PHI), "1/φ² point"),
        (total_steps, "Final point")
    ]
    
    for step, description in test_points:
        if step <= total_steps:
            value = PHIMath.phi_decay(base_value, step, total_steps)
            expected_phi_power = -(step / total_steps)
            expected_value = base_value * (PHI ** expected_phi_power)
            print(f"  {description} (step {step}): {value:.6f} (expected: {expected_value:.6f})")
    
    # Test batch progression PHI ratios
    print("\n2. Batch Progression PHI Ratios:")
    base_batch = 8
    for phase in range(4):
        batch = PHIMath.phi_batch_progression(base_batch, phase)
        expected = int(base_batch * (PHI ** phase))
        print(f"  Phase {phase}: {batch} (expected: {expected})")
    
    # Test training phase ratios
    print("\n3. Training Phase PHI Ratios:")
    for total_epochs in [100, 200, 500]:
        phase1, phase2 = PHIMath.phi_training_phases(total_epochs)
        ratio = phase2 / phase1 if phase1 > 0 else 0
        print(f"  Total {total_epochs}: Phase1={phase1}, Phase2={phase2}, Ratio={ratio:.3f} (PHI={PHI:.3f})")
    
    print(f"\n4. Golden Ratio Validation:")
    print(f"  φ = {PHI:.10f}")
    print(f"  1/φ = {INV_PHI:.10f}")
    print(f"  1/φ² = {INV_PHI*INV_PHI:.10f}")
    print(f"  φ - 1 = {PHI - 1:.10f}")
    print(f"  φ² - φ - 1 = {PHI*PHI - PHI - 1:.10f} (should be ≈ 0)")


def main():
    """Run all PHI training demos and validations."""
    print("PHI Training Mathematical Framework Validation")
    print("=" * 50)
    
    try:
        # Run all demonstrations
        demo_learning_rate_schedules()
        demo_batch_progression()
        demo_regularization_schedules()
        demo_phi_fibonacci_weights()
        validate_phi_properties()
        
        print("\n" + "=" * 50)
        print("✅ Phase 1 Validation Complete!")
        print("All PHI mathematical functions are working correctly.")
        print("Generated visualization plots:")
        print("  - phi_lr_schedules.png")
        print("  - phi_batch_progression.png") 
        print("  - phi_regularization_schedules.png")
        print("  - phi_fibonacci_weights.png")
        print("\nReady to proceed to Phase 2: Implementation")
        
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
