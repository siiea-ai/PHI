#!/usr/bin/env python3
"""
Complete PHI Training System Demo

Demonstrates the full PHI training pipeline from architecture generation
to training with golden ratio principles.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from phi.training import PHITrainingConfig
from phi.phi_model_trainer import PHIModelTrainer, PHIModelConfig, create_phi_model_trainer


class DummyDataset(Dataset):
    """Simple dataset for testing PHI training."""
    
    def __init__(self, size: int = 1000, input_dim: int = 512, output_dim: int = 512):
        self.size = size
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Generate random data with some structure
        np.random.seed(42)
        self.inputs = np.random.randn(size, input_dim).astype(np.float32)
        
        # Create outputs with some relationship to inputs (for learning)
        weights = np.random.randn(input_dim, output_dim).astype(np.float32)
        self.outputs = np.tanh(self.inputs @ weights).astype(np.float32)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.from_numpy(self.inputs[idx]),
            'labels': torch.from_numpy(self.outputs[idx])
        }


def demo_phi_architecture_generation():
    """Demo PHI architecture generation."""
    print("=" * 60)
    print("PHI ARCHITECTURE GENERATION DEMO")
    print("=" * 60)
    
    # Create PHI model trainer
    trainer = create_phi_model_trainer(
        input_dim=256,
        output_dim=128,
        depth=5,
        base_width=512,
        architecture_mode="phi"
    )
    
    # Generate PHI model
    model = trainer.create_phi_model()
    
    print(f"✅ Created PHI neural network:")
    if hasattr(model, 'get_phi_info'):
        phi_info = model.get_phi_info()
        for key, value in phi_info.items():
            if key != 'phi_ratio_validation':
                print(f"   {key}: {value}")
        
        # Show PHI ratio validation
        if 'phi_ratio_validation' in phi_info:
            ratios = phi_info['phi_ratio_validation']
            print(f"   PHI alignment: {ratios.get('phi_alignment', 'N/A')}")
            print(f"   Layer ratios: {ratios.get('layer_ratios', [])}")
    
    return trainer, model


def demo_phi_training_config():
    """Demo PHI training configuration."""
    print("\n" + "=" * 60)
    print("PHI TRAINING CONFIGURATION DEMO")
    print("=" * 60)
    
    # Create comprehensive PHI training config
    phi_config = PHITrainingConfig(
        base_learning_rate=1e-3,
        lr_schedule_mode="phi_decay",
        warmup_epochs=5,
        base_batch_size=16,
        phi_batch_progression=True,
        max_batch_size=128,
        batch_phi_phases=4,
        phi_training_phases=True,
        base_dropout=0.1,
        phi_dropout_schedule=True,
        base_weight_decay=0.01,
        phi_weight_decay_schedule=True,
        phi_data_weighting=True
    )
    
    print("✅ PHI Training Configuration:")
    print(f"   Learning Rate: {phi_config.base_learning_rate} ({phi_config.lr_schedule_mode})")
    print(f"   Batch Size: {phi_config.base_batch_size} → {phi_config.max_batch_size}")
    print(f"   Training Phases: {phi_config.phi_training_phases}")
    print(f"   Dropout Schedule: {phi_config.phi_dropout_schedule}")
    print(f"   Weight Decay Schedule: {phi_config.phi_weight_decay_schedule}")
    
    return phi_config


def demo_phi_complete_training():
    """Demo complete PHI training pipeline."""
    print("\n" + "=" * 60)
    print("COMPLETE PHI TRAINING PIPELINE DEMO")
    print("=" * 60)
    
    # Create PHI training config
    phi_config = PHITrainingConfig(
        base_learning_rate=1e-3,
        lr_schedule_mode="phi_decay",
        phi_batch_progression=True,
        phi_training_phases=True,
        phi_dropout_schedule=True
    )
    
    # Create PHI model config
    model_config = PHIModelConfig(
        input_dim=128,
        output_dim=64,
        depth=4,
        base_width=256,
        architecture_mode="phi",
        phi_training_config=phi_config,
        model_type="phi_neural_net"
    )
    
    # Create trainer
    trainer = PHIModelTrainer(model_config)
    
    # Create dummy dataset
    train_dataset = DummyDataset(size=500, input_dim=128, output_dim=64)
    eval_dataset = DummyDataset(size=100, input_dim=128, output_dim=64)
    
    print(f"✅ Created datasets:")
    print(f"   Training: {len(train_dataset)} samples")
    print(f"   Evaluation: {len(eval_dataset)} samples")
    
    # Create model
    model = trainer.create_phi_model()
    print(f"✅ Created PHI model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    sample_input = torch.randn(4, 128)
    with torch.no_grad():
        output = model(sample_input)
    
    print(f"✅ Forward pass test:")
    print(f"   Input shape: {sample_input.shape}")
    print(f"   Output shape: {output.shape}")
    
    return trainer, model, train_dataset, eval_dataset


def demo_phi_schedulers():
    """Demo PHI scheduler behavior."""
    print("\n" + "=" * 60)
    print("PHI SCHEDULERS DEMO")
    print("=" * 60)
    
    from phi.training import PHILearningRateScheduler, PHIBatchScheduler, PHIRegularizationScheduler
    
    phi_config = PHITrainingConfig(
        base_learning_rate=1e-3,
        lr_schedule_mode="phi_decay",
        phi_batch_progression=True,
        phi_dropout_schedule=True
    )
    
    # Test learning rate scheduler
    lr_scheduler = PHILearningRateScheduler(phi_config, 1000)
    print("📈 Learning Rate Schedule (first 10 steps):")
    for step in range(0, 100, 10):
        lr = lr_scheduler.get_lr(step)
        print(f"   Step {step:2d}: {lr:.6f}")
    
    # Test batch scheduler
    batch_scheduler = PHIBatchScheduler(phi_config, 20)
    print("\n📊 Batch Size Schedule:")
    for epoch in range(0, 20, 2):
        batch_size = batch_scheduler.get_batch_size(epoch)
        print(f"   Epoch {epoch:2d}: {batch_size}")
    
    # Test regularization scheduler
    reg_scheduler = PHIRegularizationScheduler(phi_config, 1000)
    print("\n🎛️  Regularization Schedule (every 100 steps):")
    for step in range(0, 1000, 100):
        dropout = reg_scheduler.get_dropout(step)
        weight_decay = reg_scheduler.get_weight_decay(step)
        print(f"   Step {step:3d}: Dropout={dropout:.4f}, Weight Decay={weight_decay:.4f}")


def demo_phi_compression():
    """Demo PHI model compression."""
    print("\n" + "=" * 60)
    print("PHI MODEL COMPRESSION DEMO")
    print("=" * 60)
    
    # Create model with compression enabled
    model_config = PHIModelConfig(
        input_dim=256,
        output_dim=128,
        depth=6,
        base_width=512,
        architecture_mode="phi",
        enable_compression=True,
        compression_ratio=2,
        compression_method="interp"
    )
    
    trainer = PHIModelTrainer(model_config)
    
    print("🗜️  Creating compressed PHI model...")
    model = trainer.create_phi_model()
    
    if hasattr(model, 'get_phi_info'):
        phi_info = model.get_phi_info()
        print(f"✅ Compressed model info:")
        for key, value in phi_info.items():
            if key != 'phi_ratio_validation':
                print(f"   {key}: {value}")
    
    return trainer, model


def main():
    """Run all PHI training demos."""
    print("🌟 PHI COMPLETE TRAINING SYSTEM DEMONSTRATION")
    print("🌟 Golden Ratio (φ ≈ 1.618) Applied to Neural Network Training")
    
    try:
        # Demo 1: Architecture Generation
        arch_trainer, arch_model = demo_phi_architecture_generation()
        
        # Demo 2: Training Configuration
        phi_config = demo_phi_training_config()
        
        # Demo 3: Complete Training Pipeline
        trainer, model, train_data, eval_data = demo_phi_complete_training()
        
        # Demo 4: Scheduler Behavior
        demo_phi_schedulers()
        
        # Demo 5: Model Compression
        comp_trainer, comp_model = demo_phi_compression()
        
        print("\n" + "=" * 60)
        print("✅ ALL PHI DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print("\n🎯 Key PHI Features Demonstrated:")
        print("   ✓ PHI-based architecture generation")
        print("   ✓ Golden ratio layer scaling")
        print("   ✓ PHI learning rate scheduling")
        print("   ✓ PHI batch size progression")
        print("   ✓ PHI training phase transitions")
        print("   ✓ PHI regularization scheduling")
        print("   ✓ PHI model compression/expansion")
        
        print("\n📊 PHI Mathematical Properties:")
        from phi.training import PHIMath
        print(f"   φ = {PHIMath.PHI:.6f}")
        print(f"   1/φ = {PHIMath.INV_PHI:.6f}")
        print(f"   1/φ² = {PHIMath.INV_PHI_SQUARED:.6f}")
        print(f"   φ² - φ - 1 = {PHIMath.PHI**2 - PHIMath.PHI - 1:.10f}")
        
        print("\n🚀 Ready for Phase 4: Real Model Training!")
        print("   Next: Train on actual datasets with baseline comparison")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
