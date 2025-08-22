"""
Complete PHI model training system integrating architecture generation with training.

This module combines PHI-based architecture generation with PHI-based training
to create a unified golden ratio training system.
"""

from __future__ import annotations

import os
import json
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

try:
    from transformers import PreTrainedModel, PreTrainedTokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    PreTrainedModel = object
    PreTrainedTokenizer = object

from .ai import generate_full_model, compress_model, expand_model, export_to_keras
from .training import PHITrainingConfig, PHIMath
from .trainer import PHITrainer, create_phi_trainer


class PHIModelConfig:
    """Configuration for PHI model architecture and training."""
    
    def __init__(
        self,
        # Architecture parameters
        input_dim: int = 512,
        output_dim: int = 512,
        depth: int = 6,
        base_width: int = 512,
        architecture_mode: str = "phi",  # phi, fibonacci, fixed
        min_width: int = 64,
        
        # Training parameters
        phi_training_config: Optional[PHITrainingConfig] = None,
        
        # Model type
        model_type: str = "phi_neural_net",  # phi_neural_net, pretrained_hf
        pretrained_model_name: Optional[str] = None,
        
        # Compression settings
        enable_compression: bool = False,
        compression_ratio: int = 2,
        compression_method: str = "interp"
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.base_width = base_width
        self.architecture_mode = architecture_mode
        self.min_width = min_width
        
        self.phi_training_config = phi_training_config or PHITrainingConfig()
        
        self.model_type = model_type
        self.pretrained_model_name = pretrained_model_name
        
        self.enable_compression = enable_compression
        self.compression_ratio = compression_ratio
        self.compression_method = compression_method


class PHINeuralNetwork(nn.Module):
    """
    PyTorch neural network with PHI-based architecture.
    
    Converts PHI AI model bundle to PyTorch module for training.
    """
    
    def __init__(self, model_bundle: Dict):
        super().__init__()
        self.model_bundle = model_bundle
        self.layers = nn.ModuleList()
        
        # Extract layer information from bundle
        input_dim = model_bundle['input_dim']
        output_dim = model_bundle['output_dim']
        hidden_dims = model_bundle.get('hidden', [])
        
        # Build layers
        layer_dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(layer_dims) - 1):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
        
        # Set activation functions
        self.hidden_activation = self._get_activation(model_bundle.get('act_hidden', 'relu'))
        self.output_activation = self._get_activation(model_bundle.get('act_output', 'linear'))
        
        # Initialize weights from bundle if available
        self._load_weights_from_bundle()
    
    def _get_activation(self, activation_name: str):
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'linear': nn.Identity(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU()
        }
        return activations.get(activation_name.lower(), nn.ReLU())
    
    def _load_weights_from_bundle(self):
        """Load weights from PHI model bundle."""
        if 'layers' not in self.model_bundle:
            return
        
        bundle_layers = self.model_bundle['layers']
        if len(bundle_layers) != len(self.layers):
            print(f"Warning: Bundle has {len(bundle_layers)} layers, model has {len(self.layers)}")
            return
        
        try:
            from .ai import _b64_to_arr
            
            for i, (layer, bundle_layer) in enumerate(zip(self.layers, bundle_layers)):
                if 'W' in bundle_layer and 'b' in bundle_layer:
                    weights = _b64_to_arr(bundle_layer['W'])
                    biases = _b64_to_arr(bundle_layer['b'])
                    
                    # Transpose weights for PyTorch (PHI uses input x weight format)
                    weights = weights.T
                    
                    with torch.no_grad():
                        layer.weight.copy_(torch.from_numpy(weights).float())
                        layer.bias.copy_(torch.from_numpy(biases).float())
            
            print("✅ Loaded PHI weights into PyTorch model")
            
        except Exception as e:
            print(f"Warning: Could not load PHI weights: {e}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through PHI neural network."""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # Apply activation (hidden layers use hidden_activation, output uses output_activation)
            if i < len(self.layers) - 1:
                x = self.hidden_activation(x)
            else:
                x = self.output_activation(x)
        
        return x
    
    def get_phi_info(self) -> Dict[str, Any]:
        """Get PHI-specific information about the model."""
        return {
            'architecture_type': 'phi_neural_network',
            'input_dim': self.model_bundle['input_dim'],
            'output_dim': self.model_bundle['output_dim'],
            'hidden_dims': self.model_bundle.get('hidden', []),
            'total_params': sum(p.numel() for p in self.parameters()),
            'phi_ratio_validation': self._validate_phi_ratios()
        }
    
    def _validate_phi_ratios(self) -> Dict[str, float]:
        """Validate that layer dimensions follow PHI ratios."""
        hidden_dims = self.model_bundle.get('hidden', [])
        if len(hidden_dims) < 2:
            return {}
        
        ratios = []
        for i in range(len(hidden_dims) - 1):
            if hidden_dims[i + 1] > 0:
                ratio = hidden_dims[i] / hidden_dims[i + 1]
                ratios.append(ratio)
        
        return {
            'layer_ratios': ratios,
            'mean_ratio': sum(ratios) / len(ratios) if ratios else 0,
            'phi_target': PHIMath.PHI,
            'phi_alignment': abs(sum(ratios) / len(ratios) - PHIMath.PHI) if ratios else float('inf')
        }


class PHIModelTrainer:
    """
    Complete PHI model trainer integrating architecture and training.
    """
    
    def __init__(self, config: PHIModelConfig):
        self.config = config
        self.model = None
        self.phi_model_bundle = None
        self.trainer = None
    
    def create_phi_model(self) -> Union[nn.Module, PreTrainedModel]:
        """Create PHI-based model according to configuration."""
        if self.config.model_type == "phi_neural_net":
            return self._create_phi_neural_network()
        elif self.config.model_type == "pretrained_hf" and HF_AVAILABLE:
            return self._create_pretrained_model()
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
    
    def _create_phi_neural_network(self) -> PHINeuralNetwork:
        """Create PHI neural network from scratch."""
        print(f"🏗️  Creating PHI neural network...")
        print(f"   Architecture: {self.config.architecture_mode}")
        print(f"   Dimensions: {self.config.input_dim} → {self.config.output_dim}")
        print(f"   Depth: {self.config.depth}, Base width: {self.config.base_width}")
        
        # Generate PHI model bundle
        self.phi_model_bundle = generate_full_model(
            input_dim=self.config.input_dim,
            output_dim=self.config.output_dim,
            depth=self.config.depth,
            base_width=self.config.base_width,
            mode=self.config.architecture_mode,
            min_width=self.config.min_width
        )
        
        print(f"✅ Generated PHI model bundle:")
        print(f"   Hidden layers: {self.phi_model_bundle.get('hidden', [])}")
        print(f"   Total parameters: {self.phi_model_bundle.get('param_count', 0):,}")
        
        # Create PyTorch model from bundle
        model = PHINeuralNetwork(self.phi_model_bundle)
        
        # Apply compression if enabled
        if self.config.enable_compression:
            model = self._apply_compression(model)
        
        self.model = model
        return model
    
    def _create_pretrained_model(self) -> PreTrainedModel:
        """Create pretrained HuggingFace model."""
        if not HF_AVAILABLE:
            raise ImportError("HuggingFace Transformers required for pretrained models")
        
        from transformers import AutoModel
        
        model_name = self.config.pretrained_model_name
        print(f"🤗 Loading pretrained model: {model_name}")
        
        model = AutoModel.from_pretrained(model_name)
        self.model = model
        return model
    
    def _apply_compression(self, model: PHINeuralNetwork) -> PHINeuralNetwork:
        """Apply PHI compression to the model."""
        print(f"🗜️  Applying PHI compression (ratio: {self.config.compression_ratio})...")
        
        # Compress the model bundle
        from .ai import AIConfig
        compression_config = AIConfig(
            strategy="ratio",
            ratio=self.config.compression_ratio,
            method=self.config.compression_method
        )
        
        compressed_bundle = compress_model(self.phi_model_bundle, compression_config)
        
        # Expand back to create compressed model
        expanded_bundle = expand_model(
            compressed_bundle,
            method=self.config.compression_method
        )
        
        print(f"✅ Compression complete:")
        print(f"   Original params: {self.phi_model_bundle.get('param_count', 0):,}")
        print(f"   Compressed params: {compressed_bundle.get('ds_param_count', 0):,}")
        
        # Create new model with compressed weights
        compressed_model = PHINeuralNetwork(expanded_bundle)
        return compressed_model
    
    def create_trainer(
        self,
        train_dataset,
        eval_dataset=None,
        output_dir: str = "./phi_model_training",
        num_epochs: int = 3,
        **kwargs
    ) -> PHITrainer:
        """Create PHI trainer for the model."""
        if self.model is None:
            self.model = self.create_phi_model()
        
        # For PHI neural networks, we need to wrap them for HuggingFace compatibility
        if isinstance(self.model, PHINeuralNetwork):
            # Create a simple wrapper for compatibility
            wrapped_model = self._wrap_phi_model_for_hf()
        else:
            wrapped_model = self.model
        
        # Create PHI trainer
        self.trainer = create_phi_trainer(
            model=wrapped_model,
            phi_config=self.config.phi_training_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=output_dir,
            num_epochs=num_epochs,
            **kwargs
        )
        
        return self.trainer
    
    def _wrap_phi_model_for_hf(self):
        """Wrap PHI neural network for HuggingFace Trainer compatibility."""
        # For now, return the model as-is
        # In a full implementation, we'd create a proper HF-compatible wrapper
        return self.model
    
    def train(self, train_dataset, eval_dataset=None, **kwargs):
        """Train the PHI model."""
        if self.trainer is None:
            self.trainer = self.create_trainer(train_dataset, eval_dataset, **kwargs)
        
        print("🚀 Starting PHI model training...")
        
        # Print model info
        if isinstance(self.model, PHINeuralNetwork):
            phi_info = self.model.get_phi_info()
            print(f"📊 PHI Model Info:")
            for key, value in phi_info.items():
                if key != 'phi_ratio_validation':
                    print(f"   {key}: {value}")
            
            if 'phi_ratio_validation' in phi_info:
                ratios = phi_info['phi_ratio_validation']
                if ratios.get('layer_ratios'):
                    print(f"   PHI ratio alignment: {ratios['phi_alignment']:.4f}")
                    print(f"   Mean layer ratio: {ratios['mean_ratio']:.3f} (target: {ratios['phi_target']:.3f})")
        
        # Start training
        result = self.trainer.train()
        
        print("✅ PHI model training complete!")
        return result
    
    def save_complete_model(self, output_dir: str):
        """Save complete PHI model including architecture and training config."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the trained model
        if self.trainer:
            self.trainer.save_model(output_dir)
        
        # Save PHI model bundle
        if self.phi_model_bundle:
            bundle_path = os.path.join(output_dir, "phi_model_bundle.json")
            with open(bundle_path, 'w') as f:
                json.dump(self.phi_model_bundle, f, indent=2)
        
        # Save PHI model config
        config_path = os.path.join(output_dir, "phi_model_config.json")
        config_dict = {
            'input_dim': self.config.input_dim,
            'output_dim': self.config.output_dim,
            'depth': self.config.depth,
            'base_width': self.config.base_width,
            'architecture_mode': self.config.architecture_mode,
            'min_width': self.config.min_width,
            'model_type': self.config.model_type,
            'pretrained_model_name': self.config.pretrained_model_name,
            'enable_compression': self.config.enable_compression,
            'compression_ratio': self.config.compression_ratio,
            'compression_method': self.config.compression_method
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Export to Keras if possible
        if self.phi_model_bundle:
            try:
                keras_path = os.path.join(output_dir, "phi_model_keras.h5")
                export_to_keras(self.phi_model_bundle, keras_path)
                print(f"✅ Exported to Keras: {keras_path}")
            except Exception as e:
                print(f"⚠️  Could not export to Keras: {e}")
        
        print(f"💾 Complete PHI model saved to: {output_dir}")
        print("📁 Generated files:")
        print(f"   - PyTorch model: {output_dir}/pytorch_model.bin")
        print(f"   - PHI bundle: {output_dir}/phi_model_bundle.json")
        print(f"   - PHI config: {output_dir}/phi_model_config.json")
        print(f"   - Training config: {output_dir}/phi_training_config.json")


def create_phi_model_trainer(
    input_dim: int = 512,
    output_dim: int = 512,
    depth: int = 6,
    base_width: int = 512,
    phi_training_config: Optional[PHITrainingConfig] = None,
    **kwargs
) -> PHIModelTrainer:
    """
    Convenience function to create PHI model trainer.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension  
        depth: Network depth
        base_width: Base layer width
        phi_training_config: PHI training configuration
        **kwargs: Additional PHIModelConfig parameters
    
    Returns:
        PHI model trainer
    """
    config = PHIModelConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        depth=depth,
        base_width=base_width,
        phi_training_config=phi_training_config or PHITrainingConfig(),
        **kwargs
    )
    
    return PHIModelTrainer(config)
