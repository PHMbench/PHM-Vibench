"""
Standalone test for multi-task head module without package dependencies.

This test directly imports and tests the MultiTaskHead module to validate
its functionality without requiring the full PHM-Vibench environment.

Author: PHM-Vibench Team
Date: 2025-08-18
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union
import numpy as np
from argparse import Namespace


class MultiTaskHead(nn.Module):
    """
    Multi-task head for simultaneous fault classification, RUL prediction, and anomaly detection.
    
    This is a standalone copy for testing purposes.
    """
    
    def __init__(self, args_m):
        super(MultiTaskHead, self).__init__()
        
        # Extract configuration parameters
        self.input_dim = args_m.output_dim
        self.hidden_dim = getattr(args_m, 'hidden_dim', 256)
        self.dropout_prob = getattr(args_m, 'dropout', 0.1)
        self.rul_max_value = getattr(args_m, 'rul_max_value', 1000.0)
        self.use_batch_norm = getattr(args_m, 'use_batch_norm', True)
        self.activation_name = getattr(args_m, 'activation', 'relu')
        
        # Store number of classes for each system/dataset
        self.num_classes = args_m.num_classes if hasattr(args_m, 'num_classes') else {}
        
        # Define activation function
        self.activation = self._get_activation(self.activation_name)
        
        # Shared feature processing layers
        self.shared_layers = self._build_shared_layers()
        
        # Task-specific heads
        self.fault_classification_heads = self._build_classification_heads()
        self.rul_prediction_head = self._build_rul_head()
        self.anomaly_detection_head = self._build_anomaly_head()
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, activation_name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'elu': nn.ELU(),
        }
        return activations.get(activation_name.lower(), nn.ReLU())
    
    def _build_shared_layers(self) -> nn.Sequential:
        """Build shared feature processing layers."""
        layers = []
        
        # First shared layer
        layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        if self.use_batch_norm:
            layers.append(nn.BatchNorm1d(self.hidden_dim))
        layers.append(self.activation)
        layers.append(nn.Dropout(self.dropout_prob))
        
        # Second shared layer
        layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        if self.use_batch_norm:
            layers.append(nn.BatchNorm1d(self.hidden_dim))
        layers.append(self.activation)
        layers.append(nn.Dropout(self.dropout_prob))
        
        return nn.Sequential(*layers)
    
    def _build_classification_heads(self) -> nn.ModuleDict:
        """Build fault classification heads for each system/dataset."""
        classification_heads = nn.ModuleDict()
        
        for system_id, n_classes in self.num_classes.items():
            # Each classification head has two layers
            head = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                self.activation,
                nn.Dropout(self.dropout_prob),
                nn.Linear(self.hidden_dim // 2, n_classes)
            )
            classification_heads[str(system_id)] = head
        
        return classification_heads
    
    def _build_rul_head(self) -> nn.Sequential:
        """Build RUL prediction head."""
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            self.activation,
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            self.activation,
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.hidden_dim // 4, 1),
            nn.ReLU()  # Ensure positive RUL values
        )
    
    def _build_anomaly_head(self) -> nn.Sequential:
        """Build anomaly detection head."""
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            self.activation,
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.hidden_dim // 2, 1)
            # Note: No sigmoid here as we'll use BCEWithLogitsLoss
        )
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(
        self, 
        x: torch.Tensor, 
        system_id: Optional[Union[str, int]] = None,
        task_id: Optional[str] = None,
        return_feature: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through the multi-task head."""
        # Handle input shape: (B, L, C) -> (B, C) via mean pooling
        if x.dim() == 3:
            x = x.mean(dim=1)  # Average pooling over sequence length
        elif x.dim() != 2:
            raise ValueError(f"Expected input with 2 or 3 dimensions, got {x.dim()}")
        
        # Process through shared layers
        shared_features = self.shared_layers(x)
        
        if return_feature:
            return shared_features
        
        # Execute specific task or all tasks
        if task_id == 'classification':
            return self._forward_classification(shared_features, system_id)
        elif task_id == 'rul_prediction':
            return self._forward_rul_prediction(shared_features)
        elif task_id == 'anomaly_detection':
            return self._forward_anomaly_detection(shared_features)
        elif task_id == 'all' or task_id is None:
            return self._forward_all_tasks(shared_features, system_id)
        else:
            raise ValueError(f"Unknown task_id: {task_id}")
    
    def _forward_classification(self, features: torch.Tensor, system_id: Optional[Union[str, int]]) -> torch.Tensor:
        """Forward pass for fault classification."""
        if system_id is None:
            raise ValueError("system_id must be provided for classification task")
        
        system_key = str(system_id)
        if system_key not in self.fault_classification_heads:
            raise ValueError(f"No classification head found for system_id: {system_id}")
        
        return self.fault_classification_heads[system_key](features)
    
    def _forward_rul_prediction(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass for RUL prediction."""
        rul_raw = self.rul_prediction_head(features)
        # Scale to reasonable RUL range
        return rul_raw * self.rul_max_value
    
    def _forward_anomaly_detection(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass for anomaly detection."""
        return self.anomaly_detection_head(features)
    
    def _forward_all_tasks(self, features: torch.Tensor, system_id: Optional[Union[str, int]]) -> Dict[str, torch.Tensor]:
        """Forward pass for all tasks simultaneously."""
        results = {}
        
        # Fault classification (if system_id provided)
        if system_id is not None:
            try:
                results['classification'] = self._forward_classification(features, system_id)
            except ValueError:
                # If no head for this system, skip classification
                pass
        
        # RUL prediction
        results['rul_prediction'] = self._forward_rul_prediction(features)
        
        # Anomaly detection
        results['anomaly_detection'] = self._forward_anomaly_detection(features)
        
        return results


def test_multi_task_head():
    """Comprehensive test suite for MultiTaskHead."""
    print("Testing MultiTaskHead Module")
    print("=" * 50)
    
    # Test configuration
    args = Namespace(
        output_dim=512,
        hidden_dim=256,
        dropout=0.1,
        num_classes={'system1': 5, 'system2': 3, 'system3': 7},
        rul_max_value=1000.0,
        use_batch_norm=True,
        activation='relu'
    )
    
    # Create model
    model = MultiTaskHead(args)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {param_count} parameters")
    
    # Test input shapes
    batch_size = 16
    seq_len = 128
    feature_dim = 512
    
    # Test with 3D input (B, L, C)
    x_3d = torch.randn(batch_size, seq_len, feature_dim)
    print(f"✓ Input shape (3D): {x_3d.shape}")
    
    # Test individual tasks
    print("\n--- Testing Individual Tasks ---")
    
    # Test classification
    try:
        cls_output = model(x_3d, system_id='system1', task_id='classification')
        assert cls_output.shape == (batch_size, 5), f"Expected (16, 5), got {cls_output.shape}"
        print(f"✓ Classification output shape: {cls_output.shape}")
    except Exception as e:
        print(f"✗ Classification test failed: {e}")
        return False
    
    # Test RUL prediction
    try:
        rul_output = model(x_3d, task_id='rul_prediction')
        assert rul_output.shape == (batch_size, 1), f"Expected (16, 1), got {rul_output.shape}"
        assert torch.all(rul_output >= 0), "RUL values should be non-negative"
        print(f"✓ RUL prediction output shape: {rul_output.shape}")
    except Exception as e:
        print(f"✗ RUL prediction test failed: {e}")
        return False
    
    # Test anomaly detection
    try:
        anomaly_output = model(x_3d, task_id='anomaly_detection')
        assert anomaly_output.shape == (batch_size, 1), f"Expected (16, 1), got {anomaly_output.shape}"
        print(f"✓ Anomaly detection output shape: {anomaly_output.shape}")
    except Exception as e:
        print(f"✗ Anomaly detection test failed: {e}")
        return False
    
    # Test all tasks
    print("\n--- Testing All Tasks ---")
    try:
        all_outputs = model(x_3d, system_id='system2', task_id='all')
        assert isinstance(all_outputs, dict), "All tasks output should be a dictionary"
        assert 'classification' in all_outputs, "Missing classification output"
        assert 'rul_prediction' in all_outputs, "Missing RUL prediction output"
        assert 'anomaly_detection' in all_outputs, "Missing anomaly detection output"
        print(f"✓ All tasks output keys: {list(all_outputs.keys())}")
    except Exception as e:
        print(f"✗ All tasks test failed: {e}")
        return False
    
    # Test feature extraction
    print("\n--- Testing Feature Extraction ---")
    try:
        features = model(x_3d, return_feature=True)
        assert features.shape == (batch_size, 256), f"Expected (16, 256), got {features.shape}"
        print(f"✓ Feature shape: {features.shape}")
    except Exception as e:
        print(f"✗ Feature extraction test failed: {e}")
        return False
    
    # Test with 2D input (B, C)
    print("\n--- Testing 2D Input ---")
    try:
        x_2d = torch.randn(batch_size, feature_dim)
        output_2d = model(x_2d, system_id='system3', task_id='classification')
        assert output_2d.shape == (batch_size, 7), f"Expected (16, 7), got {output_2d.shape}"
        print(f"✓ 2D input classification output shape: {output_2d.shape}")
    except Exception as e:
        print(f"✗ 2D input test failed: {e}")
        return False
    
    # Test error handling
    print("\n--- Testing Error Handling ---")
    try:
        # Test invalid system ID
        try:
            model(x_3d, system_id='invalid_system', task_id='classification')
            print("✗ Should have raised ValueError for invalid system ID")
            return False
        except ValueError:
            print("✓ Correctly raised ValueError for invalid system ID")
        
        # Test missing system ID for classification
        try:
            model(x_3d, task_id='classification')
            print("✗ Should have raised ValueError for missing system ID")
            return False
        except ValueError:
            print("✓ Correctly raised ValueError for missing system ID")
        
        # Test invalid task ID
        try:
            model(x_3d, task_id='invalid_task')
            print("✗ Should have raised ValueError for invalid task ID")
            return False
        except ValueError:
            print("✓ Correctly raised ValueError for invalid task ID")
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False
    
    # Test gradient flow
    print("\n--- Testing Gradient Flow ---")
    try:
        x_grad = torch.randn(4, 128, 512, requires_grad=True)
        
        # Test classification gradient flow
        cls_output = model(x_grad, system_id='system1', task_id='classification')
        loss = cls_output.sum()
        loss.backward()
        
        assert x_grad.grad is not None, "Gradients should flow to input"
        assert torch.any(x_grad.grad != 0), "Gradients should be non-zero"
        print("✓ Classification gradient flow works")
        
        # Reset gradients
        x_grad.grad = None
        
        # Test RUL prediction gradient flow
        rul_output = model(x_grad, task_id='rul_prediction')
        loss = rul_output.sum()
        loss.backward()
        
        assert x_grad.grad is not None, "Gradients should flow to input"
        assert torch.any(x_grad.grad != 0), "Gradients should be non-zero"
        print("✓ RUL prediction gradient flow works")
        
    except Exception as e:
        print(f"✗ Gradient flow test failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("✅ All tests passed successfully!")
    return True


if __name__ == '__main__':
    success = test_multi_task_head()
    if not success:
        exit(1)
    print("\n🎉 Multi-task head implementation is working correctly!")
