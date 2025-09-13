"""
Multi-Task Head Module for PHM Foundation Model

This module implements a neural network head that simultaneously performs four distinct tasks:
1. Fault Classification: Multi-class classification to identify different types of faults
2. Remaining Useful Life (RUL) Prediction: Regression task to predict remaining operational time
3. Anomaly Detection: Binary classification to detect abnormal operating conditions
4. Signal Prediction: Time-series forecasting to predict future signal sequences

The module is designed to work with the ISFM (Intelligent Signal Foundation Model) architecture
and integrates seamlessly with the PHM-Vibench framework.

Author: PHM-Vibench Team
Date: 2025-08-18
Updated: 2025-08-29 - Added Signal Prediction task
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union
import numpy as np

# Import H_03_Linear_pred for signal prediction task
from .H_03_Linear_pred import H_03_Linear_pred


class MultiTaskHead(nn.Module):
    """
    Multi-task head for simultaneous fault classification, RUL prediction, anomaly detection, and signal prediction.
    
    This module takes feature representations from the backbone network and produces
    predictions for all four PHM tasks through separate output layers with appropriate
    dimensions and activation functions.
    
    Parameters
    ----------
    args_m : Namespace
        Model configuration containing:
        - output_dim: int, dimension of input features from backbone
        - num_classes: dict, number of classes for each dataset/system
        - hidden_dim: int, dimension of hidden layers (default: 256)
        - dropout: float, dropout probability (default: 0.1)
        - rul_max_value: float, maximum RUL value for normalization (default: 1)
        - use_batch_norm: bool, whether to use batch normalization (default: True)
        - activation: str, activation function name (default: 'relu')
    
    Attributes
    ----------
    input_dim : int
        Dimension of input features from backbone
    hidden_dim : int
        Dimension of hidden layers
    dropout_prob : float
        Dropout probability
    rul_max_value : float
        Maximum RUL value for output scaling
    """
    
    def __init__(self, args_m):
        super(MultiTaskHead, self).__init__()
        
        # Extract configuration parameters
        self.input_dim = args_m.output_dim
        self.hidden_dim = getattr(args_m, 'hidden_dim', 256)
        self.dropout_prob = getattr(args_m, 'dropout', 0.1)
        self.rul_max_value = getattr(args_m, 'rul_max_value', 1)
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
        self.signal_prediction_head = H_03_Linear_pred(args_m)
        
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
        # layers.append(nn.Dropout(self.dropout_prob))
        
        # Second shared layer
        layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        if self.use_batch_norm:
            layers.append(nn.BatchNorm1d(self.hidden_dim))
        layers.append(self.activation)
        # layers.append(nn.Dropout(self.dropout_prob))
        
        return nn.Sequential(*layers)
    
    def _build_classification_heads(self) -> nn.ModuleDict:
        """Build fault classification heads for each system/dataset."""
        classification_heads = nn.ModuleDict()
        
        for system_id, n_classes in self.num_classes.items():
            # Each classification head has two layers
            head = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                self.activation,
                # nn.Dropout(self.dropout_prob),
                nn.Linear(self.hidden_dim // 2, n_classes)
            )
            classification_heads[str(system_id)] = head
        
        return classification_heads
    
    def _build_rul_head(self) -> nn.Sequential:
        """Build RUL prediction head."""
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            self.activation,
            # nn.Dropout(self.dropout_prob),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            self.activation,
            # nn.Dropout(self.dropout_prob),
            nn.Linear(self.hidden_dim // 4, 1),
            nn.ReLU()  # Ensure positive RUL values
        )
    
    def _build_anomaly_head(self) -> nn.Sequential:
        """Build anomaly detection head."""
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            self.activation,
            # nn.Dropout(self.dropout_prob),
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
        original_x: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the multi-task head.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features from backbone with shape (B, L, C) or (B, C)
        system_id : str or int, optional
            System/dataset identifier for classification task
        task_id : str, optional
            Specific task to execute ('classification', 'rul_prediction', 'anomaly_detection', 'signal_prediction', 'all')
        return_feature : bool, optional
            If True, return intermediate features instead of final predictions
        
        Returns
        -------
        torch.Tensor or Dict[str, torch.Tensor]
            Task predictions or feature representations
        """
        # Special handling for signal prediction task - preserve sequence dimension
        if task_id == 'signal_prediction':
            # Signal prediction expects (B,L,C) input, use shape parameter from kwargs
            shape = kwargs.get('shape', (96, 2))  # Default: predict 96 timesteps, 2 channels
            return self.signal_prediction_head(x, shape=shape)
        
        # For other tasks: Handle input shape (B, L, C) -> (B, C) via mean pooling
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
            # Use original_x if provided, otherwise fall back to the processed x
            signal_input = original_x if original_x is not None else x
            return self._forward_all_tasks(shared_features, system_id, signal_input, **kwargs)
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
    
    def _forward_all_tasks(self, features: torch.Tensor, system_id: Optional[Union[str, int]], original_x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
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
        
        # Signal prediction (use original sequence, not mean-pooled features)
        shape = kwargs.get('shape', (96, 2))  # Default shape
        results['signal_prediction'] = self.signal_prediction_head(original_x, shape=shape)
        
        return results


if __name__ == '__main__':
    """Unit tests for MultiTaskHead module."""
    from argparse import Namespace
    
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
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test input shapes
    batch_size = 16
    seq_len = 128
    feature_dim = 512
    
    # Test with 3D input (B, L, C)
    x_3d = torch.randn(batch_size, seq_len, feature_dim)
    print(f"Input shape (3D): {x_3d.shape}")
    
    # Test individual tasks
    print("\n=== Testing Individual Tasks ===")
    
    # Test classification
    cls_output = model(x_3d, system_id='system1', task_id='classification')
    print(f"Classification output shape: {cls_output.shape}")
    assert cls_output.shape == (batch_size, 5), f"Expected (16, 5), got {cls_output.shape}"
    
    # Test RUL prediction
    rul_output = model(x_3d, task_id='rul_prediction')
    print(f"RUL prediction output shape: {rul_output.shape}")
    assert rul_output.shape == (batch_size, 1), f"Expected (16, 1), got {rul_output.shape}"
    assert torch.all(rul_output >= 0), "RUL values should be non-negative"
    
    # Test anomaly detection
    anomaly_output = model(x_3d, task_id='anomaly_detection')
    print(f"Anomaly detection output shape: {anomaly_output.shape}")
    assert anomaly_output.shape == (batch_size, 1), f"Expected (16, 1), got {anomaly_output.shape}"
    
    # Test all tasks
    print("\n=== Testing All Tasks ===")
    all_outputs = model(x_3d, system_id='system2', task_id='all')
    print(f"All tasks output keys: {list(all_outputs.keys())}")
    assert 'classification' in all_outputs
    assert 'rul_prediction' in all_outputs
    assert 'anomaly_detection' in all_outputs
    
    # Test feature extraction
    print("\n=== Testing Feature Extraction ===")
    features = model(x_3d, return_feature=True)
    print(f"Feature shape: {features.shape}")
    assert features.shape == (batch_size, 256), f"Expected (16, 256), got {features.shape}"
    
    # Test with 2D input (B, C)
    print("\n=== Testing 2D Input ===")
    x_2d = torch.randn(batch_size, feature_dim)
    output_2d = model(x_2d, system_id='system3', task_id='classification')
    print(f"2D input classification output shape: {output_2d.shape}")
    assert output_2d.shape == (batch_size, 7), f"Expected (16, 7), got {output_2d.shape}"
    
    print("\n=== All Tests Passed! ===")
