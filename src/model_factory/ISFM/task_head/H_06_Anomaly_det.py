"""
Anomaly Detection Task Head

This module implements a neural network head specifically designed for binary anomaly detection.
It takes feature representations from the backbone network and produces binary classification
outputs to distinguish normal from anomalous operating conditions.

Author: PHM-Vibench Team
Date: 2025-09-05
"""

import torch
import torch.nn as nn


class H_06_Anomaly_det(nn.Module):
    """
    Anomaly detection head for binary classification of normal vs anomalous conditions.
    
    This module takes feature representations and outputs logits for binary classification.
    The output is designed to work with BCEWithLogitsLoss (no sigmoid activation).
    
    Parameters
    ----------
    args : Namespace
        Configuration containing:
        - output_dim: int, dimension of input features from backbone
        - hidden_dim: int, hidden layer dimension (default: output_dim // 2)
        - dropout: float, dropout probability (default: 0.1)
        - activation: str, activation function name (default: 'relu')
        - use_batch_norm: bool, whether to use batch normalization (default: False)
    """
    
    def __init__(self, args):
        super(H_06_Anomaly_det, self).__init__()
        
        # Extract configuration parameters
        self.input_dim = args.output_dim
        self.hidden_dim = getattr(args, 'hidden_dim', args.output_dim // 2)
        self.dropout_prob = getattr(args, 'dropout', 0.1)
        self.activation_name = getattr(args, 'activation', 'relu')
        self.use_batch_norm = getattr(args, 'use_batch_norm', False)
        
        # Define activation function
        self.activation = self._get_activation(self.activation_name)
        
        # Build anomaly detection network
        self.anomaly_net = self._build_network()
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, activation_name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'elu': nn.ELU(),
        }
        return activations.get(activation_name.lower(), nn.ReLU())
    
    def _build_network(self) -> nn.Sequential:
        """Build the anomaly detection network."""
        layers = []
        
        # First hidden layer
        layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        if self.use_batch_norm:
            layers.append(nn.BatchNorm1d(self.hidden_dim))
        layers.append(self.activation)
        layers.append(nn.Dropout(self.dropout_prob))
        
        # Second hidden layer
        layers.append(nn.Linear(self.hidden_dim, self.hidden_dim // 2))
        if self.use_batch_norm:
            layers.append(nn.BatchNorm1d(self.hidden_dim // 2))
        layers.append(self.activation)
        layers.append(nn.Dropout(self.dropout_prob))
        
        # Output layer (no activation - raw logits for BCEWithLogitsLoss)
        layers.append(nn.Linear(self.hidden_dim // 2, 1))
        
        return nn.Sequential(*layers)
    
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
    
    def forward(self, x, return_feature=False, **kwargs):
        """
        Forward pass through the anomaly detection head.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features from backbone with shape (B, L, C) or (B, C)
        return_feature : bool, optional
            If True, return intermediate features instead of final predictions
        
        Returns
        -------
        torch.Tensor
            Anomaly detection logits with shape (B,)
            Use with BCEWithLogitsLoss for training
        """
        # Handle input shape: (B, L, C) -> (B, C) via mean pooling
        if x.dim() == 3:
            x = x.mean(dim=1)  # Average pooling over sequence length
        elif x.dim() != 2:
            raise ValueError(f"Expected input with 2 or 3 dimensions, got {x.dim()}")
        
        # Pass through anomaly detection network
        logits = self.anomaly_net(x)
        
        if return_feature:
            # Return pre-activation features (before final linear layer)
            feature_layers = self.anomaly_net[:-1]
            features = nn.Sequential(*feature_layers)(x)
            return features
        
        # Squeeze the output dimension for binary classification compatibility
        return logits.squeeze(-1)  # Shape: (B, 1) -> (B,)
    
    def predict_proba(self, x):
        """
        Get anomaly probabilities (after sigmoid activation).
        
        Parameters
        ----------
        x : torch.Tensor
            Input features
        
        Returns
        -------
        torch.Tensor
            Anomaly probabilities with shape (B,), values in [0, 1]
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)
    
    def predict(self, x, threshold=0.5):
        """
        Get binary anomaly predictions.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features
        threshold : float, optional
            Decision threshold (default: 0.5)
        
        Returns
        -------
        torch.Tensor
            Binary predictions (0=normal, 1=anomaly) with shape (B,)
        """
        proba = self.predict_proba(x)
        return (proba > threshold).float()


if __name__ == '__main__':
    """Unit tests for H_06_Anomaly_det module."""
    from argparse import Namespace
    
    # Test configuration
    args = Namespace(
        output_dim=512,
        hidden_dim=256,
        dropout=0.1,
        activation='relu',
        use_batch_norm=True
    )
    
    # Create model
    model = H_06_Anomaly_det(args)
    print(f"Anomaly detection head created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test input shapes
    batch_size = 16
    seq_len = 128
    feature_dim = 512
    
    # Test with 3D input (B, L, C)
    x_3d = torch.randn(batch_size, seq_len, feature_dim)
    print(f"Input shape (3D): {x_3d.shape}")
    
    # Test logits output
    logits = model(x_3d)
    print(f"Logits output shape: {logits.shape}")
    assert logits.shape == (batch_size,), f"Expected (16,), got {logits.shape}"
    
    # Test probability output
    proba = model.predict_proba(x_3d)
    print(f"Probability output shape: {proba.shape}")
    assert proba.shape == (batch_size,), f"Expected (16,), got {proba.shape}"
    assert torch.all(proba >= 0) and torch.all(proba <= 1), "Probabilities should be in [0, 1]"
    
    # Test binary predictions
    predictions = model.predict(x_3d, threshold=0.5)
    print(f"Predictions shape: {predictions.shape}")
    assert predictions.shape == (batch_size,), f"Expected (16,), got {predictions.shape}"
    assert torch.all((predictions == 0) | (predictions == 1)), "Predictions should be binary (0 or 1)"
    
    # Test with 2D input (B, C)
    x_2d = torch.randn(batch_size, feature_dim)
    output_2d = model(x_2d)
    print(f"2D input output shape: {output_2d.shape}")
    assert output_2d.shape == (batch_size,), f"Expected (16,), got {output_2d.shape}"
    
    # Test feature extraction
    features = model(x_3d, return_feature=True)
    print(f"Feature shape: {features.shape}")
    assert features.shape == (batch_size, 128), f"Expected (16, 128), got {features.shape}"
    
    print("=== All Anomaly Detection Head Tests Passed! ===")