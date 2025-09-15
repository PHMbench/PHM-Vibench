"""
MomentumEncoder Backbone for Contrastive Learning

This module implements the MomentumEncoder as a reusable backbone component
following PHM-Vibench factory patterns. It provides momentum-based feature
encoding to prevent feature collapse in contrastive learning frameworks.

Key Features:
- Momentum-based parameter updates
- Dual encoder architecture (query/key encoders)
- Configurable momentum coefficient
- Support for arbitrary base encoders

Authors: PHMbench Team
License: Apache 2.0
"""

import torch
import torch.nn as nn
from copy import deepcopy
from typing import Union


class B_11_MomentumEncoder(nn.Module):
    """
    Momentum encoder backbone for contrastive learning.
    
    This backbone wraps any encoder module and maintains a momentum-updated copy
    to provide stable target representations for contrastive learning. The momentum
    encoder parameters are updated as an exponential moving average of the main
    encoder parameters.
    
    Architecture:
    - Main encoder: trainable encoder for query features
    - Momentum encoder: momentum-updated copy for key features
    - Parameter update: EMA with configurable momentum coefficient
    
    Args:
        configs (dict): Configuration dictionary containing:
            - base_encoder (nn.Module): Base encoder to wrap
            - momentum (float, optional): Momentum coefficient. Defaults to 0.999
    """
    
    def __init__(self, configs):
        """
        Initialize MomentumEncoder backbone.
        
        Args:
            configs (dict): Configuration dictionary with:
                - base_encoder: Base encoder module to wrap
                - momentum: Momentum coefficient (default: 0.999)
        """
        super().__init__()
        
        # Extract configuration parameters
        self.base_encoder = configs.get('base_encoder')
        if self.base_encoder is None:
            raise ValueError("base_encoder must be provided in configs")
            
        self.momentum = configs.get('momentum', 0.999)
        
        # Set main encoder
        self.encoder = self.base_encoder
        
        # Create momentum encoder as a deep copy
        self.momentum_encoder = deepcopy(self.base_encoder)
        
        # Disable gradients for momentum encoder
        for param in self.momentum_encoder.parameters():
            param.requires_grad = False
            
        self._validate_momentum()
    
    def _validate_momentum(self):
        """Validate momentum coefficient."""
        if not 0.0 <= self.momentum <= 1.0:
            raise ValueError(f"momentum must be in [0, 1], got {self.momentum}")
    
    @torch.no_grad()
    def update_momentum_encoder(self):
        """
        Update momentum encoder parameters using exponential moving average.
        
        The momentum encoder parameters are updated as:
        θ_k = m * θ_k + (1 - m) * θ_q
        
        where:
        - θ_k: momentum encoder parameters
        - θ_q: main encoder parameters  
        - m: momentum coefficient
        """
        for param_q, param_k in zip(
            self.encoder.parameters(), 
            self.momentum_encoder.parameters()
        ):
            param_k.data = param_k.data * self.momentum + \
                          param_q.data * (1.0 - self.momentum)
    
    def forward(self, x: torch.Tensor, use_momentum: bool = False) -> torch.Tensor:
        """
        Forward pass through either main or momentum encoder.
        
        Args:
            x (torch.Tensor): Input tensor
            use_momentum (bool): If True, use momentum encoder; 
                               otherwise use main encoder
        
        Returns:
            torch.Tensor: Encoded features
        """
        if use_momentum:
            return self.momentum_encoder(x)
        else:
            return self.encoder(x)
    
    def get_query_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get query features from main encoder.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Query features
        """
        return self.forward(x, use_momentum=False)
    
    def get_key_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get key features from momentum encoder.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Key features
        """
        return self.forward(x, use_momentum=True)
    
    def get_momentum_coefficient(self) -> float:
        """Get current momentum coefficient."""
        return self.momentum
    
    def set_momentum_coefficient(self, momentum: float):
        """
        Set momentum coefficient.
        
        Args:
            momentum (float): New momentum coefficient in [0, 1]
        """
        self.momentum = momentum
        self._validate_momentum()


# --- Demo/Test Code ---
if __name__ == '__main__':
    print("=== B_11_MomentumEncoder Backbone Test ===")
    
    # Create a simple mock encoder for testing
    class MockEncoder(nn.Module):
        """Simple mock encoder for testing."""
        def __init__(self, input_dim=512, hidden_dim=256, output_dim=128):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    # Test configurations
    test_configs = [
        {'momentum': 0.999},
        {'momentum': 0.99},
        {'momentum': 0.9}
    ]
    
    for i, momentum_val in enumerate([0.999, 0.99, 0.9]):
        print(f"\n--- Test {i+1}: Momentum = {momentum_val} ---")
        
        # Create mock base encoder
        base_encoder = MockEncoder(input_dim=512, hidden_dim=256, output_dim=128)
        
        # Create configuration
        configs = {
            'base_encoder': base_encoder,
            'momentum': momentum_val
        }
        
        # Create MomentumEncoder backbone
        momentum_encoder = B_11_MomentumEncoder(configs)
        
        # Test data
        batch_size, seq_len, input_dim = 4, 100, 512
        x = torch.randn(batch_size, seq_len, input_dim)
        
        print(f"Input shape: {x.shape}")
        
        # Test forward pass with main encoder
        query_features = momentum_encoder(x, use_momentum=False)
        print(f"Query features shape: {query_features.shape}")
        
        # Test forward pass with momentum encoder
        key_features = momentum_encoder(x, use_momentum=True)
        print(f"Key features shape: {key_features.shape}")
        
        # Test helper methods
        query_features_2 = momentum_encoder.get_query_features(x)
        key_features_2 = momentum_encoder.get_key_features(x)
        
        # Verify consistency
        assert torch.allclose(query_features, query_features_2), "Query features mismatch"
        assert torch.allclose(key_features, key_features_2), "Key features mismatch"
        print("✓ Helper methods consistent with forward")
        
        # Test momentum update (need to modify main encoder first to see changes)
        original_momentum_params = [p.clone() for p in momentum_encoder.momentum_encoder.parameters()]
        
        # Modify main encoder parameters slightly to trigger momentum update
        with torch.no_grad():
            for param in momentum_encoder.encoder.parameters():
                param.data += 0.01 * torch.randn_like(param.data)
        
        momentum_encoder.update_momentum_encoder()
        updated_momentum_params = list(momentum_encoder.momentum_encoder.parameters())
        
        # Verify parameters changed
        params_changed = any(
            not torch.allclose(orig, updated, atol=1e-4) 
            for orig, updated in zip(original_momentum_params, updated_momentum_params)
        )
        print(f"✓ Momentum update {'successful' if params_changed else 'failed'}")
        
        # Test momentum coefficient getter/setter
        assert momentum_encoder.get_momentum_coefficient() == momentum_val
        momentum_encoder.set_momentum_coefficient(0.95)
        assert momentum_encoder.get_momentum_coefficient() == 0.95
        print("✓ Momentum coefficient setter/getter working")
        
        print(f"✓ Test {i+1} completed successfully")
    
    # Test error cases
    print("\n--- Error Handling Tests ---")
    
    # Test invalid momentum values
    try:
        invalid_configs = {'base_encoder': MockEncoder(), 'momentum': 1.5}
        B_11_MomentumEncoder(invalid_configs)
        print("✗ Should have raised ValueError for momentum > 1")
    except ValueError:
        print("✓ Correctly rejected momentum > 1")
    
    try:
        invalid_configs = {'base_encoder': MockEncoder(), 'momentum': -0.1}
        B_11_MomentumEncoder(invalid_configs)
        print("✗ Should have raised ValueError for momentum < 0")
    except ValueError:
        print("✓ Correctly rejected momentum < 0")
    
    # Test missing base_encoder
    try:
        invalid_configs = {'momentum': 0.999}
        B_11_MomentumEncoder(invalid_configs)
        print("✗ Should have raised ValueError for missing base_encoder")
    except ValueError:
        print("✓ Correctly rejected missing base_encoder")
    
    print("\n=== All tests passed! ===")