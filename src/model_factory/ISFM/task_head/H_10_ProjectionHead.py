import torch
import torch.nn as nn


class H_10_ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning.
    
    This component extracts features to a lower-dimensional projection space
    commonly used in contrastive learning methods. It supports configurable
    dimensions, normalization, and activation functions.
    
    Extracted from hse_contrastive.py and made reusable.
    """
    
    def __init__(self, args):
        super(H_10_ProjectionHead, self).__init__()
        
        # Extract configuration parameters with defaults
        self.input_dim = getattr(args, 'input_dim', None)
        if self.input_dim is None:
            # Fall back to common dimension names
            self.input_dim = getattr(args, 'd_model', getattr(args, 'output_dim', 256))
        
        self.hidden_dim = getattr(args, 'hidden_dim', self.input_dim)
        self.output_dim = getattr(args, 'output_dim', 128)
        
        # Configuration options
        self.use_norm = getattr(args, 'use_norm', True)
        self.activation = getattr(args, 'activation', 'relu')
        self.dropout = getattr(args, 'dropout', 0.0)
        
        # Build projection layers
        layers = []
        
        # First linear layer
        layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        
        # Activation function
        if self.activation.lower() == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif self.activation.lower() == 'gelu':
            layers.append(nn.GELU())
        elif self.activation.lower() == 'tanh':
            layers.append(nn.Tanh())
        elif self.activation.lower() == 'sigmoid':
            layers.append(nn.Sigmoid())
        else:
            layers.append(nn.ReLU(inplace=True))  # Default to ReLU
        
        # Dropout if specified
        if self.dropout > 0:
            layers.append(nn.Dropout(self.dropout))
        
        # Second linear layer (projection layer)
        layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        
        # Layer normalization if enabled
        if self.use_norm:
            layers.append(nn.LayerNorm(self.output_dim))
        
        self.projection = nn.Sequential(*layers)
        
    def forward(self, x, **kwargs):
        """
        Forward pass through projection head.
        
        Args:
            x: Input tensor, shape can be:
               - (B, d_model) for already pooled features
               - (B, T, d_model) for temporal features (will be pooled)
            **kwargs: Additional keyword arguments (for compatibility)
        
        Returns:
            torch.Tensor: Projected features of shape (B, output_dim)
        """
        # Handle different input shapes
        if x.dim() == 3:
            # (B, T, d_model) -> pool temporal dimension
            x = x.mean(dim=1)  # (B, d_model)
        elif x.dim() == 2:
            # (B, d_model) -> already pooled
            pass
        else:
            raise ValueError(f"Expected input dimension 2 or 3, got {x.dim()}")
        
        # Apply projection
        projected = self.projection(x)  # (B, output_dim)
        
        return projected
    
    def get_output_dim(self):
        """Get the output dimension of the projection head."""
        return self.output_dim


# ---------------------------- Self-Testing -----------------------------
if __name__ == "__main__":
    import torch
    
    print("Testing H_10_ProjectionHead...")
    
    # Test Case 1: Basic configuration
    class BasicArgs:
        input_dim = 256
        hidden_dim = 256
        output_dim = 128
        use_norm = True
        activation = 'relu'
        dropout = 0.0
    
    print("\n1. Testing basic configuration...")
    args1 = BasicArgs()
    head1 = H_10_ProjectionHead(args1)
    
    # Test with 2D input (already pooled)
    x_2d = torch.randn(4, 256)
    out1 = head1(x_2d)
    print(f"Input shape: {x_2d.shape}, Output shape: {out1.shape}")
    assert out1.shape == (4, 128), f"Expected (4, 128), got {out1.shape}"
    
    # Test with 3D input (temporal features)
    x_3d = torch.randn(4, 100, 256)
    out2 = head1(x_3d)
    print(f"Input shape: {x_3d.shape}, Output shape: {out2.shape}")
    assert out2.shape == (4, 128), f"Expected (4, 128), got {out2.shape}"
    
    # Test Case 2: Without normalization
    class NoNormArgs:
        input_dim = 512
        hidden_dim = 256
        output_dim = 64
        use_norm = False
        activation = 'gelu'
        dropout = 0.1
    
    print("\n2. Testing without normalization...")
    args2 = NoNormArgs()
    head2 = H_10_ProjectionHead(args2)
    
    x = torch.randn(2, 512)
    out = head2(x)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")
    assert out.shape == (2, 64), f"Expected (2, 64), got {out.shape}"
    
    # Test Case 3: Using d_model fallback
    class FallbackArgs:
        d_model = 384  # Should be used as input_dim
        output_dim = 96
    
    print("\n3. Testing d_model fallback...")
    args3 = FallbackArgs()
    head3 = H_10_ProjectionHead(args3)
    
    x = torch.randn(3, 50, 384)
    out = head3(x)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")
    assert out.shape == (3, 96), f"Expected (3, 96), got {out.shape}"
    print(f"Input dim inferred as: {head3.input_dim}")
    
    # Test Case 4: Different activation functions
    activation_tests = ['relu', 'gelu', 'tanh', 'sigmoid', 'unknown']
    print("\n4. Testing different activation functions...")
    
    for act in activation_tests:
        class ActArgs:
            input_dim = 128
            output_dim = 64
            activation = act
        
        args = ActArgs()
        head = H_10_ProjectionHead(args)
        x = torch.randn(2, 128)
        out = head(x)
        print(f"Activation '{act}': Input {x.shape} -> Output {out.shape}")
        assert out.shape == (2, 64), f"Failed for activation {act}"
    
    # Test Case 5: Test output dimension getter
    print("\n5. Testing output dimension getter...")
    output_dim = head1.get_output_dim()
    print(f"Output dimension: {output_dim}")
    assert output_dim == 128, f"Expected 128, got {output_dim}"
    
    # Test Case 6: Error handling for wrong input dimensions
    print("\n6. Testing error handling...")
    try:
        x_wrong = torch.randn(4, 256, 128, 64)  # 4D input
        head1(x_wrong)
        assert False, "Should have raised ValueError for 4D input"
    except ValueError as e:
        print(f"Correctly caught error: {e}")
    
    # Test Case 7: GPU compatibility (if available)
    if torch.cuda.is_available():
        print("\n7. Testing GPU compatibility...")
        head_gpu = H_10_ProjectionHead(BasicArgs()).cuda()
        x_gpu = torch.randn(2, 256).cuda()
        out_gpu = head_gpu(x_gpu)
        print(f"GPU test successful: {out_gpu.device}")
        assert out_gpu.is_cuda, "Output should be on GPU"
    
    print("\n✅ All tests passed! H_10_ProjectionHead is working correctly.")
    
    # Performance test
    print("\n8. Performance test...")
    import time
    
    head_perf = H_10_ProjectionHead(BasicArgs())
    x_perf = torch.randn(1000, 100, 256)
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = head_perf(x_perf)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100 * 1000  # ms
    print(f"Average inference time: {avg_time:.2f}ms for batch size 1000")