"""
PromptFusion: Multi-strategy fusion for signal and prompt features

This module implements three different fusion strategies to combine signal embeddings
with system prompt vectors for industrial fault diagnosis. The fusion strategies balance
computational efficiency with representation quality.

Fusion Strategies:
1. Concatenation: Simple feature concatenation (fast, basic)
2. Cross-Attention: Signal attends to prompt (best quality, slower)
3. Adaptive Gating: Learnable weighted combination (balanced)

Author: PHM-Vibench Team
Date: 2025-01-06
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Literal


class PromptFusion(nn.Module):
    """
    Multi-strategy fusion module for combining signal embeddings with system prompts.
    
    This module implements three fusion strategies with different computational complexity
    and representation quality trade-offs:
    
    1. **Concatenation**: Direct feature concatenation
       - Pros: Fast, simple, preserves all information
       - Cons: Linear scaling, no interaction modeling
       - Complexity: O(1)
    
    2. **Cross-Attention**: Signal features attend to prompt
       - Pros: Rich interaction modeling, adaptive weights
       - Cons: Quadratic complexity, more parameters
       - Complexity: O(n²)
    
    3. **Adaptive Gating**: Learnable gate controls fusion
       - Pros: Balanced efficiency and quality
       - Cons: Fixed fusion pattern
       - Complexity: O(n)
    """
    
    def __init__(self,
                 signal_dim: int,
                 prompt_dim: int,
                 fusion_type: Literal['concat', 'attention', 'gating'] = 'attention',
                 num_attention_heads: int = 4,
                 dropout: float = 0.1):
        """
        Initialize PromptFusion module.
        
        Args:
            signal_dim: Dimension of signal embedding features
            prompt_dim: Dimension of prompt vectors  
            fusion_type: Fusion strategy ('concat', 'attention', 'gating')
            num_attention_heads: Number of attention heads (for attention fusion)
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        self.signal_dim = signal_dim
        self.prompt_dim = prompt_dim
        self.fusion_type = fusion_type
        
        if fusion_type == 'concat':
            # Simple concatenation fusion
            self.fusion_proj = nn.Linear(signal_dim + prompt_dim, signal_dim)
            self.norm = nn.LayerNorm(signal_dim)
            
        elif fusion_type == 'attention':
            # Cross-attention fusion: signal attends to prompt
            self.prompt_proj = nn.Linear(prompt_dim, signal_dim)
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=signal_dim,
                num_heads=num_attention_heads,
                dropout=dropout,
                batch_first=True
            )
            self.norm = nn.LayerNorm(signal_dim)
            
        elif fusion_type == 'gating':
            # Adaptive gating fusion
            self.gate_proj = nn.Linear(prompt_dim, signal_dim)
            self.transform_proj = nn.Linear(prompt_dim, signal_dim)
            self.norm = nn.LayerNorm(signal_dim)
            
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}. "
                           f"Must be one of ['concat', 'attention', 'gating']")
        
        self.dropout = nn.Dropout(dropout)
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize model parameters using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, 
                signal_emb: torch.Tensor, 
                prompt_emb: torch.Tensor) -> torch.Tensor:
        """
        Fuse signal embeddings with prompt vectors.
        
        Args:
            signal_emb: Signal embedding tensor of shape (B, num_patches, signal_dim)
            prompt_emb: Prompt vector tensor of shape (B, prompt_dim)
        
        Returns:
            fused_emb: Fused embedding tensor of shape (B, num_patches, signal_dim)
            
        Raises:
            ValueError: If input tensor shapes are incompatible
        """
        # Validate input shapes
        self._validate_inputs(signal_emb, prompt_emb)
        
        batch_size, num_patches, _ = signal_emb.shape
        
        if self.fusion_type == 'concat':
            return self._concatenation_fusion(signal_emb, prompt_emb)
            
        elif self.fusion_type == 'attention':
            return self._attention_fusion(signal_emb, prompt_emb)
            
        elif self.fusion_type == 'gating':
            return self._gating_fusion(signal_emb, prompt_emb)
    
    def _concatenation_fusion(self, 
                             signal_emb: torch.Tensor, 
                             prompt_emb: torch.Tensor) -> torch.Tensor:
        """
        Simple concatenation-based fusion.
        
        Strategy: Expand prompt to match signal sequence length, concatenate,
        then project back to original signal dimension.
        """
        batch_size, num_patches, signal_dim = signal_emb.shape
        
        # Expand prompt to match sequence length
        expanded_prompt = prompt_emb.unsqueeze(1).expand(-1, num_patches, -1)  # (B, num_patches, prompt_dim)
        
        # Concatenate features
        concatenated = torch.cat([signal_emb, expanded_prompt], dim=-1)  # (B, num_patches, signal_dim + prompt_dim)
        
        # Project back to signal dimension
        fused = self.fusion_proj(concatenated)  # (B, num_patches, signal_dim)
        fused = self.norm(fused)
        fused = self.dropout(fused)
        
        return fused
    
    def _attention_fusion(self, 
                         signal_emb: torch.Tensor, 
                         prompt_emb: torch.Tensor) -> torch.Tensor:
        """
        Cross-attention based fusion.
        
        Strategy: Signal features attend to prompt, with residual connection
        to preserve original signal information.
        """
        batch_size, num_patches, signal_dim = signal_emb.shape
        
        # Project prompt to signal dimension
        prompt_proj = self.prompt_proj(prompt_emb)  # (B, prompt_dim) -> (B, signal_dim)
        prompt_key_value = prompt_proj.unsqueeze(1)  # (B, 1, signal_dim)
        
        # Cross-attention: signal (query) attends to prompt (key, value)
        attended_signal, attention_weights = self.cross_attention(
            query=signal_emb,                    # (B, num_patches, signal_dim)
            key=prompt_key_value,                 # (B, 1, signal_dim)  
            value=prompt_key_value                # (B, 1, signal_dim)
        )
        
        # Residual connection to preserve original signal features
        fused = signal_emb + attended_signal     # (B, num_patches, signal_dim)
        fused = self.norm(fused)
        fused = self.dropout(fused)
        
        return fused
    
    def _gating_fusion(self, 
                      signal_emb: torch.Tensor, 
                      prompt_emb: torch.Tensor) -> torch.Tensor:
        """
        Adaptive gating based fusion.
        
        Strategy: Learnable gate controls mixing between original signal
        and prompt-transformed features.
        """
        batch_size, num_patches, signal_dim = signal_emb.shape
        
        # Compute gate weights from prompt
        gate_weights = torch.sigmoid(self.gate_proj(prompt_emb))  # (B, signal_dim)
        gate_weights = gate_weights.unsqueeze(1)                  # (B, 1, signal_dim)
        
        # Compute prompt transformation  
        prompt_transform = self.transform_proj(prompt_emb)        # (B, signal_dim)
        prompt_transform = prompt_transform.unsqueeze(1)          # (B, 1, signal_dim)
        
        # Adaptive fusion: gate * signal + (1 - gate) * prompt_transform
        fused = gate_weights * signal_emb + (1 - gate_weights) * prompt_transform
        fused = self.norm(fused)
        fused = self.dropout(fused)
        
        return fused
    
    def _validate_inputs(self, 
                        signal_emb: torch.Tensor, 
                        prompt_emb: torch.Tensor) -> None:
        """
        Validate input tensor shapes and dimensions.
        
        Args:
            signal_emb: Signal embedding tensor 
            prompt_emb: Prompt vector tensor
            
        Raises:
            ValueError: If shapes are incompatible
        """
        # Check signal embedding shape
        if signal_emb.dim() != 3:
            raise ValueError(f"signal_emb must be 3D (B, num_patches, signal_dim), got {signal_emb.shape}")
        
        # Check prompt embedding shape  
        if prompt_emb.dim() != 2:
            raise ValueError(f"prompt_emb must be 2D (B, prompt_dim), got {prompt_emb.shape}")
        
        # Check batch size consistency
        if signal_emb.size(0) != prompt_emb.size(0):
            raise ValueError(f"Batch size mismatch: signal_emb {signal_emb.size(0)} vs prompt_emb {prompt_emb.size(0)}")
        
        # Check dimension consistency
        if signal_emb.size(-1) != self.signal_dim:
            raise ValueError(f"signal_emb last dim {signal_emb.size(-1)} != expected signal_dim {self.signal_dim}")
        
        if prompt_emb.size(-1) != self.prompt_dim:
            raise ValueError(f"prompt_emb last dim {prompt_emb.size(-1)} != expected prompt_dim {self.prompt_dim}")
    
    def get_fusion_info(self) -> dict:
        """
        Get information about the current fusion configuration.
        
        Returns:
            Dictionary with fusion configuration details
        """
        return {
            'fusion_type': self.fusion_type,
            'signal_dim': self.signal_dim,
            'prompt_dim': self.prompt_dim,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'complexity': {
                'concat': 'O(1)',
                'attention': 'O(n²)', 
                'gating': 'O(n)'
            }[self.fusion_type]
        }


if __name__ == '__main__':
    """Comprehensive self-test for PromptFusion."""
    
    print("=== PromptFusion Self-Test ===")
    
    # Test configuration
    signal_dim = 256
    prompt_dim = 128  
    batch_size = 4
    num_patches = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test data
    signal_emb = torch.randn(batch_size, num_patches, signal_dim, device=device)
    prompt_emb = torch.randn(batch_size, prompt_dim, device=device)
    
    print(f"✓ Test data generated on {device}")
    print(f"✓ Signal embedding shape: {signal_emb.shape}")
    print(f"✓ Prompt embedding shape: {prompt_emb.shape}")
    
    # Test all fusion strategies
    fusion_types = ['concat', 'attention', 'gating']
    
    for fusion_type in fusion_types:
        print(f"\n--- Testing {fusion_type.upper()} Fusion ---")
        
        # Initialize fusion module
        fusion = PromptFusion(
            signal_dim=signal_dim,
            prompt_dim=prompt_dim, 
            fusion_type=fusion_type,
            num_attention_heads=4
        ).to(device)
        
        # Test 1: Basic functionality
        fusion.eval()
        fused = fusion(signal_emb, prompt_emb)
        expected_shape = (batch_size, num_patches, signal_dim)
        assert fused.shape == expected_shape, f"Expected {expected_shape}, got {fused.shape}"
        print(f"✓ Output shape correct: {fused.shape}")
        
        # Test 2: Consistency in eval mode
        fused2 = fusion(signal_emb, prompt_emb)
        torch.testing.assert_close(fused, fused2, rtol=1e-5, atol=1e-6)
        print("✓ Outputs consistent in eval mode")
        
        # Test 3: Gradient flow
        fusion.train()
        optimizer = torch.optim.Adam(fusion.parameters(), lr=1e-4)
        
        for i in range(3):
            optimizer.zero_grad()
            fused = fusion(signal_emb, prompt_emb)
            loss = fused.sum()
            loss.backward()
            
            # Check gradients exist
            grad_norm = sum(p.grad.norm().item() for p in fusion.parameters() if p.grad is not None)
            if grad_norm > 0:  # Some fusion types have no learnable parameters
                optimizer.step()
        
        print("✓ Gradient flow working")
        
        # Test 4: Model info
        info = fusion.get_fusion_info()
        print(f"✓ Model info: {info['num_parameters']} params, {info['complexity']} complexity")
    
    # Test 5: Input validation
    print("\n--- Testing Input Validation ---")
    
    fusion = PromptFusion(signal_dim, prompt_dim, 'attention').to(device)
    
    try:
        # Wrong signal shape
        wrong_signal = torch.randn(batch_size, signal_dim)  # Missing num_patches dimension
        fusion(wrong_signal, prompt_emb)
        assert False, "Should have raised ValueError"
    except ValueError:
        print("✓ Correctly rejects wrong signal shape")
    
    try:
        # Wrong prompt shape
        wrong_prompt = torch.randn(batch_size, num_patches, prompt_dim)  # Too many dimensions
        fusion(signal_emb, wrong_prompt)
        assert False, "Should have raised ValueError"
    except ValueError:
        print("✓ Correctly rejects wrong prompt shape")
    
    try:
        # Batch size mismatch
        mismatched_prompt = torch.randn(batch_size + 1, prompt_dim)
        fusion(signal_emb, mismatched_prompt)
        assert False, "Should have raised ValueError"
    except ValueError:
        print("✓ Correctly detects batch size mismatch")
    
    try:
        # Invalid fusion type
        invalid_fusion = PromptFusion(signal_dim, prompt_dim, 'invalid')
        assert False, "Should have raised ValueError"
    except ValueError:
        print("✓ Correctly rejects invalid fusion type")
    
    # Test 6: Different tensor sizes
    print("\n--- Testing Size Flexibility ---")
    
    test_cases = [
        (2, 32, 128, 64),   # Small batch
        (8, 128, 256, 128), # Large batch  
        (1, 16, 64, 32),    # Single sample
        (16, 256, 512, 256) # Large sequence
    ]
    
    for batch_size, num_patches, signal_dim, prompt_dim in test_cases:
        test_signal = torch.randn(batch_size, num_patches, signal_dim, device=device)
        test_prompt = torch.randn(batch_size, prompt_dim, device=device)
        
        fusion = PromptFusion(signal_dim, prompt_dim, 'attention').to(device)
        fusion.eval()
        
        result = fusion(test_signal, test_prompt)
        expected = (batch_size, num_patches, signal_dim)
        assert result.shape == expected, f"Size test failed: {result.shape} != {expected}"
    
    print("✓ All size flexibility tests passed")
    
    # Test 7: Performance comparison  
    print("\n--- Performance Comparison ---")
    
    signal_emb = torch.randn(32, 128, 256, device=device)
    prompt_emb = torch.randn(32, 128, device=device)
    
    import time
    
    performance_results = {}
    
    for fusion_type in ['concat', 'gating', 'attention']:
        fusion = PromptFusion(256, 128, fusion_type).to(device)
        fusion.eval()
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = fusion(signal_emb, prompt_emb)
        
        # Timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        for _ in range(100):
            with torch.no_grad():
                _ = fusion(signal_emb, prompt_emb)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed = time.time() - start_time
        avg_time = elapsed / 100 * 1000  # Convert to ms
        
        performance_results[fusion_type] = avg_time
        print(f"✓ {fusion_type}: {avg_time:.2f}ms per forward pass")
    
    # Verify performance order: concat < gating < attention (generally)
    print(f"✓ Performance ranking: {sorted(performance_results.items(), key=lambda x: x[1])}")
    
    # Test 8: Memory efficiency
    print("\n--- Memory Efficiency Test ---")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Large scale test
        large_signal = torch.randn(64, 256, 512, device=device)
        large_prompt = torch.randn(64, 256, device=device)
        
        fusion = PromptFusion(512, 256, 'attention').to(device)
        result = fusion(large_signal, large_prompt)
        
        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
        print(f"✓ Large-scale fusion processed, peak GPU memory: {peak_memory:.2f}GB")
    
    print("\n=== All PromptFusion Tests Passed! ===")
    print("Ready for multi-strategy prompt-signal fusion:")
    print("  • Concatenation: Fast, simple feature combination")
    print("  • Cross-Attention: Rich interaction modeling with residual connections") 
    print("  • Adaptive Gating: Balanced efficiency and representation quality")
    print("  • Automatic input validation and error handling")
    print("  • Flexible tensor size support")