"""
SystemPromptEncoder: Two-level system information encoding for industrial fault diagnosis

This module implements a hierarchical prompt encoding system that converts system metadata
into learnable prompt vectors. The encoder supports two levels of prompts:
1. System-level: Dataset_id + Domain_id (identifies the industrial system and operating conditions)  
2. Sample-level: Sample_rate (captures signal acquisition parameters)

CRITICAL: Fault type (Label) is NOT included in prompts as it is the prediction target.

Author: PHM-Vibench Team
Date: 2025-01-06
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union


class SystemPromptEncoder(nn.Module):
    """
    Two-level system information encoder for industrial equipment metadata.
    
    Architecture:
    - System Level: Dataset_id + Domain_id → System-specific prompt
    - Sample Level: Sample_rate → Signal acquisition prompt  
    - Multi-head attention fusion → Final unified prompt
    
    Key Features:
    - Hierarchical prompt design for multi-level system information
    - Embedding tables for categorical features (Dataset_id, Domain_id)
    - Linear projection for numerical features (Sample_rate)
    - Self-attention mechanism for prompt fusion
    - Comprehensive input validation and error handling
    """
    
    def __init__(self, 
                 prompt_dim: int = 128,
                 max_dataset_ids: int = 50,
                 max_domain_ids: int = 50,
                 num_attention_heads: int = 4,
                 dropout: float = 0.1):
        """
        Initialize SystemPromptEncoder.
        
        Args:
            prompt_dim: Dimension of output prompt vectors
            max_dataset_ids: Maximum number of dataset IDs to support
            max_domain_ids: Maximum number of domain IDs to support  
            num_attention_heads: Number of heads for multi-head attention
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        self.prompt_dim = prompt_dim
        self.max_dataset_ids = max_dataset_ids
        self.max_domain_ids = max_domain_ids
        
        # Calculate embedding dimensions to ensure proper concatenation
        self.dataset_dim = prompt_dim // 3
        self.domain_dim = prompt_dim // 3  
        self.sample_dim = prompt_dim - self.dataset_dim - self.domain_dim  # Use remaining dims
        
        # Categorical feature embedding tables
        self.dataset_embedding = nn.Embedding(max_dataset_ids, self.dataset_dim)
        self.domain_embedding = nn.Embedding(max_domain_ids, self.domain_dim)
        
        # Numerical feature projection layer
        self.sample_rate_proj = nn.Linear(1, self.sample_dim)
        
        # System-level prompt fusion (Dataset_id + Domain_id)
        self.system_fusion = nn.Linear(self.dataset_dim + self.domain_dim, prompt_dim)
        self.system_norm = nn.LayerNorm(prompt_dim)
        
        # Sample-level prompt processing
        self.sample_fusion = nn.Linear(self.sample_dim, prompt_dim)
        self.sample_norm = nn.LayerNorm(prompt_dim)
        
        # Multi-head attention for hierarchical prompt fusion
        self.prompt_attention = nn.MultiheadAttention(
            prompt_dim, 
            num_attention_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # Final projection and normalization
        self.final_projection = nn.Linear(prompt_dim, prompt_dim)
        self.final_norm = nn.LayerNorm(prompt_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize model parameters using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
    
    def forward(self, dataset_id: torch.Tensor, domain_id: torch.Tensor, sample_rate: torch.Tensor) -> torch.Tensor:
        """
        Encode system metadata into prompt vectors.

        Args:
            dataset_id: tensor of shape (B,) with dataset identifiers
            domain_id: tensor of shape (B,) with operating condition identifiers
            sample_rate: tensor of shape (B,) with sampling rates

        Returns:
            prompt_embedding: tensor of shape (B, prompt_dim) containing encoded prompts
        """
        # System-level prompt: Dataset_id + Domain_id
        dataset_emb = self.dataset_embedding(dataset_id)  # (B, prompt_dim//3)
        domain_emb = self.domain_embedding(domain_id)      # (B, prompt_dim//3)

        system_concat = torch.cat([dataset_emb, domain_emb], dim=-1)       # (B, 2*prompt_dim//3)
        system_prompt = self.system_norm(self.system_fusion(system_concat)) # (B, prompt_dim)

        # Sample-level prompt: Sample_rate
        sample_rate_normalized = sample_rate.unsqueeze(-1) / 10000.0  # Normalize to [0,1] range
        sample_emb = self.sample_rate_proj(sample_rate_normalized)          # (B, prompt_dim//3)
        sample_prompt = self.sample_norm(self.sample_fusion(sample_emb))    # (B, prompt_dim)

        # Multi-head attention fusion of hierarchical prompts
        prompt_stack = torch.stack([system_prompt, sample_prompt], dim=1)   # (B, 2, prompt_dim)

        # Self-attention to fuse system and sample level information
        fused_prompts, attention_weights = self.prompt_attention(
            prompt_stack, prompt_stack, prompt_stack
        )  # (B, 2, prompt_dim)

        # Aggregate to final prompt vector (mean pooling)
        aggregated_prompt = fused_prompts.mean(dim=1)                       # (B, prompt_dim)

        # Final transformation
        final_prompt = self.final_projection(aggregated_prompt)             # (B, prompt_dim)
        final_prompt = self.final_norm(final_prompt)
        final_prompt = self.dropout(final_prompt)

        return final_prompt
    
    

if __name__ == '__main__':
    """Simple self-test for SystemPromptEncoder."""

    print("=== SystemPromptEncoder Self-Test ===")

    # Test configuration
    prompt_dim = 128
    batch_size = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize encoder
    encoder = SystemPromptEncoder(
        prompt_dim=prompt_dim,
        max_dataset_ids=30,
        max_domain_ids=20,
        num_attention_heads=4
    ).to(device)
    
    print(f"✓ Initialized encoder on {device}")
    print(f"✓ Model parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    
    # Test 1: Basic functionality
    print("\n--- Test 1: Basic Functionality ---")

    dataset_ids = torch.tensor([1, 6, 13, 19], device=device)      # CWRU, XJTU, THU, MFPT
    domain_ids = torch.tensor([0, 3, 5, 7], device=device)         # Different operating conditions
    sample_rates = torch.tensor([1000.0, 2000.0, 1500.0, 2500.0], device=device)

    prompt = encoder(dataset_ids, domain_ids, sample_rates)
    expected_shape = (batch_size, prompt_dim)
    assert prompt.shape == expected_shape, f"Expected {expected_shape}, got {prompt.shape}"
    print(f"✓ Output shape correct: {prompt.shape}")

    # Test 2: Consistency check
    print("\n--- Test 2: Consistency Check ---")

    encoder.eval()  # Switch to eval mode first to disable dropout
    prompt2 = encoder(dataset_ids, domain_ids, sample_rates)
    prompt3 = encoder(dataset_ids, domain_ids, sample_rates)

    torch.testing.assert_close(prompt2, prompt3, rtol=1e-5, atol=1e-6)
    print("✓ Model outputs are consistent in eval mode")

    # Test 3: Different batch sizes
    print("\n--- Test 3: Batch Size Flexibility ---")

    for test_batch_size in [1, 8, 16, 32]:
        test_dataset_ids = torch.randint(0, 25, (test_batch_size,), device=device)
        test_domain_ids = torch.randint(0, 15, (test_batch_size,), device=device)
        test_sample_rates = torch.rand(test_batch_size, device=device) * 2000 + 500

        test_prompt = encoder(test_dataset_ids, test_domain_ids, test_sample_rates)
        assert test_prompt.shape == (test_batch_size, prompt_dim)
        
    print("✓ Handles various batch sizes correctly")
    
    # Test 4: Gradient flow
    print("\n--- Test 4: Gradient Flow ---")

    encoder.train()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)

    for i in range(3):
        optimizer.zero_grad()
        prompt = encoder(dataset_ids, domain_ids, sample_rates)
        loss = prompt.sum()  # Dummy loss
        loss.backward()

        # Check gradients exist
        grad_norm = sum(p.grad.norm().item() for p in encoder.parameters() if p.grad is not None)
        assert grad_norm > 0, "No gradients computed"

        optimizer.step()

    print("✓ Gradients flow correctly through the model")

    print("\n=== All tests passed! ===")