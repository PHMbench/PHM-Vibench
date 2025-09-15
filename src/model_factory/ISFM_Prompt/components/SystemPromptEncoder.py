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
    
    def forward(self, metadata_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode system metadata into prompt vectors.
        
        Args:
            metadata_dict: Dictionary containing system metadata tensors
                Required keys:
                - 'Dataset_id': tensor of shape (B,) with dataset identifiers
                - 'Domain_id': tensor of shape (B,) with operating condition identifiers
                - 'Sample_rate': tensor of shape (B,) with sampling rates
                
                CRITICAL: Does NOT contain 'Label' - fault type is prediction target!
        
        Returns:
            prompt_embedding: tensor of shape (B, prompt_dim) containing encoded prompts
        
        Raises:
            KeyError: If required metadata fields are missing
            ValueError: If tensor shapes or values are invalid
        """
        # Validate inputs
        self._validate_metadata_dict(metadata_dict)
        
        batch_size = metadata_dict['Dataset_id'].size(0)
        device = metadata_dict['Dataset_id'].device
        
        # System-level prompt: Dataset_id + Domain_id  
        dataset_emb = self.dataset_embedding(metadata_dict['Dataset_id'])  # (B, prompt_dim//3)
        domain_emb = self.domain_embedding(metadata_dict['Domain_id'])      # (B, prompt_dim//3)
        
        system_concat = torch.cat([dataset_emb, domain_emb], dim=-1)       # (B, 2*prompt_dim//3)
        system_prompt = self.system_norm(self.system_fusion(system_concat)) # (B, prompt_dim)
        
        # Sample-level prompt: Sample_rate
        sample_rate_normalized = metadata_dict['Sample_rate'].unsqueeze(-1) / 10000.0  # Normalize to [0,1] range
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
    
    def _validate_metadata_dict(self, metadata_dict: Dict[str, torch.Tensor]) -> None:
        """
        Validate metadata dictionary for required fields and correct formats.
        
        Args:
            metadata_dict: Dictionary to validate
            
        Raises:
            KeyError: If required fields are missing
            ValueError: If field formats are incorrect
        """
        required_fields = ['Dataset_id', 'Domain_id', 'Sample_rate']
        
        for field in required_fields:
            if field not in metadata_dict:
                raise KeyError(f"Required metadata field '{field}' is missing")
        
        # Validate batch consistency
        batch_sizes = [metadata_dict[field].size(0) for field in required_fields]
        if not all(bs == batch_sizes[0] for bs in batch_sizes):
            raise ValueError(f"Inconsistent batch sizes: {dict(zip(required_fields, batch_sizes))}")
        
        # Validate ID ranges
        dataset_ids = metadata_dict['Dataset_id']
        domain_ids = metadata_dict['Domain_id']
        
        if dataset_ids.min() < 0 or dataset_ids.max() >= self.max_dataset_ids:
            raise ValueError(f"Dataset_id out of range [0, {self.max_dataset_ids}): {dataset_ids.min()}-{dataset_ids.max()}")
        
        if domain_ids.min() < 0 or domain_ids.max() >= self.max_domain_ids:
            raise ValueError(f"Domain_id out of range [0, {self.max_domain_ids}): {domain_ids.min()}-{domain_ids.max()}")
        
        # Validate sample rates are positive
        sample_rates = metadata_dict['Sample_rate']
        if sample_rates.min() <= 0:
            raise ValueError(f"Sample_rate must be positive, got min: {sample_rates.min()}")
        
        # Ensure no Label field is accidentally included
        if 'Label' in metadata_dict:
            raise ValueError("Label field detected in metadata! Fault type should not be in prompt - it's the prediction target.")
    
    @staticmethod
    def create_metadata_dict(dataset_ids: Union[List[int], torch.Tensor],
                           domain_ids: Union[List[int], torch.Tensor], 
                           sample_rates: Union[List[float], torch.Tensor],
                           device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        """
        Utility function to create properly formatted metadata dictionary.
        
        Args:
            dataset_ids: List or tensor of dataset identifiers
            domain_ids: List or tensor of domain identifiers  
            sample_rates: List or tensor of sampling rates
            device: Target device for tensors (auto-detect if None)
        
        Returns:
            Dictionary with properly formatted metadata tensors
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Convert inputs to tensors
        if not isinstance(dataset_ids, torch.Tensor):
            dataset_ids = torch.tensor(dataset_ids, dtype=torch.long, device=device)
        
        if not isinstance(domain_ids, torch.Tensor):
            domain_ids = torch.tensor(domain_ids, dtype=torch.long, device=device)
            
        if not isinstance(sample_rates, torch.Tensor):
            sample_rates = torch.tensor(sample_rates, dtype=torch.float32, device=device)
        
        return {
            'Dataset_id': dataset_ids.to(device),
            'Domain_id': domain_ids.to(device),
            'Sample_rate': sample_rates.to(device)
        }


if __name__ == '__main__':
    """Comprehensive self-test for SystemPromptEncoder."""
    
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
    
    metadata_dict = SystemPromptEncoder.create_metadata_dict(
        dataset_ids=[1, 6, 13, 19],      # CWRU, XJTU, THU, MFPT
        domain_ids=[0, 3, 5, 7],         # Different operating conditions
        sample_rates=[1000.0, 2000.0, 1500.0, 2500.0],  # Different sampling rates
        device=device
    )
    
    prompt = encoder(metadata_dict)
    expected_shape = (batch_size, prompt_dim)
    assert prompt.shape == expected_shape, f"Expected {expected_shape}, got {prompt.shape}"
    print(f"✓ Output shape correct: {prompt.shape}")
    
    # Test 2: Consistency check
    print("\n--- Test 2: Consistency Check ---")
    
    encoder.eval()  # Switch to eval mode first to disable dropout
    prompt2 = encoder(metadata_dict)
    prompt3 = encoder(metadata_dict)
    
    torch.testing.assert_close(prompt2, prompt3, rtol=1e-5, atol=1e-6)
    print("✓ Model outputs are consistent in eval mode")
    
    # Test 3: Different batch sizes
    print("\n--- Test 3: Batch Size Flexibility ---")
    
    for test_batch_size in [1, 8, 16, 32]:
        test_metadata = SystemPromptEncoder.create_metadata_dict(
            dataset_ids=torch.randint(0, 25, (test_batch_size,)),
            domain_ids=torch.randint(0, 15, (test_batch_size,)),
            sample_rates=torch.rand(test_batch_size) * 2000 + 500,
            device=device
        )
        
        test_prompt = encoder(test_metadata)
        assert test_prompt.shape == (test_batch_size, prompt_dim)
        
    print("✓ Handles various batch sizes correctly")
    
    # Test 4: Gradient flow
    print("\n--- Test 4: Gradient Flow ---")
    
    encoder.train()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)
    
    for i in range(3):
        optimizer.zero_grad()
        prompt = encoder(metadata_dict)
        loss = prompt.sum()  # Dummy loss
        loss.backward()
        
        # Check gradients exist
        grad_norm = sum(p.grad.norm().item() for p in encoder.parameters() if p.grad is not None)
        assert grad_norm > 0, "No gradients computed"
        
        optimizer.step()
    
    print("✓ Gradients flow correctly through the model")
    
    # Test 5: Input validation
    print("\n--- Test 5: Input Validation ---")
    
    try:
        # Missing required field
        incomplete_metadata = {'Dataset_id': torch.tensor([1]), 'Domain_id': torch.tensor([2])}
        encoder(incomplete_metadata)
        assert False, "Should have raised KeyError"
    except KeyError:
        print("✓ Correctly rejects incomplete metadata")
    
    try:
        # ID out of range
        invalid_metadata = SystemPromptEncoder.create_metadata_dict(
            dataset_ids=[100],  # Out of range
            domain_ids=[1], 
            sample_rates=[1000.0]
        )
        encoder(invalid_metadata)
        assert False, "Should have raised ValueError"
    except ValueError:
        print("✓ Correctly validates ID ranges")
    
    try:
        # Label field included (should be rejected)
        metadata_with_label = SystemPromptEncoder.create_metadata_dict(
            dataset_ids=[1], domain_ids=[2], sample_rates=[1000.0]
        )
        metadata_with_label['Label'] = torch.tensor([1])  # Add forbidden Label
        encoder(metadata_with_label)
        assert False, "Should have rejected Label field"
    except ValueError as e:
        assert "Label field detected" in str(e)
        print("✓ Correctly rejects Label field (fault type is prediction target)")
    
    # Test 6: Real-world system scenarios  
    print("\n--- Test 6: Real-world System Scenarios ---")
    
    # Scenario 1: Cross-system generalization (CWRU → XJTU)
    cwru_metadata = SystemPromptEncoder.create_metadata_dict(
        dataset_ids=[1] * 4,    # CWRU system
        domain_ids=[0, 1, 2, 3],  # Different operating conditions
        sample_rates=[1000.0] * 4,
        device=device
    )
    
    xjtu_metadata = SystemPromptEncoder.create_metadata_dict(
        dataset_ids=[6] * 4,    # XJTU system  
        domain_ids=[0, 1, 2, 3],  # Same operating conditions
        sample_rates=[2000.0] * 4,  # Different sampling rate
        device=device
    )
    
    cwru_prompts = encoder(cwru_metadata)
    xjtu_prompts = encoder(xjtu_metadata)
    
    # Prompts should be different (different systems)
    similarity = F.cosine_similarity(cwru_prompts, xjtu_prompts, dim=-1).mean()
    assert similarity < 0.8, f"Cross-system prompts too similar: {similarity:.3f}"
    print(f"✓ Cross-system prompts are sufficiently different (similarity: {similarity:.3f})")
    
    # Scenario 2: Same system, different conditions
    same_system_prompts = encoder(SystemPromptEncoder.create_metadata_dict(
        dataset_ids=[1] * 2,    # Same system
        domain_ids=[0, 5],      # Different conditions  
        sample_rates=[1000.0] * 2,
        device=device
    ))
    
    condition_similarity = F.cosine_similarity(
        same_system_prompts[0:1], same_system_prompts[1:2], dim=-1
    ).item()
    print(f"✓ Same system, different conditions similarity: {condition_similarity:.3f}")
    
    # Test 7: Memory efficiency
    print("\n--- Test 7: Memory Efficiency ---")
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
        # Large batch test
        large_metadata = SystemPromptEncoder.create_metadata_dict(
            dataset_ids=torch.randint(0, 25, (1000,)),
            domain_ids=torch.randint(0, 15, (1000,)),
            sample_rates=torch.rand(1000) * 2000 + 500,
            device=device
        )
        
        large_prompt = encoder(large_metadata)
        assert large_prompt.shape == (1000, prompt_dim)
        
        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"✓ Large batch (1000) processed successfully, peak GPU memory: {peak_memory:.2f}GB")
    
    print("\n=== All SystemPromptEncoder Tests Passed! ===")
    print(f"Model ready for two-level prompt encoding:")
    print(f"  • System Level: Dataset_id + Domain_id → System context")  
    print(f"  • Sample Level: Sample_rate → Signal acquisition context")
    print(f"  • NO fault-level prompts (Label is prediction target)")
    print(f"  • Output dimension: {prompt_dim}")
    print(f"  • Supports {encoder.max_dataset_ids} datasets, {encoder.max_domain_ids} domains")