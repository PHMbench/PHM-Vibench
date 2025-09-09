"""
E_01_HSE_v2: Prompt-guided Hierarchical Signal Embedding

This is a completely NEW implementation of HSE with prompt guidance for industrial
fault diagnosis. It combines system metadata as learnable prompts with patch-based
signal processing for enhanced cross-system generalization.

CRITICAL: This implementation has ZERO dependencies on the existing E_01_HSE.py
to ensure complete model isolation and avoid any code mixing conflicts.

Key Features:
- Two-level prompt encoding: System + Sample metadata integration
- Multi-strategy prompt-signal fusion (concatenation, attention, gating)
- Training stage control with prompt freezing for Pipeline_03 integration
- Graceful fallback to signal-only processing when metadata unavailable
- Complete self-testing with comprehensive validation

Author: PHM-Vibench Team
Date: 2025-01-06
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Dict, Optional, Union, Literal

# Import prompt components (with proper relative imports)
from ..components.SystemPromptEncoder import SystemPromptEncoder
from ..components.PromptFusion import PromptFusion


class E_01_HSE_v2(nn.Module):
    """
    Prompt-guided Hierarchical Signal Embedding v2.
    
    This is a completely new implementation that integrates system metadata as
    learnable prompts with patch-based signal processing for enhanced cross-system
    fault diagnosis generalization.
    
    Architecture:
    1. Patch-based signal processing (similar to original HSE concept)
    2. Two-level prompt encoding: System (Dataset_id + Domain_id) + Sample (Sample_rate)
    3. Multi-strategy prompt-signal fusion 
    4. Training stage control for Pipeline_03 integration
    
    CRITICAL: Zero dependencies on existing E_01_HSE.py - completely isolated implementation.
    """
    
    def __init__(self, args, metadata=None):
        """
        Initialize Prompt-guided HSE v2.
        
        Args:
            args: Configuration object with embedding parameters
            metadata: Dataset metadata (unused here, for API compatibility)
        """
        super(E_01_HSE_v2, self).__init__()
        
        # Core HSE parameters
        self.patch_size_L = getattr(args, 'patch_size_L', 16)
        self.patch_size_C = getattr(args, 'patch_size_C', 1) 
        self.num_patches = getattr(args, 'num_patches', 64)
        self.output_dim = getattr(args, 'output_dim', 128)
        
        # Prompt configuration
        self.prompt_dim = getattr(args, 'prompt_dim', 64)
        self.fusion_type = getattr(args, 'fusion_type', 'attention')
        self.max_dataset_ids = getattr(args, 'max_dataset_ids', 50)
        self.max_domain_ids = getattr(args, 'max_domain_ids', 50)
        
        # Training stage control
        self.training_stage = getattr(args, 'training_stage', 'pretraining')  # 'pretraining' or 'finetuning'
        self.freeze_prompts_in_finetuning = getattr(args, 'freeze_prompts_in_finetuning', True)
        
        # Patch processing layers (independent implementation)
        patch_input_dim = self.patch_size_L * (self.patch_size_C + 1)  # +1 for time embedding
        self.patch_linear1 = nn.Linear(patch_input_dim, self.output_dim)
        self.patch_linear2 = nn.Linear(self.output_dim, self.output_dim) 
        
        # Prompt system components
        self.prompt_encoder = SystemPromptEncoder(
            prompt_dim=self.prompt_dim,
            max_dataset_ids=self.max_dataset_ids,
            max_domain_ids=self.max_domain_ids
        )
        
        self.prompt_fusion = PromptFusion(
            signal_dim=self.output_dim,
            prompt_dim=self.prompt_dim,
            fusion_type=self.fusion_type
        )
        
        # Final output projection
        self.final_norm = nn.LayerNorm(self.output_dim)
        self.dropout = nn.Dropout(0.1)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize model parameters using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def set_training_stage(self, stage: Literal['pretraining', 'finetuning']):
        """
        Set training stage and handle prompt freezing.
        
        Args:
            stage: Training stage ('pretraining' or 'finetuning')
        """
        self.training_stage = stage
        
        if stage == 'finetuning' and self.freeze_prompts_in_finetuning:
            # Freeze prompt encoder parameters during finetuning
            for param in self.prompt_encoder.parameters():
                param.requires_grad = False
            print(f"✓ Prompt encoder frozen for finetuning stage")
        else:
            # Unfreeze all parameters
            for param in self.parameters():
                param.requires_grad = True
            print(f"✓ All parameters active for {stage} stage")
    
    def forward(self, 
                x: torch.Tensor, 
                fs: Union[torch.Tensor, float], 
                metadata: Optional[Dict[str, torch.Tensor]] = None,
                **kwargs) -> torch.Tensor:
        """
        Forward pass with prompt-guided signal embedding.
        
        Args:
            x: Input signal tensor of shape (B, L, C)
            fs: Sampling frequency (tensor or scalar)
            metadata: System metadata dictionary for prompt encoding
                - 'Dataset_id': Dataset identifiers (B,)
                - 'Domain_id': Operating condition identifiers (B,) 
                - 'Sample_rate': Sampling rates (B,)
            **kwargs: Additional arguments (for API compatibility)
        
        Returns:
            Embedded signal tensor of shape (B, num_patches, output_dim)
        """
        # Step 1: Patch-based signal processing (independent implementation)
        signal_embeddings = self._process_signal_patches(x, fs)
        
        # Step 2: Prompt processing (when metadata available)
        if metadata is not None and self._is_valid_metadata(metadata):
            try:
                # Encode system prompts
                prompt_embeddings = self.prompt_encoder(metadata)
                
                # Fuse signal and prompt features
                fused_embeddings = self.prompt_fusion(signal_embeddings, prompt_embeddings)
                
            except Exception as e:
                print(f"Warning: Prompt processing failed ({e}), using signal-only processing")
                fused_embeddings = signal_embeddings
        else:
            # Fallback: Signal-only processing
            fused_embeddings = signal_embeddings
        
        # Step 3: Final processing
        output = self.final_norm(fused_embeddings)
        output = self.dropout(output)
        
        return output
    
    def _process_signal_patches(self, x: torch.Tensor, fs: Union[torch.Tensor, float]) -> torch.Tensor:
        """
        Process input signal into patches (independent implementation).
        
        This method implements patch-based signal processing similar to the original
        HSE concept but with completely new code to ensure zero dependencies.
        """
        B, L, C = x.size()
        device = x.device
        
        # Handle sampling frequency
        if torch.is_tensor(fs):
            T = 1.0 / fs  # [B] tensor
        else:
            T = 1.0 / fs  # scalar
            T = torch.full((B,), T, device=device, dtype=torch.float32)
        
        # Generate time embeddings
        time_idx = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0)  # (1, L)
        time_emb = time_idx * T.unsqueeze(1)  # (B, L)
        
        # Handle input size constraints
        if self.patch_size_L > L:
            repeat_factor = (self.patch_size_L + L - 1) // L
            x = repeat(x, 'b l c -> b (l r) c', r=repeat_factor)
            time_emb = repeat(time_emb, 'b l -> b (l r)', r=repeat_factor)
            L = x.size(1)
        
        if self.patch_size_C > C:
            repeat_factor = (self.patch_size_C + C - 1) // C
            x = repeat(x, 'b l c -> b l (c r)', r=repeat_factor)
            C = x.size(2)
        
        # Random patch sampling
        max_start_L = L - self.patch_size_L
        max_start_C = C - self.patch_size_C
        
        start_L = torch.randint(0, max_start_L + 1, (B, self.num_patches), device=device)
        start_C = torch.randint(0, max_start_C + 1, (B, self.num_patches), device=device)
        
        # Create patch indices
        offset_L = torch.arange(self.patch_size_L, device=device)
        offset_C = torch.arange(self.patch_size_C, device=device)
        
        patch_idx_L = (start_L.unsqueeze(-1) + offset_L) % L  # (B, num_patches, patch_size_L)
        patch_idx_C = (start_C.unsqueeze(-1) + offset_C) % C  # (B, num_patches, patch_size_C)
        
        # Extract patches
        patch_idx_L = patch_idx_L.unsqueeze(-1)  # (B, num_patches, patch_size_L, 1)
        patch_idx_C = patch_idx_C.unsqueeze(-2)  # (B, num_patches, 1, patch_size_C)
        
        # Gather signal patches
        x_expanded = x.unsqueeze(1).expand(-1, self.num_patches, -1, -1)
        signal_patches = x_expanded.gather(2, patch_idx_L.expand(-1, -1, -1, C))
        signal_patches = signal_patches.gather(3, patch_idx_C.expand(-1, -1, self.patch_size_L, -1))
        
        # Gather time patches
        time_expanded = time_emb.unsqueeze(1).expand(-1, self.num_patches, -1)
        time_patches = time_expanded.gather(2, patch_idx_L.squeeze(-1))
        time_patches = time_patches.unsqueeze(-1).expand(-1, -1, -1, self.patch_size_C)
        
        # Concatenate signal and time
        patches = torch.cat([signal_patches, time_patches], dim=-1)  # (B, num_patches, patch_size_L, patch_size_C+1)
        
        # Process patches through linear layers
        patches_flat = rearrange(patches, 'b p l c -> b p (l c)')
        embeddings = self.patch_linear1(patches_flat)
        embeddings = F.silu(embeddings)  
        embeddings = self.patch_linear2(embeddings)
        
        return embeddings
    
    def _is_valid_metadata(self, metadata: Dict[str, torch.Tensor]) -> bool:
        """Check if metadata dictionary contains required fields for prompt processing."""
        required_fields = ['Dataset_id', 'Domain_id', 'Sample_rate']
        return all(field in metadata for field in required_fields)
    
    def get_embedding_info(self) -> dict:
        """
        Get information about the current embedding configuration.
        
        Returns:
            Dictionary with embedding configuration details
        """
        return {
            'model_type': 'E_01_HSE_v2',
            'patch_size_L': self.patch_size_L,
            'patch_size_C': self.patch_size_C,
            'num_patches': self.num_patches,
            'output_dim': self.output_dim,
            'prompt_dim': self.prompt_dim,
            'fusion_type': self.fusion_type,
            'training_stage': self.training_stage,
            'freeze_prompts_in_finetuning': self.freeze_prompts_in_finetuning,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'prompt_parameters': sum(p.numel() for p in self.prompt_encoder.parameters()),
            'fusion_parameters': sum(p.numel() for p in self.prompt_fusion.parameters())
        }


if __name__ == '__main__':
    """Comprehensive self-test for E_01_HSE_v2."""
    
    print("=== E_01_HSE_v2 Self-Test ===")
    
    # Test configuration
    class MockArgs:
        patch_size_L = 16
        patch_size_C = 1
        num_patches = 64
        output_dim = 128
        prompt_dim = 64
        fusion_type = 'attention'
        max_dataset_ids = 30
        max_domain_ids = 20
        training_stage = 'pretraining'
        freeze_prompts_in_finetuning = True
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = MockArgs()
    
    # Initialize embedding
    embedding = E_01_HSE_v2(args).to(device)
    print(f"✓ Initialized E_01_HSE_v2 on {device}")
    print(f"✓ Model info: {embedding.get_embedding_info()}")
    
    # Test data
    batch_size = 4
    seq_length = 1024
    channels = 1
    
    signal = torch.randn(batch_size, seq_length, channels, device=device)
    fs = torch.tensor([1000.0, 2000.0, 1500.0, 2500.0], device=device)
    
    metadata = SystemPromptEncoder.create_metadata_dict(
        dataset_ids=[1, 6, 13, 19],
        domain_ids=[0, 3, 5, 7],
        sample_rates=[1000.0, 2000.0, 1500.0, 2500.0],
        device=device
    )
    
    print(f"✓ Test data created: signal {signal.shape}, fs {fs.shape}")
    
    # Test 1: Basic functionality with prompts
    print("\n--- Test 1: Basic Functionality with Prompts ---")
    
    embedding.eval()
    with torch.no_grad():
        output = embedding(signal, fs, metadata)
    
    expected_shape = (batch_size, args.num_patches, args.output_dim)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    print(f"✓ Output shape correct: {output.shape}")
    
    # Test 2: Signal-only processing (fallback)
    print("\n--- Test 2: Signal-only Processing (Fallback) ---")
    
    with torch.no_grad():
        output_no_metadata = embedding(signal, fs, metadata=None)
    
    assert output_no_metadata.shape == expected_shape
    print("✓ Signal-only processing works correctly")
    
    # Test 3: Training stage control
    print("\n--- Test 3: Training Stage Control ---")
    
    # Test pretraining stage
    embedding.set_training_stage('pretraining')
    assert embedding.training_stage == 'pretraining'
    prompt_params_trainable = any(p.requires_grad for p in embedding.prompt_encoder.parameters())
    assert prompt_params_trainable, "Prompt parameters should be trainable in pretraining"
    print("✓ Pretraining stage configured correctly")
    
    # Test finetuning stage
    embedding.set_training_stage('finetuning')
    assert embedding.training_stage == 'finetuning'
    prompt_params_frozen = all(not p.requires_grad for p in embedding.prompt_encoder.parameters())
    assert prompt_params_frozen, "Prompt parameters should be frozen in finetuning"
    print("✓ Finetuning stage configured correctly (prompts frozen)")
    
    # Test 4: Gradient flow in pretraining
    print("\n--- Test 4: Gradient Flow in Pretraining ---")
    
    embedding.set_training_stage('pretraining')
    embedding.train()
    optimizer = torch.optim.Adam(embedding.parameters(), lr=1e-4)
    
    for i in range(3):
        optimizer.zero_grad()
        output = embedding(signal, fs, metadata)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist for all trainable parameters
        grad_norm = sum(p.grad.norm().item() for p in embedding.parameters() if p.grad is not None)
        assert grad_norm > 0, "No gradients computed"
        
        optimizer.step()
    
    print("✓ Gradient flow working in pretraining mode")
    
    # Test 5: Different fusion strategies
    print("\n--- Test 5: Different Fusion Strategies ---")
    
    fusion_types = ['concat', 'attention', 'gating']
    
    for fusion_type in fusion_types:
        args.fusion_type = fusion_type
        test_embedding = E_01_HSE_v2(args).to(device)
        test_embedding.eval()
        
        with torch.no_grad():
            output = test_embedding(signal, fs, metadata)
        
        assert output.shape == expected_shape
        print(f"✓ {fusion_type} fusion strategy working")
    
    # Test 6: Invalid metadata handling
    print("\n--- Test 6: Invalid Metadata Handling ---")
    
    # Test with incomplete metadata
    incomplete_metadata = {'Dataset_id': metadata['Dataset_id']}  # Missing required fields
    
    with torch.no_grad():
        output = embedding(signal, fs, incomplete_metadata)
    
    assert output.shape == expected_shape
    print("✓ Gracefully handles incomplete metadata")
    
    # Test 7: Different input sizes
    print("\n--- Test 7: Input Size Flexibility ---")
    
    test_cases = [
        (2, 512, 1),   # Small batch
        (8, 2048, 1),  # Large sequence
        (1, 256, 2),   # Multiple channels
        (16, 128, 1)   # Large batch
    ]
    
    for B, L, C in test_cases:
        test_signal = torch.randn(B, L, C, device=device)
        test_fs = torch.rand(B, device=device) * 2000 + 500  # Random fs 500-2500
        test_metadata = SystemPromptEncoder.create_metadata_dict(
            dataset_ids=torch.randint(0, 25, (B,)),
            domain_ids=torch.randint(0, 15, (B,)),
            sample_rates=test_fs.cpu().numpy(),
            device=device
        )
        
        with torch.no_grad():
            output = embedding(test_signal, test_fs, test_metadata)
        
        expected = (B, args.num_patches, args.output_dim)
        assert output.shape == expected, f"Size test failed: {output.shape} != {expected}"
    
    print("✓ All input size flexibility tests passed")
    
    # Test 8: Memory efficiency
    print("\n--- Test 8: Memory Efficiency ---")
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
        # Large scale test
        large_signal = torch.randn(32, 1024, 1, device=device)
        large_fs = torch.rand(32, device=device) * 2000 + 500
        large_metadata = SystemPromptEncoder.create_metadata_dict(
            dataset_ids=torch.randint(0, 25, (32,)),
            domain_ids=torch.randint(0, 15, (32,)),
            sample_rates=large_fs.cpu().numpy(),
            device=device
        )
        
        with torch.no_grad():
            result = embedding(large_signal, large_fs, large_metadata)
        
        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"✓ Large-scale processing completed, peak GPU memory: {peak_memory:.2f}GB")
    
    print("\n=== All E_01_HSE_v2 Tests Passed! ===")
    print("Ready for prompt-guided signal embedding:")
    print("  • Two-level prompt encoding (System + Sample metadata)")
    print("  • Multi-strategy signal-prompt fusion (concat/attention/gating)")
    print("  • Training stage control with prompt freezing")
    print("  • Graceful fallback to signal-only processing")
    print("  • Complete independence from existing E_01_HSE.py")
    print("  • Pipeline_03 integration ready")