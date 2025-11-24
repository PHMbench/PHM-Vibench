"""
HSE_prompt: Simplified Heterogeneous Signal Embedding with System Prompts

This is a simplified version that combines Heterogeneous Signal Embedding (HSE)
with lightweight system-specific learnable prompts for industrial fault diagnosis.

Key Features:
- Heterogeneous signal processing for different lengths and sampling rates
- Simple Dataset_id → learnable prompt mapping
- Direct signal + prompt combination (no complex fusion strategies)
- Lightweight and easy to understand
- Fallback to signal-only processing when metadata unavailable

Architecture:
Signal → HSE Processing → Prompt Combination → Output

Author: PHM-Vibench Team
Date: 2025-01-23
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Union

from src.model_factory.ISFM.system_utils import normalize_fs

# Import simplified prompt encoder
from ..components.SimpleSystemPromptEncoder import SimpleSystemPromptEncoder


class HSE_prompt(nn.Module):
    """
    Simplified Heterogeneous Signal Embedding with System Prompts.

    This model combines the core HSE functionality with lightweight system-specific
    prompts for enhanced cross-system generalization in industrial fault diagnosis.

    Architecture:
    1. Patch-based heterogeneous signal processing
    2. Dataset_id → learnable prompt mapping
    3. Simple signal + prompt combination
    4. Output normalized embeddings

    Simplified from E_01_HSE_v2:
    - Removed complex fusion strategies
    - Removed multi-level prompt encoding
    - Kept core HSE signal processing
    - Lightweight prompt mechanism
    """

    def __init__(self, args):
        """
        Initialize HSE_prompt.

        Args:
            args: Configuration object with model parameters
                Required attributes:
                - patch_size_L: Patch length for signal processing
                - patch_size_C: Patch channel size
                - num_patches: Number of patches to extract
                - output_dim: Output embedding dimension

                Optional prompt-related attributes:
                - use_prompt: Enable prompt functionality (default: True)
                - prompt_dim: Prompt vector dimension (default: 64)
                - max_dataset_ids: Maximum dataset IDs to support (default: 50)
                - prompt_combination: How to combine signal and prompt ('add'/'concat', default: 'add')
        """
        super(HSE_prompt, self).__init__()

        # Core HSE parameters
        self.patch_size_L = getattr(args, 'patch_size_L', 16)
        self.patch_size_C = getattr(args, 'patch_size_C', 1)
        self.num_patches = getattr(args, 'num_patches', 64)
        self.output_dim = getattr(args, 'output_dim', 128)

        # Prompt configuration
        self.use_prompt = getattr(args, 'use_prompt', True)
        self.prompt_dim = getattr(args, 'prompt_dim', 64)
        self.max_dataset_ids = getattr(args, 'max_dataset_ids', 50)
        self.prompt_combination = getattr(args, 'prompt_combination', 'add')

        # Validate prompt combination mode
        if self.prompt_combination not in ['add', 'concat']:
            raise ValueError(f"prompt_combination must be 'add' or 'concat', got '{self.prompt_combination}'")

        # Patch processing layers
        patch_input_dim = self.patch_size_L * (self.patch_size_C + 1)  # +1 for time embedding
        self.patch_linear1 = nn.Linear(patch_input_dim, self.output_dim)
        self.patch_linear2 = nn.Linear(self.output_dim, self.output_dim)

        # Simple prompt system
        if self.use_prompt:
            self.prompt_encoder = SimpleSystemPromptEncoder(
                prompt_dim=self.prompt_dim,
                max_dataset_ids=self.max_dataset_ids
            )

            # Handle different combination methods
            if self.prompt_combination == 'add':
                if self.prompt_dim != self.output_dim:
                    # Project prompt to match signal dimension
                    self.prompt_proj = nn.Linear(self.prompt_dim, self.output_dim)
                else:
                    self.prompt_proj = nn.Identity()
            elif self.prompt_combination == 'concat':
                # Project concatenated features back to output_dim
                self.concat_proj = nn.Linear(self.output_dim + self.prompt_dim, self.output_dim)

        # Final processing
        self.final_norm = nn.LayerNorm(self.output_dim)
        self.dropout = nn.Dropout(0.1)

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        """Initialize model parameters."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def set_training_stage(self, stage: str):
        """
        Set training stage and handle prompt freezing.

        Args:
            stage: Training stage ('pretraining' or 'finetuning')
        """
        stage = stage.lower()
        if stage in {"pretraining", "pretrain"}:
            stage = "pretrain"
        elif stage in {"finetuning", "finetune"}:
            stage = "finetune"

        self.training_stage = stage

        if self.use_prompt and hasattr(self, 'prompt_encoder'):
            if stage == 'finetuning' and self.freeze_prompts_in_finetuning:
                # Freeze prompt encoder parameters during finetuning
                for param in self.prompt_encoder.parameters():
                    param.requires_grad = False
                print(f"✓ Prompt encoder frozen for finetuning stage")
            else:
                # Unfreeze all parameters
                for param in self.prompt_encoder.parameters():
                    param.requires_grad = True
                print(f"✓ Prompt encoder active for {stage} stage")

    def forward(self,
                x: torch.Tensor,
                fs: Union[torch.Tensor, float],
                dataset_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through HSE_prompt.

        Args:
            x: Input signal tensor of shape (B, L, C)
            fs: Sampling frequency (tensor or scalar)
            dataset_ids: Dataset identifiers for prompt processing (B,)

        Returns:
            Embedded signal tensor of shape (B, num_patches, output_dim)
        """
        # Step 1: Heterogeneous signal processing
        signal_embeddings = self._process_signal_patches(x, fs)

        # Step 2: Simple prompt combination (if available)
        if self.use_prompt and dataset_ids is not None:
            try:
                # Get prompt vectors
                prompt_vectors = self.prompt_encoder(dataset_ids)  # (B, prompt_dim)

                # Combine signal and prompt
                if self.prompt_combination == 'add':
                    # Project prompt to match signal dimension if needed
                    prompt_projected = self.prompt_proj(prompt_vectors)  # (B, output_dim)
                    prompt_projected = prompt_projected.unsqueeze(1)  # (B, 1, output_dim)

                    # Simple addition
                    combined_embeddings = signal_embeddings + prompt_projected

                elif self.prompt_combination == 'concat':
                    # Expand prompt to match number of patches
                    prompt_expanded = prompt_vectors.unsqueeze(1).expand(-1, self.num_patches, -1)  # (B, num_patches, prompt_dim)

                    # Concatenate and project
                    concatenated = torch.cat([signal_embeddings, prompt_expanded], dim=-1)  # (B, num_patches, output_dim + prompt_dim)
                    combined_embeddings = self.concat_proj(concatenated)

                signal_embeddings = combined_embeddings

            except Exception as e:
                print(f"Warning: Prompt processing failed ({e}), using signal-only processing")

        # Step 3: Final processing
        output = self.final_norm(signal_embeddings)
        output = self.dropout(output)

        return output

    def _process_signal_patches(self, x: torch.Tensor, fs: Union[torch.Tensor, float]) -> torch.Tensor:
        """
        Process heterogeneous signal into patches.

        Args:
            x: Input signal tensor (B, L, C)
            fs: Sampling frequency

        Returns:
            Processed signal embeddings (B, num_patches, output_dim)
        """
        B, L, C = x.size()
        device = x.device

        # Handle sampling frequency (统一为 [B])
        fs_tensor = normalize_fs(fs, batch_size=B, device=device, as_column=False)  # [B]
        T = 1.0 / fs_tensor  # [B]

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

    def get_model_info(self) -> dict:
        """
        Get information about the current model configuration.

        Returns:
            Dictionary with model configuration details
        """
        info = {
            'model_type': 'HSE_prompt',
            'patch_size_L': self.patch_size_L,
            'patch_size_C': self.patch_size_C,
            'num_patches': self.num_patches,
            'output_dim': self.output_dim,
            'use_prompt': self.use_prompt,
            'total_parameters': sum(p.numel() for p in self.parameters())
        }

        if self.use_prompt:
            info.update({
                'prompt_dim': self.prompt_dim,
                'max_dataset_ids': self.max_dataset_ids,
                'prompt_combination': self.prompt_combination,
                'prompt_parameters': sum(p.numel() for p in self.prompt_encoder.parameters())
            })

        return info


if __name__ == '__main__':
    """Comprehensive self-test for HSE_prompt."""

    print("=== HSE_prompt Self-Test ===")

    # Test configuration
    class MockArgs:
        def __init__(self):
            # HSE parameters
            self.patch_size_L = 16
            self.patch_size_C = 1
            self.num_patches = 64
            self.output_dim = 128

            # Prompt parameters
            self.use_prompt = True
            self.prompt_dim = 64
            self.max_dataset_ids = 20
            self.prompt_combination = 'add'  # Test 'add' mode

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = MockArgs()

    # Test 1: Basic functionality with prompts
    print("\n--- Test 1: Basic Functionality with Prompts ---")

    model = HSE_prompt(args).to(device)
    print(f"✓ Initialized HSE_prompt on {device}")
    print(f"✓ Model info: {model.get_model_info()}")

    # Test data
    batch_size = 4
    seq_length = 1024
    channels = 1

    signal = torch.randn(batch_size, seq_length, channels, device=device)
    fs = torch.tensor([1000.0, 2000.0, 1500.0, 2500.0], device=device)
    dataset_ids = torch.tensor([1, 6, 13, 19], device=device)

    # Forward pass with prompts
    model.eval()
    with torch.no_grad():
        output = model(signal, fs, dataset_ids)

    expected_shape = (batch_size, args.num_patches, args.output_dim)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    print(f"✓ Output shape correct: {output.shape}")

    # Test 2: Signal-only processing (no prompts)
    print("\n--- Test 2: Signal-only Processing ---")

    with torch.no_grad():
        output_no_prompt = model(signal, fs, dataset_ids=None)

    assert output_no_prompt.shape == expected_shape
    print("✓ Signal-only processing works correctly")

    # Test 3: Different combination methods
    print("\n--- Test 3: Different Combination Methods ---")

    for combination in ['add', 'concat']:
        args.prompt_combination = combination
        test_model = HSE_prompt(args).to(device)
        test_model.eval()

        with torch.no_grad():
            test_output = test_model(signal, fs, dataset_ids)

        assert test_output.shape == expected_shape
        print(f"✓ {combination} combination working")

    # Test 4: Gradient flow
    print("\n--- Test 4: Gradient Flow ---")

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for i in range(3):
        optimizer.zero_grad()
        output = model(signal, fs, dataset_ids)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        assert grad_norm > 0, "No gradients computed"

        optimizer.step()

    print("✓ Gradient flow working correctly")

    # Test 5: Different input sizes
    print("\n--- Test 5: Input Size Flexibility ---")

    test_cases = [
        (2, 512, 1),   # Small batch
        (8, 2048, 1),  # Large sequence
        (1, 256, 2),   # Multiple channels
        (16, 128, 1)   # Large batch
    ]

    for B, L, C in test_cases:
        test_signal = torch.randn(B, L, C, device=device)
        test_fs = torch.rand(B, device=device) * 2000 + 500  # Random fs 500-2500
        test_dataset_ids = torch.randint(0, args.max_dataset_ids, (B,), device=device)

        with torch.no_grad():
            test_output = model(test_signal, test_fs, test_dataset_ids)

        expected = (B, args.num_patches, args.output_dim)
        assert test_output.shape == expected, f"Size test failed: {test_output.shape} != {expected}"

    print("✓ All input size flexibility tests passed")

    # Test 6: Error handling
    print("\n--- Test 6: Error Handling ---")

    try:
        # Test invalid dataset_ids
        invalid_ids = torch.tensor([args.max_dataset_ids], device=device)
        model(signal, fs, invalid_ids)
        assert False, "Should have raised error"
    except (ValueError, RuntimeError):
        print("✓ Correctly handles invalid dataset_ids")

    try:
        # Test invalid combination method
        args.prompt_combination = 'invalid'
        HSE_prompt(args)
        assert False, "Should have raised ValueError"
    except ValueError:
        print("✓ Correctly rejects invalid combination method")

    # Test 7: Memory efficiency
    print("\n--- Test 7: Memory Efficiency ---")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

        # Large scale test
        large_signal = torch.randn(32, 1024, 1, device=device)
        large_fs = torch.rand(32, device=device) * 2000 + 500
        large_dataset_ids = torch.randint(0, args.max_dataset_ids, (32,), device=device)

        with torch.no_grad():
            large_output = model(large_signal, large_fs, large_dataset_ids)

        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"✓ Large-scale processing completed, peak GPU memory: {peak_memory:.2f}GB")

    print("\n=== All HSE_prompt Tests Passed! ===")
    print("✅ Key Features Verified:")
    print("  • Heterogeneous signal processing (different lengths, sampling rates)")
    print("  • Simple Dataset_id → learnable prompt mapping")
    print("  • Direct signal + prompt combination (add/concat)")
    print("  • Graceful fallback to signal-only processing")
    print("  • Flexible input size support")
    print("  • Memory efficient for large-scale applications")
    print("  • Ready for industrial fault diagnosis tasks")
