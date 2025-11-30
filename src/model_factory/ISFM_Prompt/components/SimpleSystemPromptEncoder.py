"""
SimpleSystemPromptEncoder: Lightweight Dataset_id to Learnable Prompt Mapping

This is a simplified version of SystemPromptEncoder that focuses only on mapping
Dataset_id to learnable prompt vectors for different industrial systems.

Key Features:
- Lightweight implementation with only Dataset_id → prompt mapping
- No complex multi-level encoding (Domain_id, Sample_rate removed)
- Simple embedding table approach
- Easy to understand and maintain
- Minimal memory footprint

Author: PHM-Vibench Team
Date: 2025-01-23
License: MIT
"""

import torch
import torch.nn as nn
from typing import Optional


class SimpleSystemPromptEncoder(nn.Module):
    """
    Lightweight system prompt encoder for Dataset_id to learnable prompt mapping.

    This simplified version only handles Dataset_id → prompt vector mapping,
    removing the complexity of multi-level metadata encoding.

    Architecture:
    Dataset_id → Embedding Table → Learnable Prompt Vector

    Use Cases:
    - Cross-system generalization in industrial fault diagnosis
    - System-specific feature adaptation
    - Lightweight prompt learning
    """

    def __init__(self,
                 prompt_dim: int = 64,
                 max_dataset_ids: int = 50):
        """
        Initialize SimpleSystemPromptEncoder.

        Args:
            prompt_dim: Dimensionality of prompt vectors
            max_dataset_ids: Maximum number of dataset IDs to support
        """
        super().__init__()

        self.prompt_dim = prompt_dim
        self.max_dataset_ids = max_dataset_ids

        # Simple embedding table: Dataset_id → prompt vector
        self.prompt_embedding = nn.Embedding(max_dataset_ids, prompt_dim)

        # Initialize parameters
        self._init_parameters()

        # Add standard PyTorch attribute for compatibility
        self.num_embeddings = self.prompt_embedding.num_embeddings

    def _init_parameters(self):
        """Initialize embedding parameters."""
        nn.init.normal_(self.prompt_embedding.weight, mean=0.0, std=0.1)

    def forward(self, dataset_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode dataset IDs to prompt vectors.

        Args:
            dataset_ids: Tensor of shape (B,) with dataset identifiers

        Returns:
            Prompt vectors of shape (B, prompt_dim)
        """
        # Validate input
        if dataset_ids.dim() != 1:
            raise ValueError(f"dataset_ids must be 1D tensor, got shape {dataset_ids.shape}")

        if dataset_ids.max() >= self.max_dataset_ids:
            raise ValueError(f"dataset_id {dataset_ids.max().item()} exceeds max_dataset_ids {self.max_dataset_ids}")

        if dataset_ids.min() < 0:
            raise ValueError(f"dataset_id {dataset_ids.min().item()} must be non-negative")

        # Get prompt vectors from embedding table
        prompt_vectors = self.prompt_embedding(dataset_ids)

        return prompt_vectors

    def get_prompt_info(self) -> dict:
        """
        Get information about the prompt encoder configuration.

        Returns:
            Dictionary with encoder configuration details
        """
        return {
            'encoder_type': 'SimpleSystemPromptEncoder',
            'prompt_dim': self.prompt_dim,
            'max_dataset_ids': self.max_dataset_ids,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }

    def get_prompt_vector(self, dataset_id: int) -> torch.Tensor:
        """
        Get prompt vector for a specific dataset ID.

        Args:
            dataset_id: Single dataset identifier

        Returns:
            Prompt vector of shape (prompt_dim,)
        """
        if dataset_id >= self.max_dataset_ids or dataset_id < 0:
            raise ValueError(f"dataset_id {dataset_id} out of range [0, {self.max_dataset_ids})")

        dataset_tensor = torch.tensor([dataset_id], dtype=torch.long)
        with torch.no_grad():
            prompt_vector = self.forward(dataset_tensor)

        return prompt_vector.squeeze(0)


def create_simple_prompt_encoder(prompt_dim: int = 64,
                                max_dataset_ids: int = 50) -> SimpleSystemPromptEncoder:
    """
    Factory function to create SimpleSystemPromptEncoder.

    Args:
        prompt_dim: Dimensionality of prompt vectors
        max_dataset_ids: Maximum number of dataset IDs to support

    Returns:
        Configured SimpleSystemPromptEncoder instance
    """
    return SimpleSystemPromptEncoder(
        prompt_dim=prompt_dim,
        max_dataset_ids=max_dataset_ids
    )


if __name__ == '__main__':
    """Simple self-test for SimpleSystemPromptEncoder."""

    print("=== SimpleSystemPromptEncoder Self-Test ===")

    # Test configuration
    prompt_dim = 64
    max_dataset_ids = 20
    batch_size = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize encoder
    encoder = SimpleSystemPromptEncoder(
        prompt_dim=prompt_dim,
        max_dataset_ids=max_dataset_ids
    ).to(device)

    print(f"✓ Initialized encoder on {device}")
    print(f"✓ Encoder info: {encoder.get_prompt_info()}")

    # Test 1: Basic functionality
    print("\n--- Test 1: Basic Functionality ---")

    dataset_ids = torch.tensor([1, 6, 13, 19], device=device)
    prompt_vectors = encoder(dataset_ids)
    expected_shape = (batch_size, prompt_dim)

    assert prompt_vectors.shape == expected_shape, f"Expected {expected_shape}, got {prompt_vectors.shape}"
    print(f"✓ Output shape correct: {prompt_vectors.shape}")

    # Test 2: Consistency check
    print("\n--- Test 2: Consistency Check ---")

    encoder.eval()
    prompt_vectors_2 = encoder(dataset_ids)
    prompt_vectors_3 = encoder(dataset_ids)

    torch.testing.assert_close(prompt_vectors_2, prompt_vectors_3, rtol=1e-5, atol=1e-6)
    print("✓ Model outputs are consistent in eval mode")

    # Test 3: Gradient flow
    print("\n--- Test 3: Gradient Flow ---")

    encoder.train()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)

    for i in range(3):
        optimizer.zero_grad()
        prompt_vectors = encoder(dataset_ids)
        loss = prompt_vectors.sum()  # Dummy loss
        loss.backward()

        # Check gradients exist
        grad_norm = sum(p.grad.norm().item() for p in encoder.parameters() if p.grad is not None)
        assert grad_norm > 0, "No gradients computed"

        optimizer.step()

    print("✓ Gradients flow correctly through the model")

    # Test 4: Different batch sizes
    print("\n--- Test 4: Batch Size Flexibility ---")

    for test_batch_size in [1, 8, 16]:
        test_dataset_ids = torch.randint(0, max_dataset_ids, (test_batch_size,), device=device)
        test_prompt_vectors = encoder(test_dataset_ids)

        expected_shape = (test_batch_size, prompt_dim)
        assert test_prompt_vectors.shape == expected_shape, f"Expected {expected_shape}, got {test_prompt_vectors.shape}"

    print("✓ Handles various batch sizes correctly")

    # Test 5: Error handling
    print("\n--- Test 5: Error Handling ---")

    try:
        # Test invalid dataset_id (too large)
        invalid_ids = torch.tensor([max_dataset_ids], device=device)
        encoder(invalid_ids)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print("✓ Correctly rejects dataset_id >= max_dataset_ids")

    try:
        # Test invalid dataset_id (negative)
        invalid_ids = torch.tensor([-1], device=device)
        encoder(invalid_ids)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print("✓ Correctly rejects negative dataset_id")

    try:
        # Test wrong input dimension
        invalid_ids = torch.tensor([[1, 2]], device=device)
        encoder(invalid_ids)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print("✓ Correctly rejects wrong input dimensions")

    # Test 6: Single prompt retrieval
    print("\n--- Test 6: Single Prompt Retrieval ---")

    dataset_id = 5
    single_prompt = encoder.get_prompt_vector(dataset_id)
    assert single_prompt.shape == (prompt_dim,), f"Expected ({prompt_dim},), got {single_prompt.shape}"
    assert single_prompt.device == device, f"Prompt vector on wrong device: {single_prompt.device}"
    print(f"✓ Single prompt retrieval works: dataset_id={dataset_id} → {single_prompt.shape}")

    # Test 7: Memory efficiency
    print("\n--- Test 7: Memory Efficiency ---")

    # Large scale test
    large_batch_size = 1024
    large_dataset_ids = torch.randint(0, max_dataset_ids, (large_batch_size,), device=device)

    with torch.no_grad():
        large_prompt_vectors = encoder(large_dataset_ids)

    expected_shape = (large_batch_size, prompt_dim)
    assert large_prompt_vectors.shape == expected_shape
    print(f"✓ Large batch processing works: {large_batch_size} samples → {large_prompt_vectors.shape}")

    # Parameter count check
    total_params = sum(p.numel() for p in encoder.parameters())
    expected_params = max_dataset_ids * prompt_dim  # Embedding table parameters
    assert total_params == expected_params, f"Expected {expected_params} parameters, got {total_params}"
    print(f"✓ Parameter count correct: {total_params:,} (embedding table: {max_dataset_ids} × {prompt_dim})")

    print("\n=== All SimpleSystemPromptEncoder Tests Passed! ===")
    print("✅ Key Features Verified:")
    print("  • Lightweight Dataset_id → prompt vector mapping")
    print("  • Simple embedding table implementation")
    print("  • Flexible batch size support")
    print("  • Robust error handling and validation")
    print("  • Memory efficient for large-scale applications")
    print("  • Ready for HSE_prompt integration")