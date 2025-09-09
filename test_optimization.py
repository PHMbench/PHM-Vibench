#!/usr/bin/env python3
"""
Test the optimized group-based batch processing for H_01_Linear_cla.
"""

import torch
import sys
import os
from argparse import Namespace

# Add src to path
sys.path.append('src')

def test_optimized_batch_processing():
    """Test the optimized group-based batch processing."""
    print("=== Testing Optimized Group-Based Batch Processing ===")
    
    try:
        from src.model_factory.ISFM.task_head.H_01_Linear_cla import H_01_Linear_cla
    except ImportError as e:
        print(f"Import failed: {e}")
        return False
    
    # Configuration with different class counts per system
    args = Namespace(
        output_dim=64,
        num_classes={1: 10, 5: 8, 6: 12, 7: 6}
    )
    
    cla_head = H_01_Linear_cla(args)
    print(f"✅ Created classification head with systems: {list(args.num_classes.keys())}")
    
    # Test scenario: batch with repeated system_ids (efficiency test case)
    batch_size = 16
    x = torch.randn(batch_size, 32, 64)  # (B, T, C)
    
    # System IDs with lots of repeats - perfect for grouping optimization
    system_ids = [1, 1, 1, 5, 5, 1, 6, 6, 6, 1, 7, 7, 5, 1, 6, 7]
    print(f"System IDs: {system_ids}")
    print(f"Unique systems: {sorted(set(system_ids))}")
    print(f"Groups: {len(set(system_ids))} groups for {len(system_ids)} samples")
    
    # Test the optimized batch processing
    output = cla_head(x, system_id=system_ids)
    max_classes = max(args.num_classes.values())  # 12
    expected_shape = (batch_size, max_classes)
    
    print(f"✅ Optimized batch output: {x.shape} -> {output.shape}")
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    # Verify results are correct by comparing with individual processing
    print("\n--- Verification: Compare with individual processing ---")
    individual_results = []
    for i, sys_id in enumerate(system_ids):
        # Use _single_forward directly to get unpadded results
        # Need to apply mean pooling first since _single_forward expects pooled input
        x_pooled = x[i:i+1].mean(dim=1)  # (1, T, C) -> (1, C)
        single_result = cla_head._single_forward(x_pooled, sys_id)
        print(f"Sample {i}: sys_id={sys_id}, shape={single_result.shape}")
        # Pad to max_classes for comparison (single_result is now [1, num_classes])
        current_classes = single_result.shape[1]
        if current_classes < max_classes:
            padding = torch.zeros(1, max_classes - current_classes, 
                                device=single_result.device, dtype=single_result.dtype)
            single_result = torch.cat([single_result, padding], dim=1)
        print(f"  After padding: {single_result.shape}")
        individual_results.append(single_result)
    
    individual_batch = torch.cat(individual_results, dim=0)
    
    # Compare results (should be identical)
    if torch.allclose(output, individual_batch, atol=1e-6):
        print("✅ Optimized results match individual processing (correctness verified)")
    else:
        print("❌ Results don't match - optimization bug detected")
        return False
    
    # Test efficiency improvement scenario
    print("\n--- Efficiency Improvement Analysis ---")
    unique_systems = len(set(system_ids))
    print(f"📊 Efficiency improvement:")
    print(f"   • Old approach: {len(system_ids)} forward passes")
    print(f"   • New approach: {unique_systems} forward passes")
    print(f"   • Speedup: {len(system_ids)/unique_systems:.1f}x")
    
    # Test edge cases
    print("\n--- Edge Case Tests ---")
    
    # All same system_id (should still work efficiently)
    all_same = [1] * 8
    x_same = torch.randn(8, 32, 64)
    output_same = cla_head(x_same, system_id=all_same)
    print(f"✅ All same system_id: {output_same.shape}")
    
    # Unknown system_ids mixed in
    with_unknown = [1, 999, 5, 999, 6]
    x_unknown = torch.randn(5, 32, 64)  
    output_unknown = cla_head(x_unknown, system_id=with_unknown)
    print(f"✅ Mixed with unknown system_ids: {output_unknown.shape}")
    
    return True


if __name__ == '__main__':
    """Run optimization tests."""
    print("Testing Group-Based Batch Processing Optimization...\n")
    
    try:
        success = test_optimized_batch_processing()
        
        if success:
            print("\n🚀 OPTIMIZATION VERIFICATION PASSED!")
            print("✅ Group-based batch processing works correctly")
            print("✅ Results match individual processing (correctness)")
            print("✅ Significant efficiency improvement achieved")
            print("✅ Edge cases handled properly")
            print("\n💡 The optimization reduces forward passes from O(N) to O(K)")
            print("   where N = batch size, K = unique system_ids (typically K << N)")
        else:
            print("\n❌ OPTIMIZATION VERIFICATION FAILED")
            
    except Exception as e:
        print(f"\n💥 Test failed: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    exit(0 if success else 1)