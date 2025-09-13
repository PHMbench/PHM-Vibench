#!/usr/bin/env python3
"""Debug script to test multi-task forward pass."""

import torch
import sys
import os
sys.path.append('/home/lq/LQcode/2_project/PHMBench/PHM-Vibench')

from argparse import Namespace
from src.model_factory.ISFM.M_01_ISFM import Model

# Create a simple test to debug the multi-task issue
def test_multitask_forward():
    print("=== Testing Multi-task Forward Pass ===")
    
    # Create mock metadata
    class MockMetadata:
        def __init__(self):
            self.num_classes = {'1': 5, '2': 3}  # Mock system IDs and classes
        
        def __getitem__(self, file_id):
            return {
                'Dataset_id': '1',
                'Sample_rate': 1000.0,
                'RUL_label': 50.0
            }
    
    metadata = MockMetadata()
    
    # Create model args - use simpler embedding
    args_m = Namespace(
        embedding='E_03_Patch_DPOT',
        backbone='B_04_Dlinear',
        task_head='MultiTaskHead',
        # Backbone args (B_04_Dlinear requirements)
        seq_len=1024,
        output_dim=256,
        individual=False,
        # MultiTaskHead specific args
        hidden_dim=128,
        dropout=0.1,
        num_classes={'1': 5, '2': 3},
        rul_max_value=1000.0,
        use_batch_norm=True,
        activation='relu'
    )
    
    # Create model
    print("Creating model...")
    model = Model(args_m, metadata)
    model.eval()
    
    # Create test input
    batch_size = 2
    seq_len = 1024
    channels = 1
    x = torch.randn(batch_size, seq_len, channels)
    
    print(f"Input shape: {x.shape}")
    
    # Test different task_id values
    test_cases = [
        ('classification', 1),
        ('multi_task', 1),
        (None, 1)
    ]
    
    for task_id, file_id in test_cases:
        print(f"\n--- Testing task_id='{task_id}' ---")
        try:
            with torch.no_grad():
                output = model.forward(x, file_id=file_id, task_id=task_id)
            
            print(f"Output type: {type(output)}")
            if output is None:
                print("❌ Output is None!")
            elif isinstance(output, dict):
                print(f"✅ Dictionary output with keys: {list(output.keys())}")
                for key, value in output.items():
                    if value is not None:
                        print(f"  {key}: shape {value.shape}")
                    else:
                        print(f"  {key}: None")
            else:
                print(f"✅ Tensor output with shape: {output.shape}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    test_multitask_forward()