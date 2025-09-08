#!/usr/bin/env python3
"""
Test script to calculate model parameter counts for different backbone configurations.
Tests the four multi-task models to identify memory usage.
"""

import torch
import torch.nn as nn
from argparse import Namespace
import pandas as pd
import numpy as np
import sys
import os

# Add project path
sys.path.append('/home/lq/LQcode/2_project/PHMBench/PHM-Vibench')

# Import model components
from src.model_factory.ISFM.M_01_ISFM import Model

class MockMetadata:
    """Mock metadata for testing - simulates subset of data"""
    def __init__(self, target_system_ids):
        # Only include target systems, not all 19 systems
        self.df = pd.DataFrame({
            'Dataset_id': target_system_ids * 10,  # Repeat for multiple samples
            'Label': list(range(10)) * len(target_system_ids),
            'Sample_rate': [12000] * (10 * len(target_system_ids))
        })
    
    def __getitem__(self, file_id):
        return {
            'Dataset_id': np.int64(self.df['Dataset_id'].iloc[0]),
            'Sample_rate': 12000
        }

def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def estimate_memory_usage(model, batch_size=32, seq_len=4096, channels=2):
    """Estimate GPU memory usage for model"""
    # Model parameters (4 bytes per float32)
    param_memory = sum(p.numel() * 4 for p in model.parameters()) / (1024**3)  # GB
    
    # Estimate activations memory (rough estimate: 2x parameters for activations)
    activation_memory = param_memory * 2
    
    # Input tensor memory
    input_memory = (batch_size * seq_len * channels * 4) / (1024**3)  # GB
    
    # Gradient memory (same as parameters)
    gradient_memory = param_memory
    
    total_memory = param_memory + activation_memory + input_memory + gradient_memory
    
    return {
        'param_memory_gb': param_memory,
        'activation_memory_gb': activation_memory,
        'input_memory_gb': input_memory,
        'gradient_memory_gb': gradient_memory,
        'total_memory_gb': total_memory
    }

def test_model_config(config_name, args):
    """Test a specific model configuration"""
    print(f"\n{'='*60}")
    print(f"Testing: {config_name}")
    print('='*60)
    
    try:
        # Create mock metadata with only target systems
        target_systems = [1, 2, 5, 6, 13, 19]  # From YAML configs
        metadata = MockMetadata(target_systems)
        
        # Create model
        model = Model(args, metadata)
        
        # Count parameters
        total_params, trainable_params = count_parameters(model)
        
        # Estimate memory
        memory_info = estimate_memory_usage(
            model, 
            batch_size=args.batch_size if hasattr(args, 'batch_size') else 32,
            seq_len=4096,
            channels=2
        )
        
        # Print results
        print(f"Model Architecture:")
        print(f"  Embedding: {args.embedding}")
        print(f"  Backbone: {args.backbone}")
        print(f"  Task Head: {args.task_head}")
        print(f"\nModel Dimensions:")
        print(f"  output_dim: {args.output_dim}")
        print(f"  hidden_dim: {getattr(args, 'hidden_dim', 512)}")
        print(f"  num_layers: {getattr(args, 'num_layers', 2)}")
        print(f"  num_heads: {getattr(args, 'num_heads', 4)}")
        print(f"  batch_size: {getattr(args, 'batch_size', 32)}")
        
        print(f"\nParameter Count:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Size in MB: {total_params * 4 / (1024**2):.2f} MB")
        
        print(f"\nMemory Estimation (per batch):")
        print(f"  Model parameters: {memory_info['param_memory_gb']:.3f} GB")
        print(f"  Activations (est): {memory_info['activation_memory_gb']:.3f} GB")
        print(f"  Input tensors: {memory_info['input_memory_gb']:.3f} GB")
        print(f"  Gradients: {memory_info['gradient_memory_gb']:.3f} GB")
        print(f"  Total estimated: {memory_info['total_memory_gb']:.3f} GB")
        
        # Check if it fits in 24GB
        if memory_info['total_memory_gb'] < 24:
            print(f"  ✅ Fits in 24GB GPU")
        else:
            print(f"  ❌ Exceeds 24GB GPU (need {memory_info['total_memory_gb']:.1f}GB)")
            
        # Module breakdown
        print(f"\nModule Parameter Breakdown:")
        for name, module in model.named_children():
            module_params = sum(p.numel() for p in module.parameters())
            print(f"  {name}: {module_params:,} params ({module_params*4/(1024**2):.2f} MB)")
            
            # For task_heads, show detail
            if name == 'task_heads' and hasattr(module, 'items'):
                for task_name, task_module in module.items():
                    task_params = sum(p.numel() for p in task_module.parameters())
                    print(f"    - {task_name}: {task_params:,} params")
        
        return total_params, memory_info['total_memory_gb']
        
    except Exception as e:
        print(f"  ❌ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    print("PHM-Vibench Multi-Task Model Parameter Analysis")
    print("Target: Fit models in 24GB GPU memory")
    
    # Test configurations from YAML files
    configs = {
        'B_04_Dlinear': Namespace(
            embedding='E_01_HSE',
            backbone='B_04_Dlinear',
            task_head='MultiTaskHead',
            output_dim=256,  # Reduced from 1024
            d_model=128,
            num_layers=2,    # Reduced from 3
            num_heads=4,     # Reduced from 8
            d_ff=512,        # Reduced from 2048
            dropout=0.1,
            num_patches=64,  # Reduced from 128
            patch_size_L=128,  # Reduced from 256
            patch_size_C=1,
            hidden_dim=64,  # MEMORY FIX: Reduced from 512
            activation='gelu',
            rul_max_value=2000.0,
            use_batch_norm=True,
            batch_size=16,   # MEMORY FIX: Reduced from 128
            # For multi-task
            enabled_tasks=['classification', 'anomaly_detection', 'signal_prediction', 'rul_prediction'],
            classification_head='H_02_distance_cla',
            prediction_head='H_03_Linear_pred',
            max_len=4096,   # Corrected to match YAML configs
            max_out=2,     # MEMORY FIX: Reduced from 3
            act='gelu'
        ),
        
        'B_06_TimesNet': Namespace(
            embedding='E_01_HSE',
            backbone='B_06_TimesNet',
            task_head='MultiTaskHead',
            output_dim=256,
            d_model=128,
            num_layers=2,
            num_heads=4,
            d_ff=512,
            dropout=0.1,
            num_patches=64,
            patch_size_L=128,
            patch_size_C=1,
            hidden_dim=64,  # MEMORY FIX: Reduced from 512
            activation='gelu',
            rul_max_value=2000.0,
            use_batch_norm=True,
            batch_size=32,  # Already reduced in YAML
            # TimesNet specific
            e_layers=2,
            factor=5,
            # Multi-task
            enabled_tasks=['classification', 'anomaly_detection', 'signal_prediction', 'rul_prediction'],
            classification_head='H_02_distance_cla',
            prediction_head='H_03_Linear_pred',
            max_len=4096,   # Corrected to match YAML configs
            max_out=2,     # MEMORY FIX: Reduced from 3
            act='gelu'
        ),
        
        'B_08_PatchTST': Namespace(
            embedding='E_01_HSE',
            backbone='B_08_PatchTST',
            task_head='MultiTaskHead',
            output_dim=256,
            d_model=128,
            num_layers=2,
            num_heads=4,
            d_ff=512,
            dropout=0.1,
            num_patches=64,
            patch_size_L=128,
            patch_size_C=1,
            hidden_dim=64,  # MEMORY FIX: Reduced from 512
            activation='gelu',
            rul_max_value=2000.0,
            use_batch_norm=True,
            batch_size=32,  # Already reduced
            # PatchTST specific
            e_layers=2,
            factor=1,
            patch_len=16,
            stride=8,
            # Multi-task
            enabled_tasks=['classification', 'anomaly_detection', 'signal_prediction', 'rul_prediction'],
            classification_head='H_02_distance_cla',
            prediction_head='H_03_Linear_pred',
            max_len=4096,   # Corrected to match YAML configs
            max_out=2,     # MEMORY FIX: Reduced from 3
            act='gelu'
        ),
        
        'B_09_FNO': Namespace(
            embedding='E_01_HSE',
            backbone='B_09_FNO',
            task_head='MultiTaskHead',
            output_dim=256,
            d_model=128,
            num_layers=2,
            num_heads=4,
            d_ff=512,
            dropout=0.1,
            num_patches=64,
            patch_size_L=128,
            patch_size_C=1,
            hidden_dim=64,  # MEMORY FIX: Reduced from 512
            activation='gelu',
            rul_max_value=2000.0,
            use_batch_norm=True,
            batch_size=16,  # Most aggressive reduction
            # FNO specific
            modes=32,
            width=128,
            # Multi-task
            enabled_tasks=['classification', 'anomaly_detection', 'signal_prediction', 'rul_prediction'],
            classification_head='H_02_distance_cla',
            prediction_head='H_03_Linear_pred',
            max_len=4096,   # Corrected to match YAML configs
            max_out=2,     # MEMORY FIX: Reduced from 3
            act='gelu'
        ),
    }
    
    results = {}
    for name, config in configs.items():
        params, memory = test_model_config(name, config)
        results[name] = (params, memory)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    
    print("\nModel Comparison:")
    print(f"{'Model':<15} {'Parameters':<15} {'Memory (GB)':<15} {'Status':<10}")
    print('-'*55)
    for name, (params, memory) in results.items():
        if params and memory:
            status = "✅ OK" if memory < 24 else "❌ OOM"
            print(f"{name:<15} {params:>12,}  {memory:>10.2f} GB   {status}")
        else:
            print(f"{name:<15} {'ERROR':<15} {'N/A':<15} ❌")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS FOR 24GB GPU")
    print("="*60)
    
    print("\n1. Batch Size Adjustments:")
    print("   - B_04_Dlinear: Reduce from 128 to 32 or 16")
    print("   - B_06_TimesNet: Keep at 32 (already optimized)")
    print("   - B_08_PatchTST: Keep at 32 (already optimized)")
    print("   - B_09_FNO: Keep at 16 (already optimized)")
    
    print("\n2. Model Dimension Reductions (if still OOM):")
    print("   - output_dim: 256 -> 128")
    print("   - hidden_dim: 512 -> 256")
    print("   - d_ff: 512 -> 256")
    print("   - num_layers: 2 -> 1")
    
    print("\n3. Additional Optimizations:")
    print("   - Enable gradient checkpointing")
    print("   - Use mixed precision (fp16)")
    print("   - Reduce num_window in data config")
    print("   - Use gradient accumulation with smaller batches")

if __name__ == '__main__':
    main()