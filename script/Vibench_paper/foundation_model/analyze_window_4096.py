#!/usr/bin/env python3
"""
Window Sampling Analysis for Foundation Model Training
Analyzes feasibility of using window_size=4096 with 6 key datasets

Author: PHM-Vibench Team
Date: 2025-09-10
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def analyze_window_sampling():
    """Comprehensive analysis of window sampling with 4096 window size."""
    
    # Load metadata
    data_dir = Path("/home/lq/LQcode/2_project/PHMBench/PHM-Vibench/data")
    df = pd.read_excel(data_dir / "metadata_6_11.xlsx")
    
    # Configuration parameters
    window_size = 4096
    stride = 4  # From current config
    num_channels = 2  # input_dim from config
    num_tasks = 4  # Multi-task learning
    
    # Select 6 main datasets based on training importance
    target_datasets = [1, 2, 5, 6, 13, 19]
    
    print('='*80)
    print('📊 Window Sampling Analysis Report')
    print('Configuration: window_size=4096, stride=4, window_sampling_strategy')
    print('='*80)
    print()
    
    # Detailed dataset analysis
    print('## Dataset Analysis')
    print('-'*80)
    dataset_summary = []
    
    for dataset_id in sorted(target_datasets):
        subset = df[df['Dataset_id'] == dataset_id]
        num_samples = len(subset)
        
        # Get sample lengths
        sample_lengths = subset['Sample_lenth'].dropna().unique()
        
        if len(sample_lengths) > 0:
            min_length = np.min(sample_lengths)
            max_length = np.max(sample_lengths)
            mean_length = subset['Sample_lenth'].mean()
            
            # Check if window_size is feasible
            can_use_window = min_length >= window_size
            
            # Calculate sliding windows with stride
            if can_use_window:
                # For sliding window: num_windows = floor((L - W) / S) + 1
                # where L = sample_length, W = window_size, S = stride
                min_windows = max(1, int((min_length - window_size) / stride + 1))
                max_windows = int((max_length - window_size) / stride + 1)
                avg_windows = int((mean_length - window_size) / stride + 1)
            else:
                min_windows = 0
                max_windows = 0
                avg_windows = 0
            
            dataset_info = {
                'id': dataset_id,
                'samples': num_samples,
                'min_length': min_length,
                'max_length': max_length,
                'mean_length': mean_length,
                'unique_lengths': len(sample_lengths),
                'can_use_window': can_use_window,
                'min_windows': min_windows,
                'max_windows': max_windows,
                'avg_windows': avg_windows
            }
            dataset_summary.append(dataset_info)
            
            print(f'### Dataset {dataset_id}')
            print(f'  Total Samples: {num_samples:,}')
            print(f'  Sample Length Range: {min_length:,.0f} - {max_length:,.0f}')
            print(f'  Mean Sample Length: {mean_length:,.0f}')
            
            if can_use_window:
                print(f'  ✅ Window Feasible: Yes')
                print(f'  Windows per Sample: {min_windows:,} - {max_windows:,} (avg: {avg_windows:,})')
                print(f'  Total Windows Available: ~{num_samples * avg_windows:,}')
            else:
                print(f'  ❌ Window Feasible: No (sample length < window size)')
                print(f'  💡 Solution: Need to use padding or smaller window size')
            
            if len(sample_lengths) > 1:
                print(f'  ⚠️  Note: {len(sample_lengths)} different sample lengths')
            print()
    
    # Memory Analysis
    print('='*80)
    print('## GPU Memory Requirements')
    print('-'*80)
    
    # Different batch configurations
    configs = [
        {'name': 'Conservative', 'batch_size': 16, 'num_window': 8},
        {'name': 'Balanced', 'batch_size': 32, 'num_window': 16},
        {'name': 'Aggressive', 'batch_size': 64, 'num_window': 32},
        {'name': 'DLinear-Optimized', 'batch_size': 64, 'num_window': 4},
        {'name': 'TimesNet-Optimized', 'batch_size': 48, 'num_window': 32},
        {'name': 'FNO-Optimized', 'batch_size': 56, 'num_window': 64}
    ]
    
    print('| Configuration | Batch | Windows | Data (MB) | 4-Task (MB) | Total Est (MB) | GPU Requirement |')
    print('|--------------|-------|---------|-----------|-------------|----------------|-----------------|')
    
    for config in configs:
        batch_size = config['batch_size']
        num_window = config['num_window']
        
        # Calculate memory
        # Data shape: [batch_size, num_window, window_size, num_channels]
        data_memory_mb = (batch_size * num_window * window_size * num_channels * 4) / (1024**2)
        
        # Multi-task overhead (4 tasks)
        multitask_memory_mb = data_memory_mb * num_tasks
        
        # Total with model parameters, optimizer states, gradients (approx 3x)
        total_memory_mb = multitask_memory_mb * 3
        
        # Determine GPU requirement
        if total_memory_mb < 8000:
            gpu_req = 'V100-16GB ✅'
        elif total_memory_mb < 40000:
            gpu_req = 'A100-40GB'
        else:
            gpu_req = 'A100-80GB'
        
        print(f'| {config["name"]:12} | {batch_size:5} | {num_window:7} | {data_memory_mb:9.1f} | {multitask_memory_mb:11.1f} | {total_memory_mb:14.1f} | {gpu_req:15} |')
    
    # Time Estimation
    print()
    print('='*80)
    print('## Training Time Estimation')
    print('-'*80)
    
    # Based on empirical data from logs
    # Adjust these based on actual hardware
    throughput = {
        'V100': {'samples_per_sec': 50, 'gpu_memory': 16384},
        'A100': {'samples_per_sec': 75, 'gpu_memory': 40960},
        'A100-80': {'samples_per_sec': 75, 'gpu_memory': 81920}
    }
    
    epochs = 200  # Standard training epochs
    gpu_type = 'V100'  # Current setup
    
    print(f'Assumptions: {epochs} epochs, {gpu_type} GPU, {throughput[gpu_type]["samples_per_sec"]} samples/sec')
    print()
    
    print('| Dataset | Samples | Windows/Sample | Time/Epoch | Total Time | Days | Status |')
    print('|---------|---------|----------------|------------|------------|------|--------|')
    
    total_training_hours = 0
    for info in dataset_summary:
        dataset_id = info['id']
        num_samples = info['samples']
        avg_windows = info['avg_windows']
        can_use = info['can_use_window']
        
        if can_use:
            # Effective samples considering windows
            effective_samples = num_samples * min(avg_windows, 100)  # Cap at 100 windows per sample
            
            # Time calculation
            time_per_epoch_sec = effective_samples / throughput[gpu_type]['samples_per_sec']
            total_time_sec = time_per_epoch_sec * epochs
            total_time_hours = total_time_sec / 3600
            total_time_days = total_time_hours / 24
            
            total_training_hours += total_time_hours
            
            if total_time_hours < 24:
                status = '✅ OK'
            elif total_time_hours < 72:
                status = '⚠️  Long'
            else:
                status = '❌ Too Long'
            
            print(f'| {dataset_id:7} | {num_samples:7,} | {avg_windows:14,} | {time_per_epoch_sec/60:10.1f}m | {total_time_hours:10.1f}h | {total_time_days:4.1f} | {status:7} |')
        else:
            print(f'| {dataset_id:7} | {num_samples:7,} | {"N/A":>14} | {"N/A":>10} | {"N/A":>10} | {"N/A":>4} | ❌ Skip |')
    
    print()
    print(f'**Total Training Time (all 6 datasets): {total_training_hours:.1f} hours ({total_training_hours/24:.1f} days)**')
    
    # Recommendations
    print()
    print('='*80)
    print('## Recommendations')
    print('-'*80)
    
    print('### 1. Dataset-Specific Strategies')
    for info in dataset_summary:
        dataset_id = info['id']
        if not info['can_use_window']:
            print(f'   Dataset {dataset_id}: ❌ Cannot use window_size=4096')
            print(f'      - Sample length ({info["min_length"]:.0f}) < window size (4096)')
            print(f'      - Solution: Use padding or reduce window_size to {int(info["min_length"] * 0.8)}')
        elif info['avg_windows'] > 10000:
            print(f'   Dataset {dataset_id}: ⚠️  Very high window count ({info["avg_windows"]:,})')
            print(f'      - Solution: Use window sampling (e.g., random sample 1000 windows)')
    
    print()
    print('### 2. Optimal Configuration')
    print('   For window_size=4096 with multi-task learning:')
    print('   - Batch Size: 32-48 (memory permitting)')
    print('   - Num Windows: 16-32 (balance memory vs coverage)')
    print('   - GPU: V100-16GB minimum, A100-40GB recommended')
    print('   - Strategy: Use gradient accumulation for larger effective batch')
    
    print()
    print('### 3. Memory Optimization')
    print('   - Use mixed precision (FP16) to halve memory usage')
    print('   - Implement gradient checkpointing for large models')
    print('   - Use dataset-specific batch sizes based on window count')
    
    print()
    print('### 4. Time Optimization')
    print('   - Parallelize across multiple GPUs for large datasets')
    print('   - Use early stopping based on validation metrics')
    print('   - Consider reducing epochs for smaller datasets')
    
    # Save results to JSON
    results = {
        'config': {
            'window_size': window_size,
            'stride': stride,
            'num_channels': num_channels,
            'num_tasks': num_tasks,
            'target_datasets': target_datasets
        },
        'dataset_summary': dataset_summary,
        'memory_configs': configs,
        'total_training_hours': total_training_hours
    }
    
    output_file = Path('script/Vibench_paper/foundation_model/window_4096_analysis.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print()
    print(f'✅ Analysis results saved to: {output_file}')
    
    return results

if __name__ == '__main__':
    analyze_window_sampling()