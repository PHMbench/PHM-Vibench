"""
Essential utilities for the two-stage multi-task pipeline.

This module provides core utility functions for weight loading and result summarization.
Configuration management is now handled directly through YAML files.

Author: PHM-Vibench Team
Date: 2025-08-18
"""

from typing import Dict
import torch
import os


def load_pretrained_weights(model, checkpoint_path: str, strict: bool = False) -> bool:
    """
    Load pretrained weights into a model.
    
    Parameters
    ----------
    model : nn.Module
        Model to load weights into
    checkpoint_path : str
        Path to the checkpoint file
    strict : bool, optional
        Whether to strictly enforce that the keys in state_dict match
        
    Returns
    -------
    bool
        True if weights were loaded successfully, False otherwise
    """
    import torch
    import os
    
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint path does not exist: {checkpoint_path}")
        return False
    
    try:
        print(f"Loading pretrained weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract backbone weights from checkpoint
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Filter backbone weights (exclude task head weights)
            backbone_weights = {
                k.replace('network.', ''): v for k, v in state_dict.items() 
                if k.startswith('network.') and not k.startswith('network.task_head')
            }
            
            # Load backbone weights with strict=False to allow missing task head weights
            missing_keys, unexpected_keys = model.load_state_dict(backbone_weights, strict=strict)
            
            if not strict and (missing_keys or unexpected_keys):
                print(f"Loaded pretrained weights with {len(missing_keys)} missing keys and {len(unexpected_keys)} unexpected keys")
                if missing_keys:
                    print(f"Missing keys: {missing_keys[:5]}...")  # Show first 5
                if unexpected_keys:
                    print(f"Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5
            
            print("✓ Pretrained backbone weights loaded successfully")
            return True
        else:
            print("Warning: No 'state_dict' found in checkpoint")
            return False
            
    except Exception as e:
        print(f"Error loading pretrained weights: {e}")
        return False


def generate_pipeline_summary(checkpoint_paths: Dict[str, str], finetuning_results: Dict) -> Dict:
    """
    Generate a summary of pipeline results.
    
    Parameters
    ----------
    checkpoint_paths : Dict[str, str]
        Dictionary mapping backbone names to checkpoint paths
    finetuning_results : Dict
        Dictionary containing fine-tuning results
        
    Returns
    -------
    Dict
        Summary dictionary with statistics and text summary
    """
    summary = {
        'successful_pretraining': sum(1 for path in checkpoint_paths.values() if path is not None),
        'total_backbones': len(checkpoint_paths),
        'successful_finetuning': 0,
        'total_finetuning_experiments': 0,
        'best_backbone': None,
        'text': ""
    }
    
    # Count successful fine-tuning experiments
    for system_results in finetuning_results.values():
        for backbone_results in system_results.values():
            if backbone_results is not None:
                summary['total_finetuning_experiments'] += 1
                if isinstance(backbone_results, dict):
                    # Multi-task or single-task with multiple metrics
                    summary['successful_finetuning'] += 1
                elif backbone_results:  # Single result
                    summary['successful_finetuning'] += 1
    
    # Determine best backbone (simplified - first successful one)
    successful_backbones = [k for k, v in checkpoint_paths.items() if v is not None]
    if successful_backbones:
        summary['best_backbone'] = successful_backbones[0]
    
    # Generate text summary
    text_lines = [
        f"Pretraining: {summary['successful_pretraining']}/{summary['total_backbones']} backbones successful",
        f"Fine-tuning: {summary['successful_finetuning']}/{summary['total_finetuning_experiments']} experiments successful",
        "",
        "Backbone Performance Summary:",
    ]
    
    for backbone, checkpoint_path in checkpoint_paths.items():
        status = "✓" if checkpoint_path else "✗"
        text_lines.append(f"  {status} {backbone}: {'Success' if checkpoint_path else 'Failed'}")
    
    summary['text'] = "\n".join(text_lines)
    return summary