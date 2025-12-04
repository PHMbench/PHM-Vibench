"""
⚠️ DEPRECATED: pipeline_config.py

This module is deprecated and will be removed in a future version.
The weight loading functionality has been moved to maintain better organization.

NEW: from src.utils.utils import load_pretrained_weights
NEW: Use result summarization in training orchestration modules

Migration Guide:
- Replace 'from src.utils import pipeline_config' with appropriate imports
- Use load_pretrained_weights from src.utils.utils
- Pipeline summary functionality now integrated in orchestrator

Last updated: 2025-11-20
Removal timeline: v2.1.0
"""

import warnings
from typing import Dict


def load_pretrained_weights(model, checkpoint_path: str, strict: bool = False) -> bool:
    """
    ⚠️ DEPRECATED: Use src.utils.utils.load_pretrained_weights instead

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
    # Issue deprecation warning
    warnings.warn(
        "pipeline_config.load_pretrained_weights is deprecated. "
        "Please use src.utils.utils.load_pretrained_weights instead. "
        "This function will be removed in v2.1.0.",
        DeprecationWarning,
        stacklevel=2
    )

    # Delegate to the new implementation
    from .utils import load_pretrained_weights as new_load_pretrained_weights
    return new_load_pretrained_weights(model, checkpoint_path, strict)


def generate_pipeline_summary(checkpoint_paths: Dict[str, str], finetuning_results: Dict) -> Dict:
    """
    ⚠️ DEPRECATED: This functionality is now integrated in training orchestration modules

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
    # Issue deprecation warning
    warnings.warn(
        "pipeline_config.generate_pipeline_summary is deprecated. "
        "This functionality is now integrated in training orchestration modules. "
        "This function will be removed in v2.1.0.",
        DeprecationWarning,
        stacklevel=2
    )
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
