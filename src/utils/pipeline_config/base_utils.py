"""
Essential utilities for the two-stage multi-task pipeline.

This module provides core utility functions for weight loading and result summarization.
Configuration management is now handled directly through YAML files.

Author: PHM-Vibench Team
Date: 2025-08-18
"""

from typing import Dict
import os

import torch


def load_pretrained_weights(model, checkpoint_path: str, strict: bool = False) -> None:
    """Load pretrained backbone weights into ``model``.

    ``strict=False`` is the intentional stage-transfer mode. A configured
    checkpoint must exist, contain a Lightning ``state_dict``, and provide at
    least one compatible ``network.`` backbone parameter. Failures propagate to
    the caller; successful loading returns normally.
    """
    checkpoint_file = os.fspath(checkpoint_path or "")
    if not checkpoint_file or not os.path.isfile(checkpoint_file):
        raise FileNotFoundError(
            "Configured pretrained checkpoint does not exist or is not a file: "
            f"{checkpoint_file or '<empty>'}"
        )

    try:
        checkpoint = torch.load(
            checkpoint_file,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(checkpoint, dict) or not isinstance(
            checkpoint.get("state_dict"), dict
        ):
            raise TypeError(
                "Expected a Lightning checkpoint containing a mapping at "
                "'state_dict'."
            )

        state_dict = checkpoint["state_dict"]
        backbone_weights = {
            key.replace("network.", "", 1): value
            for key, value in state_dict.items()
            if key.startswith("network.")
            and not key.startswith("network.task_head")
        }
        if not backbone_weights:
            raise RuntimeError(
                "Checkpoint contains no transferable 'network.' backbone "
                "parameters."
            )

        incompatible = model.load_state_dict(backbone_weights, strict=strict)
        if not strict:
            missing_keys = list(incompatible.missing_keys)
            unexpected_keys = list(incompatible.unexpected_keys)
            loaded_count = len(backbone_weights) - len(unexpected_keys)
            if loaded_count <= 0:
                raise RuntimeError(
                    "Checkpoint matched zero model parameters after backbone "
                    "filtering."
                )
            if missing_keys or unexpected_keys:
                print(
                    f"Loaded pretrained backbone from '{checkpoint_file}' with "
                    f"{len(missing_keys)} missing and "
                    f"{len(unexpected_keys)} unexpected keys."
                )
    except Exception as exc:
        if isinstance(exc, FileNotFoundError):
            raise
        raise RuntimeError(
            f"Failed to load pretrained checkpoint '{checkpoint_file}': {exc}"
        ) from exc


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
