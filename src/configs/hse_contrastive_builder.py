"""
HSE Contrastive Learning Configuration Builder
===========================================

Simple, intuitive interface for creating HSE contrastive learning configurations.
Provides clean abstractions while maintaining access to advanced features.

This builder addresses the complexity concerns by providing:
1. Simple functions for common configurations
2. Clear parameter names with sensible defaults
3. Hierarchical configuration building
4. Automatic validation and error checking

Usage Examples:
    # Basic configuration
    config = create_hse_contrastive_config()

    # Custom configuration
    config = create_hse_contrastive_config(
        contrast_weight=0.2,
        prompt_fusion='attention',
        backbone='B_04_Dlinear'
    )

    # Advanced ensemble configuration
    config = create_hse_ensemble_config(['INFONCE', 'SUPCON'])

Authors: PHM-Vibench Team
Date: 2025-01-06
"""

from typing import Dict, Any, List, Optional, Union
import copy


def create_hse_contrastive_config(
    contrast_weight: float = 0.15,
    loss_type: str = "INFONCE",
    prompt_fusion: str = "attention",
    prompt_weight: float = 0.1,
    temperature: float = 0.07,
    margin: float = 0.3,
    backbone: str = "B_08_PatchTST",
    use_system_sampling: bool = True,
    target_systems: List[int] = [1, 2, 6, 5, 12],
    training_stage: str = "pretrain"
) -> Dict[str, Any]:
    """
    Create a basic HSE contrastive learning configuration.

    Args:
        contrast_weight: Weight for contrastive loss (0.0 to 1.0)
        loss_type: Type of contrastive loss ("INFONCE", "SUPCON", "TRIPLET")
        prompt_fusion: Prompt fusion strategy ("add", "concat", "attention", "gate")
        prompt_weight: Weight for prompt integration
        temperature: Temperature parameter for InfoNCE/SupCon
        margin: Margin parameter for Triplet loss
        backbone: Backbone network to use
        use_system_sampling: Enable system-aware sampling
        target_systems: List of target system IDs
        training_stage: Training stage ("pretrain" or "finetune")

    Returns:
        Complete configuration dictionary
    """
    # Validate inputs
    _validate_basic_config(contrast_weight, loss_type, prompt_fusion, temperature, margin)

    config = {
        "model": {
            "name": "M_02_ISFM_Prompt",
            "type": "ISFM_Prompt",
            "embedding": "E_01_HSE_v2",
            "backbone": backbone,
            "task_head": "H_01_Linear_cla",
            "training_stage": training_stage,
            "freeze_prompt": training_stage == "finetune"
        },
        "task": {
            "name": "hse_contrastive",
            "type": "CDDG",
            "loss_fn": "CE",
            "contrast_weight": contrast_weight,
            "contrastive_strategy": {
                "type": "single",
                "loss_type": loss_type,
                "prompt_fusion": prompt_fusion,
                "prompt_weight": prompt_weight,
                "use_system_sampling": use_system_sampling,
                "enable_cross_system_contrast": use_system_sampling,
                "target_system_id": target_systems
            },
            "target_system_id": target_systems
        },
        "trainer": {
            "max_epochs": 100,
            "batch_size": 64,
            "learning_rate": 0.001,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "early_stopping": {
                "patience": 15,
                "monitor": "val_loss",
                "mode": "min"
            }
        },
        "data": {
            "normalization": True,
            "augmentation": True,
            "enable_system_metadata": True,
            "prompt_integration": True
        }
    }

    # Add loss-specific parameters
    if loss_type in ["INFONCE", "SUPCON"]:
        config["task"]["contrastive_strategy"]["temperature"] = temperature
    elif loss_type == "TRIPLET":
        config["task"]["contrastive_strategy"]["margin"] = margin

    return config


def create_hse_ensemble_config(
    loss_types: List[str],
    weights: Optional[List[float]] = None,
    prompt_fusion: str = "attention",
    backbone: str = "B_08_PatchTST",
    system_sampling_strategy: str = "balanced",
    target_systems: List[int] = [1, 2, 6, 5, 12]
) -> Dict[str, Any]:
    """
    Create an ensemble HSE contrastive learning configuration.

    Args:
        loss_types: List of contrastive loss types to combine
        weights: Optional weights for each loss type (auto-normalized if None)
        prompt_fusion: Default prompt fusion strategy
        backbone: Backbone network to use
        system_sampling_strategy: Strategy for system-aware sampling
        target_systems: List of target system IDs

    Returns:
        Complete ensemble configuration dictionary
    """
    # Validate inputs
    _validate_ensemble_config(loss_types, weights, prompt_fusion, system_sampling_strategy)

    # Auto-normalize weights if not provided
    if weights is None:
        weights = [1.0] * len(loss_types)
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # Create loss configurations
    loss_configs = []
    for i, loss_type in enumerate(loss_types):
        loss_config = {
            "name": loss_type,
            "weight": weights[i],
            "prompt_fusion": prompt_fusion
        }

        # Add loss-specific parameters
        if loss_type in ["INFONCE", "SUPCON"]:
            loss_config["temperature"] = 0.07 if loss_type == "INFONCE" else 0.05
        elif loss_type == "TRIPLET":
            loss_config["margin"] = 0.3

        loss_configs.append(loss_config)

    config = {
        "model": {
            "name": "M_02_ISFM_Prompt",
            "type": "ISFM_Prompt",
            "embedding": "E_01_HSE_v2",
            "backbone": backbone,
            "task_head": "H_01_Linear_cla",
            "training_stage": "pretrain",
            "freeze_prompt": False
        },
        "task": {
            "name": "hse_contrastive",
            "type": "CDDG",
            "loss_fn": "CE",
            "contrast_weight": 0.2,  # Higher weight for ensemble
            "contrastive_strategy": {
                "type": "ensemble",
                "auto_normalize_weights": True,
                "losses": loss_configs,
                "use_system_sampling": True,
                "enable_cross_system_contrast": True,
                "system_sampling_strategy": system_sampling_strategy,
                "adaptive_temperature": True
            },
            "target_system_id": target_systems
        },
        "trainer": {
            "max_epochs": 100,
            "batch_size": 64,
            "learning_rate": 0.001,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "early_stopping": {
                "patience": 15,
                "monitor": "val_loss",
                "mode": "min"
            }
        },
        "data": {
            "normalization": True,
            "augmentation": True,
            "enable_system_metadata": True,
            "prompt_integration": True
        }
    }

    return config


def create_hse_fewshot_config(
    num_support_shots: int = 5,
    num_query_shots: int = 10,
    backbone: str = "B_08_PatchTST",
    target_systems: List[int] = [1, 2, 6, 5, 12]
) -> Dict[str, Any]:
    """
    Create a few-shot HSE contrastive learning configuration.

    Args:
        num_support_shots: Number of support examples per class
        num_query_shots: Number of query examples per class
        backbone: Backbone network to use
        target_systems: List of target system IDs

    Returns:
        Complete few-shot configuration dictionary
    """
    config = {
        "model": {
            "name": "M_02_ISFM_Prompt",
            "type": "ISFM_Prompt",
            "embedding": "E_01_HSE_v2",
            "backbone": backbone,
            "task_head": "H_01_Linear_cla",
            "training_stage": "finetune",
            "freeze_prompt": True  # Freeze prompts for few-shot stability
        },
        "task": {
            "name": "hse_contrastive",
            "type": "GFS",  # Generalized Few-Shot
            "loss_fn": "CE",
            "contrast_weight": 0.1,  # Lower weight for few-shot stability
            "num_support_shots": num_support_shots,
            "num_query_shots": num_query_shots,
            "contrastive_strategy": {
                "type": "single",
                "loss_type": "SUPCON",  # SupCon works well for few-shot
                "temperature": 0.05,
                "prompt_fusion": "gate",  # Gated fusion for stability
                "prompt_weight": 0.05,
                "use_system_sampling": True,
                "enable_cross_system_contrast": False,  # Disable for few-shot stability
                "system_sampling_strategy": "balanced"
            },
            "target_system_id": target_systems
        },
        "trainer": {
            "max_epochs": 50,  # Fewer epochs for few-shot
            "batch_size": 16,  # Smaller batch for few-shot
            "learning_rate": 0.0005,  # Lower learning rate for finetuning
            "optimizer": "adamw",
            "scheduler": "step",
            "early_stopping": {
                "patience": 10,
                "monitor": "val_accuracy",
                "mode": "max"
            }
        },
        "data": {
            "normalization": True,
            "augmentation": False,  # Disable augmentation for few-shot stability
            "enable_system_metadata": True,
            "prompt_integration": True
        }
    }

    return config


def create_hse_research_config(
    research_focus: str = "cross_domain",
    backbone: str = "B_08_PatchTST",
    custom_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create research-oriented HSE contrastive configurations.

    Args:
        research_focus: Focus area ("cross_domain", "few_shot", "prompt_learning", "robustness")
        backbone: Backbone network to use
        custom_params: Optional custom parameters to override defaults

    Returns:
        Research-optimized configuration dictionary
    """
    # Base configuration
    base_config = create_hse_contrastive_config(backbone=backbone)

    # Research-specific modifications
    if research_focus == "cross_domain":
        # Optimize for cross-domain generalization
        base_config["task"]["contrast_weight"] = 0.25
        base_config["task"]["contrastive_strategy"]["system_sampling_strategy"] = "hard_negative"
        base_config["task"]["contrastive_strategy"]["enable_cross_system_contrast"] = True
        base_config["trainer"]["max_epochs"] = 150

    elif research_focus == "few_shot":
        # Use few-shot configuration
        base_config = create_hse_fewshot_config(backbone=backbone)

    elif research_focus == "prompt_learning":
        # Focus on prompt learning
        base_config["task"]["contrast_weight"] = 0.3
        base_config["task"]["contrastive_strategy"]["prompt_weight"] = 0.2
        base_config["task"]["contrastive_strategy"]["prompt_fusion"] = "attention"
        base_config["model"]["freeze_prompt"] = False
        base_config["trainer"]["learning_rate"] = 0.0005

    elif research_focus == "robustness":
        # Focus on robust learning
        base_config = create_hse_ensemble_config(
            ["INFONCE", "SUPCON", "TRIPLET"],
            backbone=backbone
        )
        base_config["task"]["contrast_weight"] = 0.2
        base_config["task"]["contrastive_strategy"]["system_sampling_strategy"] = "progressive_mixing"

    # Apply custom parameters
    if custom_params:
        base_config = _apply_custom_params(base_config, custom_params)

    return base_config


def _validate_basic_config(
    contrast_weight: float,
    loss_type: str,
    prompt_fusion: str,
    temperature: float,
    margin: float
) -> None:
    """Validate basic configuration parameters."""
    if not 0.0 <= contrast_weight <= 1.0:
        raise ValueError("contrast_weight must be between 0.0 and 1.0")

    valid_losses = ["INFONCE", "SUPCON", "TRIPLET"]
    if loss_type not in valid_losses:
        raise ValueError(f"loss_type must be one of {valid_losses}")

    valid_fusions = ["add", "concat", "attention", "gate"]
    if prompt_fusion not in valid_fusions:
        raise ValueError(f"prompt_fusion must be one of {valid_fusions}")

    if temperature <= 0:
        raise ValueError("temperature must be positive")

    if margin <= 0:
        raise ValueError("margin must be positive")


def _validate_ensemble_config(
    loss_types: List[str],
    weights: Optional[List[float]],
    prompt_fusion: str,
    system_sampling_strategy: str
) -> None:
    """Validate ensemble configuration parameters."""
    if not loss_types:
        raise ValueError("loss_types cannot be empty")

    valid_losses = ["INFONCE", "SUPCON", "TRIPLET", "PROTOTYPICAL", "BARLOWTWINS", "VICREG"]
    for loss_type in loss_types:
        if loss_type not in valid_losses:
            raise ValueError(f"loss_type '{loss_type}' must be one of {valid_losses}")

    if weights is not None and len(weights) != len(loss_types):
        raise ValueError("weights must have the same length as loss_types")

    valid_fusions = ["add", "concat", "attention", "gate"]
    if prompt_fusion not in valid_fusions:
        raise ValueError(f"prompt_fusion must be one of {valid_fusions}")

    valid_strategies = ["balanced", "hard_negative", "progressive_mixing"]
    if system_sampling_strategy not in valid_strategies:
        raise ValueError(f"system_sampling_strategy must be one of {valid_strategies}")


def _apply_custom_params(
    config: Dict[str, Any],
    custom_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Apply custom parameters to configuration using dot notation."""
    result = copy.deepcopy(config)

    def set_nested_param(d: Dict[str, Any], key: str, value: Any) -> None:
        keys = key.split('.')
        current = d
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value

    for key, value in custom_params.items():
        set_nested_param(result, key, value)

    return result


# Convenience functions for quick setup
def quick_basic_config() -> Dict[str, Any]:
    """Get a quick basic configuration with sensible defaults."""
    return create_hse_contrastive_config()


def quick_advanced_config() -> Dict[str, Any]:
    """Get a quick advanced configuration with ensemble losses."""
    return create_hse_ensemble_config(["INFONCE", "SUPCON"])


def quick_fewshot_config() -> Dict[str, Any]:
    """Get a quick few-shot configuration."""
    return create_hse_fewshot_config()


def quick_cross_domain_config() -> Dict[str, Any]:
    """Get a quick cross-domain generalization configuration."""
    return create_hse_research_config("cross_domain")


# Example usage (as comments for reference):
"""
# Basic usage
config = create_hse_contrastive_config()
config = create_hse_contrastive_config(contrast_weight=0.2, prompt_fusion='attention')

# Ensemble usage
config = create_hse_ensemble_config(['INFONCE', 'SUPCON'], weights=[0.6, 0.4])

# Research usage
config = create_hse_research_config('cross_domain')
config = create_hse_research_config('few_shot', custom_params={'trainer.learning_rate': 0.0001})

# Quick setup
config = quick_basic_config()
config = quick_advanced_config()
"""