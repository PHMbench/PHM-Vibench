"""
Configuration schema and validation for PHM-Vibench research framework.

This module provides comprehensive configuration management with validation,
type checking, and scientific reproducibility features.
"""

from __future__ import annotations

import os
import platform
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import yaml
import torch
import numpy as np
from pydantic import BaseModel, Field, validator, root_validator


class ReproducibilityConfig(BaseModel):
    """Configuration for ensuring experimental reproducibility."""

    # Random seed configuration
    global_seed: int = Field(42, description="Global random seed for all operations")
    torch_deterministic: bool = Field(True, description="Enable PyTorch deterministic operations")
    torch_benchmark: bool = Field(False, description="Enable PyTorch CUDNN benchmark")
    numpy_seed: Optional[int] = Field(None, description="NumPy random seed (defaults to global_seed)")

    # Environment tracking
    track_environment: bool = Field(True, description="Track and log environment details")
    track_git_commit: bool = Field(True, description="Track git commit hash")
    track_dependencies: bool = Field(True, description="Track package versions")

    @validator('numpy_seed', always=True)
    def set_numpy_seed(cls, v, values):
        return v if v is not None else values.get('global_seed', 42)


class DataConfig(BaseModel):
    """Configuration for data loading and preprocessing."""

    # Data paths (relative to project root)
    data_root: Path = Field(..., description="Root directory for all datasets")
    metadata_file: str = Field(..., description="Metadata file name")
    cache_dir: Optional[Path] = Field(None, description="Cache directory for preprocessed data")

    # Data loading parameters
    batch_size: int = Field(64, ge=1, description="Batch size for training")
    num_workers: int = Field(4, ge=0, description="Number of data loading workers")
    pin_memory: bool = Field(True, description="Pin memory for faster GPU transfer")

    # Data preprocessing
    normalization: str = Field("standardization", regex="^(standardization|minmax|none)$")
    window_size: int = Field(4096, ge=1, description="Signal window size")
    stride: int = Field(1, ge=1, description="Sliding window stride")

    # Data splits
    train_ratio: float = Field(0.8, ge=0.1, le=0.9, description="Training data ratio")
    val_ratio: float = Field(0.1, ge=0.05, le=0.5, description="Validation data ratio")

    @validator('cache_dir', always=True)
    def set_cache_dir(cls, v, values):
        if v is None:
            return values['data_root'] / 'cache'
        return v

    @root_validator
    def validate_data_splits(cls, values):
        train_ratio = values.get('train_ratio', 0.8)
        val_ratio = values.get('val_ratio', 0.1)
        if train_ratio + val_ratio >= 1.0:
            raise ValueError("train_ratio + val_ratio must be < 1.0")
        return values


class ModelConfig(BaseModel):
    """Configuration for model architecture and parameters."""

    # Model identification
    name: str = Field(..., description="Model name (must match implementation)")
    type: str = Field(..., description="Model type/category")

    # Model architecture parameters
    input_dim: int = Field(..., ge=1, description="Input feature dimension")
    hidden_dim: Optional[int] = Field(None, ge=1, description="Hidden layer dimension")
    num_layers: Optional[int] = Field(None, ge=1, description="Number of layers")
    dropout: float = Field(0.1, ge=0.0, le=0.9, description="Dropout probability")

    # Task-specific parameters
    num_classes: Optional[int] = Field(None, ge=2, description="Number of output classes")
    output_dim: Optional[int] = Field(None, ge=1, description="Output dimension for regression")

    # Model initialization
    weight_init: str = Field("xavier_uniform", description="Weight initialization method")
    bias_init: str = Field("zeros", description="Bias initialization method")

    # Pre-trained weights
    pretrained_path: Optional[Path] = Field(None, description="Path to pre-trained weights")
    freeze_backbone: bool = Field(False, description="Freeze backbone parameters")


class OptimizationConfig(BaseModel):
    """Configuration for optimization and training parameters."""

    # Optimizer configuration
    optimizer: str = Field("adam", regex="^(adam|adamw|sgd|rmsprop)$")
    learning_rate: float = Field(1e-3, gt=0, description="Initial learning rate")
    weight_decay: float = Field(1e-4, ge=0, description="L2 regularization coefficient")
    momentum: float = Field(0.9, ge=0, le=1, description="SGD momentum (if applicable)")

    # Learning rate scheduling
    scheduler: Optional[str] = Field(None, regex="^(cosine|step|plateau|exponential)$")
    scheduler_params: Dict[str, Any] = Field(default_factory=dict, description="Scheduler parameters")

    # Training parameters
    max_epochs: int = Field(100, ge=1, description="Maximum training epochs")
    early_stopping: bool = Field(True, description="Enable early stopping")
    patience: int = Field(10, ge=1, description="Early stopping patience")
    min_delta: float = Field(1e-4, ge=0, description="Minimum improvement for early stopping")

    # Gradient clipping
    gradient_clip_val: Optional[float] = Field(None, gt=0, description="Gradient clipping value")
    gradient_clip_algorithm: str = Field("norm", regex="^(norm|value)$")


class TaskConfig(BaseModel):
    """Configuration for task-specific parameters."""

    # Task identification
    name: str = Field(..., description="Task name")
    type: str = Field(..., description="Task type (DG, FS, etc.)")

    # Loss function configuration
    loss_function: str = Field("cross_entropy", description="Loss function name")
    loss_params: Dict[str, Any] = Field(default_factory=dict, description="Loss function parameters")

    # Metrics configuration
    metrics: List[str] = Field(["accuracy"], description="Evaluation metrics")

    # Task-specific parameters
    domain_config: Optional[Dict[str, Any]] = Field(None, description="Domain adaptation configuration")
    few_shot_config: Optional[Dict[str, Any]] = Field(None, description="Few-shot learning configuration")


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    # Experiment metadata
    name: str = Field(..., description="Experiment name")
    description: str = Field("", description="Experiment description")
    tags: List[str] = Field(default_factory=list, description="Experiment tags")

    # Configuration sections
    reproducibility: ReproducibilityConfig = Field(default_factory=ReproducibilityConfig)
    data: DataConfig = Field(..., description="Data configuration")
    model: ModelConfig = Field(..., description="Model configuration")
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    task: TaskConfig = Field(..., description="Task configuration")

    # Execution parameters
    num_runs: int = Field(1, ge=1, description="Number of experimental runs")
    output_dir: Path = Field(Path("results"), description="Output directory")

    # Logging configuration
    log_level: str = Field("INFO", regex="^(DEBUG|INFO|WARNING|ERROR)$")
    log_interval: int = Field(10, ge=1, description="Logging interval (steps)")

    # Hardware configuration
    device: str = Field("auto", description="Device for computation (auto/cpu/cuda)")
    mixed_precision: bool = Field(False, description="Enable mixed precision training")

    @validator('device', always=True)
    def validate_device(cls, v):
        if v == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return v

    @root_validator
    def validate_task_model_compatibility(cls, values):
        """Validate compatibility between task and model configurations."""
        task_config = values.get('task')
        model_config = values.get('model')

        if task_config and model_config:
            # Check classification task has num_classes
            if 'classification' in task_config.name.lower():
                if model_config.num_classes is None:
                    raise ValueError("Classification tasks require model.num_classes to be specified")

            # Check regression task has output_dim
            if 'regression' in task_config.name.lower():
                if model_config.output_dim is None:
                    raise ValueError("Regression tasks require model.output_dim to be specified")

        return values


def load_and_validate_config(config_path: Union[str, Path]) -> ExperimentConfig:
    """Load and validate configuration from YAML file.

    Parameters
    ----------
    config_path : Union[str, Path]
        Path to configuration YAML file

    Returns
    -------
    ExperimentConfig
        Validated configuration object

    Raises
    ------
    FileNotFoundError
        If configuration file doesn't exist
    ValidationError
        If configuration is invalid
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        raw_config = yaml.safe_load(f)

    return ExperimentConfig(**raw_config)


def save_config(config: ExperimentConfig, output_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.

    Parameters
    ----------
    config : ExperimentConfig
        Configuration object to save
    output_path : Union[str, Path]
        Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config.dict(), f, default_flow_style=False, allow_unicode=True)


def create_environment_tracker() -> Dict[str, Any]:
    """Create comprehensive environment tracking information.

    Returns
    -------
    Dict[str, Any]
        Environment information dictionary
    """
    import subprocess
    import sys

    env_info = {
        'timestamp': datetime.now().isoformat(),
        'hostname': socket.gethostname(),
        'platform': {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor()
        },
        'python': {
            'version': sys.version,
            'executable': sys.executable,
            'path': sys.path[:5]  # First 5 paths only
        },
        'pytorch': {
            'version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    }

    # Try to get git information
    try:
        git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                           stderr=subprocess.DEVNULL).decode().strip()
        git_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                                           stderr=subprocess.DEVNULL).decode().strip()
        env_info['git'] = {
            'commit': git_commit,
            'branch': git_branch
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        env_info['git'] = None

    return env_info


if __name__ == "__main__":
    # Example usage and validation
    example_config = {
        "name": "resnet1d_classification_experiment",
        "description": "ResNet1D classification on CWRU bearing dataset",
        "data": {
            "data_root": "data/",
            "metadata_file": "metadata.xlsx",
            "batch_size": 64,
            "window_size": 4096
        },
        "model": {
            "name": "ResNet1D",
            "type": "CNN",
            "input_dim": 3,
            "num_classes": 10
        },
        "task": {
            "name": "classification",
            "type": "DG"
        }
    }

    config = ExperimentConfig(**example_config)
    print("✅ Configuration validation passed!")
    print(f"Experiment: {config.name}")
    print(f"Device: {config.device}")
    print(f"Global seed: {config.reproducibility.global_seed}")

    # Test environment tracking
    env_info = create_environment_tracker()
    print(f"Environment tracked: {env_info['timestamp']}")
    print(f"Platform: {env_info['platform']['system']} {env_info['platform']['release']}")
    print(f"PyTorch: {env_info['pytorch']['version']}")
    print(f"CUDA available: {env_info['pytorch']['cuda_available']}")