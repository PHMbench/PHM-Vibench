"""
Comprehensive Reproducibility Framework for PHM-Vibench

This module provides a complete framework for ensuring experimental reproducibility
in scientific machine learning research, following best practices for computational
research and meeting publication standards.

Key Features:
- Deterministic random seed management across all libraries
- Comprehensive environment tracking and logging
- Experiment configuration versioning
- Result validation and comparison tools
- Automatic dependency tracking
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import socket
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything as pl_seed_everything

logger = logging.getLogger(__name__)


class ReproducibilityManager:
    """
    Comprehensive manager for experimental reproducibility.

    This class handles all aspects of reproducibility including:
    - Deterministic random seed management
    - Environment tracking and logging
    - Configuration hashing and versioning
    - Result validation and comparison

    The manager ensures that experiments can be exactly reproduced
    across different machines and environments.

    Parameters
    ----------
    config : ReproducibilityConfig
        Reproducibility configuration settings
    experiment_name : str
        Name of the current experiment
    output_dir : Path
        Directory for storing reproducibility artifacts

    Examples
    --------
    >>> config = ReproducibilityConfig(global_seed=42)
    >>> manager = ReproducibilityManager(config, "test_experiment", Path("results"))
    >>> manager.setup_reproducibility()
    >>> # Run experiment...
    >>> manager.save_reproducibility_report()
    """

    def __init__(
        self,
        config: Any,  # ReproducibilityConfig
        experiment_name: str,
        output_dir: Path
    ):
        self.config = config
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize tracking variables
        self.environment_info: Dict[str, Any] = {}
        self.config_hash: Optional[str] = None
        self.setup_timestamp: Optional[str] = None

        # Setup logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{experiment_name}")

    def setup_reproducibility(self) -> None:
        """
        Setup complete reproducibility environment.

        This method should be called before any model training or data loading
        to ensure deterministic behavior across all components.
        """
        self.setup_timestamp = datetime.now().isoformat()

        # 1. Setup deterministic random seeds
        self._setup_random_seeds()

        # 2. Configure PyTorch for deterministic operations
        self._setup_pytorch_determinism()

        # 3. Track environment information
        if getattr(self.config, 'track_environment', True):
            self.environment_info = self._collect_environment_info()

        # 4. Setup warnings and logging
        self._setup_warnings()

        self.logger.info(f"Reproducibility setup completed for experiment: {self.experiment_name}")
        self.logger.info(f"Global seed: {getattr(self.config, 'global_seed', 42)}")
        self.logger.info(f"PyTorch deterministic: {getattr(self.config, 'torch_deterministic', True)}")

    def _setup_random_seeds(self) -> None:
        """Setup deterministic random seeds across all libraries."""
        global_seed = getattr(self.config, 'global_seed', 42)
        numpy_seed = getattr(self.config, 'numpy_seed', global_seed)

        # Set Python random seed
        random.seed(global_seed)

        # Set NumPy random seed
        np.random.seed(numpy_seed)

        # Set PyTorch seeds (CPU and CUDA)
        torch.manual_seed(global_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(global_seed)
            torch.cuda.manual_seed_all(global_seed)

        # Set PyTorch Lightning seed
        pl_seed_everything(global_seed, workers=True)

        self.logger.debug(f"Random seeds set: global={global_seed}, numpy={numpy_seed}")

    def _setup_pytorch_determinism(self) -> None:
        """Configure PyTorch for deterministic operations."""
        torch_deterministic = getattr(self.config, 'torch_deterministic', True)
        torch_benchmark = getattr(self.config, 'torch_benchmark', False)

        if torch_deterministic:
            # Enable deterministic algorithms
            torch.use_deterministic_algorithms(True)

            # Set environment variable for deterministic behavior
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

            # Disable CUDNN benchmark for deterministic behavior
            torch.backends.cudnn.benchmark = torch_benchmark
            torch.backends.cudnn.deterministic = True

            self.logger.debug("PyTorch deterministic mode enabled")
        else:
            # Enable benchmark for performance (non-deterministic)
            torch.backends.cudnn.benchmark = torch_benchmark
            self.logger.debug(f"PyTorch benchmark mode: {torch_benchmark}")

    def _collect_environment_info(self) -> Dict[str, Any]:
        """Collect comprehensive environment information."""
        env_info = {
            'timestamp': self.setup_timestamp,
            'experiment_name': self.experiment_name,
            'hostname': socket.gethostname(),
            'user': os.getenv('USER', 'unknown'),
            'working_directory': str(Path.cwd()),

            # Platform information
            'platform': {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'architecture': platform.architecture(),
                'python_implementation': platform.python_implementation(),
            },

            # Python environment
            'python': {
                'version': sys.version,
                'executable': sys.executable,
                'path': sys.path[:10],  # First 10 paths only
                'prefix': sys.prefix,
            },

            # PyTorch information
            'pytorch': {
                'version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
                'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'device_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
            },

            # PyTorch Lightning information
            'pytorch_lightning': {
                'version': pl.__version__,
            },

            # Reproducibility settings
            'reproducibility': {
                'global_seed': getattr(self.config, 'global_seed', 42),
                'numpy_seed': getattr(self.config, 'numpy_seed', 42),
                'torch_deterministic': getattr(self.config, 'torch_deterministic', True),
                'torch_benchmark': getattr(self.config, 'torch_benchmark', False),
            }
        }

        # Add git information if available
        if getattr(self.config, 'track_git_commit', True):
            env_info['git'] = self._get_git_info()

        # Add package versions if requested
        if getattr(self.config, 'track_dependencies', True):
            env_info['packages'] = self._get_package_versions()

        return env_info

    def _get_git_info(self) -> Optional[Dict[str, str]]:
        """Get git repository information."""
        try:
            git_info = {}

            # Get commit hash
            git_info['commit'] = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()

            # Get branch name
            git_info['branch'] = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()

            # Get remote URL
            try:
                git_info['remote'] = subprocess.check_output(
                    ['git', 'config', '--get', 'remote.origin.url'],
                    stderr=subprocess.DEVNULL,
                    text=True
                ).strip()
            except subprocess.CalledProcessError:
                git_info['remote'] = None

            # Check for uncommitted changes
            try:
                subprocess.check_output(
                    ['git', 'diff-index', '--quiet', 'HEAD', '--'],
                    stderr=subprocess.DEVNULL
                )
                git_info['dirty'] = False
            except subprocess.CalledProcessError:
                git_info['dirty'] = True

            return git_info

        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.warning("Git information not available")
            return None

    def _get_package_versions(self) -> Dict[str, str]:
        """Get versions of key packages."""
        packages = {}

        # Key packages for PHM-Vibench
        key_packages = [
            'torch', 'torchvision', 'torchaudio', 'pytorch-lightning',
            'numpy', 'pandas', 'scikit-learn', 'scipy', 'matplotlib',
            'h5py', 'pydantic', 'yaml', 'wandb', 'tensorboard'
        ]

        for package_name in key_packages:
            try:
                if package_name == 'yaml':
                    import yaml
                    packages[package_name] = getattr(yaml, '__version__', 'unknown')
                else:
                    module = __import__(package_name.replace('-', '_'))
                    packages[package_name] = getattr(module, '__version__', 'unknown')
            except ImportError:
                packages[package_name] = 'not_installed'
            except Exception as e:
                packages[package_name] = f'error: {str(e)}'

        return packages

    def _setup_warnings(self) -> None:
        """Setup warning filters for reproducibility."""
        # Filter out common non-deterministic warnings
        warnings.filterwarnings('ignore', message='.*deterministic.*')
        warnings.filterwarnings('ignore', message='.*non-deterministic.*')

        # But keep important warnings
        warnings.filterwarnings('default', category=UserWarning)
        warnings.filterwarnings('default', category=FutureWarning)

    def compute_config_hash(self, config: Any) -> str:
        """
        Compute a hash of the experiment configuration.

        This hash can be used to identify identical configurations
        and ensure reproducibility across runs.

        Parameters
        ----------
        config : Any
            Experiment configuration object

        Returns
        -------
        str
            SHA-256 hash of the configuration
        """
        # Convert config to dictionary if needed
        if hasattr(config, 'dict'):
            config_dict = config.dict()
        elif hasattr(config, '__dict__'):
            config_dict = config.__dict__
        else:
            config_dict = dict(config)

        # Sort keys for consistent hashing
        config_str = json.dumps(config_dict, sort_keys=True, default=str)

        # Compute hash
        self.config_hash = hashlib.sha256(config_str.encode()).hexdigest()

        return self.config_hash

    def save_reproducibility_report(self, config: Any, results: Optional[Dict[str, Any]] = None) -> Path:
        """
        Save a comprehensive reproducibility report.

        Parameters
        ----------
        config : Any
            Experiment configuration
        results : Optional[Dict[str, Any]]
            Experiment results to include in the report

        Returns
        -------
        Path
            Path to the saved report
        """
        # Compute configuration hash
        config_hash = self.compute_config_hash(config)

        # Create report
        report = {
            'experiment_info': {
                'name': self.experiment_name,
                'timestamp': self.setup_timestamp,
                'config_hash': config_hash,
            },
            'environment': self.environment_info,
            'configuration': config.dict() if hasattr(config, 'dict') else dict(config),
            'results': results or {},
            'reproducibility_notes': {
                'deterministic_setup': getattr(self.config, 'torch_deterministic', True),
                'global_seed': getattr(self.config, 'global_seed', 42),
                'environment_tracked': getattr(self.config, 'track_environment', True),
                'git_tracked': getattr(self.config, 'track_git_commit', True),
                'dependencies_tracked': getattr(self.config, 'track_dependencies', True),
            }
        }

        # Save report
        report_path = self.output_dir / f"reproducibility_report_{config_hash[:8]}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)

        self.logger.info(f"Reproducibility report saved: {report_path}")
        return report_path

    def validate_reproducibility(self, other_report_path: Path) -> Dict[str, bool]:
        """
        Validate reproducibility against another experiment report.

        Parameters
        ----------
        other_report_path : Path
            Path to another reproducibility report

        Returns
        -------
        Dict[str, bool]
            Validation results for different aspects
        """
        # Load other report
        with open(other_report_path, 'r', encoding='utf-8') as f:
            other_report = json.load(f)

        validation = {
            'same_config_hash': False,
            'same_pytorch_version': False,
            'same_cuda_version': False,
            'same_platform': False,
            'same_git_commit': False,
            'reproducible': False
        }

        # Check configuration hash
        if self.config_hash == other_report['experiment_info']['config_hash']:
            validation['same_config_hash'] = True

        # Check PyTorch version
        current_pytorch = self.environment_info.get('pytorch', {}).get('version')
        other_pytorch = other_report.get('environment', {}).get('pytorch', {}).get('version')
        validation['same_pytorch_version'] = current_pytorch == other_pytorch

        # Check CUDA version
        current_cuda = self.environment_info.get('pytorch', {}).get('cuda_version')
        other_cuda = other_report.get('environment', {}).get('pytorch', {}).get('cuda_version')
        validation['same_cuda_version'] = current_cuda == other_cuda

        # Check platform
        current_platform = self.environment_info.get('platform', {}).get('system')
        other_platform = other_report.get('environment', {}).get('platform', {}).get('system')
        validation['same_platform'] = current_platform == other_platform

        # Check git commit
        current_commit = self.environment_info.get('git', {}).get('commit') if self.environment_info.get('git') else None
        other_commit = other_report.get('environment', {}).get('git', {}).get('commit') if other_report.get('environment', {}).get('git') else None
        validation['same_git_commit'] = current_commit == other_commit

        # Overall reproducibility assessment
        validation['reproducible'] = (
            validation['same_config_hash'] and
            validation['same_pytorch_version'] and
            validation['same_git_commit']
        )

        return validation


def setup_experiment_reproducibility(
    config: Any,
    experiment_name: str,
    output_dir: Path
) -> ReproducibilityManager:
    """
    Convenience function to setup reproducibility for an experiment.

    Parameters
    ----------
    config : Any
        Experiment configuration with reproducibility settings
    experiment_name : str
        Name of the experiment
    output_dir : Path
        Output directory for reproducibility artifacts

    Returns
    -------
    ReproducibilityManager
        Configured reproducibility manager
    """
    # Extract reproducibility config
    repro_config = getattr(config, 'reproducibility', None)
    if repro_config is None:
        # Create default reproducibility config
        from types import SimpleNamespace
        repro_config = SimpleNamespace(
            global_seed=42,
            torch_deterministic=True,
            torch_benchmark=False,
            track_environment=True,
            track_git_commit=True,
            track_dependencies=True
        )

    # Create and setup manager
    manager = ReproducibilityManager(repro_config, experiment_name, output_dir)
    manager.setup_reproducibility()

    return manager


if __name__ == "__main__":
    # Example usage and testing
    from types import SimpleNamespace
    from pathlib import Path

    # Create test configuration
    repro_config = SimpleNamespace(
        global_seed=42,
        torch_deterministic=True,
        torch_benchmark=False,
        track_environment=True,
        track_git_commit=True,
        track_dependencies=True
    )

    # Test reproducibility manager
    manager = ReproducibilityManager(
        config=repro_config,
        experiment_name="test_experiment",
        output_dir=Path("test_results")
    )

    # Setup reproducibility
    manager.setup_reproducibility()

    # Create mock experiment config
    experiment_config = SimpleNamespace(
        model=SimpleNamespace(name="ResNet1D", input_dim=3),
        data=SimpleNamespace(batch_size=64),
        optimization=SimpleNamespace(learning_rate=0.001)
    )

    # Save reproducibility report
    report_path = manager.save_reproducibility_report(experiment_config)
    print(f"✅ Reproducibility report saved: {report_path}")

    # Test configuration hashing
    config_hash = manager.compute_config_hash(experiment_config)
    print(f"Configuration hash: {config_hash}")

    print("✅ Reproducibility framework test completed!")