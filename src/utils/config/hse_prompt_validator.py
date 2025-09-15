"""
HSE Prompt Configuration Validator for PHM-Vibench.

Enhanced P1 version with comprehensive validation and fixing utilities for HSE prompt-guided
training configurations, including Pipeline_03 integration support and ablation study validation.

Features:
- Enhanced configuration validation with detailed error reporting
- Automatic configuration fixing with clear explanations
- Pipeline_03 integration validation
- Ablation study configuration support
- Parameter range checking and optimization suggestions
- Multi-stage configuration validation (pretraining/finetuning)

Author: PHM-Vibench Team
Date: 2025-09-13
License: MIT
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Union, Optional, List, Tuple

try:
    from .path_standardizer import PathStandardizer
except ImportError:
    # When running as main, use absolute import
    import sys
    sys.path.insert(0, '.')
    from src.utils.config.path_standardizer import PathStandardizer


class HSEPromptConfigValidator:
    """Enhanced configuration validator for HSE prompt-guided contrastive learning."""

    VALID_FUSION_STRATEGIES = ['concat', 'attention', 'gating']
    VALID_CONTRASTIVE_LOSSES = ['INFONCE', 'TRIPLET', 'SUPCON', 'PROTOTYPICAL', 'BARLOWTWINS', 'VICREG']
    VALID_TRAINING_STAGES = ['pretrain', 'finetune', 'both']
    VALID_BACKBONES = ['B_08_PatchTST', 'B_04_Dlinear', 'B_06_TimesNet', 'B_09_FNO']
    VALID_MODELS = ['M_02_ISFM_Prompt', 'M_01_ISFM']  # For model validation
    VALID_EMBEDDINGS = ['E_01_HSE_v2', 'E_01_HSE', 'E_02_HSE_v2']  # For embedding validation
    VALID_TASK_HEADS = ['H_01_Linear_cla', 'H_02_distance_cla', 'H_09_multiple_task', 'H_10_ProjectionHead']

    # Parameter ranges for validation
    PARAMETER_RANGES = {
        'prompt_dim': (16, 512),
        'batch_size': (1, 512),
        'learning_rate': (1e-6, 1e-1),
        'temperature': (0.01, 1.0),
        'contrast_weight': (0.0, 1.0),
        'prompt_similarity_weight': (0.0, 1.0),
        'max_epochs': (1, 1000),
        'num_workers': (0, 32)
    }

    # Pipeline_03 specific fields
    PIPELINE_03_FIELDS = ['stage_1_pretraining', 'stage_2_finetuning', 'zero_shot_evaluation']

    # Standard datasets for unified metric learning
    STANDARD_DATASETS = ['CWRU', 'XJTU', 'THU', 'Ottawa', 'JNU']
    
    def __init__(self, data_root: Optional[str] = None):
        """
        Initialize HSE prompt configuration validator.
        
        Args:
            data_root: Root directory for data paths
        """
        self.path_standardizer = PathStandardizer(data_root)
        self.validation_errors = []
        self.validation_warnings = []
    
    def validate_config(self, config: Dict[str, Any], config_type: str = 'general') -> Tuple[bool, List[str], List[str]]:
        """
        Validate HSE prompt configuration.
        
        Args:
            config: Configuration dictionary
            config_type: Type of config ('pretraining', 'finetuning', 'general')
            
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.validation_errors = []
        self.validation_warnings = []
        
        # Validate basic structure
        self._validate_basic_structure(config)
        
        # Validate data configuration
        self._validate_data_config(config.get('data', {}))
        
        # Validate model configuration
        self._validate_model_config(config.get('model', {}))
        
        # Validate task configuration
        self._validate_task_config(config.get('task', {}))
        
        # Validate trainer configuration
        self._validate_trainer_config(config.get('trainer', {}))
        
        # Validate environment configuration
        self._validate_environment_config(config.get('environment', {}))
        
        # Stage-specific validation
        if config_type == 'pretraining':
            self._validate_pretraining_config(config)
        elif config_type == 'finetuning':
            self._validate_finetuning_config(config)
        
        # Validate HSE prompt specific settings
        self._validate_hse_prompt_config(config)

        # Enhanced P1 validations
        self._validate_parameter_ranges(config)
        self._validate_pipeline_03_integration(config)
        self._validate_unified_metric_learning(config)

        is_valid = len(self.validation_errors) == 0
        return is_valid, self.validation_errors, self.validation_warnings
    
    def fix_config(self, config: Dict[str, Any], config_type: str = 'general') -> Dict[str, Any]:
        """
        Automatically fix common configuration issues.
        
        Args:
            config: Configuration dictionary
            config_type: Type of config ('pretraining', 'finetuning', 'general')
            
        Returns:
            Fixed configuration dictionary
        """
        fixed_config = config.copy()
        
        # Fix paths
        fixed_config = self.path_standardizer.standardize_config_paths(fixed_config)
        
        # Fix model configuration
        fixed_config = self._fix_model_config(fixed_config)
        
        # Fix task configuration
        fixed_config = self._fix_task_config(fixed_config)
        
        # Fix trainer configuration
        fixed_config = self._fix_trainer_config(fixed_config)
        
        # Fix HSE prompt specific settings
        fixed_config = self._fix_hse_prompt_config(fixed_config)
        
        return fixed_config
    
    def _validate_basic_structure(self, config: Dict[str, Any]) -> None:
        """Validate basic configuration structure."""
        required_sections = ['data', 'model', 'task', 'trainer', 'environment']
        
        for section in required_sections:
            if section not in config:
                self.validation_errors.append(f"Missing required section: {section}")
            elif not isinstance(config[section], dict):
                self.validation_errors.append(f"Section '{section}' must be a dictionary")
    
    def _validate_data_config(self, data_config: Dict[str, Any]) -> None:
        """Validate data configuration."""
        required_fields = ['data_dir', 'metadata_file', 'batch_size']
        
        for field in required_fields:
            if field not in data_config:
                self.validation_errors.append(f"Missing required data field: {field}")
        
        # Validate data_dir exists
        if 'data_dir' in data_config:
            data_dir = Path(data_config['data_dir'])
            if not data_dir.exists():
                self.validation_errors.append(f"Data directory does not exist: {data_dir}")
        
        # Validate metadata_file exists
        if 'data_dir' in data_config and 'metadata_file' in data_config:
            metadata_path = Path(data_config['data_dir']) / data_config['metadata_file']
            if not metadata_path.exists():
                self.validation_errors.append(f"Metadata file does not exist: {metadata_path}")
        
        # Validate batch_size
        if 'batch_size' in data_config:
            batch_size = data_config['batch_size']
            if not isinstance(batch_size, int) or batch_size <= 0:
                self.validation_errors.append(f"batch_size must be a positive integer, got: {batch_size}")
    
    def _validate_model_config(self, model_config: Dict[str, Any]) -> None:
        """Validate model configuration."""
        # Validate required fields
        if 'embedding' not in model_config:
            self.validation_errors.append("Missing required model field: embedding")
        elif model_config['embedding'] != 'E_01_HSE_v2':
            self.validation_warnings.append(
                f"Expected E_01_HSE_v2 embedding for HSE prompt, got: {model_config['embedding']}"
            )
        
        if 'backbone' not in model_config:
            self.validation_errors.append("Missing required model field: backbone")
        elif model_config['backbone'] not in self.VALID_BACKBONES:
            self.validation_warnings.append(
                f"Backbone '{model_config['backbone']}' not in recommended list: {self.VALID_BACKBONES}"
            )
        
        # Validate HSE prompt specific model settings
        if 'fusion_strategy' in model_config:
            if model_config['fusion_strategy'] not in self.VALID_FUSION_STRATEGIES:
                self.validation_errors.append(
                    f"Invalid fusion_strategy: {model_config['fusion_strategy']}. "
                    f"Must be one of: {self.VALID_FUSION_STRATEGIES}"
                )
        
        # Validate training stage
        if 'training_stage' in model_config:
            if model_config['training_stage'] not in self.VALID_TRAINING_STAGES:
                self.validation_errors.append(
                    f"Invalid training_stage: {model_config['training_stage']}. "
                    f"Must be one of: {self.VALID_TRAINING_STAGES}"
                )
    
    def _validate_task_config(self, task_config: Dict[str, Any]) -> None:
        """Validate task configuration."""
        # Validate contrastive learning settings
        if 'contrast_loss' in task_config:
            if task_config['contrast_loss'] not in self.VALID_CONTRASTIVE_LOSSES:
                self.validation_errors.append(
                    f"Invalid contrast_loss: {task_config['contrast_loss']}. "
                    f"Must be one of: {self.VALID_CONTRASTIVE_LOSSES}"
                )
        
        # Validate contrast weight
        if 'contrast_weight' in task_config:
            weight = task_config['contrast_weight']
            if not isinstance(weight, (int, float)) or not 0 <= weight <= 1:
                self.validation_errors.append(
                    f"contrast_weight must be a float between 0 and 1, got: {weight}"
                )
        
        # Validate temperature
        if 'temperature' in task_config:
            temp = task_config['temperature']
            if not isinstance(temp, (int, float)) or temp <= 0:
                self.validation_errors.append(
                    f"temperature must be a positive number, got: {temp}"
                )
        
        # Validate prompt similarity weight
        if 'prompt_similarity_weight' in task_config:
            weight = task_config['prompt_similarity_weight']
            if not isinstance(weight, (int, float)) or not 0 <= weight <= 1:
                self.validation_errors.append(
                    f"prompt_similarity_weight must be a float between 0 and 1, got: {weight}"
                )
    
    def _validate_trainer_config(self, trainer_config: Dict[str, Any]) -> None:
        """Validate trainer configuration."""
        # Validate max_epochs
        if 'max_epochs' in trainer_config:
            epochs = trainer_config['max_epochs']
            if not isinstance(epochs, int) or epochs <= 0:
                self.validation_errors.append(
                    f"max_epochs must be a positive integer, got: {epochs}"
                )
        
        # Validate devices
        if 'devices' in trainer_config:
            devices = trainer_config['devices']
            if not isinstance(devices, (int, list)):
                self.validation_errors.append(
                    f"devices must be an integer or list, got: {type(devices)}"
                )
    
    def _validate_environment_config(self, env_config: Dict[str, Any]) -> None:
        """Validate environment configuration."""
        # Validate output directory
        if 'output_dir' in env_config:
            output_dir = Path(env_config['output_dir'])
            parent_dir = output_dir.parent
            if not parent_dir.exists():
                self.validation_warnings.append(
                    f"Output directory parent does not exist: {parent_dir}"
                )
    
    def _validate_pretraining_config(self, config: Dict[str, Any]) -> None:
        """Validate pretraining-specific configuration."""
        model_config = config.get('model', {})
        task_config = config.get('task', {})
        
        # Pretraining should not freeze prompts
        if model_config.get('freeze_prompt', False):
            self.validation_warnings.append(
                "freeze_prompt=True in pretraining config may prevent prompt learning"
            )
        
        # Pretraining should have contrastive learning
        if not task_config.get('contrast_weight', 0) > 0:
            self.validation_warnings.append(
                "contrast_weight should be > 0 for pretraining to enable contrastive learning"
            )
    
    def _validate_finetuning_config(self, config: Dict[str, Any]) -> None:
        """Validate finetuning-specific configuration."""
        model_config = config.get('model', {})
        
        # Finetuning typically freezes prompts
        if not model_config.get('freeze_prompt', False):
            self.validation_warnings.append(
                "freeze_prompt=False in finetuning config may lead to overfitting"
            )
    
    def _validate_hse_prompt_config(self, config: Dict[str, Any]) -> None:
        """Validate HSE prompt specific configuration."""
        model_config = config.get('model', {})
        
        # Check for prompt-related parameters
        prompt_params = ['fusion_strategy', 'prompt_dim', 'system_prompt_dim', 'sample_prompt_dim']
        has_prompt_config = any(param in model_config for param in prompt_params)
        
        if not has_prompt_config:
            self.validation_warnings.append(
                "No HSE prompt configuration found. Consider adding fusion_strategy, prompt_dim, etc."
            )
        
        # Validate prompt dimensions
        for dim_param in ['prompt_dim', 'system_prompt_dim', 'sample_prompt_dim']:
            if dim_param in model_config:
                dim = model_config[dim_param]
                if not isinstance(dim, int) or dim <= 0:
                    self.validation_errors.append(
                        f"{dim_param} must be a positive integer, got: {dim}"
                    )
    
    def _fix_model_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Fix model configuration issues."""
        if 'model' not in config:
            config['model'] = {}
        
        model_config = config['model']
        
        # Set default embedding if missing or incorrect
        if 'embedding' not in model_config:
            model_config['embedding'] = 'E_01_HSE_v2'
            
        # Set default fusion strategy
        if 'fusion_strategy' not in model_config:
            model_config['fusion_strategy'] = 'attention'
        elif model_config['fusion_strategy'] not in self.VALID_FUSION_STRATEGIES:
            model_config['fusion_strategy'] = 'attention'
        
        # Set default training stage
        if 'training_stage' not in model_config:
            model_config['training_stage'] = 'pretrain'
        
        return config
    
    def _fix_task_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Fix task configuration issues."""
        if 'task' not in config:
            config['task'] = {}
        
        task_config = config['task']
        
        # Set default contrastive loss
        if 'contrast_loss' not in task_config:
            task_config['contrast_loss'] = 'INFONCE'
        elif task_config['contrast_loss'] not in self.VALID_CONTRASTIVE_LOSSES:
            task_config['contrast_loss'] = 'INFONCE'
        
        # Set default contrast weight
        if 'contrast_weight' not in task_config:
            task_config['contrast_weight'] = 0.15
        
        # Set default temperature
        if 'temperature' not in task_config:
            task_config['temperature'] = 0.07
        
        # Set default prompt similarity weight
        if 'prompt_similarity_weight' not in task_config:
            task_config['prompt_similarity_weight'] = 0.1
        
        return config
    
    def _fix_trainer_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Fix trainer configuration issues."""
        if 'trainer' not in config:
            config['trainer'] = {}
        
        trainer_config = config['trainer']
        
        # Set default max_epochs
        if 'max_epochs' not in trainer_config:
            trainer_config['max_epochs'] = 100
        
        return config
    
    def _fix_hse_prompt_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Fix HSE prompt specific configuration."""
        if 'model' not in config:
            config['model'] = {}
        
        model_config = config['model']
        
        # Set default prompt dimensions
        if 'prompt_dim' not in model_config:
            model_config['prompt_dim'] = 64
        
        if 'system_prompt_dim' not in model_config:
            model_config['system_prompt_dim'] = 32
        
        if 'sample_prompt_dim' not in model_config:
            model_config['sample_prompt_dim'] = 32

        return config

    def _validate_parameter_ranges(self, config: Dict[str, Any]) -> None:
        """Validate parameter values against reasonable ranges."""
        all_config = {}
        all_config.update(config.get('data', {}))
        all_config.update(config.get('model', {}))
        all_config.update(config.get('task', {}))
        all_config.update(config.get('trainer', {}))

        for param_name, (min_val, max_val) in self.PARAMETER_RANGES.items():
            if param_name in all_config:
                value = all_config[param_name]
                if isinstance(value, (int, float)):
                    if not min_val <= value <= max_val:
                        self.validation_errors.append(
                            f"Parameter '{param_name}' value {value} is outside valid range [{min_val}, {max_val}]"
                        )

                        # Provide optimization suggestions
                        if param_name == 'batch_size' and value > max_val:
                            self.validation_warnings.append(
                                f"Large batch size ({value}) may cause GPU memory issues. "
                                f"Consider reducing to {min(64, max_val)} for initial experiments."
                            )
                        elif param_name == 'learning_rate':
                            if value > 0.01:
                                self.validation_warnings.append(
                                    f"High learning rate ({value}) may cause training instability. "
                                    f"Consider starting with 0.001 for HSE prompt training."
                                )
                            elif value < 1e-5:
                                self.validation_warnings.append(
                                    f"Very low learning rate ({value}) may slow convergence. "
                                    f"Consider using at least 1e-4 for prompt learning."
                                )

    def _validate_pipeline_03_integration(self, config: Dict[str, Any]) -> None:
        """Validate Pipeline_03 integration requirements."""
        # Check if this is a Pipeline_03 configuration
        has_pipeline_03_fields = any(field in config for field in self.PIPELINE_03_FIELDS)

        if has_pipeline_03_fields:
            # Validate Pipeline_03 structure
            if 'stage_1_pretraining' in config:
                stage1_config = config['stage_1_pretraining']
                self._validate_pretraining_stage_config(stage1_config)

            if 'stage_2_finetuning' in config:
                stage2_config = config['stage_2_finetuning']
                self._validate_finetuning_stage_config(stage2_config)

            # Validate zero-shot evaluation configuration
            if 'zero_shot_evaluation' in config:
                zero_shot_config = config['zero_shot_evaluation']
                self._validate_zero_shot_config(zero_shot_config)

            # Check model consistency across stages
            self._validate_cross_stage_consistency(config)

    def _validate_pretraining_stage_config(self, stage_config: Dict[str, Any]) -> None:
        """Validate pretraining stage configuration."""
        # Model should use M_02_ISFM_Prompt for HSE prompt training
        model_config = stage_config.get('model', {})
        if 'name' in model_config and model_config['name'] not in self.VALID_MODELS:
            self.validation_errors.append(
                f"Invalid model for pretraining: {model_config['name']}. "
                f"Expected one of: {self.VALID_MODELS}"
            )

        # Embedding should be E_01_HSE_v2 for prompt training
        if 'embedding' in model_config:
            if model_config['embedding'] == 'E_01_HSE_v2':
                # Check prompt-specific parameters
                if not model_config.get('use_prompt', True):
                    self.validation_warnings.append(
                        "use_prompt=False with E_01_HSE_v2 embedding. "
                        "Consider using E_01_HSE for non-prompt training."
                    )

                # Ensure prompts are not frozen during pretraining
                if model_config.get('freeze_prompt', False):
                    self.validation_warnings.append(
                        "freeze_prompt=True in pretraining may prevent prompt learning. "
                        "Consider setting freeze_prompt=False for stage 1."
                    )

        # Task should enable contrastive learning
        task_config = stage_config.get('task', {})
        contrast_weight = task_config.get('contrast_weight', 0.0)
        if contrast_weight <= 0:
            self.validation_warnings.append(
                "contrast_weight=0 in pretraining disables contrastive learning. "
                "HSE prompt training benefits from contrastive learning (suggest 0.15)."
            )

    def _validate_finetuning_stage_config(self, stage_config: Dict[str, Any]) -> None:
        """Validate finetuning stage configuration."""
        model_config = stage_config.get('model', {})

        # Prompt freezing is recommended for finetuning
        if model_config.get('embedding') == 'E_01_HSE_v2':
            if not model_config.get('freeze_prompt', False):
                self.validation_warnings.append(
                    "freeze_prompt=False in finetuning may cause prompt overfitting. "
                    "Consider setting freeze_prompt=True for stage 2."
                )

        # Reduced contrast weight for finetuning
        task_config = stage_config.get('task', {})
        contrast_weight = task_config.get('contrast_weight', 0.15)
        if contrast_weight > 0.1:
            self.validation_warnings.append(
                f"High contrast_weight ({contrast_weight}) in finetuning may interfere with task learning. "
                "Consider reducing to 0.05 or 0.0 for finetuning."
            )

    def _validate_zero_shot_config(self, zero_shot_config: Dict[str, Any]) -> None:
        """Validate zero-shot evaluation configuration."""
        if 'target_datasets' not in zero_shot_config:
            self.validation_warnings.append(
                "No target_datasets specified for zero-shot evaluation. "
                f"Consider adding: {self.STANDARD_DATASETS}"
            )
        else:
            target_datasets = zero_shot_config['target_datasets']
            for dataset in target_datasets:
                if dataset not in self.STANDARD_DATASETS:
                    self.validation_warnings.append(
                        f"Unknown dataset '{dataset}' in zero-shot evaluation. "
                        f"Standard datasets: {self.STANDARD_DATASETS}"
                    )

    def _validate_cross_stage_consistency(self, config: Dict[str, Any]) -> None:
        """Validate consistency between pretraining and finetuning stages."""
        stage1_config = config.get('stage_1_pretraining', {})
        stage2_config = config.get('stage_2_finetuning', {})

        if not stage1_config or not stage2_config:
            return

        # Check model architecture consistency
        stage1_model = stage1_config.get('model', {})
        stage2_model = stage2_config.get('model', {})

        for key in ['embedding', 'backbone']:
            if key in stage1_model and key in stage2_model:
                if stage1_model[key] != stage2_model[key]:
                    self.validation_errors.append(
                        f"Inconsistent {key} between stages: "
                        f"stage1={stage1_model[key]}, stage2={stage2_model[key]}"
                    )

    def _validate_unified_metric_learning(self, config: Dict[str, Any]) -> None:
        """Validate unified metric learning configuration."""
        data_config = config.get('data', {})

        # Check for unified dataset configuration
        if 'unified_datasets' in data_config:
            unified_datasets = data_config['unified_datasets']
            if not isinstance(unified_datasets, list):
                self.validation_errors.append("unified_datasets must be a list")
            else:
                # Validate dataset names
                for dataset in unified_datasets:
                    if dataset not in self.STANDARD_DATASETS:
                        self.validation_warnings.append(
                            f"Unknown dataset '{dataset}' in unified_datasets. "
                            f"Standard datasets: {self.STANDARD_DATASETS}"
                        )

                # Check if all 5 datasets are included for full unified learning
                if len(unified_datasets) == len(self.STANDARD_DATASETS):
                    missing = set(self.STANDARD_DATASETS) - set(unified_datasets)
                    if missing:
                        self.validation_warnings.append(
                            f"Missing datasets for full unified learning: {list(missing)}"
                        )
                elif len(unified_datasets) < 2:
                    self.validation_warnings.append(
                        "Unified metric learning requires at least 2 datasets for effectiveness."
                    )

    def validate_ablation_config(self, config: Dict[str, Any], ablation_type: str) -> Tuple[bool, List[str], List[str]]:
        """
        Validate ablation study configuration.

        Args:
            config: Configuration dictionary
            ablation_type: Type of ablation ('system_prompt_only', 'sample_prompt_only', 'no_prompt_baseline')

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.validation_errors = []
        self.validation_warnings = []

        # First run standard validation
        is_valid, errors, warnings = self.validate_config(config)

        # Add ablation-specific validation
        model_config = config.get('model', {})

        if ablation_type == 'system_prompt_only':
            # Should disable sample-level prompts
            if model_config.get('use_sample_prompt', True):
                self.validation_warnings.append(
                    "System-only ablation should set use_sample_prompt=False"
                )

        elif ablation_type == 'sample_prompt_only':
            # Should disable system-level prompts
            if model_config.get('use_system_prompt', True):
                self.validation_warnings.append(
                    "Sample-only ablation should set use_system_prompt=False"
                )

        elif ablation_type == 'no_prompt_baseline':
            # Should use standard embedding without prompts
            if model_config.get('embedding') == 'E_01_HSE_v2':
                self.validation_warnings.append(
                    "No-prompt baseline should use E_01_HSE instead of E_01_HSE_v2"
                )
            if model_config.get('use_prompt', False):
                self.validation_warnings.append(
                    "No-prompt baseline should set use_prompt=False"
                )

        # Ensure experimental controls are identical
        self._validate_ablation_controls(config, ablation_type)

        final_valid = is_valid and len(self.validation_errors) == 0
        return final_valid, self.validation_errors + errors, self.validation_warnings + warnings

    def _validate_ablation_controls(self, config: Dict[str, Any], ablation_type: str) -> None:
        """Validate that ablation study maintains proper experimental controls."""
        # Check that non-prompt parameters are standardized
        task_config = config.get('task', {})
        trainer_config = config.get('trainer', {})

        # Standard settings for fair comparison
        expected_settings = {
            'contrast_loss': 'INFONCE',  # Same contrastive loss
            'temperature': 0.07,         # Same temperature
            'max_epochs': 100,           # Same training duration
        }

        for setting, expected_value in expected_settings.items():
            # Find the setting in task or trainer config
            actual_value = task_config.get(setting) or trainer_config.get(setting)
            if actual_value is not None and actual_value != expected_value:
                self.validation_warnings.append(
                    f"Ablation study should use standardized {setting}={expected_value} "
                    f"for fair comparison, got {actual_value}"
                )

    def get_optimization_suggestions(self, config: Dict[str, Any]) -> List[str]:
        """
        Generate optimization suggestions for configuration.

        Args:
            config: Configuration dictionary

        Returns:
            List of optimization suggestions
        """
        suggestions = []

        model_config = config.get('model', {})
        task_config = config.get('task', {})
        data_config = config.get('data', {})

        # Memory optimization suggestions
        batch_size = data_config.get('batch_size', 32)
        if batch_size > 64:
            suggestions.append(
                f"Consider reducing batch_size from {batch_size} to 64 or lower "
                "to prevent GPU memory issues with HSE prompt models"
            )

        # Training efficiency suggestions
        num_workers = data_config.get('num_workers', 4)
        if num_workers > 8:
            suggestions.append(
                f"num_workers={num_workers} may be excessive. Consider 4-8 for optimal performance"
            )

        # HSE prompt optimization suggestions
        if model_config.get('embedding') == 'E_01_HSE_v2':
            fusion_type = model_config.get('fusion_type', 'attention')
            if fusion_type == 'concat':
                suggestions.append(
                    "fusion_type='concat' is fastest but may limit representation quality. "
                    "Consider 'attention' for better performance or 'gating' for balanced efficiency"
                )

        # Contrastive learning optimization
        contrast_weight = task_config.get('contrast_weight', 0.15)
        if contrast_weight > 0.3:
            suggestions.append(
                f"High contrast_weight ({contrast_weight}) may overwhelm task learning. "
                "Consider 0.15-0.25 for balanced training"
            )

        # Training stage suggestions
        training_stage = model_config.get('training_stage', 'pretrain')
        freeze_prompt = model_config.get('freeze_prompt', False)

        if training_stage == 'pretrain' and freeze_prompt:
            suggestions.append(
                "freeze_prompt=True in pretraining prevents prompt learning. "
                "Set freeze_prompt=False for stage 1, True for stage 2"
            )

        return suggestions
    
    def validate_yaml_file(self, yaml_path: Union[str, Path], 
                          config_type: str = 'general') -> Tuple[bool, List[str], List[str]]:
        """
        Validate a YAML configuration file.
        
        Args:
            yaml_path: Path to YAML file
            config_type: Type of config ('pretraining', 'finetuning', 'general')
            
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            return False, [f"Configuration file does not exist: {yaml_path}"], []
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            return self.validate_config(config, config_type)
            
        except yaml.YAMLError as e:
            return False, [f"YAML parsing error: {e}"], []
        except Exception as e:
            return False, [f"Error reading configuration file: {e}"], []
    
    def fix_yaml_file(self, yaml_path: Union[str, Path], 
                     output_path: Optional[Union[str, Path]] = None,
                     config_type: str = 'general',
                     backup: bool = True) -> None:
        """
        Fix a YAML configuration file.
        
        Args:
            yaml_path: Path to YAML file
            output_path: Output path (if None, overwrites input)
            config_type: Type of config ('pretraining', 'finetuning', 'general')
            backup: Whether to create backup of original file
        """
        yaml_path = Path(yaml_path)
        output_path = Path(output_path) if output_path else yaml_path
        
        # Create backup if requested
        if backup and output_path == yaml_path:
            backup_path = yaml_path.with_suffix(yaml_path.suffix + '.bak')
            if yaml_path.exists():
                yaml_path.rename(backup_path)
                yaml_path = backup_path
        
        # Load configuration
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Fix configuration
        fixed_config = self.fix_config(config, config_type)
        
        # Save fixed configuration
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(fixed_config, f, default_flow_style=False, sort_keys=False,
                     allow_unicode=True, indent=2)


def validate_hse_prompt_directory(config_dir: Union[str, Path],
                                 data_root: Optional[str] = None,
                                 pattern: str = "*.yaml") -> Dict[str, Tuple[bool, List[str], List[str]]]:
    """
    Validate all HSE prompt configuration files in a directory.
    
    Args:
        config_dir: Directory containing configuration files
        data_root: Root directory for data paths
        pattern: File pattern to match (default: "*.yaml")
        
    Returns:
        Dictionary mapping file paths to validation results
    """
    config_dir = Path(config_dir)
    validator = HSEPromptConfigValidator(data_root)
    results = {}
    
    for yaml_file in config_dir.glob(pattern):
        # Determine config type from filename
        filename = yaml_file.name.lower()
        if 'pretrain' in filename:
            config_type = 'pretraining'
        elif 'finetune' in filename or 'fine_tune' in filename:
            config_type = 'finetuning'
        else:
            config_type = 'general'
        
        try:
            is_valid, errors, warnings = validator.validate_yaml_file(yaml_file, config_type)
            results[str(yaml_file)] = (is_valid, errors, warnings)
        except Exception as e:
            results[str(yaml_file)] = (False, [f"Validation error: {e}"], [])
    
    return results


if __name__ == '__main__':
    """Self-test for HSE prompt configuration validator."""
    import tempfile
    
    print("=== HSE Prompt Config Validator Self-Test ===")
    
    # Test configuration
    test_config = {
        'environment': {
            'project': 'HSE-Prompt-Test',
            'output_dir': 'save/test_experiment'
        },
        'data': {
            'data_dir': '/home/user/data/PHMbenchdata/PHM-Vibench',
            'metadata_file': 'metadata_6_1.xlsx',
            'batch_size': 32,
            'num_workers': 4
        },
        'model': {
            'embedding': 'E_01_HSE_v2',
            'backbone': 'B_08_PatchTST',
            'task_head': 'H_01_Linear_cla',
            'fusion_strategy': 'attention',
            'prompt_dim': 64,
            'training_stage': 'pretrain',
            'freeze_prompt': False
        },
        'task': {
            'task_type': 'CDDG',
            'loss': 'CE',
            'contrast_loss': 'INFONCE',
            'contrast_weight': 0.15,
            'temperature': 0.07,
            'prompt_similarity_weight': 0.1
        },
        'trainer': {
            'max_epochs': 100,
            'devices': 1,
            'accelerator': 'auto'
        }
    }
    
    # Test validator
    validator = HSEPromptConfigValidator()
    
    print("\n1. Testing configuration validation:")
    is_valid, errors, warnings = validator.validate_config(test_config, 'pretraining')
    
    print(f"   Valid: {is_valid}")
    print(f"   Errors ({len(errors)}):")
    for error in errors[:5]:  # Show first 5 errors
        print(f"     - {error}")
    print(f"   Warnings ({len(warnings)}):")
    for warning in warnings[:5]:  # Show first 5 warnings
        print(f"     - {warning}")
    
    print("\n2. Testing configuration fixing:")
    # Create invalid config
    invalid_config = {
        'data': {'batch_size': -1},
        'model': {'fusion_strategy': 'invalid'},
        'task': {'contrast_weight': 2.0}
    }
    
    fixed_config = validator.fix_config(invalid_config)
    print(f"   Fixed batch_size: {fixed_config['data'].get('batch_size')}")
    print(f"   Fixed fusion_strategy: {fixed_config['model'].get('fusion_strategy')}")
    print(f"   Fixed contrast_weight: {fixed_config['task'].get('contrast_weight')}")
    
    print("\n3. Testing YAML file validation:")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config, f, default_flow_style=False)
        test_yaml = f.name
    
    try:
        is_valid, errors, warnings = validator.validate_yaml_file(test_yaml, 'pretraining')
        print(f"   YAML validation: {'✓' if is_valid else '✗'}")
        print(f"   Errors: {len(errors)}, Warnings: {len(warnings)}")
        
    except Exception as e:
        print(f"   YAML validation: ✗ {e}")
    finally:
        os.unlink(test_yaml)
    
    print("\n4. Testing invalid configurations:")
    test_cases = [
        ({'model': {'fusion_strategy': 'invalid'}}, "Invalid fusion strategy"),
        ({'task': {'contrast_loss': 'UNKNOWN'}}, "Invalid contrast loss"),
        ({'data': {'batch_size': 0}}, "Invalid batch size"),
        ({'trainer': {'max_epochs': -10}}, "Invalid max epochs")
    ]
    
    for test_config_invalid, description in test_cases:
        is_valid, errors, warnings = validator.validate_config(test_config_invalid)
        has_expected_error = len(errors) > 0
        print(f"   {description}: {'✓' if has_expected_error else '✗'}")
    
    print("\n=== All tests completed ===")