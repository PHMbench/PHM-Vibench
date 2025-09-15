"""
HSE Prompt Configuration Validator for PHM-Vibench.
Provides comprehensive validation and fixing utilities for HSE prompt-guided training configurations.
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
    """Configuration validator for HSE prompt-guided contrastive learning."""
    
    VALID_FUSION_STRATEGIES = ['concat', 'attention', 'gating']
    VALID_CONTRASTIVE_LOSSES = ['INFONCE', 'TRIPLET', 'SUPCON', 'PROTOTYPICAL', 'BARLOWTWINS', 'VICREG']
    VALID_TRAINING_STAGES = ['pretrain', 'finetune', 'both']
    VALID_BACKBONES = ['B_08_PatchTST', 'B_04_Dlinear', 'B_06_TimesNet', 'B_09_FNO']
    
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