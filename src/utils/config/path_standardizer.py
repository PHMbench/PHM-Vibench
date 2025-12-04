"""
Path standardization utility for PHM-Vibench configurations.
Ensures consistent data path handling across different environments.
"""

import os
from pathlib import Path
from typing import Dict, Any, Union, Optional
import yaml


class PathStandardizer:
    """Utility class for standardizing data paths in PHM-Vibench configurations."""
    
    DEFAULT_DATA_ROOT = "/home/user/data/PHMbenchdata/PHM-Vibench"
    
    def __init__(self, data_root: Optional[str] = None):
        """
        Initialize path standardizer.
        
        Args:
            data_root: Root directory for data. If None, uses DEFAULT_DATA_ROOT
        """
        self.data_root = Path(data_root or self.DEFAULT_DATA_ROOT)
    
    def standardize_data_path(self, path: Union[str, Path]) -> str:
        """
        Standardize a data path to use consistent root.
        
        Args:
            path: Input path to standardize
            
        Returns:
            Standardized path as string
        """
        path = Path(path)
        
        # If already absolute and exists, return as-is
        if path.is_absolute():
            return str(path)
        
        # If relative, resolve against data_root
        return str(self.data_root / path)
    
    def standardize_config_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardize all data-related paths in a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configuration with standardized paths
        """
        config = config.copy()
        
        # Standardize data section paths
        if 'data' in config and isinstance(config['data'], dict):
            data_config = config['data']
            
            # Standardize data_dir
            if 'data_dir' in data_config:
                data_config['data_dir'] = self.standardize_data_path(data_config['data_dir'])
            
            # Standardize metadata_file path if it's absolute
            if 'metadata_file' in data_config:
                metadata_path = data_config['metadata_file']
                if not os.path.isabs(metadata_path):
                    # Keep relative metadata files as-is
                    pass
                else:
                    data_config['metadata_file'] = self.standardize_data_path(metadata_path)
        
        # Standardize environment paths
        if 'environment' in config and isinstance(config['environment'], dict):
            env_config = config['environment']
            
            # Standardize output_dir if relative
            if 'output_dir' in env_config:
                output_dir = env_config['output_dir']
                if not os.path.isabs(output_dir):
                    # Keep relative output dirs as-is for flexibility
                    pass
        
        return config
    
    def standardize_yaml_file(self, yaml_path: Union[str, Path], 
                            output_path: Optional[Union[str, Path]] = None,
                            backup: bool = True) -> None:
        """
        Standardize paths in a YAML configuration file.
        
        Args:
            yaml_path: Path to YAML file
            output_path: Output path (if None, overwrites input)
            backup: Whether to create backup of original file
        """
        yaml_path = Path(yaml_path)
        output_path = Path(output_path) if output_path else yaml_path
        
        # Create backup if requested
        if backup and output_path == yaml_path:
            backup_path = yaml_path.with_suffix(yaml_path.suffix + '.bak')
            yaml_path.rename(backup_path)
            yaml_path = backup_path
        
        # Load configuration
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Standardize paths
        config = self.standardize_config_paths(config)
        
        # Save standardized configuration
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, 
                     allow_unicode=True, indent=2)
    
    def validate_data_paths(self, config: Dict[str, Any]) -> Dict[str, bool]:
        """
        Validate that data paths in configuration exist.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Dictionary mapping path keys to existence status
        """
        validation_results = {}
        
        if 'data' in config and isinstance(config['data'], dict):
            data_config = config['data']
            
            # Check data_dir
            if 'data_dir' in data_config:
                data_dir = Path(data_config['data_dir'])
                validation_results['data_dir'] = data_dir.exists()
            
            # Check metadata_file
            if 'metadata_file' in data_config and 'data_dir' in data_config:
                metadata_path = Path(data_config['data_dir']) / data_config['metadata_file']
                validation_results['metadata_file'] = metadata_path.exists()
        
        return validation_results


def standardize_config_directory(config_dir: Union[str, Path],
                               data_root: Optional[str] = None,
                               pattern: str = "*.yaml",
                               backup: bool = True) -> None:
    """
    Standardize all YAML files in a directory.
    
    Args:
        config_dir: Directory containing configuration files
        data_root: Root directory for data paths
        pattern: File pattern to match (default: "*.yaml")
        backup: Whether to create backups
    """
    config_dir = Path(config_dir)
    standardizer = PathStandardizer(data_root)
    
    for yaml_file in config_dir.glob(pattern):
        print(f"Standardizing {yaml_file}")
        try:
            standardizer.standardize_yaml_file(yaml_file, backup=backup)
            print(f"✓ Successfully standardized {yaml_file}")
        except Exception as e:
            print(f"✗ Error standardizing {yaml_file}: {e}")


if __name__ == '__main__':
    """Self-test for path standardizer utility."""
    import tempfile
    
    # Test configuration
    test_config = {
        'data': {
            'data_dir': '/old/path/to/data',
            'metadata_file': 'metadata_6_1.xlsx',
            'batch_size': 32
        },
        'environment': {
            'output_dir': 'results/test',
            'project': 'TestProject'
        },
        'model': {
            'name': 'M_01_ISFM'
        }
    }
    
    # Test path standardization
    standardizer = PathStandardizer('/home/user/data/PHMbenchdata/PHM-Vibench')
    
    print("=== Path Standardizer Self-Test ===")
    
    # Test 1: Path standardization
    print("\n1. Testing path standardization:")
    test_paths = [
        '/absolute/path/to/data',
        'relative/path',
        './current/relative',
        '../parent/relative'
    ]
    
    for path in test_paths:
        standardized = standardizer.standardize_data_path(path)
        print(f"  {path} -> {standardized}")
    
    # Test 2: Config standardization
    print("\n2. Testing configuration standardization:")
    print("Original config:")
    print(f"  data_dir: {test_config['data']['data_dir']}")
    
    standardized_config = standardizer.standardize_config_paths(test_config)
    print("Standardized config:")
    print(f"  data_dir: {standardized_config['data']['data_dir']}")
    
    # Test 3: YAML file processing
    print("\n3. Testing YAML file processing:")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config, f, default_flow_style=False)
        test_yaml = f.name
    
    try:
        standardizer.standardize_yaml_file(test_yaml, backup=False)
        
        with open(test_yaml, 'r') as f:
            processed_config = yaml.safe_load(f)
        
        print(f"  YAML processing: ✓")
        print(f"  New data_dir: {processed_config['data']['data_dir']}")
        
    except Exception as e:
        print(f"  YAML processing: ✗ {e}")
    finally:
        os.unlink(test_yaml)
    
    # Test 4: Path validation
    print("\n4. Testing path validation:")
    validation_results = standardizer.validate_data_paths(standardized_config)
    for path_key, exists in validation_results.items():
        status = "✓" if exists else "✗"
        print(f"  {path_key}: {status}")
    
    print("\n=== All tests completed ===")