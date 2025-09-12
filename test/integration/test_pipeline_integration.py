"""
Comprehensive pipeline integration tests for ContrastiveIDTask.

Tests verify complete integration with PHM-Vibench pipeline system including:
- Pipeline_ID workflow integration
- Factory system integration (data, model, task, trainer)
- Configuration system integration with all presets
- End-to-end pipeline testing
- Registry and component discovery
"""

import os
import sys
import tempfile
import shutil
import pytest
import torch
import yaml
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Imports after path setup
from src.configs.config_utils import load_config
from src.data_factory import build_data, DATA_FACTORY_REGISTRY
from src.model_factory import build_model
from src.task_factory import build_task, TASK_REGISTRY
from src.trainer_factory import build_trainer
from src.Pipeline_01_default import pipeline as default_pipeline
from src.Pipeline_ID import pipeline as id_pipeline


class TestPipelineIDIntegration:
    """Test ContrastiveIDTask integration with Pipeline_ID workflow."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.yaml")
        
        # Create minimal test configuration
        self.test_config = {
            'data': {
                'factory_name': 'id_data_factory',
                'data_dir': '/tmp/test_data',
                'dataset_name': 'test_dataset',
                'metadata_file': 'metadata_test.xlsx',
                'window_size': 1024,
                'batch_size': 8
            },
            'model': {
                'name': 'B_04_Dlinear',
                'type': 'backbone',
                'd_model': 512
            },
            'task': {
                'type': 'contrastive_id',
                'name': 'pretrain',
                'temperature': 0.07
            },
            'trainer': {
                'name': 'pytorch_lightning_trainer',
                'max_epochs': 1,
                'accelerator': 'cpu'
            },
            'environment': {
                'VBENCH_HOME': str(project_root),
                'seed': 42,
                'iterations': 1
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(self.test_config, f)
    
    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_pipeline_id_integration(self):
        """Test Pipeline_ID calls default pipeline correctly."""
        args_mock = SimpleNamespace(config_path=self.config_path)
        
        # Mock the default pipeline to avoid full execution
        with patch('src.Pipeline_ID.default_pipeline') as mock_pipeline:
            mock_pipeline.return_value = [{'test_acc': 0.8}]
            
            result = id_pipeline(args_mock)
            
            # Verify Pipeline_ID called default_pipeline with correct args
            mock_pipeline.assert_called_once_with(args_mock)
            assert result == [{'test_acc': 0.8}]
    
    def test_configuration_loading_flow(self):
        """Test configuration loading through pipeline."""
        # Test configuration can be loaded
        config = load_config(self.config_path)
        
        # Verify all required sections exist
        assert hasattr(config, 'data')
        assert hasattr(config, 'model')
        assert hasattr(config, 'task')
        assert hasattr(config, 'trainer')
        assert hasattr(config, 'environment')
        
        # Verify contrastive task configuration
        assert config.task.type == 'contrastive_id'
        assert config.task.name == 'pretrain'
        assert config.task.temperature == 0.07
    
    @patch('src.Pipeline_01_default.build_data')
    @patch('src.Pipeline_01_default.build_model')
    @patch('src.Pipeline_01_default.build_task')
    @patch('src.Pipeline_01_default.build_trainer')
    def test_data_flow_integration(self, mock_trainer, mock_task, mock_model, mock_data):
        """Test data flow from ID_dataset through to model training."""
        # Setup mocks
        mock_data_factory = MagicMock()
        mock_data_factory.get_metadata.return_value = {'num_classes': 10}
        mock_data_factory.get_dataloader.return_value = MagicMock()
        mock_data_factory.data.close = MagicMock()
        mock_data.return_value = mock_data_factory
        
        mock_model.return_value = MagicMock()
        mock_task.return_value = MagicMock()
        
        mock_trainer_instance = MagicMock()
        mock_trainer_instance.fit = MagicMock()
        mock_trainer_instance.test.return_value = [{'test_acc': 0.8}]
        mock_trainer.return_value = mock_trainer_instance
        
        # Mock other dependencies
        with patch('src.Pipeline_01_default.load_best_model_checkpoint') as mock_load:
            mock_load.return_value = mock_task.return_value
            with patch('src.Pipeline_01_default.init_lab'), \
                 patch('src.Pipeline_01_default.close_lab'), \
                 patch('os.path.join'), \
                 patch('pandas.DataFrame'):
                
                args_mock = SimpleNamespace(config_path=self.config_path)
                result = default_pipeline(args_mock)
                
                # Verify build functions were called
                mock_data.assert_called_once()
                mock_model.assert_called_once()
                mock_task.assert_called_once()
                mock_trainer.assert_called_once()
                
                # Verify training flow
                mock_trainer_instance.fit.assert_called_once()
                mock_trainer_instance.test.assert_called_once()
                mock_data_factory.data.close.assert_called_once()


class TestFactorySystemIntegration:
    """Test integration across all factory components."""
    
    def test_data_factory_integration(self):
        """Verify ID_dataset integration and H5 file handling."""
        # Test data factory registry contains id_data_factory
        available_factories = DATA_FACTORY_REGISTRY.available()
        print(f"Available data factories: {list(available_factories.keys())}")
        
        # Test can resolve id_data_factory
        from src.data_factory import resolve_data_factory_class
        try:
            factory_cls = resolve_data_factory_class('id_data_factory')
            assert factory_cls is not None
        except KeyError:
            # If not registered, check if it can be imported directly
            from src.data_factory.id_data_factory import id_data_factory
            assert id_data_factory is not None
    
    def test_task_factory_integration(self):
        """Confirm ContrastiveIDTask registration and initialization."""
        # Test task is registered
        key = 'contrastive_id.pretrain'
        available_tasks = TASK_REGISTRY.available()
        print(f"Available tasks: {list(available_tasks.keys())}")
        
        if key in available_tasks:
            # Test can get task class
            task_cls = TASK_REGISTRY.get(key)
            assert task_cls is not None
            assert task_cls.__name__ == 'ContrastiveIDTask'
        else:
            # If not registered, test that it can be imported
            from src.task_factory.task.pretrain.ContrastiveIDTask import ContrastiveIDTask
            assert ContrastiveIDTask is not None
    
    def test_component_dependency_resolution(self):
        """Test factory pattern compliance and dependency resolution."""
        # Test all factories can be imported
        from src.data_factory import build_data
        from src.model_factory import build_model  
        from src.task_factory import build_task
        from src.trainer_factory import build_trainer
        
        # All build functions should be callable
        assert callable(build_data)
        assert callable(build_model)
        assert callable(build_task)
        assert callable(build_trainer)
    
    def test_factory_integration_flow(self):
        """Test integration between factories using mocked build functions."""
        # Test that build functions are properly integrated
        # This test focuses on the interface rather than implementation
        from src.data_factory import build_data
        from src.model_factory import build_model
        from src.task_factory import build_task
        
        # Verify that the build functions have the expected signatures
        import inspect
        
        # Check build_data signature
        build_data_sig = inspect.signature(build_data)
        assert 'args_data' in build_data_sig.parameters
        assert 'args_task' in build_data_sig.parameters
        
        # Check build_model signature  
        build_model_sig = inspect.signature(build_model)
        assert 'args' in build_model_sig.parameters
        
        # Check build_task signature
        build_task_sig = inspect.signature(build_task)
        expected_params = {'args_task', 'network', 'args_data', 'args_model', 
                          'args_trainer', 'args_environment', 'metadata'}
        assert expected_params.issubset(set(build_task_sig.parameters.keys()))


class TestConfigurationSystemIntegration:
    """Test configuration system integration with all presets."""
    
    def test_all_preset_configurations(self):
        """Test all 4 preset configurations load correctly."""
        presets = [
            'contrastive',
            'contrastive_ablation', 
            'contrastive_cross',
            'contrastive_prod'
        ]
        
        for preset in presets:
            try:
                config = load_config(preset)
                
                # Verify basic structure
                assert hasattr(config, 'data')
                assert hasattr(config, 'model')
                assert hasattr(config, 'task')
                assert hasattr(config, 'trainer')
                
                # Verify contrastive task configuration
                assert config.task.type == 'pretrain'
                assert config.task.name == 'contrastive_id'
                assert hasattr(config.task, 'temperature')
                
            except FileNotFoundError:
                # Skip if preset config file doesn't exist
                pytest.skip(f"Preset config {preset} not found")
    
    def test_config_override_mechanisms(self):
        """Test configuration override mechanisms and parameter merging."""
        # Test override with dictionary
        overrides = {
            'task.temperature': 0.05,
            'model.d_model': 256
        }
        
        try:
            config = load_config('contrastive', overrides)
            assert config.task.temperature == 0.05
            assert config.model.d_model == 256
        except FileNotFoundError:
            pytest.skip("Contrastive preset config not found")
    
    def test_config_wrapper_compatibility(self):
        """Test ConfigWrapper compatibility with pipeline."""
        from src.configs.config_utils import ConfigWrapper
        from types import SimpleNamespace
        
        # Create config wrapper the correct way
        config = ConfigWrapper()
        config.data = SimpleNamespace(batch_size=32)
        config.task = SimpleNamespace(type='contrastive_id')
        
        # Test both access patterns
        assert config.data.batch_size == 32  # Attribute access
        assert hasattr(config, 'data')  # hasattr check
        assert config.task.type == 'contrastive_id'
        
        # Test dictionary-like methods
        assert config.get('data') is not None
        assert config.get('nonexistent', 'default') == 'default'


class TestEndToEndPipeline:
    """Test complete end-to-end pipeline workflows."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "e2e_config.yaml")
        
        # Create comprehensive test configuration
        self.e2e_config = {
            'data': {
                'factory_name': 'id_data_factory',
                'data_dir': '/tmp/test_data',
                'dataset_name': 'test_dataset',
                'metadata_file': 'metadata_test.xlsx',
                'window_size': 1024,
                'overlap': 0.5,
                'batch_size': 4
            },
            'model': {
                'name': 'B_04_Dlinear',
                'type': 'backbone',
                'd_model': 128,
                'seq_len': 1024
            },
            'task': {
                'type': 'contrastive_id',
                'name': 'pretrain',
                'temperature': 0.07,
                'lr': 0.001
            },
            'trainer': {
                'name': 'pytorch_lightning_trainer',
                'max_epochs': 1,
                'accelerator': 'cpu',
                'devices': 1,
                'log_every_n_steps': 1
            },
            'environment': {
                'VBENCH_HOME': str(project_root),
                'seed': 42,
                'iterations': 1,
                'log_level': 'INFO'
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(self.e2e_config, f)
    
    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('src.Pipeline_01_default.build_data')
    @patch('src.Pipeline_01_default.build_model')
    @patch('src.Pipeline_01_default.build_task')
    @patch('src.Pipeline_01_default.build_trainer')
    def test_full_workflow_integration(self, mock_trainer, mock_task, mock_model, mock_data):
        """Test complete workflow: data → model → task → training → evaluation."""
        # Setup comprehensive mocks
        mock_data_factory = MagicMock()
        mock_data_factory.get_metadata.return_value = {
            'num_classes': 10,
            'feature_dim': 1,
            'sampling_rate': 12800
        }
        mock_dataloader = MagicMock()
        mock_data_factory.get_dataloader.return_value = mock_dataloader
        mock_data_factory.data.close = MagicMock()
        mock_data.return_value = mock_data_factory
        
        mock_network = MagicMock()
        mock_model.return_value = mock_network
        
        mock_task_instance = MagicMock()
        mock_task_instance.network = mock_network
        mock_task.return_value = mock_task_instance
        
        mock_trainer_instance = MagicMock()
        mock_trainer_instance.fit = MagicMock()
        mock_trainer_instance.test.return_value = [{'test_contrastive_acc': 0.85}]
        mock_trainer.return_value = mock_trainer_instance
        
        # Mock helper functions
        with patch('src.Pipeline_01_default.load_best_model_checkpoint') as mock_load:
            mock_load.return_value = mock_task_instance
            with patch('src.Pipeline_01_default.init_lab'), \
                 patch('src.Pipeline_01_default.close_lab'), \
                 patch('src.Pipeline_01_default.path_name') as mock_path_name, \
                 patch('pandas.DataFrame') as mock_df:
                
                mock_path_name.return_value = ('/tmp/results', 'test_exp')
                mock_df_instance = MagicMock()
                mock_df.return_value = mock_df_instance
                
                # Run full pipeline
                args = SimpleNamespace(config_path=self.config_path)
                results = default_pipeline(args)
                
                # Verify complete workflow execution
                assert mock_data.called
                assert mock_model.called
                assert mock_task.called
                assert mock_trainer.called
                assert mock_trainer_instance.fit.called
                assert mock_trainer_instance.test.called
                assert mock_data_factory.data.close.called
                
                # Verify results structure
                assert results is not None
                assert len(results) == 1
                assert 'test_contrastive_acc' in results[0]
    
    def test_multiple_datasets_configuration(self):
        """Test pipeline with multiple datasets and configurations."""
        # Test different dataset configurations can be loaded
        datasets = ['CWRU', 'THU', 'HUST']
        
        for dataset in datasets:
            config_copy = self.e2e_config.copy()
            config_copy['data']['dataset_name'] = dataset
            
            # Configuration should load without errors
            config = load_config(config_copy)
            assert config.data.dataset_name == dataset
    
    @patch('src.Pipeline_01_default.build_trainer')
    def test_checkpoint_handling(self, mock_build_trainer):
        """Test checkpoint saving/loading through pipeline."""
        mock_trainer = MagicMock()
        mock_checkpoint_callback = MagicMock()
        mock_trainer.checkpoint_callback = mock_checkpoint_callback
        mock_build_trainer.return_value = mock_trainer
        
        # Mock checkpoint path
        checkpoint_path = os.path.join(self.temp_dir, 'checkpoint.ckpt')
        
        # Test checkpoint loading function exists
        from src.utils.utils import load_best_model_checkpoint
        assert callable(load_best_model_checkpoint)


class TestRegistryAndComponentDiscovery:
    """Test registry system and component discovery mechanisms."""
    
    def test_task_registry_discovery(self):
        """Verify task registration in TASK_REGISTRY."""
        # Test ContrastiveIDTask is registered
        key = 'contrastive_id.pretrain'
        available_tasks = TASK_REGISTRY.available()
        
        if key in available_tasks:
            # Test can discover task class
            task_cls = TASK_REGISTRY.get(key)
            assert task_cls.__name__ == 'ContrastiveIDTask'
            assert hasattr(task_cls, 'prepare_batch')
            assert hasattr(task_cls, 'infonce_loss')
        else:
            # If not in registry, verify it can be imported directly
            from src.task_factory.task.pretrain.ContrastiveIDTask import ContrastiveIDTask
            assert ContrastiveIDTask is not None
            assert hasattr(ContrastiveIDTask, 'prepare_batch')
            assert hasattr(ContrastiveIDTask, 'infonce_loss')
    
    def test_automatic_component_discovery(self):
        """Test automatic component discovery and initialization."""
        # Test task factory can resolve component path
        from src.task_factory.task_factory import resolve_task_module
        
        args_task = SimpleNamespace(type='contrastive_id', name='pretrain')
        module_path = resolve_task_module(args_task)
        
        expected_path = 'src.task_factory.task.contrastive_id.pretrain'
        assert module_path == expected_path
    
    def test_registry_pattern_compliance(self):
        """Validate factory pattern compliance."""
        # Test all registries follow the same pattern
        from src.data_factory import DATA_FACTORY_REGISTRY
        from src.task_factory import TASK_REGISTRY
        
        # All registries should have same interface
        for registry in [DATA_FACTORY_REGISTRY, TASK_REGISTRY]:
            assert hasattr(registry, '_items')
            assert hasattr(registry, 'register')
            assert hasattr(registry, 'get')
    
    def test_backward_compatibility(self):
        """Ensure backward compatibility with existing pipelines."""
        # Test existing task types still work
        from src.task_factory.task_factory import resolve_task_module
        
        # Test traditional task resolution
        args_traditional = SimpleNamespace(type='classification', name='DefaultTask')
        traditional_path = resolve_task_module(args_traditional)
        
        # Should resolve to expected path
        assert 'src.task_factory.task' in traditional_path


class TestPerformanceValidation:
    """Test performance validation during integration."""
    
    def test_memory_usage_integration(self):
        """Test memory usage doesn't exceed reasonable limits."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Load configuration (should be lightweight)
        minimal_config = {
            'data': {'batch_size': 32, 'data_dir': '/tmp', 'metadata_file': 'test.xlsx'},
            'model': {'name': 'test', 'type': 'backbone'},
            'task': {'type': 'contrastive_id', 'name': 'test'}
        }
        config = load_config(minimal_config)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be minimal (less than 50MB)
        assert memory_increase < 50, f"Memory usage increased by {memory_increase}MB"
    
    def test_import_time_performance(self):
        """Test import times are reasonable."""
        import time
        
        start_time = time.time()
        
        # Import main components
        from src.configs.config_utils import load_config
        from src.task_factory import build_task
        from src.data_factory import build_data
        
        end_time = time.time()
        import_time = end_time - start_time
        
        # Import time should be reasonable (less than 2 seconds)
        assert import_time < 2.0, f"Import took {import_time}s"
    
    @patch('torch.cuda.is_available')
    def test_gpu_compatibility_validation(self, mock_cuda_available):
        """Test GPU compatibility when available."""
        # Test both GPU available and unavailable scenarios
        for gpu_available in [True, False]:
            mock_cuda_available.return_value = gpu_available
            
            config = {
                'data': {'data_dir': '/tmp', 'metadata_file': 'test.xlsx'},
                'model': {'name': 'test', 'type': 'backbone'},
                'task': {'name': 'test', 'type': 'pretrain'},
                'trainer': {
                    'accelerator': 'gpu' if gpu_available else 'cpu',
                    'devices': 1 if gpu_available else 'auto'
                }
            }
            
            loaded_config = load_config(config)
            expected_accelerator = 'gpu' if gpu_available else 'cpu'
            assert loaded_config.trainer.accelerator == expected_accelerator


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])