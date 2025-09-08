"""
End-to-end integration tests for multi-task PHM training pipeline.

Tests the complete workflow from configuration loading through model training
to ensure all components work together correctly after the OOM fixes.

Author: PHM-Vibench Team  
Date: 2025-09-08
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import torch
import torch.nn as nn
import tempfile
import shutil
from pathlib import Path
from argparse import Namespace
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from src.configs import load_config
from src.model_factory.model_factory import model_factory
from src.task_factory.task.In_distribution.multi_task_phm import task as MultiTaskPHM


class MockDataLoader:
    """Mock dataloader for testing."""
    def __init__(self, batch_size=4, num_batches=3):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.current_batch = 0
        
    def __iter__(self):
        self.current_batch = 0
        return self
        
    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration
            
        self.current_batch += 1
        
        # Return batch in expected format: {'x': tensor, 'y': tensor, 'file_id': tensor}
        batch = {
            'x': torch.randn(self.batch_size, 4096, 2),  # [B, L, C]
            'y': torch.randint(0, 4, (self.batch_size,)),  # Classification labels
            'file_id': torch.tensor([1, 2, 3, 4])  # File IDs for metadata lookup
        }
        return batch
    
    def __len__(self):
        return self.num_batches


class MockMetadata:
    """Mock metadata object with required structure."""
    def __init__(self):
        # Create mock DataFrame for get_num_classes
        self.df = pd.DataFrame({
            'Dataset_id': [1, 1, 2, 2],
            'Label': [0, 1, 2, 3],
            'Name': ['dataset1', 'dataset1', 'dataset2', 'dataset2'],
            'Sample_rate': [12000, 12000, 12000, 12000]
        })
        
        # Create mock metadata dictionary for task module
        self.metadata_dict = {
            1: {'Name': 'dataset1', 'Label': 0, 'RUL_label': 500.0, 'Sample_rate': 12000},
            2: {'Name': 'dataset1', 'Label': 1, 'RUL_label': 750.0, 'Sample_rate': 12000},
            3: {'Name': 'dataset2', 'Label': 2, 'RUL_label': 300.0, 'Sample_rate': 12000},
            4: {'Name': 'dataset2', 'Label': 3, 'RUL_label': 900.0, 'Sample_rate': 12000},
        }
        
        # Also make this object behave as the metadata dict for model access
        self.update(self.metadata_dict)
        
    def get(self, key, default=None):
        return self.metadata_dict.get(key, default)
    
    def __getitem__(self, key):
        return self.metadata_dict[key]
    
    def __contains__(self, key):
        return key in self.metadata_dict
    
    def update(self, other):
        """Allow this object to be used as the metadata directly"""
        for key, value in other.items():
            setattr(self, f'_item_{key}', value)
    
    def values(self):
        return self.metadata_dict.values()


class TestEndToEndIntegration(unittest.TestCase):
    """End-to-end integration tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config_file = "script/Vibench_paper/foundation_model/multitask_B_08_PatchTST.yaml"
        self.temp_dir = tempfile.mkdtemp()
        self.mock_metadata = MockMetadata()
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_config_to_model_pipeline(self):
        """Test the complete pipeline from config to model creation."""
        # Load configuration
        config = load_config(self.config_file)
        # Config can be dict or ConfigWrapper
        self.assertTrue(hasattr(config, '__getitem__') or isinstance(config, dict))
        self.assertIn('model', config)
        self.assertIn('task', config)
        
        # Convert to args
        args_model = Namespace(**dict(config['model']))
        
        # Create model
        model = model_factory(args_model, self.mock_metadata)
        self.assertIsNotNone(model)
        
        # Check model has parameters
        total_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(total_params, 1000, "Model should have reasonable number of parameters")
        
        # Check model can do forward pass (use single file_id, not list)
        x = torch.randn(2, 4096, 2)  # [B, L, C]
        with torch.no_grad():
            output = model(x, file_id=1, task_id=['classification', 'rul_prediction'])
        
        # Check output structure
        self.assertIsNotNone(output)
        print(f"Model created successfully with {total_params:,} parameters")
    
    def test_config_to_task_module_pipeline(self):
        """Test the complete pipeline from config to task module creation."""
        # Load configuration
        config = load_config(self.config_file)
        
        # Convert to args objects
        args_model = Namespace(**dict(config['model']))
        args_data = Namespace(**dict(config.get('data', {})))
        args_task = Namespace(**dict(config.get('task', {})))
        args_trainer = Namespace(**dict(config.get('trainer', {})))
        args_environment = Namespace(**dict(config.get('environment', {})))
        
        # Create network
        network = model_factory(args_model, self.mock_metadata)
        
        # Create task module
        task_module = MultiTaskPHM(
            network, args_data, args_model, args_task, 
            args_trainer, args_environment, self.mock_metadata.metadata_dict
        )
        
        # Check task module properties
        self.assertIsNotNone(task_module)
        self.assertIsInstance(task_module.enabled_tasks, list)
        self.assertIsInstance(task_module.task_weights, dict)
        self.assertIsInstance(task_module.task_loss_fns, dict)
        self.assertIsInstance(task_module.task_metrics, dict)
        
        print(f"Task module created with {len(task_module.enabled_tasks)} enabled tasks")
    
    def test_training_step_execution(self):
        """Test that a training step can execute without errors."""
        # Load configuration
        config = load_config(self.config_file)
        
        # Convert to args objects
        args_model = Namespace(**dict(config['model']))
        args_data = Namespace(**dict(config.get('data', {})))
        args_task = Namespace(**dict(config.get('task', {})))
        args_trainer = Namespace(gpus=False)  # Force CPU
        args_environment = Namespace(**dict(config.get('environment', {})))
        
        # Create network and task module
        network = model_factory(args_model, self.mock_metadata)
        task_module = MultiTaskPHM(
            network, args_data, args_model, args_task, 
            args_trainer, args_environment, self.mock_metadata.metadata_dict
        )
        
        # Create mock batch
        batch = {
            'x': torch.randn(2, 4096, 2),  # [B, L, C]
            'y': torch.tensor([0, 1]),     # Classification labels
            'file_id': torch.tensor([1, 2])  # File IDs
        }
        
        # Execute training step
        with torch.no_grad():  # Avoid gradient computation for test
            loss = task_module.training_step(batch, batch_idx=0)
        
        # Check loss is valid
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Scalar loss
        self.assertFalse(torch.isnan(loss), "Loss should not be NaN")
        self.assertFalse(torch.isinf(loss), "Loss should not be infinite")
        
        print(f"Training step executed successfully with loss: {loss.item():.4f}")
    
    def test_validation_step_execution(self):
        """Test that a validation step can execute without errors."""
        # Load configuration
        config = load_config(self.config_file)
        
        # Convert to args objects
        args_model = Namespace(**dict(config['model']))
        args_data = Namespace(**dict(config.get('data', {})))
        args_task = Namespace(**dict(config.get('task', {})))
        args_trainer = Namespace(gpus=False)  # Force CPU
        args_environment = Namespace(**dict(config.get('environment', {})))
        
        # Create network and task module
        network = model_factory(args_model, self.mock_metadata)
        task_module = MultiTaskPHM(
            network, args_data, args_model, args_task, 
            args_trainer, args_environment, self.mock_metadata.metadata_dict
        )
        
        # Create mock batch
        batch = {
            'x': torch.randn(2, 4096, 2),  # [B, L, C]
            'y': torch.tensor([2, 3]),     # Classification labels
            'file_id': torch.tensor([3, 4])  # File IDs
        }
        
        # Execute validation step
        with torch.no_grad():
            loss = task_module.validation_step(batch, batch_idx=0)
        
        # Check loss is valid
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Scalar loss
        self.assertFalse(torch.isnan(loss), "Validation loss should not be NaN")
        self.assertFalse(torch.isinf(loss), "Validation loss should not be infinite")
        
        print(f"Validation step executed successfully with loss: {loss.item():.4f}")
    
    def test_multi_batch_training(self):
        """Test training on multiple batches to ensure stability."""
        # Load configuration
        config = load_config(self.config_file)
        
        # Convert to args objects
        args_model = Namespace(**dict(config['model']))
        args_data = Namespace(**dict(config.get('data', {})))
        args_task = Namespace(**dict(config.get('task', {})))
        args_trainer = Namespace(gpus=False)
        args_environment = Namespace(**dict(config.get('environment', {})))
        
        # Create network and task module
        network = model_factory(args_model, self.mock_metadata)
        task_module = MultiTaskPHM(
            network, args_data, args_model, args_task, 
            args_trainer, args_environment, self.mock_metadata.metadata_dict
        )
        
        # Create mock dataloader
        dataloader = MockDataLoader(batch_size=2, num_batches=5)
        
        losses = []
        
        # Simulate mini training loop
        for batch_idx, batch in enumerate(dataloader):
            with torch.no_grad():
                loss = task_module.training_step(batch, batch_idx)
                losses.append(loss.item())
        
        # Check all losses are valid
        self.assertEqual(len(losses), 5, "Should have 5 losses from 5 batches")
        for i, loss in enumerate(losses):
            self.assertFalse(np.isnan(loss), f"Loss at batch {i} should not be NaN: {loss}")
            self.assertFalse(np.isinf(loss), f"Loss at batch {i} should not be infinite: {loss}")
            self.assertGreater(loss, 0, f"Loss at batch {i} should be positive: {loss}")
        
        print(f"Multi-batch training successful. Losses: {[f'{l:.4f}' for l in losses]}")
    
    def test_task_specific_outputs(self):
        """Test that all task-specific outputs are generated correctly."""
        # Load configuration
        config = load_config(self.config_file)
        
        # Convert to args objects
        args_model = Namespace(**dict(config['model']))
        args_data = Namespace(**dict(config.get('data', {})))
        args_task = Namespace(**dict(config.get('task', {})))
        args_trainer = Namespace(gpus=False)
        args_environment = Namespace(**dict(config.get('environment', {})))
        
        # Create network and task module
        network = model_factory(args_model, self.mock_metadata)
        task_module = MultiTaskPHM(
            network, args_data, args_model, args_task, 
            args_trainer, args_environment, self.mock_metadata.metadata_dict
        )
        
        # Test each enabled task
        enabled_tasks = task_module.enabled_tasks
        self.assertGreater(len(enabled_tasks), 1, "Should have multiple enabled tasks")
        
        x = torch.randn(1, 4096, 2)
        
        for task_name in enabled_tasks:
            with torch.no_grad():
                # Test network output for specific task
                output = network(x, file_id=1, task_id=task_name)
                self.assertIsNotNone(output)
                
                # Check that task-specific output exists
                task_output = getattr(output, f'{task_name}_logits', None)
                if task_output is None:
                    task_output = getattr(output, task_name, output)
                
                self.assertIsNotNone(task_output, f"Should have output for task {task_name}")
                self.assertIsInstance(task_output, torch.Tensor)
                
                print(f"Task {task_name}: output shape {task_output.shape}")
    
    def test_memory_usage_stability(self):
        """Test that memory usage remains stable during training."""
        # Load configuration
        config = load_config(self.config_file)
        
        # Convert to args objects
        args_model = Namespace(**dict(config['model']))
        args_data = Namespace(**dict(config.get('data', {})))
        args_task = Namespace(**dict(config.get('task', {})))
        args_trainer = Namespace(gpus=False)
        args_environment = Namespace(**dict(config.get('environment', {})))
        
        # Create network and task module
        network = model_factory(args_model, self.mock_metadata)
        task_module = MultiTaskPHM(
            network, args_data, args_model, args_task, 
            args_trainer, args_environment, self.mock_metadata.metadata_dict
        )
        
        # Monitor memory usage during training
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run multiple training steps
        for i in range(10):
            batch = {
                'x': torch.randn(4, 4096, 2),
                'y': torch.randint(0, 4, (4,)),
                'file_id': torch.tensor([1, 2, 3, 4])
            }
            
            with torch.no_grad():
                loss = task_module.training_step(batch, batch_idx=i)
            
            # Force garbage collection
            gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 500MB for this test)
        self.assertLess(memory_increase, 500, 
                       f"Memory increase {memory_increase:.1f}MB should be reasonable")
        
        print(f"Memory usage: {initial_memory:.1f}MB → {final_memory:.1f}MB "
              f"(+{memory_increase:.1f}MB)")
    
    def test_configuration_variations(self):
        """Test different configuration variations work correctly."""
        configs_to_test = [
            "script/Vibench_paper/foundation_model/multitask_B_04_Dlinear.yaml",
            "script/Vibench_paper/foundation_model/multitask_B_06_TimesNet.yaml",
            "script/Vibench_paper/foundation_model/multitask_B_09_FNO.yaml"
        ]
        
        for config_file in configs_to_test:
            if not os.path.exists(config_file):
                print(f"Skipping {config_file} (not found)")
                continue
                
            with self.subTest(config=config_file):
                # Load configuration
                config = load_config(config_file)
                
                # Convert to args
                args_model = Namespace(**dict(config['model']))
                args_task = Namespace(**dict(config.get('task', {})))
                args_trainer = Namespace(gpus=False)
                
                # Create model
                model = model_factory(args_model, self.mock_metadata)
                
                # Test basic forward pass
                x = torch.randn(1, 4096, 2)
                with torch.no_grad():
                    output = model(x, file_id=1, task_id=['classification'])
                
                self.assertIsNotNone(output)
                print(f"Configuration {Path(config_file).name} tested successfully")


if __name__ == '__main__':
    unittest.main()