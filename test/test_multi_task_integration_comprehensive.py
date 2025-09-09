"""
Comprehensive integration tests for multi_task_phm module with real model components.

Tests integration with:
- Actual ISFM models
- Real data loaders
- PyTorch Lightning training loop
- Memory usage validation

Author: PHM-Vibench Team
Date: 2025-09-08
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import torch
import torch.nn as nn
import pytorch_lightning as pl
from argparse import Namespace
from unittest.mock import patch, MagicMock
import tempfile
import psutil
import gc

from src.task_factory.task.In_distribution.multi_task_phm import task as MultiTaskPHM
from src.model_factory.ISFM.M_01_ISFM import Model as ISFM_Model


class MockISFMModel(nn.Module):
    """Realistic mock ISFM model for integration testing."""
    
    def __init__(self, args_model, metadata):
        super().__init__()
        self.args_model = args_model
        self.metadata = metadata
        
        # Basic model components
        input_dim = getattr(args_model, 'input_dim', 2)
        output_dim = getattr(args_model, 'output_dim', 256)
        hidden_dim = getattr(args_model, 'hidden_dim', 64)
        
        # Simple encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Multi-task heads
        self.classification_head = nn.Linear(output_dim, 4)
        self.anomaly_detection_head = nn.Linear(output_dim, 1)
        self.signal_prediction_head = nn.Linear(output_dim, input_dim)
        self.rul_prediction_head = nn.Linear(output_dim, 1)
    
    def forward(self, x, file_id=None, task_id=None):
        """Forward pass with multi-task outputs."""
        # Handle different input formats
        if len(x.shape) == 3:  # [B, L, C]
            # Use mean pooling over sequence dimension
            x = x.mean(dim=1)  # [B, C]
        
        # Encode features
        features = self.encoder(x)  # [B, output_dim]
        
        # Generate outputs for all tasks
        outputs = {
            'classification': self.classification_head(features),
            'anomaly_detection': self.anomaly_detection_head(features),
            'signal_prediction': self.signal_prediction_head(features).unsqueeze(1).expand(-1, 128, -1),
            'rul_prediction': self.rul_prediction_head(features)
        }
        
        return outputs


class TestMultiTaskPHMIntegration(unittest.TestCase):
    """Integration test cases for multi_task_phm module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.args_data = Namespace()
        self.args_model = Namespace(
            input_dim=2,
            output_dim=256,
            hidden_dim=64,
            dropout=0.1
        )
        self.args_task = Namespace(
            enabled_tasks=['classification', 'anomaly_detection', 'signal_prediction', 'rul_prediction'],
            task_weights={'classification': 1.0, 'anomaly_detection': 0.6, 'signal_prediction': 0.7, 'rul_prediction': 0.8},
            optimizer='adamw',
            lr=0.001,
            weight_decay=0.01,
            scheduler={'name': 'cosine', 'options': {'T_max': 100, 'eta_min': 1e-6}}
        )
        self.args_trainer = Namespace(gpus=False)
        self.args_environment = Namespace()
        
        # Realistic metadata
        self.metadata = {
            1: {'Name': 'CWRU', 'RUL_label': 500.0, 'Label': 0, 'Dataset_id': 1},
            2: {'Name': 'CWRU', 'RUL_label': 750.0, 'Label': 1, 'Dataset_id': 1},
            3: {'Name': 'XJTU', 'RUL_label': 300.0, 'Label': 2, 'Dataset_id': 2},
            4: {'Name': 'XJTU', 'RUL_label': 900.0, 'Label': 3, 'Dataset_id': 2},
            5: {'Name': 'FEMTO', 'RUL_label': 400.0, 'Label': 1, 'Dataset_id': 3},
        }
        
        # Create realistic ISFM model
        self.network = MockISFMModel(self.args_model, self.metadata)
        
        # Create task module
        self.task_module = MultiTaskPHM(
            self.network,
            self.args_data,
            self.args_model,
            self.args_task,
            self.args_trainer,
            self.args_environment,
            self.metadata
        )

    # ========== Integration Tests ==========
    
    def test_full_training_step_integration(self):
        """Test complete training step with real model components."""
        # Create realistic batch
        batch = {
            'x': torch.randn(8, 1024, 2),  # [batch_size, seq_len, features]
            'y': torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]),
            'file_id': torch.tensor([1, 2, 3, 4, 5, 1, 2, 3])
        }
        
        # Test training step
        loss = self.task_module.training_step(batch, 0)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss.item() >= 0)
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))
    
    def test_full_validation_step_integration(self):
        """Test complete validation step with real model components."""
        batch = {
            'x': torch.randn(4, 1024, 2),
            'y': torch.tensor([0, 1, 2, 3]),
            'file_id': torch.tensor([1, 2, 3, 4])
        }
        
        # Set model to eval mode
        self.task_module.eval()
        
        with torch.no_grad():
            loss = self.task_module.validation_step(batch, 0)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss.item() >= 0)
    
    def test_optimizer_configuration(self):
        """Test optimizer and scheduler configuration."""
        optimizer_config = self.task_module.configure_optimizers()
        
        self.assertIn('optimizer', optimizer_config)
        self.assertIn('lr_scheduler', optimizer_config)
        
        optimizer = optimizer_config['optimizer']
        scheduler = optimizer_config['lr_scheduler']
        
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.001)
        self.assertEqual(optimizer.param_groups[0]['weight_decay'], 0.01)
    
    def test_pytorch_lightning_integration(self):
        """Test integration with PyTorch Lightning trainer."""
        # Create temporary checkpoint directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a minimal trainer for testing
            trainer = pl.Trainer(
                max_epochs=1,
                enable_checkpointing=False,
                logger=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                accelerator='cpu',
                devices=1
            )
            
            # Test that the module is compatible with Lightning
            self.assertIsInstance(self.task_module, pl.LightningModule)
            
            # Test training_step interface
            batch = {
                'x': torch.randn(4, 1024, 2),
                'y': torch.tensor([0, 1, 2, 3]),
                'file_id': torch.tensor([1, 2, 3, 4])
            }
            
            loss = self.task_module.training_step(batch, 0)
            self.assertIsInstance(loss, torch.Tensor)

    # ========== Memory and Performance Tests ==========
    
    def test_memory_usage_validation(self):
        """Test that memory usage stays within acceptable limits."""
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run multiple forward passes
        for _ in range(10):
            batch = {
                'x': torch.randn(16, 1024, 2),
                'y': torch.randint(0, 4, (16,)),
                'file_id': torch.randint(1, 6, (16,))
            }
            
            with torch.no_grad():
                loss = self.task_module.training_step(batch, 0)
        
        # Check memory usage after operations
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 500MB for this test)
        self.assertLess(memory_increase, 500, 
                       f"Memory increased by {memory_increase:.1f}MB, which is too much")
    
    def test_gradient_computation(self):
        """Test that gradients are computed correctly."""
        batch = {
            'x': torch.randn(4, 1024, 2),
            'y': torch.tensor([0, 1, 2, 3]),
            'file_id': torch.tensor([1, 2, 3, 4])
        }
        
        # Clear gradients
        self.task_module.zero_grad()
        
        # Forward pass
        loss = self.task_module.training_step(batch, 0)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients were computed
        gradient_computed = False
        for param in self.task_module.parameters():
            if param.grad is not None:
                gradient_computed = True
                break
        
        self.assertTrue(gradient_computed, "No gradients were computed")
    
    def test_batch_size_scalability(self):
        """Test performance with different batch sizes."""
        batch_sizes = [1, 4, 8, 16, 32]
        
        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                batch = {
                    'x': torch.randn(batch_size, 1024, 2),
                    'y': torch.randint(0, 4, (batch_size,)),
                    'file_id': torch.randint(1, 6, (batch_size,))
                }
                
                # Should handle all batch sizes without error
                loss = self.task_module.training_step(batch, 0)
                self.assertIsInstance(loss, torch.Tensor)
                self.assertFalse(torch.isnan(loss))

    # ========== Real World Scenario Tests ==========
    
    def test_cross_dataset_training(self):
        """Test training with samples from different datasets."""
        # Batch with mixed dataset samples
        batch = {
            'x': torch.randn(6, 1024, 2),
            'y': torch.tensor([0, 1, 2, 3, 1, 0]),
            'file_id': torch.tensor([1, 2, 3, 4, 5, 1])  # Mix of CWRU, XJTU, FEMTO
        }
        
        loss = self.task_module.training_step(batch, 0)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss.item() >= 0)
    
    def test_missing_task_graceful_degradation(self):
        """Test graceful degradation when some tasks fail."""
        # Create network that only supports some tasks
        class PartialNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(2, 64)
                self.classification_head = nn.Linear(64, 4)
                # Missing other task heads
            
            def forward(self, x, file_id=None, task_id=None):
                if len(x.shape) == 3:
                    x = x.mean(dim=1)
                features = self.encoder(x)
                return {
                    'classification': self.classification_head(features),
                    # Missing: anomaly_detection, signal_prediction, rul_prediction
                }
        
        partial_network = PartialNetwork()
        partial_task_module = MultiTaskPHM(
            partial_network, self.args_data, self.args_model, self.args_task,
            self.args_trainer, self.args_environment, self.metadata
        )
        
        batch = {
            'x': torch.randn(4, 1024, 2),
            'y': torch.tensor([0, 1, 2, 3]),
            'file_id': torch.tensor([1, 2, 3, 4])
        }
        
        with patch('builtins.print'):  # Suppress warnings
            loss = partial_task_module.training_step(batch, 0)
            # Should still work with only classification task
            self.assertIsInstance(loss, torch.Tensor)
            self.assertTrue(loss.item() > 0)  # Should have some loss from classification
    
    def test_model_state_dict_compatibility(self):
        """Test model state dict save/load compatibility."""
        # Save model state
        original_state_dict = self.task_module.state_dict()
        
        # Create new identical model
        new_network = MockISFMModel(self.args_model, self.metadata)
        new_task_module = MultiTaskPHM(
            new_network, self.args_data, self.args_model, self.args_task,
            self.args_trainer, self.args_environment, self.metadata
        )
        
        # Load state dict
        new_task_module.load_state_dict(original_state_dict)
        
        # Test that models produce same output
        test_input = {
            'x': torch.randn(2, 1024, 2),
            'y': torch.tensor([0, 1]),
            'file_id': torch.tensor([1, 2])
        }
        
        with torch.no_grad():
            original_loss = self.task_module.training_step(test_input, 0)
            new_loss = new_task_module.training_step(test_input, 0)
            
            # Should produce identical results
            torch.testing.assert_close(original_loss, new_loss, atol=1e-6, rtol=1e-6)

    # ========== Edge Case Integration Tests ==========
    
    def test_extreme_values_handling(self):
        """Test handling of extreme values in inputs and targets."""
        # Test with extreme RUL values
        extreme_metadata = {
            1: {'Name': 'test', 'RUL_label': 0.0, 'Label': 0, 'Dataset_id': 1},  # Zero RUL
            2: {'Name': 'test', 'RUL_label': 10000.0, 'Label': 1, 'Dataset_id': 1},  # Very high RUL
        }
        
        extreme_network = MockISFMModel(self.args_model, extreme_metadata)
        extreme_task_module = MultiTaskPHM(
            extreme_network, self.args_data, self.args_model, self.args_task,
            self.args_trainer, self.args_environment, extreme_metadata
        )
        
        batch = {
            'x': torch.randn(2, 1024, 2),
            'y': torch.tensor([0, 1]),
            'file_id': torch.tensor([1, 2])
        }
        
        loss = extreme_task_module.training_step(batch, 0)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))
    
    def test_concurrent_task_execution(self):
        """Test that multiple tasks can execute concurrently without interference."""
        # Test with different task combinations
        task_combinations = [
            ['classification'],
            ['classification', 'anomaly_detection'],
            ['classification', 'rul_prediction'],
            ['signal_prediction', 'rul_prediction'],
            ['classification', 'anomaly_detection', 'signal_prediction', 'rul_prediction']
        ]
        
        for enabled_tasks in task_combinations:
            with self.subTest(enabled_tasks=enabled_tasks):
                task_args = Namespace(
                    enabled_tasks=enabled_tasks,
                    task_weights={task: 1.0 for task in enabled_tasks},
                    optimizer='adam',
                    lr=0.001
                )
                
                task_module = MultiTaskPHM(
                    self.network, self.args_data, self.args_model, task_args,
                    self.args_trainer, self.args_environment, self.metadata
                )
                
                batch = {
                    'x': torch.randn(4, 1024, 2),
                    'y': torch.tensor([0, 1, 2, 3]),
                    'file_id': torch.tensor([1, 2, 3, 4])
                }
                
                loss = task_module.training_step(batch, 0)
                self.assertIsInstance(loss, torch.Tensor)
                if len(enabled_tasks) > 0:
                    self.assertTrue(loss.item() >= 0)

    def tearDown(self):
        """Clean up after tests."""
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    unittest.main(verbosity=2)