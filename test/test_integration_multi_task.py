"""
Integration test for the complete multi-task PHM foundation model pipeline.

This test validates the integration between the multi-task head, ISFM backbone,
and the overall PHM-Vibench framework without requiring the full environment.

Author: PHM-Vibench Team
Date: 2025-08-18
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import yaml
from argparse import Namespace
from pathlib import Path


class MockISFMBackbone(nn.Module):
    """Mock ISFM backbone for integration testing."""
    
    def __init__(self, output_dim=1024):
        super().__init__()
        self.output_dim = output_dim
        self.backbone = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim)
        )
    
    def forward(self, x):
        # Simulate backbone processing
        if x.dim() == 3:
            x = x.mean(dim=1)  # Average pooling
        return self.backbone(x)


class MockISFMModel(nn.Module):
    """Mock complete ISFM model for integration testing."""
    
    def __init__(self, args_m, metadata):
        super().__init__()
        self.args_m = args_m
        self.metadata = metadata
        
        # Mock backbone
        self.backbone = MockISFMBackbone(args_m.output_dim)
        
        # Import and create the actual multi-task head
        from src.model_factory.ISFM.task_head.multi_task_head import MultiTaskHead
        self.task_head = MultiTaskHead(args_m)
        
        # Mock number of classes
        self.num_classes = getattr(args_m, 'num_classes', {})
    
    def forward(self, x, file_id=None, task_id=None, return_feature=False):
        """Forward pass through the complete model."""
        # Backbone processing
        features = self.backbone(x)
        
        # Task head processing
        if task_id == 'all':
            # For integration test, we need to determine system_id from file_id
            system_id = self._get_system_id_from_file_id(file_id)
            return self.task_head(features, system_id=system_id, task_id=task_id, return_feature=return_feature)
        else:
            system_id = self._get_system_id_from_file_id(file_id) if task_id == 'classification' else None
            return self.task_head(features, system_id=system_id, task_id=task_id, return_feature=return_feature)
    
    def _get_system_id_from_file_id(self, file_id):
        """Mock function to get system_id from file_id."""
        if file_id is None:
            return 'system1'
        
        # Mock metadata lookup
        if isinstance(file_id, torch.Tensor):
            file_id = file_id.item() if file_id.numel() == 1 else file_id[0].item()
        
        # Simple mapping for testing
        system_mapping = {0: 'system1', 1: 'system2', 2: 'system3'}
        return system_mapping.get(file_id, 'system1')


def load_config():
    """Load the multi-task configuration file."""
    config_path = Path("configs/multi_task_config.yaml")
    
    if not config_path.exists():
        print(f"Warning: Config file {config_path} not found. Using default config.")
        return create_default_config()
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config: {e}. Using default config.")
        return create_default_config()


def create_default_config():
    """Create a default configuration for testing."""
    return {
        'model': {
            'output_dim': 1024,
            'hidden_dim': 512,
            'dropout': 0.1,
            'activation': 'gelu',
            'rul_max_value': 2000.0,
            'use_batch_norm': True
        },
        'task': {
            'enabled_tasks': ['classification', 'rul_prediction', 'anomaly_detection'],
            'task_weights': {
                'classification': 1.0,
                'rul_prediction': 0.8,
                'anomaly_detection': 0.6
            }
        }
    }


def create_mock_metadata():
    """Create mock metadata for testing."""
    return {
        0: {'Name': 'dataset1', 'Label': 4, 'Dataset_id': 'system1'},
        1: {'Name': 'dataset2', 'Label': 2, 'Dataset_id': 'system2'},
        2: {'Name': 'dataset3', 'Label': 6, 'Dataset_id': 'system3'}
    }


def create_mock_batch():
    """Create a mock batch for testing."""
    batch_size = 8
    seq_len = 256
    feature_dim = 512
    
    return {
        'x': torch.randn(batch_size, seq_len, feature_dim),
        'file_id': torch.randint(0, 3, (batch_size,)),
        'y': torch.randint(0, 5, (batch_size,)),
        'rul': torch.randn(batch_size).abs() * 1000,  # Positive RUL values
        'anomaly': torch.randint(0, 2, (batch_size,)).float()
    }


def test_model_creation():
    """Test that the complete model can be created successfully."""
    print("Testing model creation...")
    
    # Load configuration
    config = load_config()
    
    # Create model arguments
    args_m = Namespace(
        output_dim=config['model']['output_dim'],
        hidden_dim=config['model']['hidden_dim'],
        dropout=config['model']['dropout'],
        activation=config['model']['activation'],
        rul_max_value=config['model']['rul_max_value'],
        use_batch_norm=config['model']['use_batch_norm'],
        num_classes={'system1': 5, 'system2': 3, 'system3': 7}
    )
    
    # Create metadata
    metadata = create_mock_metadata()
    
    # Create model
    try:
        model = MockISFMModel(args_m, metadata)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✓ Model created successfully with {param_count:,} parameters")
        return model, args_m, metadata
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return None, None, None


def test_forward_pass(model):
    """Test forward pass through the complete model."""
    print("\nTesting forward pass...")
    
    # Create mock batch
    batch = create_mock_batch()
    
    try:
        # Test individual tasks
        print("  Testing individual tasks...")
        
        # Classification
        cls_output = model(batch['x'], file_id=batch['file_id'][0], task_id='classification')
        print(f"    ✓ Classification output shape: {cls_output.shape}")
        
        # RUL prediction
        rul_output = model(batch['x'], task_id='rul_prediction')
        print(f"    ✓ RUL prediction output shape: {rul_output.shape}")
        assert torch.all(rul_output >= 0), "RUL values should be non-negative"
        
        # Anomaly detection
        anomaly_output = model(batch['x'], task_id='anomaly_detection')
        print(f"    ✓ Anomaly detection output shape: {anomaly_output.shape}")
        
        # All tasks
        print("  Testing all tasks simultaneously...")
        all_outputs = model(batch['x'], file_id=batch['file_id'][0], task_id='all')
        assert isinstance(all_outputs, dict), "All tasks output should be a dictionary"
        assert 'classification' in all_outputs, "Missing classification output"
        assert 'rul_prediction' in all_outputs, "Missing RUL prediction output"
        assert 'anomaly_detection' in all_outputs, "Missing anomaly detection output"
        print(f"    ✓ All tasks output keys: {list(all_outputs.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False


def test_loss_computation(model):
    """Test loss computation for multi-task training."""
    print("\nTesting loss computation...")
    
    try:
        # Create mock batch
        batch = create_mock_batch()
        
        # Forward pass
        outputs = model(batch['x'], file_id=batch['file_id'][0], task_id='all')
        
        # Define loss functions
        cls_loss_fn = nn.CrossEntropyLoss()
        rul_loss_fn = nn.MSELoss()
        anom_loss_fn = nn.BCEWithLogitsLoss()
        
        # Compute individual losses
        cls_loss = cls_loss_fn(outputs['classification'], batch['y'])
        rul_loss = rul_loss_fn(outputs['rul_prediction'].squeeze(), batch['rul'])
        anom_loss = anom_loss_fn(outputs['anomaly_detection'].squeeze(), batch['anomaly'])
        
        # Compute total loss with weights
        task_weights = {'classification': 1.0, 'rul_prediction': 0.8, 'anomaly_detection': 0.6}
        total_loss = (task_weights['classification'] * cls_loss + 
                     task_weights['rul_prediction'] * rul_loss + 
                     task_weights['anomaly_detection'] * anom_loss)
        
        print(f"  ✓ Classification loss: {cls_loss.item():.4f}")
        print(f"  ✓ RUL prediction loss: {rul_loss.item():.4f}")
        print(f"  ✓ Anomaly detection loss: {anom_loss.item():.4f}")
        print(f"  ✓ Total weighted loss: {total_loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        return False


def test_gradient_flow(model):
    """Test that gradients flow properly through the model."""
    print("\nTesting gradient flow...")
    
    try:
        # Create mock batch with gradients enabled
        batch = create_mock_batch()
        batch['x'].requires_grad_(True)
        
        # Forward pass
        outputs = model(batch['x'], file_id=batch['file_id'][0], task_id='all')
        
        # Compute a simple loss
        loss = (outputs['classification'].sum() + 
                outputs['rul_prediction'].sum() + 
                outputs['anomaly_detection'].sum())
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        assert batch['x'].grad is not None, "Input gradients should exist"
        assert torch.any(batch['x'].grad != 0), "Input gradients should be non-zero"
        
        # Check model parameter gradients
        param_with_grad = 0
        total_params = 0
        for param in model.parameters():
            total_params += 1
            if param.grad is not None:
                param_with_grad += 1
        
        print(f"  ✓ Parameters with gradients: {param_with_grad}/{total_params}")
        print(f"  ✓ Input gradient norm: {batch['x'].grad.norm().item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Gradient flow test failed: {e}")
        return False


def test_config_validation():
    """Test configuration file validation."""
    print("\nTesting configuration validation...")
    
    try:
        config = load_config()
        
        # Check required sections
        required_sections = ['model', 'task']
        for section in required_sections:
            assert section in config, f"Missing required section: {section}"
        
        # Check model configuration
        model_config = config['model']
        required_model_keys = ['output_dim', 'hidden_dim']
        for key in required_model_keys:
            assert key in model_config, f"Missing required model key: {key}"
        
        # Check task configuration
        task_config = config['task']
        assert 'enabled_tasks' in task_config, "Missing enabled_tasks in task config"
        assert isinstance(task_config['enabled_tasks'], list), "enabled_tasks should be a list"
        
        print("  ✓ Configuration validation passed")
        return True
        
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
        return False


def run_integration_tests():
    """Run all integration tests."""
    print("Multi-Task PHM Foundation Model Integration Tests")
    print("=" * 60)
    
    test_results = []
    
    # Test 1: Configuration validation
    test_results.append(test_config_validation())
    
    # Test 2: Model creation
    model, args_m, metadata = test_model_creation()
    if model is None:
        print("\n❌ Integration tests failed - could not create model")
        return False
    test_results.append(True)
    
    # Test 3: Forward pass
    test_results.append(test_forward_pass(model))
    
    # Test 4: Loss computation
    test_results.append(test_loss_computation(model))
    
    # Test 5: Gradient flow
    test_results.append(test_gradient_flow(model))
    
    # Summary
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print("\n" + "=" * 60)
    print(f"Integration Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("✅ All integration tests passed!")
        print("\n🎉 Multi-task PHM foundation model is ready for use!")
        return True
    else:
        print(f"❌ {total_tests - passed_tests} test(s) failed")
        return False


if __name__ == '__main__':
    success = run_integration_tests()
    exit(0 if success else 1)
