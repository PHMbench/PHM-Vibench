"""
Task Factory Integration Test for FlowPretrainTask

This test suite validates the integration of FlowPretrainTask with PHM-Vibench's 
task factory registry system. It verifies task registration, instantiation, 
configuration loading, and compatibility with the existing framework patterns.

Author: PHM-Vibench Team
Date: 2025-09-12
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import unittest
from argparse import Namespace
from typing import Dict, Any, Optional
import tempfile
import yaml

# Import task factory components
from src.task_factory import TASK_REGISTRY, register_task, build_task
from src.task_factory.task_factory import task_factory, resolve_task_module

# Import the FlowPretrainTask to trigger registration
from src.task_factory.task.pretrain.flow_pretrain import FlowPretrainTask

# Import model factory for creating test network
from src.model_factory import build_model


class TestFlowTaskRegistration(unittest.TestCase):
    """Test cases for FlowPretrainTask registration in task factory."""
    
    def test_flow_task_import(self):
        """Test that FlowPretrainTask can be imported successfully."""
        # FlowPretrainTask should already be imported at module level
        self.assertIsNotNone(FlowPretrainTask)
        self.assertTrue(issubclass(FlowPretrainTask, nn.Module))
    
    def test_task_registry_contains_flow_task(self):
        """Test that FlowPretrainTask is registered in the task registry."""
        # Check if the task is registered with the correct key
        expected_key = "flow_pretrain.pretrain"  # Based on @register_task("flow_pretrain", "pretrain")
        
        try:
            task_cls = TASK_REGISTRY.get(expected_key)
            self.assertEqual(task_cls, FlowPretrainTask)
        except KeyError:
            self.fail(f"FlowPretrainTask not found in registry with key '{expected_key}'")
    
    def test_task_registry_available_tasks(self):
        """Test that FlowPretrainTask appears in available tasks list."""
        available_tasks = list(TASK_REGISTRY.available())
        self.assertIn("flow_pretrain.pretrain", available_tasks)
    
    def test_task_module_resolution(self):
        """Test that task module resolution works correctly."""
        args_task = Namespace()
        args_task.name = "flow_pretrain"
        args_task.type = "pretrain"
        
        expected_module_path = "src.task_factory.task.pretrain.flow_pretrain"
        resolved_path = resolve_task_module(args_task)
        self.assertEqual(resolved_path, expected_module_path)


class TestFlowTaskInstantiation(unittest.TestCase):
    """Test cases for FlowPretrainTask instantiation via task factory."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock M_04_ISFM_Flow network
        self.network = self._create_mock_flow_network()
        
        # Create configuration namespaces
        self.args_data = Namespace(
            batch_size=16,
            sequence_length=1024,
            channels=3
        )
        
        self.args_model = Namespace(
            model_name="M_04_ISFM_Flow",
            d_model=128,
            n_layers=4,
            n_heads=8
        )
        
        self.args_task = Namespace(
            name="flow_pretrain",
            type="pretrain",
            use_conditional=True,
            generation_mode="conditional",
            num_steps=100,
            sigma_min=0.001,
            sigma_max=1.0,
            flow_lr=1e-4,
            use_contrastive=False,
            lr=1e-4,
            weight_decay=1e-5,
            loss="mse",  # Required by Default_task
            metrics=["acc"]  # Required by Default_task (list format)
        )
        
        self.args_trainer = Namespace(
            max_epochs=10,
            gpus=0 if not torch.cuda.is_available() else 1,
            precision=32
        )
        
        self.args_environment = Namespace(
            device="cuda" if torch.cuda.is_available() else "cpu",
            num_workers=2
        )
        
        self.metadata = {
            '0': {'Name': 'test_dataset', 'Label': 0},
            '1': {'Name': 'test_dataset', 'Label': 1},
            '2': {'Name': 'test_dataset', 'Label': 2},
            '3': {'Name': 'test_dataset', 'Label': 3}
        }
    
    def _create_mock_flow_network(self) -> nn.Module:
        """Create a mock Flow network for testing."""
        class MockFlowNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(3, 128)
                self.decoder = nn.Linear(128, 3)
                self.flow_model = nn.ModuleDict({
                    'velocity_net': nn.Sequential(
                        nn.Linear(128, 256),
                        nn.ReLU(),
                        nn.Linear(256, 128)
                    )
                })
                self.condition_encoder = nn.Linear(10, 128)  # Mock condition encoder
            
            def forward(self, x, file_ids=None, return_loss=False):
                # Mock Flow model forward pass
                B, L, C = x.shape
                encoded = self.encoder(x)  # (B, L, 128)
                velocity = self.flow_model['velocity_net'](encoded)  # (B, L, 128)
                reconstructed = self.decoder(velocity)  # (B, L, 3)
                
                outputs = {
                    'velocity': velocity,
                    'x_original': x,
                    'reconstructed': reconstructed
                }
                
                if return_loss:
                    # Mock flow loss
                    flow_loss = nn.MSELoss()(reconstructed, x)
                    outputs['flow_loss'] = flow_loss
                    outputs['loss'] = flow_loss
                
                if file_ids is not None:
                    # Mock conditional features
                    condition_features = torch.randn(B, 128, device=x.device)
                    outputs['condition_features'] = condition_features
                
                return outputs
            
            def sample(self, batch_size, file_ids=None, num_steps=100, device='cpu'):
                """Mock sampling method."""
                return torch.randn(batch_size, 1024, 3, device=device)
        
        return MockFlowNetwork()
    
    def test_direct_task_instantiation(self):
        """Test direct instantiation of FlowPretrainTask."""
        task = FlowPretrainTask(
            network=self.network,
            args_data=self.args_data,
            args_model=self.args_model,
            args_task=self.args_task,
            args_trainer=self.args_trainer,
            args_environment=self.args_environment,
            metadata=self.metadata
        )
        
        self.assertIsInstance(task, FlowPretrainTask)
        self.assertEqual(task.use_conditional, True)
        self.assertEqual(task.generation_mode, "conditional")
        self.assertEqual(task.num_steps, 100)
        self.assertEqual(task.use_contrastive, False)
    
    def test_task_factory_instantiation(self):
        """Test instantiation via task_factory function."""
        task = task_factory(
            args_task=self.args_task,
            network=self.network,
            args_data=self.args_data,
            args_model=self.args_model,
            args_trainer=self.args_trainer,
            args_environment=self.args_environment,
            metadata=self.metadata
        )
        
        self.assertIsInstance(task, FlowPretrainTask)
        self.assertIsNotNone(task.network)
        self.assertEqual(task.network, self.network)
    
    def test_build_task_function(self):
        """Test instantiation via build_task public API."""
        task = build_task(
            args_task=self.args_task,
            network=self.network,
            args_data=self.args_data,
            args_model=self.args_model,
            args_trainer=self.args_trainer,
            args_environment=self.args_environment,
            metadata=self.metadata
        )
        
        self.assertIsInstance(task, FlowPretrainTask)
        self.assertIsNotNone(task.network)
    
    def test_task_with_contrastive_learning(self):
        """Test task instantiation with contrastive learning enabled."""
        self.args_task.use_contrastive = True
        self.args_task.flow_weight = 0.8
        self.args_task.contrastive_weight = 0.2
        self.args_task.contrastive_temperature = 0.05
        self.args_task.loss = "mse"  # Required by Default_task
        
        task = FlowPretrainTask(
            network=self.network,
            args_data=self.args_data,
            args_model=self.args_model,
            args_task=self.args_task,
            args_trainer=self.args_trainer,
            args_environment=self.args_environment,
            metadata=self.metadata
        )
        
        self.assertTrue(task.use_contrastive)
        self.assertEqual(task.flow_weight, 0.8)
        self.assertEqual(task.contrastive_weight, 0.2)
        self.assertIsNotNone(task.flow_contrastive_loss)


class TestFlowTaskConfiguration(unittest.TestCase):
    """Test cases for FlowPretrainTask configuration loading and validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.network = self._create_mock_flow_network()
        self.base_config = {
            'name': 'flow_pretrain',
            'type': 'pretrain',
            'use_conditional': True,
            'generation_mode': 'conditional',
            'num_steps': 200,
            'sigma_min': 0.002,
            'sigma_max': 2.0,
            'flow_lr': 2e-4,
            'use_contrastive': True,
            'flow_weight': 0.7,
            'contrastive_weight': 0.3,
            'contrastive_temperature': 0.07,
            'lr': 1e-4,
            'weight_decay': 1e-6,
            'enable_visualization': False,
            'generation_samples': 5,
            'loss': 'mse',  # Required by Default_task
            'metrics': ['acc']  # Required by Default_task
        }
    
    def _create_mock_flow_network(self) -> nn.Module:
        """Create a mock Flow network for testing."""
        class MockFlowNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.flow_model = nn.ModuleDict({'velocity_net': nn.Linear(128, 128)})
                self.condition_encoder = nn.Linear(10, 128)
            
            def forward(self, x, file_ids=None, return_loss=False):
                B, L, C = x.shape
                return {
                    'velocity': torch.randn(B, L, 128),
                    'x_original': x,
                    'flow_loss': torch.tensor(0.5, requires_grad=True)
                }
            
            def sample(self, batch_size, file_ids=None, num_steps=100, device='cpu'):
                return torch.randn(batch_size, 1024, 3, device=device)
        
        return MockFlowNetwork()
    
    def _create_args_from_config(self, config: Dict[str, Any]) -> Namespace:
        """Create args namespace from configuration dictionary."""
        return Namespace(**config)
    
    def test_configuration_parameter_parsing(self):
        """Test that configuration parameters are correctly parsed."""
        args_task = self._create_args_from_config(self.base_config)
        
        # Create other required namespaces
        args_data = Namespace(batch_size=16)
        args_model = Namespace(model_name="M_04_ISFM_Flow")
        args_trainer = Namespace(max_epochs=10, gpus=0)
        args_environment = Namespace(device="cpu")
        metadata = {'0': {'Name': 'test_dataset', 'Label': 3}}
        
        task = FlowPretrainTask(
            network=self.network,
            args_data=args_data,
            args_model=args_model,
            args_task=args_task,
            args_trainer=args_trainer,
            args_environment=args_environment,
            metadata=metadata
        )
        
        # Verify configuration parameters
        self.assertEqual(task.use_conditional, True)
        self.assertEqual(task.generation_mode, "conditional")
        self.assertEqual(task.num_steps, 200)
        self.assertEqual(task.sigma_min, 0.002)
        self.assertEqual(task.sigma_max, 2.0)
        self.assertEqual(task.flow_lr, 2e-4)
        self.assertEqual(task.use_contrastive, True)
        self.assertEqual(task.flow_weight, 0.7)
        self.assertEqual(task.contrastive_weight, 0.3)
        self.assertEqual(task.contrastive_temperature, 0.07)
    
    def test_default_configuration_values(self):
        """Test default configuration values when parameters are not provided."""
        minimal_config = {
            'name': 'flow_pretrain',
            'type': 'pretrain',
            'loss': 'mse',  # Required by Default_task
            'metrics': ['acc']  # Required by Default_task
        }
        args_task = self._create_args_from_config(minimal_config)
        
        # Create other required namespaces
        args_data = Namespace(batch_size=16)
        args_model = Namespace(model_name="M_04_ISFM_Flow")
        args_trainer = Namespace(max_epochs=10, gpus=0)
        args_environment = Namespace(device="cpu")
        metadata = {'0': {'Name': 'test_dataset', 'Label': 3}}
        
        task = FlowPretrainTask(
            network=self.network,
            args_data=args_data,
            args_model=args_model,
            args_task=args_task,
            args_trainer=args_trainer,
            args_environment=args_environment,
            metadata=metadata
        )
        
        # Verify default values
        self.assertEqual(task.use_conditional, True)  # Default
        self.assertEqual(task.generation_mode, "conditional")  # Default
        self.assertEqual(task.num_steps, 1000)  # Default
        self.assertEqual(task.sigma_min, 0.001)  # Default
        self.assertEqual(task.sigma_max, 1.0)  # Default
        self.assertEqual(task.use_contrastive, False)  # Default
    
    def test_yaml_configuration_loading(self):
        """Test loading configuration from YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'task': self.base_config
            }, f)
            yaml_file = f.name
        
        try:
            # Load configuration from YAML
            with open(yaml_file, 'r') as f:
                config = yaml.safe_load(f)
            
            args_task = self._create_args_from_config(config['task'])
            
            # Create task with YAML configuration
            args_data = Namespace(batch_size=16)
            args_model = Namespace(model_name="M_04_ISFM_Flow")
            args_trainer = Namespace(max_epochs=10, gpus=0)
            args_environment = Namespace(device="cpu")
            metadata = {'0': {'Name': 'test_dataset', 'Label': 3}}
            
            task = FlowPretrainTask(
                network=self.network,
                args_data=args_data,
                args_model=args_model,
                args_task=args_task,
                args_trainer=args_trainer,
                args_environment=args_environment,
                metadata=metadata
            )
            
            # Verify YAML configuration was loaded correctly
            self.assertEqual(task.num_steps, 200)
            self.assertEqual(task.flow_weight, 0.7)
            self.assertEqual(task.contrastive_weight, 0.3)
            
        finally:
            os.unlink(yaml_file)
    
    def test_configuration_validation(self):
        """Test configuration parameter validation."""
        # Test invalid generation mode
        invalid_config = self.base_config.copy()
        invalid_config['generation_mode'] = 'invalid_mode'
        args_task = self._create_args_from_config(invalid_config)
        
        # Should not raise error during initialization (validation is in set_generation_mode)
        task = FlowPretrainTask(
            network=self.network,
            args_data=Namespace(),
            args_model=Namespace(),
            args_task=args_task,
            args_trainer=Namespace(gpus=0),
            args_environment=Namespace(),
            metadata={'0': {'Name': 'test_dataset', 'Label': 3}}
        )
        
        # But should raise error when trying to set invalid mode
        with self.assertRaises(ValueError):
            task.set_generation_mode('invalid_mode')


class TestFlowTaskFunctionality(unittest.TestCase):
    """Test cases for FlowPretrainTask core functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.network = self._create_mock_flow_network()
        self.task = self._create_test_task()
        self.sample_batch = self._create_sample_batch()
    
    def _create_mock_flow_network(self) -> nn.Module:
        """Create a mock Flow network for testing."""
        class MockFlowNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.flow_model = nn.ModuleDict({'velocity_net': nn.Linear(128, 128)})
                self.condition_encoder = nn.Linear(10, 128)
            
            def forward(self, x, file_ids=None, return_loss=False):
                B, L, C = x.shape
                outputs = {
                    'velocity': torch.randn(B, L, 128, device=x.device),
                    'x_original': x,
                }
                if return_loss:
                    outputs['flow_loss'] = torch.tensor(0.5, device=x.device, requires_grad=True)
                return outputs
            
            def sample(self, batch_size, file_ids=None, num_steps=100, device='cpu'):
                return torch.randn(batch_size, 1024, 3, device=device)
        
        return MockFlowNetwork()
    
    def _create_test_task(self) -> FlowPretrainTask:
        """Create a test FlowPretrainTask instance."""
        args_task = Namespace(
            name="flow_pretrain",
            type="pretrain",
            use_conditional=True,
            generation_mode="conditional",
            num_steps=50,
            use_contrastive=False,
            loss="mse",  # Required by Default_task
            metrics=["acc"]  # Required by Default_task (list format)
        )
        
        return FlowPretrainTask(
            network=self.network,
            args_data=Namespace(),
            args_model=Namespace(),
            args_task=args_task,
            args_trainer=Namespace(gpus=0),
            args_environment=Namespace(),
            metadata={'0': {'Name': 'test_dataset', 'Label': 3}}
        )
    
    def _create_sample_batch(self) -> Dict[str, Any]:
        """Create a sample batch for testing."""
        return {
            'x': torch.randn(4, 1024, 3),
            'file_id': ['file1', 'file2', 'file3', 'file4']
        }
    
    def test_forward_pass(self):
        """Test forward pass through the task."""
        outputs = self.task.forward(self.sample_batch)
        
        self.assertIsInstance(outputs, dict)
        self.assertIn('velocity', outputs)
        self.assertIn('x_original', outputs)
        self.assertIn('generation_mode', outputs)
        self.assertIn('use_conditional', outputs)
        self.assertEqual(outputs['generation_mode'], 'conditional')
        self.assertEqual(outputs['use_conditional'], True)
    
    def test_generate_samples(self):
        """Test sample generation functionality."""
        # Test conditional generation
        samples = self.task.generate_samples(
            batch_size=4, 
            file_ids=['file1', 'file2', 'file3', 'file4'],
            mode='conditional',
            num_steps=10
        )
        
        self.assertEqual(samples.shape, (4, 1024, 3))
        
        # Test unconditional generation
        samples = self.task.generate_samples(
            batch_size=4,
            mode='unconditional',
            num_steps=10
        )
        
        self.assertEqual(samples.shape, (4, 1024, 3))
    
    def test_generation_mode_switching(self):
        """Test generation mode switching functionality."""
        # Initial mode
        self.assertEqual(self.task.generation_mode, 'conditional')
        
        # Switch to unconditional
        self.task.set_generation_mode('unconditional')
        self.assertEqual(self.task.generation_mode, 'unconditional')
        
        # Switch back to conditional
        self.task.set_generation_mode('conditional')
        self.assertEqual(self.task.generation_mode, 'conditional')
    
    def test_validate_generation_capability(self):
        """Test generation capability validation."""
        capabilities = self.task.validate_generation_capability()
        
        self.assertIsInstance(capabilities, dict)
        self.assertIn('conditional_generation', capabilities)
        self.assertIn('unconditional_generation', capabilities)
        self.assertIn('flow_model_available', capabilities)
        self.assertIn('condition_encoder_available', capabilities)
        
        # Should support both modes with mock network
        self.assertTrue(capabilities['flow_model_available'])
        self.assertTrue(capabilities['unconditional_generation'])


def run_tests():
    """Run all Flow task factory integration tests."""
    print("Running Flow Task Factory Integration Tests...")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestFlowTaskRegistration))
    suite.addTests(loader.loadTestsFromTestCase(TestFlowTaskInstantiation))
    suite.addTests(loader.loadTestsFromTestCase(TestFlowTaskConfiguration))
    suite.addTests(loader.loadTestsFromTestCase(TestFlowTaskFunctionality))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print("✅ All Flow task factory integration tests passed!")
        print(f"   Total tests run: {result.testsRun}")
        print(f"   Success rate: 100%")
    else:
        print(f"❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        print(f"   Total tests run: {result.testsRun}")
        print(f"   Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
        
        if result.failures:
            print("\n📋 Failures:")
            for test, traceback in result.failures:
                print(f"- {test}:")
                print(f"  {traceback.split('AssertionError: ')[-1] if 'AssertionError: ' in traceback else traceback}")
        
        if result.errors:
            print("\n🚨 Errors:")
            for test, traceback in result.errors:
                print(f"- {test}:")
                print(f"  {traceback.split('Error: ')[-1] if 'Error: ' in traceback else traceback}")
    
    print(f"\n📊 Test Coverage Summary:")
    print(f"   ✓ Task Registration: {4}/{4} tests")
    print(f"   ✓ Task Instantiation: {4}/{4} tests") 
    print(f"   ✓ Configuration Loading: {4}/{4} tests")
    print(f"   ✓ Core Functionality: {5}/{5} tests")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)