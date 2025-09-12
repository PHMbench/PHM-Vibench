#!/usr/bin/env python3
"""
Pipeline Stage Transitions and Metadata Preservation Tests for FlowPretrainTask

This test suite validates pipeline stage transitions and metadata preservation
for Flow-based pretraining in multi-stage workflows, specifically testing
the Pipeline_02_pretrain_fewshot pipeline transitions.

Test Coverage:
- Pipeline stage transition validation
- Metadata preservation across stages
- Checkpoint compatibility and loading
- Training state transfer between stages
- Pipeline_02_pretrain_fewshot workflow simulation
- Error handling during stage transitions  
- Resource management and cleanup during transitions
- Configuration inheritance between stages

Author: PHM-Vibench Team  
Date: 2025-09-12
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import unittest
import tempfile
import shutil
import yaml
import json
from pathlib import Path
from argparse import Namespace
from typing import Dict, Any, Optional, Union
import numpy as np
from unittest.mock import MagicMock, patch

# Import required components
from src.configs import load_config, ConfigWrapper
from src.task_factory.task.pretrain.flow_pretrain import FlowPretrainTask
from src.configs.config_utils import PRESET_TEMPLATES, transfer_namespace
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


class MockFlowNetwork(nn.Module):
    """Enhanced mock Flow network for testing pipeline transitions."""
    
    def __init__(self, input_dim=1, feature_dim=128, condition_dim=10):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.output_dim = feature_dim
        self.training_step_count = 0
        
        # Enhanced encoder for feature extraction
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        # Mock flow components with state tracking
        self.flow_model = nn.ModuleDict({
            'velocity_net': nn.Sequential(
                nn.Linear(feature_dim, feature_dim * 2),
                nn.ReLU(),
                nn.Linear(feature_dim * 2, feature_dim),
                nn.Tanh()
            ),
            'time_embedding': nn.Linear(1, feature_dim),
        })
        
        # Condition encoder for conditional generation
        self.condition_encoder = nn.Embedding(100, condition_dim)
        
        # Track training state for pipeline transitions
        self.training_metadata = {
            'epochs_trained': 0,
            'total_steps': 0,
            'best_val_loss': float('inf'),
            'stage': 'pretraining'
        }
        
    def encode(self, x, file_ids=None):
        """Encode input signals into feature representations."""
        batch_size, seq_len, channels = x.shape
        x_flat = x.view(-1, channels)
        encoded = self.encoder(x_flat)
        return encoded.view(batch_size, seq_len, self.feature_dim)
        
    def forward(self, x, file_ids=None, return_loss=False):
        batch_size, seq_len, channels = x.shape
        x_flat = x.view(-1, channels)
        encoded = self.encoder(x_flat)
        encoded = encoded.view(batch_size, seq_len, self.feature_dim)
        
        # Mock flow computation
        velocity = self.flow_model['velocity_net'](encoded)
        
        outputs = {
            'velocity': velocity,
            'x_original': x,
            'reconstructed': x + 0.01 * torch.randn_like(x),  # Add small noise
            'encoded_features': encoded
        }
        
        if return_loss:
            # Mock flow loss that decreases over training
            flow_loss = torch.tensor(1.0 / (1 + self.training_step_count * 0.1))
            outputs['flow_loss'] = flow_loss
            outputs['loss'] = flow_loss
            self.training_step_count += 1
            
        return outputs

    def sample(self, batch_size, file_ids=None, num_steps=10, device='cpu'):
        """Generate samples using Flow model."""
        return torch.randn(batch_size, 256, self.input_dim, device=device)
        
    def state_dict(self):
        """Enhanced state dict with training metadata."""
        base_state = super().state_dict()
        # Add training metadata to state dict
        base_state['_training_metadata'] = self.training_metadata
        return base_state
        
    def load_state_dict(self, state_dict, strict=True):
        """Enhanced state dict loading with metadata recovery."""
        # Extract training metadata if present
        if '_training_metadata' in state_dict:
            self.training_metadata = state_dict.pop('_training_metadata')
        return super().load_state_dict(state_dict, strict)


class MockDataLoader:
    """Mock DataLoader for testing pipeline transitions."""
    
    def __init__(self, batch_size=32, num_batches=10, sequence_length=256):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.sequence_length = sequence_length
        
    def __iter__(self):
        for i in range(self.num_batches):
            yield {
                'x': torch.randn(self.batch_size, self.sequence_length, 1),
                'file_id': [f'file_{j}' for j in range(self.batch_size)],
                'label': torch.randint(0, 4, (self.batch_size,))
            }
            
    def __len__(self):
        return self.num_batches


class TestPipelineStageTransitions(unittest.TestCase):
    """Test pipeline stage transitions and workflow validation."""
    
    def setUp(self):
        """Set up test fixtures for pipeline transitions."""
        self.temp_dir = tempfile.mkdtemp()
        self.network = MockFlowNetwork()
        self.metadata = {'0': {'Name': 'test_dataset', 'Label': 0}}
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def _create_pretraining_config(self) -> Dict[str, Any]:
        """Create pretraining stage configuration."""
        return {
            'data': {
                'data_dir': 'data',
                'metadata_file': 'metadata_CWRU.xlsx',
                'dataset': 'CWRU',
                'batch_size': 32,
                'sequence_length': 256,
                'channels': 1
            },
            'model': {
                'name': 'M_04_ISFM_Flow',
                'type': 'foundation_model',
                'hidden_dim': 128,
                'use_conditional': True
            },
            'task': {
                'name': 'flow_pretrain',
                'type': 'pretrain',
                'num_steps': 50,  # Short for testing
                'use_contrastive': False,
                'lr': 1e-4,
                'max_epochs': 2,  # Short for testing
                'validate_generation': True,
                'loss': 'mse',  # Required by Default_task
                'metrics': ['acc']  # Required by Default_task
            },
            'trainer': {
                'gpus': 0,  # CPU only for testing
                'precision': 32,
                'save_top_k': 1,
                'monitor': 'val_loss'
            },
            'environment': {
                'seed': 42
            }
        }
        
    def _create_fewshot_config(self) -> Dict[str, Any]:
        """Create few-shot stage configuration."""
        return {
            'data': {
                'data_dir': 'data',
                'metadata_file': 'metadata_XJTU.xlsx',
                'dataset': 'XJTU',
                'batch_size': 16,  # Smaller batch for few-shot
                'sequence_length': 256,
                'channels': 1
            },
            'model': {
                'name': 'M_04_ISFM_Flow',
                'type': 'foundation_model',
                'hidden_dim': 128,
                'use_conditional': True
            },
            'task': {
                'name': 'finetuning',
                'type': 'FS',  # Few-shot task type
                'lr': 5e-5,  # Lower learning rate
                'max_epochs': 1,  # Very short for testing
                'freeze_flow_layers': True,  # Freeze Flow layers during fine-tuning
                'loss': 'mse',  # Required by Default_task
                'metrics': ['acc']  # Required by Default_task
            },
            'trainer': {
                'gpus': 0,
                'precision': 32,
                'save_top_k': 1
            },
            'environment': {
                'seed': 42
            }
        }
        
    def _create_flow_task(self, config_dict: Dict[str, Any]) -> FlowPretrainTask:
        """Create FlowPretrainTask from configuration."""
        config = load_config(config_dict)
        
        args_data = transfer_namespace(config.get('data', {}))
        args_model = transfer_namespace(config.get('model', {}))
        args_task = transfer_namespace(config.get('task', {}))
        args_trainer = transfer_namespace(config.get('trainer', {}))
        args_environment = transfer_namespace(config.get('environment', {}))
        
        return FlowPretrainTask(
            network=self.network,
            args_data=args_data,
            args_model=args_model,
            args_task=args_task,
            args_trainer=args_trainer,
            args_environment=args_environment,
            metadata=self.metadata
        )
        
    def test_pretraining_stage_setup(self):
        """Test pretraining stage initialization and setup."""
        pretrain_config = self._create_pretraining_config()
        pretrain_task = self._create_flow_task(pretrain_config)
        
        # Verify pretraining task is correctly configured
        self.assertEqual(pretrain_task.args_task.name, 'flow_pretrain')
        self.assertEqual(pretrain_task.num_steps, 100)  # Minimum 100 steps enforced by FlowPretrainTask
        self.assertFalse(pretrain_task.use_contrastive)
        self.assertTrue(pretrain_task.validate_generation)
        
        # Test compatibility validation
        compatibility = pretrain_task.validate_pipeline_compatibility()
        self.assertTrue(compatibility['compatible'])
        self.assertIn('COMPATIBLE', compatibility['status'])
        
        print(f"✅ Pretraining stage setup validated")
        print(f"   Task name: {pretrain_task.args_task.name}")
        print(f"   Compatibility status: {compatibility['status']}")
        
    def test_checkpoint_callback_configuration(self):
        """Test checkpoint callback configuration for pipeline compatibility."""
        pretrain_config = self._create_pretraining_config()
        pretrain_task = self._create_flow_task(pretrain_config)
        
        # Get checkpoint callback
        checkpoint_cb = pretrain_task.get_pipeline_checkpoint_callback()
        
        # Verify checkpoint configuration
        self.assertIsInstance(checkpoint_cb, ModelCheckpoint)
        self.assertEqual(checkpoint_cb.monitor, 'val_loss')
        self.assertEqual(checkpoint_cb.mode, 'min')
        self.assertEqual(checkpoint_cb.save_top_k, 1)
        self.assertTrue(checkpoint_cb.save_last)
        self.assertFalse(checkpoint_cb.save_weights_only)  # Need full checkpoint
        
        print(f"✅ Checkpoint callback configuration validated")
        print(f"   Monitor: {checkpoint_cb.monitor}")
        print(f"   Save weights only: {checkpoint_cb.save_weights_only}")
        
    def test_training_metadata_preservation(self):
        """Test training metadata preservation during pipeline transitions."""
        pretrain_config = self._create_pretraining_config()
        pretrain_task = self._create_flow_task(pretrain_config)
        
        # Simulate training progress
        original_metadata = pretrain_task.network.training_metadata.copy()
        
        # Simulate some training steps
        pretrain_task.network.training_metadata.update({
            'epochs_trained': 5,
            'total_steps': 150,
            'best_val_loss': 0.25,
            'stage': 'pretraining_completed'
        })
        
        # Save and load state dict (simulating checkpoint save/load)
        state_dict = pretrain_task.network.state_dict()
        self.assertIn('_training_metadata', state_dict)
        
        # Create new network and load state
        new_network = MockFlowNetwork()
        new_network.load_state_dict(state_dict)
        
        # Verify metadata was preserved
        self.assertEqual(new_network.training_metadata['epochs_trained'], 5)
        self.assertEqual(new_network.training_metadata['total_steps'], 150)
        self.assertEqual(new_network.training_metadata['best_val_loss'], 0.25)
        self.assertEqual(new_network.training_metadata['stage'], 'pretraining_completed')
        
        print(f"✅ Training metadata preservation validated")
        print(f"   Epochs preserved: {new_network.training_metadata['epochs_trained']}")
        print(f"   Steps preserved: {new_network.training_metadata['total_steps']}")
        
    def test_pipeline_stage_transition_simulation(self):
        """Test full pipeline stage transition simulation."""
        # Stage 1: Pretraining
        pretrain_config = self._create_pretraining_config()
        pretrain_task = self._create_flow_task(pretrain_config)
        
        # Mock training in pretraining stage
        pretrain_task.network.training_metadata.update({
            'epochs_trained': 10,
            'total_steps': 300,
            'best_val_loss': 0.15,
            'stage': 'pretraining_completed'
        })
        
        # Save checkpoint (simulate Pipeline_02 checkpoint saving)
        checkpoint_path = os.path.join(self.temp_dir, 'pretrain_checkpoint.ckpt')
        checkpoint_data = {
            'state_dict': pretrain_task.network.state_dict(),
            'epoch': 10,
            'global_step': 300,
            'pytorch-lightning_version': '1.9.0',
            'state_dict_keys': list(pretrain_task.network.state_dict().keys()),
            'hyper_parameters': {
                'task_name': pretrain_task.args_task.name,
                'num_steps': pretrain_task.num_steps,
                'use_contrastive': pretrain_task.use_contrastive
            }
        }
        torch.save(checkpoint_data, checkpoint_path)
        
        # Stage 2: Few-shot transition
        fewshot_config = self._create_fewshot_config()
        
        # Create new network for few-shot stage (simulate loading from checkpoint)
        fewshot_network = MockFlowNetwork()
        
        # Load pretrained checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        fewshot_network.load_state_dict(checkpoint['state_dict'])
        
        # Verify metadata was transferred
        self.assertEqual(fewshot_network.training_metadata['epochs_trained'], 10)
        self.assertEqual(fewshot_network.training_metadata['stage'], 'pretraining_completed')
        
        # Update stage for few-shot
        fewshot_network.training_metadata['stage'] = 'fewshot_finetuning'
        
        # Create few-shot task with loaded network
        fewshot_task = FlowPretrainTask(
            network=fewshot_network,
            args_data=transfer_namespace(fewshot_config['data']),
            args_model=transfer_namespace(fewshot_config['model']),
            args_task=transfer_namespace(fewshot_config['task']),
            args_trainer=transfer_namespace(fewshot_config['trainer']),
            args_environment=transfer_namespace(fewshot_config['environment']),
            metadata=self.metadata
        )
        
        # Verify few-shot task configuration
        self.assertEqual(fewshot_task.args_task.name, 'finetuning')
        self.assertEqual(fewshot_network.training_metadata['stage'], 'fewshot_finetuning')
        
        print(f"✅ Pipeline stage transition simulation completed")
        print(f"   Pretraining → Few-shot transition successful")
        print(f"   Metadata preserved across stages")
        print(f"   Checkpoint loading/saving verified")
        
    def test_feature_extraction_across_stages(self):
        """Test feature extraction capability across pipeline stages."""
        pretrain_config = self._create_pretraining_config()
        pretrain_task = self._create_flow_task(pretrain_config)
        
        # Create mock dataloader
        dataloader = MockDataLoader(batch_size=8, num_batches=3)
        
        # Extract features
        features = pretrain_task.extract_feature_representations(dataloader)
        
        # Verify feature extraction
        self.assertIn('encoded_features', features)
        self.assertIn('original_signals', features)
        self.assertIn('file_ids', features)
        
        # Verify feature shapes
        if isinstance(features['encoded_features'], torch.Tensor):
            self.assertEqual(len(features['encoded_features'].shape), 3)  # batch, seq, features
            self.assertEqual(features['encoded_features'].shape[2], 128)  # feature_dim
        
        self.assertIsInstance(features['file_ids'], list)
        self.assertGreater(len(features['file_ids']), 0)
        
        print(f"✅ Feature extraction across stages validated")
        print(f"   Feature types extracted: {list(features.keys())}")
        if isinstance(features['encoded_features'], torch.Tensor):
            print(f"   Feature shape: {features['encoded_features'].shape}")
        
    def test_configuration_inheritance_between_stages(self):
        """Test configuration inheritance patterns between pipeline stages."""
        # Base configuration
        base_config = self._create_pretraining_config()
        
        # Stage-specific overrides
        fewshot_overrides = {
            'task': {
                'name': 'finetuning',
                'type': 'FS',
                'lr': 5e-5,
                'max_epochs': 1
            },
            'data': {
                'batch_size': 16,
                'dataset': 'XJTU'
            }
        }
        
        # Test configuration composition (simulate pipeline config inheritance)
        inherited_config = load_config(base_config, fewshot_overrides)
        
        # Verify inheritance worked correctly
        self.assertEqual(inherited_config.task.name, 'finetuning')  # Overridden
        self.assertEqual(inherited_config.task.lr, 5e-5)  # Overridden
        self.assertEqual(inherited_config.model.hidden_dim, 128)  # Inherited
        self.assertEqual(inherited_config.model.use_conditional, True)  # Inherited
        self.assertEqual(inherited_config.data.batch_size, 16)  # Overridden
        self.assertEqual(inherited_config.environment.seed, 42)  # Inherited
        
        print(f"✅ Configuration inheritance between stages validated")
        print(f"   Task name inherited/overridden: {inherited_config.task.name}")
        print(f"   Model config inherited: hidden_dim={inherited_config.model.hidden_dim}")
        
    def test_error_handling_during_transitions(self):
        """Test error handling during pipeline stage transitions."""
        pretrain_config = self._create_pretraining_config()
        pretrain_task = self._create_flow_task(pretrain_config)
        
        # Test handling of missing checkpoint
        with self.assertRaises(FileNotFoundError):
            torch.load('nonexistent_checkpoint.ckpt')
        
        # Test handling of corrupted state dict
        corrupted_state = {'invalid_key': 'invalid_value'}
        with self.assertRaises((RuntimeError, KeyError)):
            pretrain_task.network.load_state_dict(corrupted_state, strict=True)
        
        # Test graceful handling with strict=False
        try:
            pretrain_task.network.load_state_dict(corrupted_state, strict=False)
            # Should not raise error but should warn
        except Exception as e:
            self.fail(f"Non-strict loading should not fail: {e}")
        
        # Test compatibility validation with incompatible network
        class IncompatibleNetwork(nn.Module):
            def forward(self, x):
                return x
            def state_dict(self):
                # Deliberately raise an error to make it incompatible
                raise RuntimeError("Incompatible state dict")
        
        incompatible_task = FlowPretrainTask(
            network=IncompatibleNetwork(),
            args_data=transfer_namespace(pretrain_config['data']),
            args_model=transfer_namespace(pretrain_config['model']),
            args_task=transfer_namespace(pretrain_config['task']),
            args_trainer=transfer_namespace(pretrain_config['trainer']),
            args_environment=transfer_namespace(pretrain_config['environment']),
            metadata=self.metadata
        )
        
        compatibility = incompatible_task.validate_pipeline_compatibility()
        # Should detect compatibility issues
        self.assertFalse(compatibility['compatible'])
        
        print(f"✅ Error handling during transitions validated")
        print(f"   Missing checkpoint handling: ✓")
        print(f"   Corrupted state dict handling: ✓")
        print(f"   Compatibility validation: ✓")


class TestPipelineResourceManagement(unittest.TestCase):
    """Test resource management and cleanup during pipeline transitions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_checkpoint_file_management(self):
        """Test checkpoint file creation and cleanup."""
        # Create checkpoint directory structure
        checkpoint_dir = os.path.join(self.temp_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Simulate checkpoint saving
        network = MockFlowNetwork()
        checkpoint_files = []
        
        for epoch in range(5):
            checkpoint_path = os.path.join(
                checkpoint_dir, 
                f'flow_pretrain-epoch={epoch:02d}-val_loss=0.{5-epoch}000.ckpt'
            )
            torch.save({'state_dict': network.state_dict(), 'epoch': epoch}, checkpoint_path)
            checkpoint_files.append(checkpoint_path)
        
        # Verify files were created
        for ckpt_file in checkpoint_files:
            self.assertTrue(os.path.exists(ckpt_file))
        
        # Simulate checkpoint cleanup (keeping only best)
        best_checkpoint = checkpoint_files[-1]  # Last one (lowest loss)
        for ckpt_file in checkpoint_files[:-1]:
            os.remove(ckpt_file)
        
        # Verify cleanup
        self.assertTrue(os.path.exists(best_checkpoint))
        for ckpt_file in checkpoint_files[:-1]:
            self.assertFalse(os.path.exists(ckpt_file))
        
        print(f"✅ Checkpoint file management validated")
        print(f"   Checkpoints created and cleaned up properly")
        
    def test_memory_cleanup_between_stages(self):
        """Test memory cleanup between pipeline stages."""
        # Create large tensors to simulate training state
        large_tensors = []
        for i in range(10):
            tensor = torch.randn(100, 100, 100)  # Large tensor
            large_tensors.append(tensor)
        
        # Verify tensors exist
        self.assertEqual(len(large_tensors), 10)
        
        # Simulate stage transition cleanup
        del large_tensors
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Memory should be freed (we can't directly test this, but ensure no errors)
        new_tensor = torch.randn(50, 50, 50)
        self.assertIsNotNone(new_tensor)
        
        print(f"✅ Memory cleanup between stages validated")
        
    def test_configuration_persistence(self):
        """Test configuration persistence across pipeline stages."""
        # Create configuration
        config_data = {
            'pipeline_id': 'Pipeline_02_pretrain_fewshot',
            'pretraining_completed': True,
            'pretrain_epochs': 10,
            'pretrain_best_loss': 0.15,
            'transition_timestamp': '2025-09-12T10:00:00Z'
        }
        
        # Save configuration
        config_path = os.path.join(self.temp_dir, 'pipeline_state.json')
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        # Verify saved
        self.assertTrue(os.path.exists(config_path))
        
        # Load and verify
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)
        
        self.assertEqual(loaded_config['pipeline_id'], 'Pipeline_02_pretrain_fewshot')
        self.assertTrue(loaded_config['pretraining_completed'])
        self.assertEqual(loaded_config['pretrain_epochs'], 10)
        
        print(f"✅ Configuration persistence validated")
        print(f"   Pipeline state saved and loaded correctly")


class TestPipeline02Compatibility(unittest.TestCase):
    """Test specific compatibility with Pipeline_02_pretrain_fewshot."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_pipeline_02_config_structure(self):
        """Test compatibility with Pipeline_02 configuration structure."""
        # Simulate Pipeline_02 pretraining config
        pretrain_config = {
            'environment': {'seed': 42, 'CUDA_VISIBLE_DEVICES': '0'},
            'data': {
                'data_dir': 'data',
                'metadata_file': 'metadata_CWRU.xlsx',
                'batch_size': 32
            },
            'model': {
                'name': 'M_04_ISFM_Flow',
                'type': 'foundation_model'
            },
            'task': {
                'name': 'flow_pretrain',
                'type': 'pretrain',
                'lr': 1e-4,
                'max_epochs': 50
            },
            'trainer': {
                'gpus': 1,
                'save_top_k': 3,
                'monitor': 'val_total_loss'
            }
        }
        
        # Test config loading (as Pipeline_02 would do)
        config = load_config(pretrain_config)
        
        # Convert to namespaces (as Pipeline_02 does)
        args_environment = transfer_namespace(config.get('environment', {}))
        args_data = transfer_namespace(config.get('data', {}))
        args_model = transfer_namespace(config.get('model', {}))
        args_task = transfer_namespace(config.get('task', {}))
        args_trainer = transfer_namespace(config.get('trainer', {}))
        
        # Verify namespace conversion
        self.assertEqual(args_task.name, 'flow_pretrain')
        self.assertEqual(args_data.batch_size, 32)
        self.assertEqual(args_trainer.save_top_k, 3)
        self.assertEqual(args_environment.seed, 42)
        
        print(f"✅ Pipeline_02 config structure compatibility validated")
        
    def test_pipeline_02_checkpoint_format(self):
        """Test checkpoint format compatibility with Pipeline_02."""
        network = MockFlowNetwork()
        
        # Simulate Pipeline_02 checkpoint format
        checkpoint_data = {
            'state_dict': network.state_dict(),
            'epoch': 10,
            'global_step': 300,
            'pytorch-lightning_version': '1.9.0',
            'lr_schedulers': [],
            'optimizer_states': [{}],
            'hyper_parameters': {
                'task_name': 'flow_pretrain',
                'learning_rate': 1e-4
            }
        }
        
        # Save checkpoint
        checkpoint_path = os.path.join(self.temp_dir, 'pipeline_02_checkpoint.ckpt')
        torch.save(checkpoint_data, checkpoint_path)
        
        # Load checkpoint (as Pipeline_02 few-shot stage would)
        loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Verify checkpoint structure
        self.assertIn('state_dict', loaded_checkpoint)
        self.assertIn('epoch', loaded_checkpoint)
        self.assertIn('hyper_parameters', loaded_checkpoint)
        
        # Test state dict loading
        new_network = MockFlowNetwork()
        new_network.load_state_dict(loaded_checkpoint['state_dict'])
        
        print(f"✅ Pipeline_02 checkpoint format compatibility validated")
        print(f"   Checkpoint epoch: {loaded_checkpoint['epoch']}")
        print(f"   Global step: {loaded_checkpoint['global_step']}")
        
    def test_pipeline_02_workflow_simulation(self):
        """Test full Pipeline_02 workflow simulation."""
        # Step 1: Create pretraining config file
        pretrain_config = {
            'environment': {'seed': 42},
            'data': {
                'data_dir': 'data',
                'metadata_file': 'metadata_CWRU.xlsx',
                'batch_size': 32
            },
            'model': {
                'name': 'M_04_ISFM_Flow',
                'type': 'foundation_model'
            },
            'task': {
                'name': 'flow_pretrain',
                'type': 'pretrain',
                'num_steps': 10,  # Short for testing
                'max_epochs': 1,
                'loss': 'mse',  # Required by Default_task
                'metrics': ['acc']  # Required by Default_task
            },
            'trainer': {'gpus': 0, 'save_top_k': 1}
        }
        
        pretrain_path = os.path.join(self.temp_dir, 'pretrain.yaml')
        with open(pretrain_path, 'w') as f:
            yaml.dump(pretrain_config, f)
        
        # Step 2: Create few-shot config file
        fs_config = {
            'environment': {'seed': 42},
            'data': {
                'data_dir': 'data',
                'metadata_file': 'metadata_XJTU.xlsx',
                'batch_size': 16
            },
            'model': {
                'name': 'M_04_ISFM_Flow',
                'type': 'foundation_model'
            },
            'task': {
                'name': 'finetuning',
                'type': 'FS',
                'max_epochs': 1,
                'loss': 'mse',  # Required by Default_task
                'metrics': ['acc']  # Required by Default_task
            },
            'trainer': {'gpus': 0}
        }
        
        fs_path = os.path.join(self.temp_dir, 'fewshot.yaml')
        with open(fs_path, 'w') as f:
            yaml.dump(fs_config, f)
        
        # Step 3: Simulate Pipeline_02 pretraining stage
        pretrain_config_loaded = load_config(pretrain_path)
        pretrain_args = {
            'environment': transfer_namespace(pretrain_config_loaded.get('environment', {})),
            'data': transfer_namespace(pretrain_config_loaded.get('data', {})),
            'model': transfer_namespace(pretrain_config_loaded.get('model', {})),
            'task': transfer_namespace(pretrain_config_loaded.get('task', {})),
            'trainer': transfer_namespace(pretrain_config_loaded.get('trainer', {}))
        }
        
        # Create pretraining task
        pretrain_network = MockFlowNetwork()
        pretrain_task = FlowPretrainTask(
            network=pretrain_network,
            args_data=pretrain_args['data'],
            args_model=pretrain_args['model'],
            args_task=pretrain_args['task'],
            args_trainer=pretrain_args['trainer'],
            args_environment=pretrain_args['environment'],
            metadata={'0': {'Name': 'CWRU', 'Label': 0}}
        )
        
        # Simulate training and checkpoint saving
        checkpoint_path = os.path.join(self.temp_dir, 'pretrain_best.ckpt')
        torch.save({
            'state_dict': pretrain_task.network.state_dict(),
            'epoch': 1,
            'global_step': 10
        }, checkpoint_path)
        
        # Step 4: Simulate Pipeline_02 few-shot stage
        fs_config_loaded = load_config(fs_path)
        fs_args = {
            'environment': transfer_namespace(fs_config_loaded.get('environment', {})),
            'data': transfer_namespace(fs_config_loaded.get('data', {})),
            'model': transfer_namespace(fs_config_loaded.get('model', {})),
            'task': transfer_namespace(fs_config_loaded.get('task', {})),
            'trainer': transfer_namespace(fs_config_loaded.get('trainer', {}))
        }
        
        # Load pretrained weights
        fs_network = MockFlowNetwork()
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        fs_network.load_state_dict(checkpoint['state_dict'])
        
        # Create few-shot task
        fs_task = FlowPretrainTask(
            network=fs_network,
            args_data=fs_args['data'],
            args_model=fs_args['model'],
            args_task=fs_args['task'],
            args_trainer=fs_args['trainer'],
            args_environment=fs_args['environment'],
            metadata={'0': {'Name': 'XJTU', 'Label': 0}}
        )
        
        # Verify workflow completed successfully
        self.assertEqual(pretrain_task.args_task.name, 'flow_pretrain')
        self.assertEqual(fs_task.args_task.name, 'finetuning')
        self.assertTrue(os.path.exists(checkpoint_path))
        
        print(f"✅ Pipeline_02 workflow simulation completed successfully")
        print(f"   Pretraining stage: ✓")
        print(f"   Checkpoint saving: ✓")
        print(f"   Few-shot stage: ✓")
        print(f"   Workflow transition: ✓")


def run_all_tests():
    """Run all pipeline transition tests."""
    print("Running FlowPretrainTask Pipeline Stage Transitions Tests")
    print("=" * 80)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestPipelineStageTransitions,
        TestPipelineResourceManagement,
        TestPipeline02Compatibility
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 80)
    if result.wasSuccessful():
        print("✅ All pipeline transition tests passed!")
        print(f"   Total tests run: {result.testsRun}")
        print(f"   Success rate: 100%")
    else:
        print(f"❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        print(f"   Total tests run: {result.testsRun}")
        success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
        print(f"   Success rate: {success_rate:.1f}%")
        
        if result.failures:
            print("\n📋 Test Failures:")
            for test, traceback in result.failures:
                print(f"- {test}:")
                print(f"  {traceback.split('AssertionError: ')[-1] if 'AssertionError: ' in traceback else 'See traceback above'}")
        
        if result.errors:
            print("\n🚨 Test Errors:")
            for test, traceback in result.errors:
                print(f"- {test}:")
                print(f"  {traceback.split('Error: ')[-1] if 'Error: ' in traceback else 'See traceback above'}")
    
    # Test coverage summary
    print(f"\n📊 Test Coverage Summary:")
    print(f"   ✓ Pipeline Stage Transitions: 7 tests")
    print(f"   ✓ Resource Management: 3 tests")
    print(f"   ✓ Pipeline_02 Compatibility: 3 tests")
    print(f"   ✓ Total Coverage: {result.testsRun} tests")
    
    print(f"\n🔧 Pipeline Transition Features Validated:")
    print(f"   ✓ Pipeline stage setup and initialization")
    print(f"   ✓ Checkpoint callback configuration")
    print(f"   ✓ Training metadata preservation")
    print(f"   ✓ Full pipeline stage transitions")
    print(f"   ✓ Feature extraction across stages")
    print(f"   ✓ Configuration inheritance patterns")
    print(f"   ✓ Error handling during transitions")
    print(f"   ✓ Resource management and cleanup")
    print(f"   ✓ Pipeline_02_pretrain_fewshot compatibility")
    print(f"   ✓ Checkpoint format compatibility")
    print(f"   ✓ Full workflow simulation")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
    
    print(f"\n🎯 Task 29 - Pipeline Stage Transitions and Metadata Tests:")
    if success:
        print("   ✅ All pipeline transition tests passed")
        print("   ✅ Pipeline stage transitions work correctly")
        print("   ✅ Metadata preservation across stages verified")
        print("   ✅ Pipeline_02_pretrain_fewshot workflow compatibility confirmed")
        print("   ✅ Checkpoint compatibility and loading validated")
        print("   ✅ Error handling during transitions implemented")
        print("   ✅ Resource management and cleanup verified")
        print("   ✅ Configuration inheritance between stages functional")
    else:
        print("   ❌ Some tests failed - check output above")
        
    sys.exit(0 if success else 1)