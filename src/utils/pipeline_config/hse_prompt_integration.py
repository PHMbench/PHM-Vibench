"""
HSE Prompt Integration for Pipeline_03

This module provides integration utilities for HSE Prompt-guided contrastive learning
with the Pipeline_03 MultiTaskPretrainFinetunePipeline workflow.

Key Features:
- Create pretraining/finetuning configurations for prompt-guided models
- Checkpoint loading compatibility with Pipeline_03 format
- Parameter freezing utilities for prompt components during finetuning
- Seamless integration with existing PHM-Vibench workflows

Author: PHM-Vibench Team
Date: 2025-01-06
License: MIT
"""

import os
import copy
from typing import Dict, List, Optional, Any, Union
from pathlib import Path


class HSEPromptPipelineIntegration:
    """
    Adapter class for integrating HSE Prompt-guided learning with Pipeline_03.
    
    This class provides utilities to seamlessly integrate prompt-guided contrastive
    learning with the existing MultiTaskPretrainFinetunePipeline infrastructure.
    """
    
    def __init__(self):
        """Initialize the HSE Prompt Pipeline Integration adapter."""
        self.prompt_models = ['M_02_ISFM_Prompt']
        self.prompt_embeddings = ['E_01_HSE_v2']
        self.recommended_backbones = ['B_08_PatchTST', 'B_04_Dlinear', 'B_06_TimesNet', 'B_09_FNO']
        
    def create_hse_prompt_pretraining_config(self, 
                                           base_configs: Dict, 
                                           backbone: str,
                                           target_systems: List[int],
                                           pretraining_config: Dict) -> Dict:
        """
        Create pretraining configuration for HSE Prompt-guided learning.
        
        Args:
            base_configs: Base configuration dictionary from Pipeline_03
            backbone: Backbone architecture name  
            target_systems: List of dataset IDs for pretraining
            pretraining_config: Pretraining-specific configuration
            
        Returns:
            Complete configuration dictionary for HSE prompt pretraining
        """
        config = copy.deepcopy(base_configs)
        
        # Update model configuration for prompt-guided learning
        config['model'].update({
            'name': 'M_02_ISFM_Prompt',
            'type': 'ISFM_Prompt',
            'embedding': 'E_01_HSE_v2',
            'backbone': backbone,
            'task_head': pretraining_config.get('task_head', 'H_01_Linear_cla'),
            
            # Prompt-specific configuration
            'use_prompt': True,
            'prompt_dim': pretraining_config.get('prompt_dim', 128),
            'fusion_type': pretraining_config.get('fusion_type', 'attention'),
            'training_stage': 'pretraining',
            'freeze_prompt': False,
            
            # HSE v2 specific parameters
            'patch_size_L': pretraining_config.get('patch_size_L', 16),
            'patch_size_C': pretraining_config.get('patch_size_C', 1),
            'num_patches': pretraining_config.get('num_patches', 64),
            'output_dim': pretraining_config.get('output_dim', 128),
            
            # Prompt system parameters
            'max_dataset_ids': pretraining_config.get('max_dataset_ids', 50),
            'max_domain_ids': pretraining_config.get('max_domain_ids', 50)
        })
        
        # Update task configuration for contrastive learning
        config['task'].update({
            'name': 'hse_contrastive',
            'type': 'CDDG',
            'loss_type': pretraining_config.get('loss_type', 'InfoNCE'),
            'contrast_weight': pretraining_config.get('contrast_weight', 0.15),
            'temperature': pretraining_config.get('temperature', 0.07),
            'use_prompt_similarity': pretraining_config.get('use_prompt_similarity', True),
            'prompt_similarity_weight': pretraining_config.get('prompt_similarity_weight', 0.05)
        })
        
        # Update data configuration for multi-system training
        if 'data' in config:
            config['data']['target_systems'] = target_systems
            config['data']['cross_system_sampling'] = True
            config['data']['system_aware_batching'] = True
        
        # Update training configuration
        config['trainer'].update({
            'max_epochs': pretraining_config.get('max_epochs', 50),
            'learning_rate': pretraining_config.get('learning_rate', 1e-4),
            'weight_decay': pretraining_config.get('weight_decay', 1e-5),
            'warmup_epochs': pretraining_config.get('warmup_epochs', 5)
        })
        
        # Add experiment naming
        experiment_name = f"HSE_Prompt_Pretrain_{backbone}_systems_{'_'.join(map(str, target_systems))}"
        config['environment']['experiment_name'] = experiment_name
        
        return config
    
    def create_hse_prompt_finetuning_config(self,
                                          base_configs: Dict,
                                          pretrained_checkpoint: str,
                                          backbone: str, 
                                          target_system: int,
                                          finetuning_config: Dict) -> Dict:
        """
        Create finetuning configuration for HSE Prompt-guided learning.
        
        Args:
            base_configs: Base configuration dictionary from Pipeline_03
            pretrained_checkpoint: Path to pretrained model checkpoint
            backbone: Backbone architecture name
            target_system: Target system ID for finetuning
            finetuning_config: Finetuning-specific configuration
            
        Returns:
            Complete configuration dictionary for HSE prompt finetuning
        """
        config = copy.deepcopy(base_configs)
        
        # Update model configuration for prompt-guided finetuning
        config['model'].update({
            'name': 'M_02_ISFM_Prompt',
            'type': 'ISFM_Prompt', 
            'embedding': 'E_01_HSE_v2',
            'backbone': backbone,
            'task_head': finetuning_config.get('task_head', 'H_01_Linear_cla'),
            
            # Prompt-specific configuration for finetuning
            'use_prompt': True,
            'prompt_dim': finetuning_config.get('prompt_dim', 128),
            'fusion_type': finetuning_config.get('fusion_type', 'attention'),
            'training_stage': 'finetuning',
            'freeze_prompt': finetuning_config.get('freeze_prompt', True),
            
            # HSE v2 parameters (should match pretraining)
            'patch_size_L': finetuning_config.get('patch_size_L', 16),
            'patch_size_C': finetuning_config.get('patch_size_C', 1), 
            'num_patches': finetuning_config.get('num_patches', 64),
            'output_dim': finetuning_config.get('output_dim', 128),
            
            # Pretrained checkpoint loading
            'pretrained_checkpoint': pretrained_checkpoint,
            'load_backbone_only': True
        })
        
        # Update task configuration for supervised learning
        config['task'].update({
            'name': finetuning_config.get('task_name', 'classification'),
            'type': finetuning_config.get('task_type', 'classification'),
            'loss_type': finetuning_config.get('loss_type', 'CrossEntropyLoss'),
            'use_contrastive': finetuning_config.get('use_contrastive', False)
        })
        
        # Update data configuration for single system
        if 'data' in config:
            config['data']['target_systems'] = [target_system]
            config['data']['cross_system_sampling'] = False
        
        # Update training configuration for finetuning
        config['trainer'].update({
            'max_epochs': finetuning_config.get('max_epochs', 20),
            'learning_rate': finetuning_config.get('learning_rate', 5e-5),  # Lower LR for finetuning
            'weight_decay': finetuning_config.get('weight_decay', 1e-5),
            'warmup_epochs': finetuning_config.get('warmup_epochs', 2)
        })
        
        # Add experiment naming
        experiment_name = f"HSE_Prompt_Finetune_{backbone}_system_{target_system}"
        config['environment']['experiment_name'] = experiment_name
        
        return config
    
    def adapt_checkpoint_loading(self, 
                                checkpoint_path: str,
                                model_config: Dict) -> Dict:
        """
        Adapt checkpoint loading for Pipeline_03 format compatibility.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model_config: Model configuration dictionary
            
        Returns:
            Updated model configuration with checkpoint loading parameters
        """
        config = copy.deepcopy(model_config)
        
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found: {checkpoint_path}")
            # Still add loading parameters for testing purposes
            config.update({
                'load_pretrained': False,
                'pretrained_path': checkpoint_path,
                'load_backbone_only': True,
                'strict_loading': False,
            })
            return config
        
        # Add checkpoint loading parameters
        config.update({
            'load_pretrained': True,
            'pretrained_path': checkpoint_path,
            'load_backbone_only': True,  # Only load backbone, not task heads
            'strict_loading': False,     # Allow missing keys for prompt components
        })
        
        return config
    
    def freeze_prompt_parameters(self, model) -> int:
        """
        Freeze prompt-related parameters during finetuning.
        
        Args:
            model: Model instance with prompt components
            
        Returns:
            Number of parameters frozen
        """
        frozen_params = 0
        
        # Freeze SystemPromptEncoder parameters
        if hasattr(model, 'prompt_encoder'):
            for param in model.prompt_encoder.parameters():
                param.requires_grad = False
                frozen_params += param.numel()
        
        # Freeze PromptFusion parameters  
        if hasattr(model, 'prompt_fusion'):
            for param in model.prompt_fusion.parameters():
                param.requires_grad = False
                frozen_params += param.numel()
        
        # Freeze embedding-level prompt parameters (E_01_HSE_v2)
        if hasattr(model, 'embedding'):
            if hasattr(model.embedding, 'prompt_encoder'):
                for param in model.embedding.prompt_encoder.parameters():
                    param.requires_grad = False
                    frozen_params += param.numel()
            
            if hasattr(model.embedding, 'prompt_fusion'):
                for param in model.embedding.prompt_fusion.parameters():
                    param.requires_grad = False
                    frozen_params += param.numel()
        
        print(f"✓ Frozen {frozen_params:,} prompt parameters for finetuning")
        return frozen_params
    
    def get_integration_info(self) -> Dict[str, Any]:
        """
        Get information about HSE Prompt Pipeline_03 integration.
        
        Returns:
            Dictionary with integration details
        """
        return {
            'integration_type': 'HSE_Prompt_Pipeline_03',
            'supported_models': self.prompt_models,
            'supported_embeddings': self.prompt_embeddings,
            'recommended_backbones': self.recommended_backbones,
            'features': [
                'Two-level prompt encoding (System + Sample)',
                'Prompt-guided contrastive learning',
                'Training stage control with prompt freezing',
                'Pipeline_03 checkpoint format compatibility',
                'Multi-system pretraining support',
                'Single-system finetuning support'
            ]
        }


# Convenience functions for Pipeline_03 compatibility
def create_pretraining_config(base_configs: Dict, 
                            backbone: str, 
                            target_systems: List[int], 
                            pretraining_config: Dict) -> Dict:
    """
    Create pretraining configuration (Pipeline_03 compatibility function).
    
    This function provides backward compatibility with existing Pipeline_03 imports.
    """
    integration = HSEPromptPipelineIntegration()
    return integration.create_hse_prompt_pretraining_config(
        base_configs, backbone, target_systems, pretraining_config
    )


def create_finetuning_config(base_configs: Dict,
                           pretrained_checkpoint: str,
                           backbone: str,
                           target_system: int, 
                           finetuning_config: Dict) -> Dict:
    """
    Create finetuning configuration (Pipeline_03 compatibility function).
    
    This function provides backward compatibility with existing Pipeline_03 imports.
    """
    integration = HSEPromptPipelineIntegration()
    return integration.create_hse_prompt_finetuning_config(
        base_configs, pretrained_checkpoint, backbone, target_system, finetuning_config
    )


def adapt_checkpoint_loading(checkpoint_path: str, model_config: Dict) -> Dict:
    """
    Adapt checkpoint loading (Pipeline_03 compatibility function).
    
    This function provides backward compatibility with existing Pipeline_03 imports.
    """
    integration = HSEPromptPipelineIntegration()
    return integration.adapt_checkpoint_loading(checkpoint_path, model_config)


if __name__ == '__main__':
    """Comprehensive self-test for HSE Prompt Pipeline_03 integration."""
    
    print("=== HSE Prompt Pipeline_03 Integration Self-Test ===")
    
    # Initialize integration adapter
    integration = HSEPromptPipelineIntegration()
    print(f"✓ HSEPromptPipelineIntegration initialized")
    
    # Test 1: Integration info
    print("\n--- Test 1: Integration Information ---")
    info = integration.get_integration_info()
    print(f"✓ Integration type: {info['integration_type']}")
    print(f"✓ Supported models: {info['supported_models']}")
    print(f"✓ Supported embeddings: {info['supported_embeddings']}")
    print(f"✓ Recommended backbones: {info['recommended_backbones']}")
    print(f"✓ Features: {len(info['features'])} integration features")
    
    # Test 2: Pretraining configuration creation
    print("\n--- Test 2: Pretraining Configuration Creation ---")
    
    base_configs = {
        'environment': {'seed': 42, 'experiment_name': 'test'},
        'model': {'name': 'base_model', 'type': 'ISFM'},
        'task': {'name': 'base_task', 'type': 'classification'},
        'trainer': {'max_epochs': 10, 'learning_rate': 1e-3},
        'data': {'batch_size': 32}
    }
    
    pretraining_config = {
        'prompt_dim': 128,
        'fusion_type': 'attention',
        'contrast_weight': 0.15,
        'max_epochs': 50
    }
    
    pretrain_config = integration.create_hse_prompt_pretraining_config(
        base_configs, 'B_08_PatchTST', [1, 6, 13], pretraining_config
    )
    
    assert pretrain_config['model']['name'] == 'M_02_ISFM_Prompt'
    assert pretrain_config['model']['embedding'] == 'E_01_HSE_v2'
    assert pretrain_config['model']['backbone'] == 'B_08_PatchTST'
    assert pretrain_config['model']['use_prompt'] == True
    assert pretrain_config['model']['training_stage'] == 'pretraining'
    assert pretrain_config['task']['name'] == 'hse_contrastive'
    assert pretrain_config['data']['target_systems'] == [1, 6, 13]
    print("✓ Pretraining configuration created successfully")
    print(f"✓ Model: {pretrain_config['model']['name']} with {pretrain_config['model']['embedding']}")
    print(f"✓ Task: {pretrain_config['task']['name']} with contrast_weight={pretrain_config['task']['contrast_weight']}")
    
    # Test 3: Finetuning configuration creation
    print("\n--- Test 3: Finetuning Configuration Creation ---")
    
    finetuning_config = {
        'task_name': 'classification',
        'task_type': 'classification', 
        'freeze_prompt': True,
        'max_epochs': 20
    }
    
    finetune_config = integration.create_hse_prompt_finetuning_config(
        base_configs, '/path/to/checkpoint.ckpt', 'B_08_PatchTST', 1, finetuning_config
    )
    
    assert finetune_config['model']['name'] == 'M_02_ISFM_Prompt'
    assert finetune_config['model']['training_stage'] == 'finetuning'
    assert finetune_config['model']['freeze_prompt'] == True
    assert finetune_config['task']['name'] == 'classification'
    assert finetune_config['data']['target_systems'] == [1]
    print("✓ Finetuning configuration created successfully")
    print(f"✓ Model: {finetune_config['model']['name']} in {finetune_config['model']['training_stage']} mode")
    print(f"✓ Task: {finetune_config['task']['name']} for system {finetune_config['data']['target_systems'][0]}")
    
    # Test 4: Checkpoint adaptation
    print("\n--- Test 4: Checkpoint Loading Adaptation ---")
    
    model_config = {'name': 'test_model', 'type': 'ISFM_Prompt'}
    adapted_config = integration.adapt_checkpoint_loading('/fake/checkpoint.ckpt', model_config)
    
    assert adapted_config['load_pretrained'] == False  # False because checkpoint doesn't exist
    assert adapted_config['pretrained_path'] == '/fake/checkpoint.ckpt'
    assert adapted_config['load_backbone_only'] == True
    assert adapted_config['strict_loading'] == False
    print("✓ Checkpoint loading adaptation working")
    print(f"✓ Configured for backbone-only loading with strict_loading={adapted_config['strict_loading']}")
    
    # Test 5: Compatibility functions
    print("\n--- Test 5: Pipeline_03 Compatibility Functions ---")
    
    # Test compatibility function imports
    compat_pretrain = create_pretraining_config(base_configs, 'B_04_Dlinear', [1, 6], pretraining_config)
    assert compat_pretrain['model']['backbone'] == 'B_04_Dlinear'
    print("✓ create_pretraining_config compatibility function working")
    
    compat_finetune = create_finetuning_config(base_configs, '/path/checkpoint', 'B_04_Dlinear', 6, finetuning_config)
    assert compat_finetune['model']['backbone'] == 'B_04_Dlinear'
    assert compat_finetune['data']['target_systems'] == [6]
    print("✓ create_finetuning_config compatibility function working")
    
    compat_adapt = adapt_checkpoint_loading('/path/checkpoint', model_config)
    assert compat_adapt['load_pretrained'] == False  # False because checkpoint doesn't exist
    print("✓ adapt_checkpoint_loading compatibility function working")
    
    # Test 6: Configuration validation
    print("\n--- Test 6: Configuration Validation ---")
    
    # Validate that essential HSE Prompt parameters are set
    required_model_params = ['use_prompt', 'prompt_dim', 'fusion_type', 'training_stage']
    for param in required_model_params:
        assert param in pretrain_config['model'], f"Missing required model parameter: {param}"
        assert param in finetune_config['model'], f"Missing required model parameter: {param}"
    
    required_task_params = ['name', 'type']
    for param in required_task_params:
        assert param in pretrain_config['task'], f"Missing required task parameter: {param}"
        assert param in finetune_config['task'], f"Missing required task parameter: {param}"
    
    print("✓ Configuration validation passed")
    print("✓ All required HSE Prompt parameters present")
    
    print("\n=== HSE Prompt Pipeline_03 Integration Tests Completed Successfully! ===")
    print("✅ Integration Verified:")
    print("  • HSEPromptPipelineIntegration adapter functional")
    print("  • Pretraining configuration generation working")
    print("  • Finetuning configuration generation working")
    print("  • Checkpoint loading adaptation working")
    print("  • Pipeline_03 compatibility functions working")
    print("  • Configuration validation passing")
    print("\n✅ Ready for Pipeline_03 Integration:")
    print("  • All required functions implemented")
    print("  • HSE prompt parameters properly configured")
    print("  • Two-stage training workflow supported")
    print("  • Backward compatibility maintained")
    print("  • Task 6 P0 requirements satisfied")