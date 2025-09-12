#!/usr/bin/env python3
"""
End-to-end test for Flow pretraining implementation.
This script tests the complete Flow model pipeline from configuration loading to training.
"""

import torch
import torch.nn as nn
from argparse import Namespace
import numpy as np
import tempfile
import os

def create_mock_metadata():
    """Create mock metadata for testing."""
    class MockMetadata:
        def __init__(self):
            # Create a simple DataFrame-like structure
            import pandas as pd
            self.df = pd.DataFrame({
                'Dataset_id': [1, 2, 3, 4],
                'Domain_id': [1, 1, 2, 2],
                'file_id': ['file1', 'file2', 'file3', 'file4'],
                'Label': [0, 1, 0, 1]  # Capital L for Label column
            })
    return MockMetadata()

def create_test_args():
    """Create test arguments for Flow pretraining."""
    
    # Data args
    args_data = Namespace()
    args_data.batch_size = 8
    args_data.sequence_length = 256
    args_data.channels = 1
    
    # Model args  
    args_model = Namespace()
    args_model.type = "ISFM"  # Model factory type
    args_model.name = "M_04_ISFM_Flow"
    args_model.hidden_dim = 128
    args_model.time_dim = 32
    args_model.condition_dim = 32
    args_model.use_conditional = True
    args_model.sigma_min = 0.001
    args_model.sigma_max = 1.0
    
    # Task args
    args_task = Namespace()
    args_task.name = "flow_pretrain"
    args_task.type = "pretrain"
    args_task.lr = 1e-3
    args_task.weight_decay = 1e-5
    args_task.num_steps = 20  # Small for testing
    args_task.use_contrastive = False  # Start with Flow-only
    args_task.enable_visualization = False
    args_task.track_memory = False
    args_task.track_gradients = False
    
    # Trainer args
    args_trainer = Namespace()
    args_trainer.gpus = 0  # CPU-only for testing
    args_trainer.precision = 32
    args_trainer.gradient_clip_val = 1.0
    args_trainer.max_epochs = 2
    args_trainer.log_every_n_steps = 10
    
    # Environment args
    args_environment = Namespace()
    args_environment.seed = 42
    
    return args_data, args_model, args_task, args_trainer, args_environment

def test_flow_model_creation():
    """Test M_04_ISFM_Flow model creation."""
    print("🧪 Testing Flow Model Creation")
    print("="*50)
    
    try:
        from src.model_factory import build_model
        
        args_data, args_model, args_task, args_trainer, args_environment = create_test_args()
        metadata = create_mock_metadata()
        
        # Create Flow model
        model = build_model(
            args=args_model,
            metadata=metadata
        )
        
        print(f"✅ Model created successfully: {type(model)}")
        print(f"   Model name: {args_model.name}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        batch_size = 4
        seq_len = args_data.sequence_length
        channels = args_data.channels
        
        x = torch.randn(batch_size, seq_len, channels)
        file_ids = ['file1', 'file2', 'file3', 'file4']
        
        with torch.no_grad():
            outputs = model(x=x, file_ids=file_ids, return_loss=True)
            
        print(f"✅ Forward pass successful")
        print(f"   Output keys: {list(outputs.keys())}")
        
        # Check if we have expected outputs
        expected_keys = ['flow_loss', 'velocity', 'x_original']
        for key in expected_keys:
            if key in outputs:
                print(f"   ✅ {key}: {outputs[key].shape}")
            else:
                print(f"   ⚠️  Missing key: {key}")
                
        return model, True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, False

def test_flow_task_creation():
    """Test FlowPretrainTask creation."""
    print("\n🧪 Testing Flow Task Creation")
    print("="*50)
    
    try:
        from src.task_factory import build_task
        
        # First create the model
        model, model_success = test_flow_model_creation()
        if not model_success:
            print("❌ Cannot test task without model")
            return None, False
            
        args_data, args_model, args_task, args_trainer, args_environment = create_test_args()
        metadata = create_mock_metadata()
        
        # Create Flow task
        task = build_task(
            args_task=args_task,
            network=model,
            args_data=args_data,
            args_model=args_model,
            args_trainer=args_trainer,
            args_environment=args_environment,
            metadata=metadata
        )
        
        print(f"✅ Task created successfully: {type(task)}")
        print(f"   Task name: {args_task.name}")
        print(f"   Task type: {args_task.type}")
        
        # Test task forward pass
        batch = {
            'x': torch.randn(4, args_data.sequence_length, args_data.channels),
            'file_id': ['file1', 'file2', 'file3', 'file4']
        }
        
        with torch.no_grad():
            task_outputs = task.forward(batch)
            
        print(f"✅ Task forward pass successful")
        print(f"   Task output keys: {list(task_outputs.keys())}")
        
        return task, True
        
    except Exception as e:
        print(f"❌ Task creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, False

def test_flow_generation():
    """Test Flow sample generation."""
    print("\n🧪 Testing Flow Generation")
    print("="*50)
    
    try:
        # Create task
        task, task_success = test_flow_task_creation()
        if not task_success:
            print("❌ Cannot test generation without task")
            return False
            
        # Test sample generation
        batch_size = 4
        num_steps = 10  # Small for testing
        
        print(f"🎲 Generating {batch_size} samples with {num_steps} steps...")
        
        # Test unconditional generation
        with torch.no_grad():
            samples = task.generate_samples(
                batch_size=batch_size,
                mode='unconditional',
                num_steps=num_steps
            )
            
        print(f"✅ Unconditional generation successful")
        print(f"   Generated samples shape: {samples.shape}")
        print(f"   Sample statistics: mean={samples.mean():.4f}, std={samples.std():.4f}")
        
        # Test conditional generation
        file_ids = ['file1', 'file2', 'file3', 'file4']
        with torch.no_grad():
            cond_samples = task.generate_samples(
                batch_size=batch_size,
                file_ids=file_ids,
                mode='conditional',
                num_steps=num_steps
            )
            
        print(f"✅ Conditional generation successful")  
        print(f"   Conditional samples shape: {cond_samples.shape}")
        print(f"   Sample statistics: mean={cond_samples.mean():.4f}, std={cond_samples.std():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_step():
    """Test training step execution."""
    print("\n🧪 Testing Training Step")
    print("="*50)
    
    try:
        # Create task
        task, task_success = test_flow_task_creation()
        if not task_success:
            print("❌ Cannot test training without task")
            return False
            
        # Create training batch
        batch = {
            'x': torch.randn(4, 256, 1),
            'file_id': ['file1', 'file2', 'file3', 'file4'],
            'label': torch.tensor([0, 1, 0, 1])
        }
        
        batch_idx = 0
        
        print("🏋️ Testing training step...")
        
        # Test training step
        loss = task.training_step(batch, batch_idx)
        
        print(f"✅ Training step successful")
        print(f"   Loss: {loss:.6f}")
        print(f"   Loss shape: {loss.shape}")
        print(f"   Loss requires grad: {loss.requires_grad}")
        
        # Test backward pass
        print("⬅️  Testing backward pass...")
        loss.backward()
        print("✅ Backward pass successful")
        
        # Check gradients
        grad_count = 0
        total_params = 0
        for name, param in task.named_parameters():
            total_params += 1
            if param.grad is not None:
                grad_count += 1
                
        print(f"✅ Gradients computed: {grad_count}/{total_params} parameters")
        
        return True
        
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_loading():
    """Test loading Flow config file."""
    print("\n🧪 Testing Config Loading")
    print("="*50)
    
    try:
        config_path = "configs/demo/Pretraining/Flow/flow_pretrain_basic.yaml"
        
        if not os.path.exists(config_path):
            print(f"❌ Config file not found: {config_path}")
            return False
            
        # Try to load config using PHM-Vibench's config system
        try:
            from src.configs import load_config
            config = load_config(config_path)
            print(f"✅ Config loaded with load_config: {type(config)}")
        except Exception as e:
            print(f"⚠️  load_config failed: {e}")
            
            # Fallback: load with PyYAML
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✅ Config loaded with yaml: {type(config)}")
            
        print("📋 Config contents:")
        for section in ['data', 'model', 'task', 'trainer']:
            if section in config:
                print(f"   {section}: {len(config[section])} parameters")
            else:
                print(f"   {section}: missing")
                
        return True
        
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False

def main():
    """Run all end-to-end tests."""
    print("🚀 PHM-Vibench Flow Implementation End-to-End Test")
    print("="*60)
    
    test_results = []
    
    # Test 1: Config loading
    test_results.append(("Config Loading", test_config_loading()))
    
    # Test 2: Model creation
    test_results.append(("Model Creation", test_flow_model_creation()[1]))
    
    # Test 3: Task creation  
    test_results.append(("Task Creation", test_flow_task_creation()[1]))
    
    # Test 4: Generation
    test_results.append(("Generation", test_flow_generation()))
    
    # Test 5: Training step
    test_results.append(("Training Step", test_training_step()))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name:<20} {status}")
        if result:
            passed += 1
            
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Flow implementation is ready.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)