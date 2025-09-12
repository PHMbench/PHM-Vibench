#!/usr/bin/env python3
"""
Test script for Flow task registration validation.
This script tests if FlowPretrainTask is properly registered with the task factory.
"""

def test_flow_task_registration():
    """Test Flow task registration."""
    print("🧪 Testing Flow Task Registration")
    print("="*50)
    
    try:
        # Import the task to trigger registration
        from src.task_factory.task.pretrain.flow_pretrain import FlowPretrainTask
        print("✅ FlowPretrainTask imported successfully")
        
        # Check if it's registered
        from src.task_factory import TASK_REGISTRY
        
        print(f"\n📋 Registry Status:")
        try:
            task_cls = TASK_REGISTRY.get("pretrain.flow_pretrain")
            print(f"✅ Task registered as: {task_cls}")
        except KeyError:
            print("❌ Task not found in registry")
            
        # Test with alternative key format
        try:
            task_cls = TASK_REGISTRY.get("flow_pretrain.pretrain")  # Different order
            print(f"✅ Alternative registration found: {task_cls}")
        except KeyError:
            print("❌ Alternative key not found either")
            
        # Show what's actually in the registry
        print(f"\n🔍 Available tasks in registry: {list(TASK_REGISTRY.available())}")
        
    except Exception as e:
        print(f"❌ Failed to import or test: {e}")
        
    print("\n" + "="*50)

def test_task_factory_resolution():
    """Test task factory module resolution."""
    print("🏭 Testing Task Factory Resolution")
    print("="*50)
    
    try:
        from argparse import Namespace
        from src.task_factory.task_factory import resolve_task_module
        
        # Create args for Flow task
        args_task = Namespace()
        args_task.name = "flow_pretrain"
        args_task.type = "pretrain"
        
        module_path = resolve_task_module(args_task)
        print(f"📍 Resolved module path: {module_path}")
        
        # Try importing the resolved module
        import importlib
        task_module = importlib.import_module(module_path)
        print(f"✅ Module imported successfully: {task_module}")
        
        # Check if it has a 'task' attribute (old style) or registered class
        if hasattr(task_module, 'task'):
            print(f"✅ Old-style 'task' class found: {task_module.task}")
        elif hasattr(task_module, 'FlowPretrainTask'):
            print(f"✅ FlowPretrainTask class found: {task_module.FlowPretrainTask}")
        else:
            print("❌ No suitable task class found in module")
            
    except Exception as e:
        print(f"❌ Task factory resolution failed: {e}")
        
    print("\n" + "="*50)

if __name__ == "__main__":
    test_flow_task_registration()
    test_task_factory_resolution()