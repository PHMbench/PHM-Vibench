#!/usr/bin/env python3
"""
Test script to validate the simplified model factory integration.
"""

import sys
import os

# Add src to Python path
sys.path.insert(0, 'src')

def test_model_factory():
    """Test model factory integration."""
    try:
        from configs.config_utils import load_config
        from model_factory.model_factory import create_model

        # Load the simplified configuration
        config_path = "configs/demo/Simplified_Prompt/hse_prompt_demo.yaml"
        config = load_config(config_path)

        print("✅ Configuration loaded successfully!")
        print(f"🏭 Attempting to create model: {config.model.type}")

        # Try to create the model
        model = create_model(config.model, metadata=None)

        print(f"✅ Model created successfully!")
        print(f"📊 Model type: {type(model).__name__}")

        # Test model info
        if hasattr(model, 'get_model_info'):
            info = model.get_model_info()
            print(f"📈 Model parameters: {info.get('total_parameters', 'unknown'):,}")
            print(f"🎯 Use prompt: {info.get('use_prompt', 'unknown')}")
            if 'prompt_config' in info:
                prompt_info = info['prompt_config']
                print(f"🔗 Prompt dim: {prompt_info.get('prompt_dim', 'unknown')}")
                print(f"📝 Combination: {prompt_info.get('prompt_combination', 'unknown')}")

        print("🎉 Simplified ISFM_Prompt model is ready for training!")
        return 0

    except Exception as e:
        print(f"❌ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(test_model_factory())