#!/usr/bin/env python3
"""
Test script to validate the simplified HSE_prompt configuration
without requiring PyTorch installation.
"""

import sys
import os

# Add src to Python path
sys.path.insert(0, 'src')

def test_config():
    """Test configuration loading."""
    try:
        from configs.config_utils import load_config

        # Load the simplified configuration
        config_path = "configs/demo/Simplified_Prompt/hse_prompt_demo.yaml"
        config = load_config(config_path)

        print("✅ Configuration loaded successfully!")
        print(f"📁 Model name: {config.model.name}")
        print(f"🔧 Model type: {config.model.type}")
        print(f"🏗️  Embedding: {config.model.embedding}")
        print(f"📋 Task name: {config.task.name}")
        print(f"📋 Task type: {config.task.type}")
        print(f"🎯 Use prompt: {config.model.use_prompt}")
        print(f"🔗 Prompt combination: {config.model.prompt_combination}")

        # Check required fields
        required_fields = {
            'data': ['data_dir', 'metadata_file'],
            'model': ['name', 'type'],
            'task': ['name', 'type']
        }

        print("\n🔍 Validating required fields:")
        all_valid = True

        for section, fields in required_fields.items():
            section_obj = getattr(config, section)
            print(f"  📂 {section}:")
            for field in fields:
                if hasattr(section_obj, field):
                    value = getattr(section_obj, field)
                    print(f"    ✅ {field}: {value}")
                else:
                    print(f"    ❌ {field}: MISSING")
                    all_valid = False

        if all_valid:
            print("\n🎉 All required fields are present!")
            print("🚀 Configuration is ready for use!")
        else:
            print("\n❌ Some required fields are missing!")
            return 1

        return 0

    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(test_config())