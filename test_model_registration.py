#!/usr/bin/env python3
"""
Test script to validate model registration without requiring PyTorch.
"""

import sys
import os

# Add current directory to Python path for proper imports
sys.path.insert(0, '.')

def test_model_registration():
    """Test model registration."""
    try:
        from src.model_factory.ISFM_Prompt.__init__ import PROMPT_MODEL_REGISTRY

        print("✅ ISFM_Prompt module imported successfully!")

        # Check if our model is registered
        if 'M_02_ISFM_Prompt' in PROMPT_MODEL_REGISTRY:
            print("✅ M_02_ISFM_Prompt model is registered!")
            model_class = PROMPT_MODEL_REGISTRY['M_02_ISFM_Prompt']
            print(f"📋 Model class: {model_class.__name__}")

            # Try to get basic info about the model
            if hasattr(model_class, '__doc__'):
                doc = model_class.__doc__
                if doc:
                    print(f"📝 Model description: {doc.strip().split('.')[0]}...")

            # Check if the embedding is registered
            from src.model_factory.ISFM_Prompt.embedding.__init__ import PROMPT_EMBEDDINGS
            if 'HSE_prompt' in PROMPT_EMBEDDINGS:
                print("✅ HSE_prompt embedding is registered!")
                embedding_class = PROMPT_EMBEDDINGS['HSE_prompt']
                print(f"🏗️  Embedding class: {embedding_class.__name__}")
            else:
                print("❌ HSE_prompt embedding not found!")
                return 1

            # Check if the simple prompt encoder is available
            from src.model_factory.ISFM_Prompt.components.__init__ import __all__ as component_all
            if 'SimpleSystemPromptEncoder' in component_all:
                print("✅ SimpleSystemPromptEncoder is available!")
            else:
                print("❌ SimpleSystemPromptEncoder not found!")
                return 1

            print("\n🎉 All simplified components are properly registered!")
            print("🚀 The model factory should be able to create the simplified ISFM_Prompt model!")
            return 0

        else:
            print("❌ M_02_ISFM_Prompt model not found in registry!")
            print(f"📋 Available models: {list(PROMPT_MODEL_REGISTRY.keys())}")
            return 1

    except Exception as e:
        print(f"❌ Error checking model registration: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(test_model_registration())